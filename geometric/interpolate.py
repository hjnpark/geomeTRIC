import os, sys, json, copy, tempfile

from .params import parse_interpolate_args, IntpParams
from .prepare import get_molecule_engine
from .nifty import ang2bohr
from .molecule import EqualSpacing
from .internal import (
    CartesianCoordinates,
    PrimitiveInternalCoordinates,
    DelocalizedInternalCoordinates,
)
import numpy as np


class Interpolate:
    def __init__(self, params, M, engine):

        self.dir = os.getcwd()
        self.engine = engine

        if M.elem == 0:
            raise RuntimeError(
                "xyz file format is not correct. Make sure there is no space between the two structures in the xyz file."
            )

        try:
            if len(M) != 2:
                print(
                    "The provided xyz file has more than two structures. The first and last geometries will be used."
                )
        except:
            raise RuntimeError(
                "Check your xyz file to make sure the two frames have the same atom number"
            )

        self.elem = M.elem
        self.params = params

        # Coordinates in Angstrom
        self.M = M

        self.interpolated_dict = {}
        self.Energy_dict = {}
        self.coordsys_dict = {
            "cart": (CartesianCoordinates, False, False),
            "prim": (PrimitiveInternalCoordinates, True, False),
            "dlc": (DelocalizedInternalCoordinates, True, False),
            "hdlc": (DelocalizedInternalCoordinates, False, True),
            "tric-p": (PrimitiveInternalCoordinates, False, False),
            "tric": (DelocalizedInternalCoordinates, False, False),
        }
        M_reac = self.M[0]
        M_prod = self.M[-1]
        M_reac.build_topology()
        M_prod.build_topology()

        if len(M_reac.molecules) >= len(M_reac.molecules):
            self.M_ini = copy.deepcopy(M_reac)
            self.M_fin = copy.deepcopy(M_prod)

            #self.M_fin = copy.deepcopy(M_reac)
            #self.M_ini = copy.deepcopy(M_prod)
        else:
            self.M_ini = copy.deepcopy(M_prod)
            self.M_fin = copy.deepcopy(M_reac)

            #self.M_ini = copy.deepcopy(M_reac)
            #self.M_fin = copy.deepcopy(M_prod)

        self.simple_xyz = self.M[0]
        self.mix_xyz = self.M[0]
        self.fwd_M = self.M[0]
        self.bwd_M = self.M[0]

        self.reac = self.M_ini.xyzs[0].flatten() * ang2bohr
        self.prod = self.M_fin.xyzs[0].flatten() * ang2bohr

    def simple_interpolate(self, PRIM = None):
         PRIMIC, connect, addcart = self.coordsys_dict["prim"]
         if PRIM is None:
            PRIM = PRIMIC(
                self.M,
                build=True,
                connect=connect,
                addcart=addcart,
                constraints=None,
            )
         else:
            PRIM = PRIM

         for ic in self.params.coordsys:
             CoordClass, connect, addcart = self.coordsys_dict[ic.lower()]

             IC = CoordClass(
                 self.M,
                 build=True,
                 connect=connect,
                 addcart=addcart,
                 Prims=PRIM,
                 constraints=None,
             )

             dq = IC.calcDiff(self.prod, self.reac)

             nDiv = self.params.frames

             curr_coords = self.reac.copy()

             coord_list = []

             cn_info = {"Forward IC": []}
             for i in range(nDiv):
                 # ------------------Geting condition numbers here-------------------------

                 coord_list.append(curr_coords)

                 IC.build_dlc_0(curr_coords)
                 dq = IC.calcDiff(self.prod, curr_coords)
                 new_coords = IC.newCartesian(curr_coords, dq / nDiv)


                 print("getting GMatrixes from IC objects")
                 G_f_ini = IC.GMatrix(new_coords)
                 eig_f_ini, vec_f_ini = np.linalg.eigh(G_f_ini)

                 print("\n-------------------------------------------------------------")
                 con_num_if = np.real(eig_f_ini[-1] / eig_f_ini[0])
                 print("Condition number of forward %f" % con_num_if)
                 cn_info["Forward IC"].append(con_num_if)

                 curr_coords = new_coords.copy()

                 nDiv -= 1

             self.simple_xyz.xyzs = [
                 coords.reshape(-1, 3) / ang2bohr for coords in coord_list
             ]

             rough_M = copy.deepcopy(self.simple_xyz)

             json_str = json.dumps(cn_info, indent=4)
             with open("simple_cn_info.json", "w") as f:
                 f.write(json_str)
             print(
                 "Error in final interpolated vs. product structure (%s):" % ic,
                 np.linalg.norm(curr_coords - self.prod),
             )
             self.interpolated_dict["simple_" + ic] = np.array(coord_list)
             equal_spaced_M = EqualSpacing(self.simple_xyz)[
                 np.array(
                     [
                         int(round(i))
                         for i in np.linspace(
                             0, len(self.simple_xyz) - 1, self.params.frames
                         )
                     ]
                 )
             ]

             xyz_dir = os.path.join(self.dir, "interpolated")
             if not os.path.exists(xyz_dir):
                 os.makedirs(xyz_dir)
             self.simple_xyz.write(
                 os.path.join(xyz_dir, "simple_interpolated_%s.xyz" % ic)
             )
             equal_spaced_M.write(
                 os.path.join(xyz_dir, "simple_smoothed_interpolated_%s.xyz" % ic)
             )


    def mixed_interpolate(self, PRIM=None):
        PRIMIC, connect, addcart = self.coordsys_dict["prim"]
        if PRIM is None:
            ini_PRIM = PRIMIC(
                self.M,
                build=True,
                connect=connect,
                addcart=addcart,
                constraints=None,
            )
        else:
            ini_PRIM = PRIM

        M_two_ends = copy.deepcopy(self.M)
        M_reac_mid = copy.deepcopy(self.M)
        M_prod_mid = copy.deepcopy(self.M)

        #PRIM_b = PRIMIC(
        #    self.M_fin,
        #    build=True,
        #    connect=connect,
        #    addcart=addcart,
        #    constraints=None,
        #)
        for ic in self.params.coordsys:

            CoordClass, connect, addcart = self.coordsys_dict[ic.lower()]
            IC = CoordClass(
                M_two_ends,
                build=True,
                connect=connect,
                addcart=addcart,
                Prims=ini_PRIM,
                constraints=None,
            )

            #IC_b = CoordClass(
            #    self.M_fin,
            #    build=True,
            #    connect=connect,
            #    addcart=addcart,
            #    Prims=PRIM_b,
            #    constraints=None,
            #)


            #dq_f = IC_f.calcDiff(self.prod, self.reac)
            #dq_b = IC_b.calcDiff(self.reac, self.prod)

            nDiv = self.params.frames

            curr_coords_f = self.reac.copy()
            curr_coords_b = self.prod.copy()

            coord_list_f = []
            coord_list_b = []

            cn_info = {"Forward IC": [], "Backward IC": []}

            nDiv += nDiv % 2

            PRIM_most_Internals = ini_PRIM

            for i in range(nDiv//2):
                print(i)
                # ------------------Geting condition numbers here-------------------------
                coord_list_f.append(curr_coords_f)
                coord_list_b.append(curr_coords_b)

                IC.build_dlc_0(curr_coords_f)
                dq_f = IC.calcDiff(curr_coords_b, curr_coords_f)
                new_coords_f = IC.newCartesian(curr_coords_f, dq_f / nDiv)

                IC.build_dlc_0(curr_coords_b)
                dq_b = IC.calcDiff(curr_coords_f, curr_coords_b)
                new_coords_b = IC.newCartesian(curr_coords_b, dq_b / nDiv)

                #print("getting GMatrixes from IC objects")
                #G_f_ini = IC.GMatrix(new_coords_f)
                #eig_f_ini, vec_f_ini = np.linalg.eigh(G_f_ini)

                #G_b_ini = IC.GMatrix(new_coords_b)
                #eig_b_ini, vec_b_ini = np.linalg.eigh(G_b_ini)

                #print("\n-------------------------------------------------------------")
                #con_num_if = np.real(eig_f_ini[-1] / eig_f_ini[0])
                #con_num_ib = np.real(eig_b_ini[-1] / eig_b_ini[0])
                #print("Condition number of forward %f" % con_num_if)
                #print("Condition number of backward %f" % con_num_ib)
                #cn_info["Forward IC"].append(con_num_if)
                #cn_info["Backward IC"].append(con_num_ib)

                curr_coords_f = new_coords_f.copy()
                curr_coords_b = new_coords_b.copy()

                M_two_ends.xyzs = [self.reac.reshape(-1,3)/ang2bohr,
                                   curr_coords_f.reshape(-1,3)/ang2bohr,
                                   curr_coords_b.reshape(-1,3)/ang2bohr,
                                   self.prod.reshape(-1,3)/ang2bohr]
                #M_prod_mid.xyzs = [curr_coords_f.reshape(-1,3)/ang2bohr, self.prod.reshape(-1,3)/ang2bohr]
                #M_reac_mid.xyzs = [self.reac.reshape(-1,3)/ang2bohr, curr_coords_b.reshape(-1,3)/ang2bohr]
                #self.M_ini.xyzs = [curr_coords_f.reshape(-1,3)/ang2bohr
                #self.M_fin.xyzs = [curr_coords_b.reshape(-1,3)/ang2bohr]

                new_PRIM_two_ends = PRIMIC(
                    M_two_ends,
                    build=True,
                    connect=connect,
                    addcart=addcart,
                    constraints=None,
                )
                #new_PRIM_prod_mid = PRIMIC(
                #    M_prod_mid,
                #    build=True,
                #    connect=connect,
                #    addcart=addcart,
                #    constraints=None,
                #)

                #new_PRIM_reac_mid = PRIMIC(
                #    M_reac_mid,
                #    build=True,
                #    connect=connect,
                #    addcart=addcart,
                #    constraints=None,
                #)

                #PRIMS_list = [(len(new_PRIM_two_ends.Internals), new_PRIM_two_ends),
                #              (len(new_PRIM_reac_mid.Internals), new_PRIM_reac_mid),
                #              (len(new_PRIM_prod_mid.Internals), new_PRIM_prod_mid)]
                #PRIMS_list.sort(key=lambda  a: a[0], reverse=True)

                #print(PRIMS_list)
                if len(new_PRIM_two_ends.Internals) > len(PRIM_most_Internals.Internals):
                    PRIM_most_Internals = new_PRIM_two_ends
                    #self.mixed_interpolate(new_PRIM)

                #new_Internals = len(PRIM.Internals)
                #PRIM_f = PRIMIC(
                #    self.M_ini,
                #    build=True,
                #    connect=connect,
                #    addcart=addcart,
                #    constraints=None,
                #)

                #PRIM_b = PRIMIC(
                #    self.M_fin,
                #    build=True,
                #    connect=connect,
                #    addcart=addcart,
                #    constraints=None,
                #)

                #IC_f.Prims = PRIM_f
                #IC_b.Prims = PRIM_b
                #if new_Internals > ini_Internals:
                #    print(new_Internals, ini_Internals)
                #    IC.Prims = PRIM

                nDiv -= 2

            print(len(PRIM_most_Internals.Internals), len(ini_PRIM.Internals))
            #if len(PRIM_most_Internals.Internals) == len(ini_PRIM.Internals):
            #    self.simple_interpolate(PRIM_most_Internals)
            #else:
            #    self.mixed_interpolate(PRIM_most_Internals)
            self.simple_interpolate(PRIM_most_Internals)
            coord_list = coord_list_f + coord_list_b[::-1]

            self.simple_xyz.xyzs = [
                coords.reshape(-1, 3) / ang2bohr for coords in coord_list
            ]

            rough_M = copy.deepcopy(self.simple_xyz)

            json_str = json.dumps(cn_info, indent=4)
            with open("mixed_cn_info.json", "w") as f:
                f.write(json_str)
            print(
                "Difference in forward vs. backward interpolation (%s):" % ic,
                np.linalg.norm(curr_coords_f - curr_coords_b),
            )
            self.interpolated_dict["mixed_" + ic] = np.array(coord_list)
            equal_spaced_M = EqualSpacing(self.simple_xyz)[
                np.array(
                    [
                        int(round(i))
                        for i in np.linspace(
                            0, len(self.simple_xyz) - 1, self.params.frames
                        )
                    ]
                )
            ]

            xyz_dir = os.path.join(self.dir, "interpolated")
            if not os.path.exists(xyz_dir):
                os.makedirs(xyz_dir)
            self.simple_xyz.write(
                os.path.join(xyz_dir, "mixed_interpolated_%s.xyz" % ic)
            )
            equal_spaced_M.write(
                os.path.join(xyz_dir, "mixed_smoothed_interpolated_%s.xyz" % ic)
            )

    def calculate_energies(self, interpolation_type=None):
        print("Calculating Energies...")
        for ic, geos in self.interpolated_dict.items():
            e_list = []
            for xyz in geos:
                E = self.engine.calc_new(xyz.flatten(), tempfile.mktemp())["energy"]
                e_list.append(E)
            self.Energy_dict[ic] = e_list
        json_str = json.dumps(self.Energy_dict, indent=4)
        with open(
            os.path.join(self.dir, "%s_energies.json" % interpolation_type), "w"
        ) as f:
            f.write(json_str)

    def plot(self):
        import matplotlib.pyplot as plt

        coordsys_dict = {
            "simple_cart": ["Cartesian", ".--"],
            "simple_prim": ["Primitive I.C.", "2--"],
            "simple_dlc": ["Delocalized I.C.", "1--"],
            "simple_hdlc": ["HDLC", "+--"],
            "simple_tric": ["TRIC", "d--"],
            "simple_tric-p": ["TRICP", "X--"],
            "fwdbwd_cart": ["FB-Cartesian", "|--"],
            "fwdbwd_prim": ["FB-Primitive I.C.", "4--"],
            "fwdbwd_dlc": ["FB-Delocalized I.C.", "3--"],
            "fwdbwd_hdlc": ["FB-HDLC", "h--"],
            "fwdbwd_tric": ["FB-TRIC", "<--"],
            "fwdbwd_tric-p": ["FB-TRICP", "^--"],
            "mix_cart": ["M-Cartesian", "|--"],
            "mix_prim": ["M-Primitive I.C.", "4--"],
            "mix_dlc": ["M-Delocalized I.C.", "3--"],
            "mix_hdlc": ["M-HDLC", "h--"],
            "mix_tric": ["M-TRIC", "<--"],
            "mix_tric-p": ["M-TRICP", "^--"],
            "geodestic": ["Geodestic", "o--"],
        }
        label = []
        e_array = []
        for ic, es in self.Energy_dict.items():
            label.append(ic)
            e_array.append(es)

        x = np.arange(len(e_array[0]))

        plt.figure(figsize=(12, 8))
        for i in range(len(e_array)):
            plt.plot(
                x,
                e_array[i],
                coordsys_dict[label[i]][-1],
                label=coordsys_dict[label[i]][0],
            )

        plt.legend()
        plt.xlabel("Frame")
        plt.ylabel("Energy (Hartree)")
        plt.savefig(os.path.join(self.dir, "energy_plot.png"), bbox_inches="tight")


def main():
    args_dict = parse_interpolate_args(sys.argv[1:])
    args_dict["interpolation"] = True
    params = IntpParams(**args_dict)
    M, engine = get_molecule_engine(**args_dict)

    TRIC = Interpolate(params, M, engine)
    if params.type == 'simple':
        TRIC.simple_interpolate()
    else:
        TRIC.mixed_interpolate()
    # TRIC.calculate_energies("simple")
    # TRIC.fwd_bwd_interpolate()
    # TRIC.mix_interpolate()
    # TRIC.calculate_energies("mix")
    # TRIC.calculate_energies("fwd_bwd")

    # TRIC.plot()
    print("Done!")


if __name__ == "__main__":
    main()
