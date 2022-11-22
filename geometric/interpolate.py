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

        self.reac = self.M_ini.xyzs[0].flatten() * ang2bohr
        self.prod = self.M_fin.xyzs[0].flatten() * ang2bohr
        #self.fwd_M = self.M[0]
        #self.bwd_M = self.M[0]
        #self.forward_coord_list = []
        #self.backward_coord_list = []
        self.PRIMs = None

    def interpolate(self):
        PRIMIC, connect, addcart = self.coordsys_dict["tric-p"]
        for ic in self.params.coordsys:

            M = copy.deepcopy(self.M)
            nDiv = self.params.frames
            total_frames = self.params.frames
            #forward_coords = self.forward_coord_list.copy()
            #backward_coords = self.backward_coord_list.copy()
            curr_coords = self.reac.copy()
            coord_list = []
            CoordClass, connect, addcart = self.coordsys_dict[ic.lower()]
            cn_info = {"Forward IC": []}
            IC = CoordClass(
                M,
                build=True,
                connect=connect,
                addcart=addcart,
                constraints=None,
            )

            #M.xyzs = np.append(forward_coords, backward_coords, axis = 0)

            #new_PRIM = PRIMIC(
            #    M,
            #    build=True,
            #    connect=connect,
            #    addcart=addcart,
            #    constraints=None,
            #    warn=False,
            #)
            #print("PRIMS from all coords", len(new_PRIM.Internals))
            #IC.Prims = new_PRIM
            IC.Prims = self.PRIMs
            #map_size = int(total_frames * 0.05)

            for i in range(nDiv):
                #if i < map_size*2 :
                #    M.xyzs =forward_coords[:map_size*2] #np.append(forward_coords[i:i+map_size], backward_coords[i:i+map_size], axis = 0)
                #    print("1", i)
                ##elif i == total_frames -1 :
                #    #M.xyzs =backward_coords[i-map_size:i]#np.append(forward_coords[i-map_size:i], backward_coords[i-map_size:i], axis = 0)
                ##elif i < map_size:
                ##    M.xyzs = np.append(forward_coords[0:i + map_size], backward_coords[0:i + map_size], axis=0)
                #elif total_frames-1 < i + map_size*2:
                #    print("2", i)
                #    M.xyzs = backward_coords[-map_size*2:]#np.append(forward_coords[i-map_size:-1], backward_coords[i-map_size:-1], axis=0)
                #else:
                #    print("3", i)
                #    M.xyzs = np.append(forward_coords[i-map_size:i+map_size], backward_coords[i-map_size:i+map_size], axis = 0)
                #M.comms = ['comms %i' for i in range(len(M.xyzs))]
                #print(len(M.xyzs))
                #print(len(M.comms))
                #new_PRIM = PRIMIC(
                #    M,
                #    build=True,
                #    connect=connect,
                #    addcart=addcart,
                #    constraints=None,
                #    warn=False,
                #)

                # ------------------Geting condition numbers here-------------------------

                coord_list.append(curr_coords)

                IC.build_dlc_0(curr_coords)
                dq = IC.calcDiff(self.prod, curr_coords)
                new_coords = IC.newCartesian(curr_coords, dq / nDiv)


                #print("getting GMatrixes from IC objects")
                #G_f_ini = IC.GMatrix(new_coords)
                #eig_f_ini, vec_f_ini = np.linalg.eigh(G_f_ini)

                #print("\n-------------------------------------------------------------")
                #con_num_if = np.real(eig_f_ini[-1] / eig_f_ini[0])
                #print("Condition number of forward %f" % con_num_if)
                #cn_info["Forward IC"].append(con_num_if)

                curr_coords = new_coords.copy()

                nDiv -= 1

            self.simple_xyz.xyzs = [
                coords.reshape(-1, 3) / ang2bohr for coords in coord_list
            ]

            rough_M = copy.deepcopy(self.simple_xyz)

            #json_str = json.dumps(cn_info, indent=4)
            #with open("simple_cn_info.json", "w") as f:
            #    f.write(json_str)
            print(
                "Error in final interpolated vs. product structure (%s):" % ic,
                np.linalg.norm(curr_coords - self.prod),
            )
            self.interpolated_dict["simple_" + ic] = np.array(coord_list)

            xyz_dir = os.path.join(self.dir, "interpolated")
            if not os.path.exists(xyz_dir):
                os.makedirs(xyz_dir)
            self.simple_xyz.write(
                os.path.join(xyz_dir, "interpolated_%s.xyz" % ic)
            )


    def collect_PRIMs(self):
        print("Collecting Primitive Internal Coordinates...")
        PRIMIC, connect, addcart = self.coordsys_dict["tric-p"]
        #if self.PRIMS is None:
        #    ini_PRIM = PRIMIC(
        #        self.M,
        #        build=True,
        #        connect=connect,
        #        addcart=addcart,
        #        constraints=None,
        #    )
        #else:
        #    ini_PRIM = self.PRIMS

        M = copy.deepcopy(self.M)

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
                M,
                build=True,
                connect=connect,
                addcart=addcart,
                constraints=None,
            )

            nDiv = self.params.frames

            curr_coords_f = copy.deepcopy(self.reac)
            curr_coords_b = copy.deepcopy(self.prod)
            reac = copy.deepcopy(self.reac)
            prod = copy.deepcopy(self.prod)

            #coord_list_f = []
            #coord_list_b = []

            #cn_info = {"Forward IC": [], "Backward IC": []}

            #nDiv += nDiv % 2

            PRIM_most_Internals = IC.Prims

            for i in range(nDiv):
                # ------------------Geting condition numbers here-------------------------
                #coord_list_f.append(curr_coords_f.reshape(-1,3)/ang2bohr)
                #coord_list_b.append(curr_coords_b.reshape(-1,3)/ang2bohr)

                IC.build_dlc_0(curr_coords_f)
                dq_f = IC.calcDiff(prod, curr_coords_f)

                new_coords_f = IC.newCartesian(curr_coords_f, dq_f / nDiv)

                IC.build_dlc_0(curr_coords_b)
                dq_b = IC.calcDiff(reac, curr_coords_b)

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

                M.xyzs = [reac.reshape(-1,3)/ang2bohr,
                         curr_coords_f.reshape(-1,3)/ang2bohr,
                         curr_coords_b.reshape(-1,3)/ang2bohr,
                         prod.reshape(-1,3)/ang2bohr]

                new_PRIM = PRIMIC(
                    M,
                    build=True,
                    connect=connect,
                    addcart=addcart,
                    constraints=None,
                    warn=False,
                )

                if len(new_PRIM.Internals) > len(PRIM_most_Internals.Internals):
                    IC.Prims = new_PRIM
                    PRIM_most_Internals = new_PRIM

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

                nDiv -= 1

            #self.forward_coord_list = coord_list_f
            #self.backward_coord_list = coord_list_b#[::-1]
            self.PRIMs = PRIM_most_Internals
            #Coord_list = coord_list_f + coord_list_b
            #M.xyzs = [coords.reshape(-1,3)/ang2bohr for coords in coord_list]
            #New_PRIM = PRIMIC(
            #    M,
            #    build=True,
            #    connect=connect,
            #    addcart=addcart,
            #    constraints=None,
            #    warn=False,
            #)
            #if len(PRIM_most_Internals.Internals) == len(ini_PRIM.Internals):
            #    self.simple_interpolate(PRIM_most_Internals)
            #else:
            #    self.mixed_interpolate(PRIM_most_Internals)
            #self.PRIMS = IC.Prims #PRIM_most_Internals
            #self.simple_interpolate(PRIM_most_Internals)

            print("Primitive Internal Coordinates are ready.")


            #self.fwd_M.xyzs = [
            #    coords for coords in coord_list_f
            #]

            #self.bwd_M.xyzs = [
            #    coords for coords in coord_list_b[::-1]
            #]

            #self.bwd_M.xyzs = [coords.reshape(-1, 3) / ang2bohr for coords in coord_list_b]


            #json_str = json.dumps(cn_info, indent=4)
            #with open("mixed_cn_info.json", "w") as f:
            #    f.write(json_str)

            #self.interpolated_dict["mixed_" + ic] = np.array(coord_list_f)


            #xyz_dir = os.path.join(self.dir, "interpolated")
            #if not os.path.exists(xyz_dir):
            #    os.makedirs(xyz_dir)
            #self.fwd_M.write(
            #    os.path.join(xyz_dir, "fwd_interpolated_%s.xyz" % ic)
            #)

            #self.bwd_M.write(
            #    os.path.join(xyz_dir, "bwd_interpolated_%s.xyz" % ic)
            #)


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
    TRIC.collect_PRIMs()
    TRIC.interpolate()
    #TRIC.mixed_interpolate()
    # TRIC.calculate_energies("simple")
    # TRIC.fwd_bwd_interpolate()
    # TRIC.mix_interpolate()
    # TRIC.calculate_energies("mix")
    # TRIC.calculate_energies("fwd_bwd")

    # TRIC.plot()
    print("Done!")


if __name__ == "__main__":
    main()
