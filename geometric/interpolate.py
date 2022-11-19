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
        else:
            self.M_ini = copy.deepcopy(M_prod)
            self.M_fin = copy.deepcopy(M_reac)

        self.simple_xyz = self.M[0]
        self.mix_xyz = self.M[0]
        self.fwd_M = self.M[0]
        self.bwd_M = self.M[0]

        self.reac = self.M_ini.xyzs[0].flatten() * ang2bohr
        self.prod = self.M_fin.xyzs[0].flatten() * ang2bohr

    def simple_interpolate(self):

        for ic in self.params.coordsys:
            CoordClass, connect, addcart = self.coordsys_dict[ic.lower()]
            print("first IC_f_ini")
            IC_f_ini = CoordClass(
                self.M_ini,
                build=True,
                connect=connect,
                addcart=addcart,
                constraints=None,
            )
            print("now IC_b_ini")
            IC_b_ini = CoordClass(
                self.M_fin,
                build=True,
                connect=connect,
                addcart=addcart,
                constraints=None,
            )
            print("Calculating initial difference")
            dq_f = IC_f_ini.calcDiff(self.prod, self.reac)
            dq_b = IC_b_ini.calcDiff(self.reac, self.prod)
            nDiv = self.params.frames
            nDiv += nDiv % 2

            curr_coords_f_normal = self.reac.copy()
            curr_coords_b_normal = self.prod.copy()

            coord_list_f = [curr_coords_f_normal]
            coord_list_b = [curr_coords_b_normal]

            M_ini = copy.deepcopy(self.M_ini)
            M_fin = copy.deepcopy(self.M_fin)

            cn_info = {
                "Initial Forward IC": [],
                "Initial Backward IC": [],
                "New Forward IC": [],
                "New Backward IC": [],
            }

            for i in range(nDiv // 2):

                dq_f = IC_f_ini.calcDiff(curr_coords_b_normal, curr_coords_f_normal)
                dq_b = IC_b_ini.calcDiff(curr_coords_f_normal, curr_coords_b_normal)
                new_coords_f_normal = IC_f_ini.newCartesian(
                    curr_coords_f_normal, dq_f / nDiv
                )
                new_coords_b_normal = IC_b_ini.newCartesian(
                    curr_coords_b_normal, dq_b / nDiv
                )

                curr_coords_f_normal = new_coords_f_normal.copy()
                curr_coords_b_normal = new_coords_b_normal.copy()

                # ------------------Geting condition numbers here-------------------------
                print("getting GMatrixes from IC objects")
                G_f_ini = IC_f_ini.GMatrix(new_coords_f_normal)
                eig_f_ini, vec_f_ini = np.linalg.eigh(G_f_ini)

                G_b_ini = IC_b_ini.GMatrix(new_coords_b_normal)
                eig_b_ini, vec_b_ini = np.linalg.eigh(G_b_ini)

                print("\n-------------------------------------------------------------")
                con_num_if = np.real(eig_f_ini[-1] / eig_f_ini[0])
                con_num_ib = np.real(eig_b_ini[-1] / eig_b_ini[0])
                print("Condition number of initial forward %f" % con_num_if)
                print("Condition number of initial backward %f" % con_num_ib)
                # CN_ratio = con_num_if/np.mean(cn_info['Initial Forward IC'])
                # print(CN_ratio)
                cn_info["Initial Forward IC"].append(con_num_if)
                cn_info["Initial Backward IC"].append(con_num_ib)
                # --------------------Done collecting condition numbers ---------------------
                # print("b Eigvals", np.real(eig_b[-5:]))
                # print("b Largest Eigval: %f" %np.real(eig_b[-1]))
                # print("Lergest vector:", vec[0])
                # print("b Smallest Eigval: %f" %np.real(eig_b[0]))
                # print("Smallets vector:", vec[-1])
                IC_f_ini.build_dlc(curr_coords_f_normal)
                IC_b_ini.build_dlc(curr_coords_b_normal)

                coord_list_f.append(curr_coords_f_normal)
                coord_list_b.append(curr_coords_b_normal)
                nDiv -= 2
            coord_list = coord_list_f + coord_list_b[::-1]
            json_str = json.dumps(cn_info, indent=4)
            with open("cn_info.json", "w") as f:
                f.write(json_str)
            print(
                "Error in final interpolated vs. product structure (%s):" % ic,
                np.linalg.norm(curr_coords_f_normal - self.prod),
            )
            self.interpolated_dict["simple_" + ic] = np.array(coord_list)
            self.simple_xyz.xyzs = [
                coords.reshape(-1, 3) / ang2bohr for coords in coord_list
            ]

            smoothed_M = EqualSpacing(self.simple_xyz)[
                np.array(
                    [
                        int(round(i))
                        for i in np.linspace(0, len(self.simple_xyz) - 1, self.params.frames)
                    ]
                )
            ]

            xyz_dir = os.path.join(self.dir, "interpolated")
            if not os.path.exists(xyz_dir):
                os.makedirs(xyz_dir)
            self.simple_xyz.write(
                os.path.join(xyz_dir, "simple_interpolated_%s.xyz" % ic)
            )
            smoothed_M.write(os.path.join(xyz_dir, "smoothed_interpolated_%s.xyz" % ic))

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
    TRIC.simple_interpolate()
    # TRIC.calculate_energies("simple")
    # TRIC.fwd_bwd_interpolate()
    # TRIC.mix_interpolate()
    # TRIC.calculate_energies("mix")
    # TRIC.calculate_energies("fwd_bwd")

    # TRIC.plot()
    print("Done!")


if __name__ == "__main__":
    main()
