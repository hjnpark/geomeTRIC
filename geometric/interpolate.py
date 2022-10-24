import os, sys, json, tempfile

from .params import parse_interpolate_args, IntpParams
from .prepare import get_molecule_engine
from .nifty import ang2bohr
from .internal import (
    CartesianCoordinates,
    PrimitiveInternalCoordinates,
    DelocalizedInternalCoordinates,
)
import numpy as np


class TRICterpolate:
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

    def interpolate(self):
        M_reac = self.M[0]
        M_prod = self.M[-1]
        M_reac.build_topology()
        M_prod.build_topology()

        if len(M_reac.molecules) >= len(M_reac.molecules):
            M_ini = M_reac
            M_fin = M_prod
        else:
            M_ini = M_prod
            M_fin = M_reac

        reac = M_ini.xyzs[0].flatten() * ang2bohr
        prod = M_fin.xyzs[0].flatten() * ang2bohr
        if len(reac) != len(prod):
            raise RuntimeError(
                "Number of atoms of reactant and product should be same."
            )

        CART = CartesianCoordinates(M_ini, build=True, connect=False, addcart=False)
        PRIM = PrimitiveInternalCoordinates(
            M_ini, build=True, connect=True, addcart=False
        )
        TRIC = DelocalizedInternalCoordinates(
            M_ini, build=True, connect=False, addcart=False
        )
        TRIC_REV = DelocalizedInternalCoordinates(
            M_fin, build=True, connect=False, addcart=False
        )
        DLC = DelocalizedInternalCoordinates(
            M_ini, build=True, connect=True, addcart=False
        )
        HDLC = DelocalizedInternalCoordinates(
            M_ini, build=True, connect=False, addcart=True
        )

        TRICP = PrimitiveInternalCoordinates(
            M_ini, build=True, connect=False, addcart=False
        )

        ICs = {
            "cart": CART,
            "prim": PRIM,
            "tric": TRIC,
            "tric_rev":TRIC_REV,
            "dlc": DLC,
            "hdlc": HDLC,
            "tric-p": TRICP,
        }

        for ic in self.params.coordsys:
            IC = ICs[ic]
            IC_rev = ICs["tric_rev"]
            dq_forward = IC.calcDiff(prod, reac)
            dq_backward = IC_rev.calcDiff(reac, prod)
            nDiv = self.params.frames
            reac_coords = reac.copy()
            prod_coords = prod.copy()
            fwd_coord_list = [reac_coords]
            for i in range(nDiv//2):
                new_coords = IC.newCartesian(reac_coords, dq_forward / nDiv)
                fwd_coord_list.append(new_coords)
                reac_coords = new_coords.copy()

            bwd_coord_list = [prod_coords]
            for i in range(nDiv//2):
                new_coords = IC_rev.newCartesian(prod_coords, dq_backward / nDiv)
                bwd_coord_list.append(new_coords)
                prod_coords = new_coords.copy()
            coord_list = fwd_coord_list + bwd_coord_list[::-1]
            #print(
            #    "Error in final interpolated vs. product structure (%s):" % ic,
            #    np.linalg.norm(curr_coords - prod),
            #)
            print(
               "Error in TS from forward vs. TS from backward (%s):" % ic,
               np.linalg.norm(fwd_coord_list[-1] - bwd_coord_list[-1]),
            )

            self.interpolated_dict[ic] = np.array(coord_list)
            M_reac.xyzs = [coords.reshape(-1, 3) / ang2bohr for coords in coord_list]
            xyz_dir = os.path.join(self.dir, "interpolated")
            if not os.path.exists(xyz_dir):
                os.makedirs(xyz_dir)
            M_reac.write(os.path.join(xyz_dir, "interpolated_%s.xyz" % ic))

    def calculate_energies(self):
        print("Calculating Energies...")
        for ic, geos in self.interpolated_dict.items():
            e_list = []
            for xyz in geos:
                E = self.engine.calc_new(xyz.flatten(), tempfile.mktemp())["energy"]
                e_list.append(E)
            self.Energy_dict[ic] = e_list
        json_str = json.dumps(self.Energy_dict, indent=4)
        with open(os.path.join(self.dir, "energies.json"), "w") as f:
            f.write(json_str)

    def plot(self):
        import matplotlib.pyplot as plt

        coordsys_list = {
            "cart": ["Cartesian", ".--"],
            "prim": ["Primitive I.C.", "2--"],
            "dlc": ["Delocalized I.C.", "1--"],
            "hdlc": ["HDLC", "+--"],
            "tric": ["TRIC", "d--"],
            "tric-p": ["TRICP", "X--"]
        }
        label = []
        e_array = []
        for ic, es in self.Energy_dict.items():
            label.append(ic)
            e_array.append(es)

        x = np.arange(len(e_array[0]))

        plt.figure(figsize=(6, 4))
        for i in range(len(e_array)):
            plt.plot(
                x,
                e_array[i],
                coordsys_list[label[i]][-1],
                label=coordsys_list[label[i]][0],
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

    TRIC = TRICterpolate(params, M, engine)
    TRIC.interpolate()
    TRIC.calculate_energies()
    TRIC.plot()
    print("Done!")


if __name__ == "__main__":
    main()
