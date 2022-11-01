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

        self.reac = self.M_ini.xyzs[0].flatten() * ang2bohr
        self.prod = self.M_fin.xyzs[0].flatten() * ang2bohr

    def simple_interpolate(self):

        for ic in self.params.coordsys:
            CoordClass, connect, addcart = self.coordsys_dict[ic.lower()]
            IC = CoordClass(
                self.M_ini,
                build=True,
                connect=connect,
                addcart=addcart,
                constraints=None,
            )
            dq = IC.calcDiff(self.prod, self.reac)
            nDiv = self.params.frames - 1
            curr_coords = self.reac.copy()
            coord_list = [curr_coords]
            for i in range(nDiv):
                new_coords = IC.newCartesian(curr_coords, dq / nDiv)
                coord_list.append(new_coords)
                curr_coords = new_coords.copy()
            print(
                "Error in final interpolated vs. product structure (%s):" % ic,
                np.linalg.norm(curr_coords - self.prod),
            )
            self.interpolated_dict["simple_" + ic] = np.array(coord_list)
            self.simple_xyz.xyzs = [
                coords.reshape(-1, 3) / ang2bohr for coords in coord_list
            ]
            xyz_dir = os.path.join(self.dir, "interpolated")
            if not os.path.exists(xyz_dir):
                os.makedirs(xyz_dir)
            self.simple_xyz.write(
                os.path.join(xyz_dir, "simple_interpolated_%s.xyz" % ic)
            )

    def fwd_bwd_interpolate(self):

        for ic in self.params.coordsys:
            CoordClass, connect, addcart = self.coordsys_dict[ic.lower()]

            nDiv = self.params.frames - 1
            reac_coords = self.reac.copy()
            prod_coords = self.prod.copy()
            fwd_coord_list = [reac_coords]
            bwd_coord_list = [prod_coords]
            M_ini = copy.deepcopy(self.M_ini)
            M_fin = copy.deepcopy(self.M_fin)
            for i in range(nDiv // 2):
                IC_fwd = CoordClass(
                    M_ini,
                    build=True,
                    connect=connect,
                    addcart=addcart,
                    constraints=None,
                )
                IC_bwd = CoordClass(
                    M_fin,
                    build=True,
                    connect=connect,
                    addcart=addcart,
                    constraints=None,
                )
                if i == 0:
                    dq_forward = IC_fwd.calcDiff(self.prod, self.reac)
                    dq_backward = IC_bwd.calcDiff(self.reac, self.prod)
                else:
                    dq_forward = IC_fwd.calcDiff(new_bwd_coords, new_fwd_coords)
                    dq_backward = IC_bwd.calcDiff(new_fwd_coords, new_bwd_coords)
                interval = nDiv - 2 * i
                if interval == 2:
                    new_fwd_coords = IC_fwd.newCartesian(
                        reac_coords, dq_forward / interval
                    )
                    fwd_coord_list.append(new_fwd_coords)
                    break
                else:
                    new_fwd_coords = IC_fwd.newCartesian(
                        reac_coords, dq_forward / interval
                    )
                    fwd_coord_list.append(new_fwd_coords)
                    reac_coords = new_fwd_coords.copy()
                    new_bwd_coords = IC_bwd.newCartesian(
                        prod_coords, dq_backward / interval
                    )
                    bwd_coord_list.append(new_bwd_coords)
                    prod_coords = new_bwd_coords.copy()

                    M_ini.xyzs = [new_fwd_coords.reshape(-1, 3) / ang2bohr]
                    M_fin.xyzs = [new_bwd_coords.reshape(-1, 3) / ang2bohr]

            coord_list = fwd_coord_list + bwd_coord_list[::-1]
            print(
                "Difference between TS from forward and TS from backward (%s):" % ic,
                np.linalg.norm(fwd_coord_list[-1] - bwd_coord_list[-1]),
            )

            self.interpolated_dict["fwdbwd_" + ic] = np.array(coord_list)
            self.mix_xyz.xyzs = [
                coords.reshape(-1, 3) / ang2bohr for coords in coord_list
            ]
            xyz_dir = os.path.join(self.dir, "interpolated")
            if not os.path.exists(xyz_dir):
                os.makedirs(xyz_dir)
            self.mix_xyz.write(os.path.join(xyz_dir, "fwdbwd_interpolated_%s.xyz" % ic))

    def fill(self, cart1, cart2, ic, max_diff, mean_diff):
        print("filling...")
        reac_coords = cart1.copy()
        prod_coords = cart2.copy()
        filled_fwd_list = [cart1]
        fwd_cart_diff = []
        bwd_cart_diff = []
        filled_bwd_list = [cart2]
        M_ini = copy.deepcopy(self.M_ini)
        M_fin = copy.deepcopy(self.M_fin)
        M_ini.xyzs =  [cart1.reshape(-1,3)/ang2bohr]
        M_fin.xyzs = [cart2.reshape(-1,3)/ang2bohr]
        CoordClass, connect, addacart = self.coordsys_dict[ic.lower()]
        nDiv = 2 + int(max_diff//mean_diff)
        for i in range(nDiv):
            IC_fwd = CoordClass(M_ini, build=True, connect=connect, addcart=addacart, constraints=None)
            IC_bwd = CoordClass(M_fin, build=True, connect=connect, addcart=addacart, constraints=None)
            if i == 0:
                dq_fwd = IC_fwd.calcDiff(reac_coords, prod_coords)
                dq_bwd = IC_bwd.calcDiff(prod_coords, reac_coords)
            else:
                dq_fwd = IC_fwd.calcDiff(new_bwd_coords, new_fwd_coords)
                dq_bwd = IC_bwd.calcDiff(new_fwd_coords, new_bwd_coords)
            interval = nDiv - 2 * i
            step_fwd = dq_fwd / interval
            step_bwd = dq_bwd / interval
            if interval == 1 or interval ==2:
                new_fwd_coords = IC_fwd.newCartesian(reac_coords, step_fwd)
                filled_fwd_list.append(new_fwd_coords)
                break
            else:
                new_fwd_coords = IC_fwd.newCartesian(reac_coords, step_fwd)
                new_bwd_coords = IC_bwd.newCartesian(prod_coords, step_bwd)

                fwd_diff = np.linalg.norm(reac_coords.reshape(-1, 3) - new_fwd_coords.reshape(-1, 3), axis=1)
                fwd_cart_diff.append(fwd_diff)
                fwd_diff_mean = np.mean(fwd_cart_diff, axis=0)
                fwd_diff_max_ind = np.argmax(fwd_diff)
                fwd_diff_max = np.max(fwd_diff)
                fwd_pct_diff = 100 * abs(fwd_diff_max - fwd_diff_mean[fwd_diff_max_ind]) / np.max(
                    fwd_diff_mean[fwd_diff_max_ind])
                if fwd_pct_diff > 100:
                    filled_list = self.fill(reac_coords, new_fwd_coords, ic, fwd_diff_max,
                                            fwd_diff_mean[fwd_diff_max_ind])
                    filled_fwd_list += filled_list
                    new_fwd_coords = filled_list[-1]
                else:
                    filled_fwd_list.append(new_fwd_coords)

                reac_coords = new_fwd_coords.copy()

                bwd_diff = np.linalg.norm(prod_coords.reshape(-1, 3) - new_bwd_coords.reshape(-1, 3), axis=1)
                bwd_cart_diff.append(bwd_diff)
                bwd_diff_mean = np.mean(bwd_cart_diff, axis=0)
                bwd_diff_max_ind = np.argmax(bwd_diff)
                bwd_diff_max = np.max(bwd_diff)
                bwd_pct_diff = 100 * abs(np.max(bwd_diff) - bwd_diff_mean[bwd_diff_max_ind]) / np.max(
                    bwd_diff_mean[bwd_diff_max_ind])
                if bwd_pct_diff > 100:
                    filled_list = self.fill(prod_coords, new_bwd_coords, ic, bwd_diff_max,
                                            bwd_diff_mean[bwd_diff_max_ind])
                    filled_bwd_list += filled_list
                    new_bwd_coords = filled_list[-1]
                else:
                    filled_bwd_list.append(new_bwd_coords)

                prod_coords = new_bwd_coords.copy()

                M_ini.xyzs = [new_fwd_coords.reshape(-1, 3) / ang2bohr]
                M_fin.xyzs = [new_bwd_coords.reshape(-1, 3) / ang2bohr]
        filled_list = filled_fwd_list + filled_bwd_list[::-1]
        return filled_list
    def mix_interpolate(self):

        for ic in self.params.coordsys:
            CoordClass, connect, addcart = self.coordsys_dict[ic.lower()]

            nDiv = self.params.frames - 1
            reac_coords = self.reac.copy()
            prod_coords = self.prod.copy()
            fwd_coord_list = [reac_coords]
            fwd_cart_diff = []
            bwd_coord_list = [prod_coords]
            bwd_cart_diff = []
            M_ini = copy.deepcopy(self.M_ini)
            M_fin = copy.deepcopy(self.M_fin)
            for i in range(nDiv // 2):
                IC_fwd = CoordClass(
                    M_ini,
                    build=True,
                    connect=connect,
                    addcart=addcart,
                    constraints=None,
                )
                IC_bwd = CoordClass(
                    M_fin,
                    build=True,
                    connect=connect,
                    addcart=addcart,
                    constraints=None,
                )

                if i == 0:
                    dq_fwd = IC_fwd.calcDiff(self.prod, self.reac)
                    dq_bwd = IC_bwd.calcDiff(self.reac, self.prod)
                else:
                    dq_fwd = IC_fwd.calcDiff(new_bwd_coords, new_fwd_coords)
                    dq_bwd = IC_bwd.calcDiff(new_fwd_coords, new_bwd_coords)

                interval = nDiv - 2 * i
                step_fwd = dq_fwd / interval
                step_bwd = dq_bwd / interval
                if interval == 1 or interval == 2:
                    new_fwd_coords = IC_fwd.newCartesian(reac_coords, step_fwd)
                    fwd_coord_list.append(new_fwd_coords)
                    break
                else:
                    new_fwd_coords = IC_fwd.newCartesian(reac_coords, step_fwd)
                    new_bwd_coords = IC_bwd.newCartesian(prod_coords, step_bwd)

                    print('------------------------------------------------------------')
                    fwd_diff = np.linalg.norm(reac_coords.reshape(-1,3)-new_fwd_coords.reshape(-1,3), axis=1)
                    fwd_cart_diff.append(fwd_diff)
                    fwd_diff_mean = np.mean(fwd_cart_diff, axis=0)
                    fwd_diff_max_ind = np.argmax(fwd_diff)
                    fwd_diff_max = np.max(fwd_diff)
                    fwd_pct_diff = 100*abs(fwd_diff_max - fwd_diff_mean[fwd_diff_max_ind])/np.max(fwd_diff_mean[fwd_diff_max_ind])
                    print("Iteration:", i)
                    print("Max fwd dis and its mean:",fwd_diff_max, fwd_diff_mean[fwd_diff_max_ind])
                    print("Percent diff:", fwd_pct_diff)
                    print("Max bwd dis ind:", fwd_diff_max_ind)
                    if fwd_pct_diff > 100:
                        filled_list = self.fill(reac_coords, new_fwd_coords, ic, fwd_diff_max, fwd_diff_mean[fwd_diff_max_ind])
                        fwd_coord_list += filled_list
                        new_fwd_coords = filled_list[-1]
                    else:
                        fwd_coord_list.append(new_fwd_coords)
                    reac_coords = new_fwd_coords.copy()

                    bwd_diff = np.linalg.norm(prod_coords.reshape(-1,3) - new_bwd_coords.reshape(-1,3), axis=1)
                    bwd_cart_diff.append(bwd_diff)
                    bwd_diff_mean =  np.mean(bwd_cart_diff, axis=0)
                    bwd_diff_max_ind = np.argmax(bwd_diff)
                    bwd_diff_max = np.max(bwd_diff)
                    bwd_pct_diff = 100 * abs(np.max(bwd_diff) - bwd_diff_mean[bwd_diff_max_ind]) / np.max(bwd_diff_mean[bwd_diff_max_ind])
                    print("\nMax bwd dis and its mean:", np.max(bwd_diff), bwd_diff_mean[bwd_diff_max_ind])
                    print("Percent diff:", bwd_pct_diff)
                    print("Max bwd dis ind:", bwd_diff_max_ind)
                    if bwd_pct_diff > 100:
                        filled_list = self.fill(prod_coords, new_bwd_coords, ic, bwd_diff_max, bwd_diff_mean[bwd_diff_max_ind])
                        bwd_coord_list += filled_list
                        new_bwd_coords = filled_list[-1]
                    else:
                        bwd_coord_list.append(new_bwd_coords)
                    prod_coords = new_bwd_coords.copy()

                    M_ini.xyzs = [new_fwd_coords.reshape(-1, 3) / ang2bohr]
                    M_fin.xyzs = [new_bwd_coords.reshape(-1, 3) / ang2bohr]

            coord_list = fwd_coord_list + bwd_coord_list[::-1]

            self.interpolated_dict["mix_" + ic] = np.array(coord_list)
            self.mix_xyz.xyzs = [
                coords.reshape(-1, 3) / ang2bohr for coords in coord_list
            ]
            #Equal_M = EqualSpacing(self.mix_xyz)
            Equal_M = self.mix_xyz
            xyz_dir = os.path.join(self.dir, "interpolated")
            if not os.path.exists(xyz_dir):
                os.makedirs(xyz_dir)
            Equal_M.write(os.path.join(xyz_dir, "mixed_interpolated_%s.xyz" % ic))

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
    # TRIC.simple_interpolate()
    # TRIC.calculate_energies("simple")
    # TRIC.fwd_bwd_interpolate()
    TRIC.mix_interpolate()
    TRIC.calculate_energies("mix")
    # TRIC.calculate_energies("fwd_bwd")

    TRIC.plot()
    print("Done!")


if __name__ == "__main__":
    main()
