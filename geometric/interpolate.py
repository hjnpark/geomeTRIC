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
            IC_f_ini = CoordClass(
                self.M_ini,
                build=True,
                connect=connect,
                addcart=addcart,
                constraints=None,
            )

            IC_b_ini = CoordClass(
                self.M_fin,
                build=True,
                connect=connect,
                addcart=addcart,
                constraints=None,
            )
            dq_f = IC_f_ini.calcDiff(self.prod, self.reac)
            dq_b = IC_b_ini.calcDiff(self.reac, self.prod)
            nDiv = self.params.frames
            curr_coords_f_normal = self.reac.copy()
            curr_coords_b_normal = self.prod.copy()
            coord_list = [curr_coords_f_normal]
            M_ini = copy.deepcopy(self.M_ini)
            M_fin = copy.deepcopy(self.M_fin)
            M_ini_upd = copy.deepcopy(self.M_ini)
            M_fin_upd = copy.deepcopy(self.M_fin)

            cn_info = {'Initial Forward IC':[],
                       'Initial Backward IC':[],
                       'New Forward IC':[],
                       'New Backward IC':[],
                       'Updated Forward IC':[],
                       'Updated Backward IC':[]}

            for i in range(nDiv):
                IC_f_new = CoordClass(
                    M_ini,
                    build=True,
                    connect=connect,
                    addcart=addcart,
                    constraints=None,
                )

                IC_b_new = CoordClass(
                    M_fin,
                    build=True,
                    connect=connect,
                    addcart=addcart,
                    constraints=None,
                )

                IC_f_upd =  CoordClass(
                    M_ini_upd,
                    build=True,
                    connect=connect,
                    addcart=addcart,
                    constraints=None,
                )

                IC_b_upd =  CoordClass(
                    M_fin_upd,
                    build=True,
                    connect=connect,
                    addcart=addcart,
                    constraints=None,
                )

                if i != 0:
                    dq_upd_f = IC_f_upd.calcDiff(curr_coords_b_upd, curr_coords_f_upd)
                    dq_upd_b = IC_b_upd.calcDiff(curr_coords_f_upd, curr_coords_b_upd)
                    new_coords_f_upd = IC_f_upd.newCartesian(curr_coords_f_upd, dq_upd_f / (nDiv - i))
                    new_coords_b_upd = IC_b_upd.newCartesian(curr_coords_b_upd, dq_upd_b / (nDiv - i))

                    curr_coords_f_upd = new_coords_f_upd.copy()
                    curr_coords_b_upd = new_coords_b_upd.copy()

                new_coords_f_normal = IC_f_new.newCartesian(curr_coords_f_normal, dq_f / nDiv)
                new_coords_b_normal = IC_b_new.newCartesian(curr_coords_b_normal, dq_b / nDiv)



                coord_list.append(new_coords_f_normal)
                curr_coords_f_normal = new_coords_f_normal.copy()
                curr_coords_b_normal = new_coords_b_normal.copy()


                # ------------------Geting condition numbers here-------------------------
                G_f_ini = IC_f_ini.GMatrix(new_coords_f_normal)
                eig_f_ini, vec_f_ini = np.linalg.eigh(G_f_ini)

                G_b_ini = IC_b_ini.GMatrix(new_coords_f_normal)
                eig_b_ini, vec_b_ini = np.linalg.eigh(G_b_ini)

                G_f_new = IC_f_new.GMatrix(new_coords_f_normal)
                eig_f_new, vec_f_new = np.linalg.eigh(G_f_new)

                G_b_new = IC_b_new.GMatrix(new_coords_f_normal)
                eig_b_new, vec_b_new = np.linalg.eigh(G_b_new)
                if i != 0:
                    G_f_upd = IC_f_upd.GMatrix(new_coords_f_upd)
                    eig_f_upd, vec_f_upd = np.linalg.eigh(G_f_upd)

                    G_b_upd = IC_b_upd.GMatrix(new_coords_b_upd)
                    eig_b_upd, vec_b_upd = np.linalg.eigh(G_b_upd)
                print("\n-------------------------------------------------------------")
                #print("G", G)
                con_num_if = np.real(eig_f_ini[-1] / eig_f_ini[0])
                con_num_ib = np.real(eig_b_ini[-1] / eig_b_ini[0])
                con_num_nf = np.real(eig_f_new[-1] / eig_f_new[0])
                con_num_nb = np.real(eig_b_new[-1] / eig_b_new[0])
                if i != 0:
                    con_num_uf = np.real(eig_b_upd[-1] / eig_b_upd[0])
                    con_num_ub = np.real(eig_b_upd[-1] / eig_b_upd[0])

                    cn_info['Updated Forward IC'].append(con_num_uf)
                    cn_info['Updated Backward IC'].append(con_num_ub)
                print("Condition number of initial forward %f" %con_num_if)
                print("Condition number of initial backward %f" %con_num_ib)
                print("Condition number of new forward %f" %con_num_nf)
                print("Condition number of new backward %f" %con_num_nb)
                cn_info['Initial Forward IC'].append(con_num_if)
                cn_info['Initial Backward IC'].append(con_num_ib)
                cn_info['New Forward IC'].append(con_num_nf)
                cn_info['New Backward IC'].append(con_num_nb)
                #cn_info['Updated Forward IC'].append(con_num_uf)
                #cn_info['Updated Backward IC'].append(con_num_ub)
                # --------------------Done collecting condition numbers ---------------------
                #print("b Eigvals", np.real(eig_b[-5:]))
                #print("b Largest Eigval: %f" %np.real(eig_b[-1]))
                #print("Lergest vector:", vec[0])
                #print("b Smallest Eigval: %f" %np.real(eig_b[0]))
                #print("Smallets vector:", vec[-1])
                M_ini.xyzs = [new_coords_f_normal.reshape(-1, 3) / ang2bohr]
                M_fin.xyzs = [new_coords_b_normal.reshape(-1, 3) / ang2bohr]
                if i == 0 :
                    curr_coords_f_upd = new_coords_f_normal.copy()
                    curr_coords_b_upd = new_coords_b_normal.copy()
                else:
                    M_ini_upd.xyzs = [new_coords_f_upd.reshape(-1, 3) / ang2bohr]
                    M_fin_upd.xyzs = [new_coords_b_upd.reshape(-1, 3) / ang2bohr]
            json_str = json.dumps(cn_info, indent=4)
            with open('cn_info.json', "w") as f:
                f.write(json_str)
            print(
                "Error in final interpolated vs. product structure (%s):" % ic,
                np.linalg.norm(curr_coords_f_normal - self.prod),
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

    def fill(self, cart1, cart2, ic, final_diff, mean_diff):
        print("Mean difference: %.5f" % mean_diff)
        print("Max difference: %.5f" % final_diff)
        print("filling...")
        reac_coords = cart1.copy()
        prod_coords = cart2.copy()
        filled_fwd_list = []
        filled_bwd_list = []
        M_ini = copy.deepcopy(self.M_ini)
        M_fin = copy.deepcopy(self.M_fin)
        M_ini.xyzs = [cart1.reshape(-1, 3) / ang2bohr]
        M_fin.xyzs = [cart2.reshape(-1, 3) / ang2bohr]
        CoordClass, connect, addacart = self.coordsys_dict[ic.lower()]
        nDiv = int(final_diff // mean_diff)
        nDiv += nDiv % 2
        for i in range(nDiv // 2):
            IC_fwd = CoordClass(
                M_ini, build=True, connect=connect, addcart=addacart, constraints=None
            )
            IC_bwd = CoordClass(
                M_fin, build=True, connect=connect, addcart=addacart, constraints=None
            )
            if i == 0:
                dq_fwd = IC_fwd.calcDiff(reac_coords, prod_coords)
                dq_bwd = IC_bwd.calcDiff(prod_coords, reac_coords)
            else:
                dq_fwd = IC_fwd.calcDiff(new_bwd_coords, new_fwd_coords)
                dq_bwd = IC_bwd.calcDiff(new_fwd_coords, new_bwd_coords)
            interval = nDiv - 2 * i
            step_fwd = dq_fwd / interval
            step_bwd = dq_bwd / interval
            if interval == 2:
                new_fwd_coords = IC_fwd.newCartesian(reac_coords, step_fwd)
                filled_fwd_list.append(new_fwd_coords)
                new_bwd_coords = IC_bwd.newCartesian(prod_coords, step_bwd)
                filled_bwd_list.append(new_bwd_coords)
                final_diff = np.linalg.norm(new_bwd_coords - new_fwd_coords)
                if final_diff / mean_diff > 1.0:
                    filled_list = self.fill(
                        new_fwd_coords, new_bwd_coords, ic, final_diff, mean_diff
                    )
                    filled_fwd_list += filled_list

                break
            else:
                new_fwd_coords = IC_fwd.newCartesian(reac_coords, step_fwd)
                new_bwd_coords = IC_bwd.newCartesian(prod_coords, step_bwd)
                reac_coords = new_fwd_coords.copy()
                prod_coords = new_bwd_coords.copy()

                filled_fwd_list.append(new_fwd_coords)
                filled_bwd_list.append(new_bwd_coords)

                # fwd_diff = np.linalg.norm(
                #    reac_coords.reshape(-1, 3) - new_fwd_coords.reshape(-1, 3),
                #    axis=1,
                # )

                M_ini.xyzs = [new_fwd_coords.reshape(-1, 3) / ang2bohr]
                M_fin.xyzs = [new_bwd_coords.reshape(-1, 3) / ang2bohr]
        filled_list = filled_fwd_list + filled_bwd_list[::-1]
        return filled_list

    def mix_interpolate(self):
        for ic in self.params.coordsys:
            CoordClass, connect, addcart = self.coordsys_dict[ic.lower()]

            nDiv = self.params.frames
            nDiv += nDiv%2
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
                if interval == 2:
                    mean_diff = np.mean((fwd_diff_mean + bwd_diff_mean) / 2)
                    new_fwd_coords = IC_fwd.newCartesian(reac_coords, step_fwd)
                    fwd_coord_list.append(new_fwd_coords)
                    new_bwd_coords = IC_bwd.newCartesian(prod_coords, step_bwd)
                    bwd_coord_list.append(new_bwd_coords)
                    final_diff = np.linalg.norm(fwd_coord_list[-1] - bwd_coord_list[-1])
                    filled_list = self.fill(
                        new_fwd_coords, new_bwd_coords, ic, final_diff, mean_diff
                    )
                    fwd_coord_list += filled_list
                    break
                else:
                    new_fwd_coords = IC_fwd.newCartesian(reac_coords, step_fwd)
                    new_bwd_coords = IC_bwd.newCartesian(prod_coords, step_bwd)

                    fwd_diff = np.linalg.norm(
                        reac_coords.reshape(-1, 3) - new_fwd_coords.reshape(-1, 3),
                        axis=1,
                    )
                    fwd_cart_diff.append(fwd_diff)
                    fwd_diff_mean = np.mean(fwd_cart_diff, axis=0)
                    fwd_coord_list.append(new_fwd_coords)
                    reac_coords = new_fwd_coords.copy()

                    bwd_diff = np.linalg.norm(
                        prod_coords.reshape(-1, 3) - new_bwd_coords.reshape(-1, 3),
                        axis=1,
                    )
                    print("iteration %i" % i)
                    print("fwd_bwd_diffes", np.mean(fwd_diff), np.mean(bwd_diff))
                    bwd_cart_diff.append(bwd_diff)
                    bwd_diff_mean = np.mean(bwd_cart_diff, axis=0)
                    bwd_coord_list.append(new_bwd_coords)
                    prod_coords = new_bwd_coords.copy()

                    M_ini.xyzs = [new_fwd_coords.reshape(-1, 3) / ang2bohr]
                    M_fin.xyzs = [new_bwd_coords.reshape(-1, 3) / ang2bohr]

            coord_list = fwd_coord_list + bwd_coord_list[::-1]

            self.interpolated_dict["mix_" + ic] = np.array(coord_list)
            self.mix_xyz.xyzs = [
                coords.reshape(-1, 3) / ang2bohr for coords in coord_list
            ]
            tot_len = len(self.mix_xyz)
            self.fwd_M.xyzs = [
                coords.reshape(-1, 3) / ang2bohr for coords in fwd_coord_list
            ]
            self.bwd_M.xyzs = [
                coords.reshape(-1, 3) / ang2bohr for coords in bwd_coord_list
            ]
            if self.params.equal_space:
                print("Spacing frames evenly.")
                final_M = EqualSpacing(self.mix_xyz)[
                    np.array(
                        [
                            int(round(i))
                            for i in np.linspace(0, tot_len - 1, self.params.frames)
                        ]
                    )
                ]
            else:
                final_M = self.mix_xyz[
                    np.array(
                        [
                            int(round(i))
                            for i in np.linspace(0, tot_len - 1, self.params.frames)
                        ]
                    )
                ]
            xyz_dir = os.path.join(self.dir, "interpolated")
            if not os.path.exists(xyz_dir):
                os.makedirs(xyz_dir)
            final_M.write(os.path.join(xyz_dir, "mixed_interpolated_%s.xyz" % ic))
            self.fwd_M.write(
                os.path.join(xyz_dir, "mixed_fwd_interpolated_%s.xyz" % ic)
            )
            self.bwd_M.write(
                os.path.join(xyz_dir, "mixed_bwd_interpolated_%s.xyz" % ic)
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
