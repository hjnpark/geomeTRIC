import os, sys, json, copy, tempfile
import networkx as nx

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
        self.M_updated = M[0]

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
        M_reac.build_bonds()
        M_prod.build_topology()
        M_prod.build_bonds()

        if len(M_reac.molecules) >= len(M_reac.molecules):
            self.M_ini = copy.deepcopy(M_reac)
            self.M_fin = copy.deepcopy(M_prod)

        else:
            self.M_ini = copy.deepcopy(M_prod)
            self.M_fin = copy.deepcopy(M_reac)


        self.interpolated_M = self.M[0]

        self.reac = self.M_ini.xyzs[0].flatten() * ang2bohr
        self.prod = self.M_fin.xyzs[0].flatten() * ang2bohr
        self.PRIMs = None
        self.IC = None

    def analyze_M(self):
        ini_G = self.M_ini.topology
        fin_G = self.M_fin.topology
        G_diff = nx.difference(fin_G, ini_G)

        if len(G_diff.edges) > 1:
            raise RuntimeError("It can't interpolate trajectories containing more than one chemical reaction step.")


    def interpolate(self):
        PRIMIC, connect, addcart = self.coordsys_dict["tric-p"]
        for ic in self.params.coordsys:

            M = copy.deepcopy(self.M)
            M_updated = self.M_updated
            nDiv = self.params.frames
            total_frames = self.params.frames
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
            IC.Prims = self.PRIMs
            self.IC = IC

            for i in range(nDiv):

                coord_list.append(curr_coords)

                IC.build_dlc_0(curr_coords)

                dq = IC.calcDiff(self.prod, curr_coords)

                new_coords = IC.newCartesian(curr_coords, dq / nDiv)

                curr_coords = new_coords.copy()

                nDiv -= 1

            self.interpolated_M.xyzs = [
                coords.reshape(-1, 3) / ang2bohr for coords in coord_list
            ]

            self.interpolated_coords = coord_list


            print(
                "Error in final interpolated vs. product structure (%s):" % ic,
                np.linalg.norm(curr_coords - self.prod),
            )
            self.interpolated_dict["simple_" + ic] = np.array(coord_list)

            xyz_dir = os.path.join(self.dir, "interpolated")
            if not os.path.exists(xyz_dir):
                os.makedirs(xyz_dir)
            self.interpolated_M.write(
                os.path.join(xyz_dir, "interpolated_%s.xyz" % ic)
            )


    def optimize(self, stepsize = 0.1):
        def calc_E(IC, chain):
            k = 1
            E = []
            for i in range(len(chain)):
                running_E = 0
                IC.build_dlc_0(chain[i])
                d1, d2 = 0, 0
                if i > 0:
                    d2 = IC.calcDiff(chain[i-1], chain[i])
                if i < len(chain) -1:
                    d1 = IC.calcDiff(chain[i], chain[i+1])

                d1 = np.sum(np.square(d1))
                d2 = np.sum(np.square(d2))
                running_E += k*(d1 + d2)

                E.append(running_E)
                
            return np.array(E)

        IC = self.IC
        chain_coords = self.interpolated_coords.copy()

        initial_Es = calc_E(IC, chain_coords)
    
        deriv_list = []

        for coords in chain_coords:

            IC.build_dlc_0(coords)
            derivatives = IC.derivatives(coords) # derivatives (3N, N, 3)
            deriv_list.append(derivatives.reshape(derivatives.shape[0],-1)) # reshaping (3N, 3N)


        chain_coords = np.array(self.interpolated_coords.copy())

        expanded_chain_coords = np.transpose(np.repeat(chain_coords[:,:, None], np.shape(chain_coords)[-1], axis=2), (0,2,1))

        updated_coords = expanded_chain_coords + np.array(deriv_list)

        del_E_list = []
        for i in range(updated_coords.shape[-1]):
            new_Es = calc_E(IC, updated_coords[:,i])
            del_E = initial_Es - new_Es
            del_E_list.append(del_E)

        print(del_E_list)
        print(np.shape(del_E_list))

        
             
            

    def collect_PRIMs(self):
        print("Collecting Primitive Internal Coordinates...")
        PRIMIC, connect, addcart = self.coordsys_dict["tric-p"]
        M = copy.deepcopy(self.M)

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

            PRIM_most_Internals = IC.Prims

            for i in range(nDiv):
                IC.build_dlc_0(curr_coords_f)
                dq_f = IC.calcDiff(prod, curr_coords_f)

                new_coords_f = IC.newCartesian(curr_coords_f, dq_f / nDiv)

                IC.build_dlc_0(curr_coords_b)
                dq_b = IC.calcDiff(reac, curr_coords_b)

                new_coords_b = IC.newCartesian(curr_coords_b, dq_b / nDiv)

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

                nDiv -= 1

            self.PRIMs = PRIM_most_Internals
            print("Primitive Internal Coordinates are ready.")


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
    TRIC.analyze_M()
    TRIC.collect_PRIMs()
    TRIC.interpolate()
    TRIC.optimize()
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
