import os, sys, json, copy, tempfile
import networkx as nx

from .params import parse_interpolate_args, IntpParams
from .prepare import get_molecule_engine
from .nifty import ang2bohr
from .molecule import EqualSpacing, Molecule
from .internal import (
    CartesianCoordinates,
    PrimitiveInternalCoordinates,
    DelocalizedInternalCoordinates, Handoff, ReducedDistance,
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
        self.DistanceIC = []
        self.atoms_ind = []

    def analyze_M(self):
        ini_G = self.M_ini.topology
        fin_G = self.M_fin.topology
        G_diff = nx.difference(fin_G, ini_G)

        #if len(G_diff.edges) > 1:
        #    raise RuntimeError("It can't interpolate trajectories containing more than one chemical reaction step.")

        PRIMIC, connect, addcart = self.coordsys_dict["tric-p"]
        PRIM1 = PRIMIC(self.M_ini,
                       build = True,
                       connect = connect,
                       addcart = addcart,
                       constraints=None,
                       warn=False)

        PRIM2 = PRIMIC(self.M_fin,
                       build = True,
                       connect = connect,
                       addcart = addcart,
                       constraints=None,
                       warn=False)

        bond1 = [x.__repr__() for x in PRIM1.Internals if x not in PRIM2.Internals and x.__repr__().split()[0] == 'Distance']
        bond2 = [x.__repr__() for x in PRIM2.Internals if x not in PRIM1.Internals and x.__repr__().split()[0] == 'Distance']

        if len(bond1) == 1 and len(bond2) == 1:
            atoms_ind = set([int(x)-1 for x in bond1[0].split()[-1].split('-') + bond2[0].split()[-1].split('-')])
            if len(atoms_ind) == 3:
                print("Handoff and ReducedDistance IC will be used")
                self.atoms_ind = [x for x in atoms_ind]
                self.DistanceIC = [bond1[0], bond2[0]]

    def simple_interpolate(self):
        ic = self.params.coordsys
        M = copy.deepcopy(self.M)
        nDiv = self.params.frames - 1
        curr_coords = self.reac.copy()
        coord_list = [curr_coords]
        CoordClass, connect, addcart = self.coordsys_dict[ic.lower()]
        IC = CoordClass(
            M,
            build=True,
            connect=connect,
            addcart=addcart,
            constraints=None,
        )

        #IC.Prims = self.PRIMs
        dq = IC.calcDiff(self.prod, self.reac)
        for i in range(nDiv):


            new_coords = IC.newCartesian(curr_coords, dq / nDiv)

            curr_coords = new_coords.copy()

            coord_list.append(curr_coords)

        self.interpolated_M.xyzs = [
            coords.reshape(-1, 3) / ang2bohr for coords in coord_list
        ]

        self.interpolated_coords = coord_list

        print(
            "Error in final interpolated vs. product structure (%s):" % ic,
            np.linalg.norm(curr_coords - self.prod),
        )

        xyz_dir = os.path.join(self.dir, "interpolated")
        if not os.path.exists(xyz_dir):
            os.makedirs(xyz_dir)
        self.interpolated_M.write(
            os.path.join(xyz_dir, "interpolated_%s.xyz" % ic)
        )

    def RTRIC_interpolate(self):
        ic = self.params.coordsys
        M = copy.deepcopy(self.M)
        nDiv = self.params.frames - 1
        curr_coords = self.reac.copy()
        coord_list = [curr_coords]
        CoordClass, connect, addcart = self.coordsys_dict[ic.lower()]
        IC = CoordClass(
            M,
            connect=connect,
            addcart=addcart,
            constraints=None,
            Prims=self.PRIMs,
        )


        for i in range(nDiv):

            IC.build_dlc_0(curr_coords)

            dq = IC.calcDiff(self.prod, curr_coords)
            new_coords = IC.newCartesian(curr_coords, dq / nDiv)

            curr_coords = new_coords.copy()

            coord_list.append(curr_coords)
            nDiv -= 1

        self.interpolated_M.xyzs = [
            coords.reshape(-1, 3) / ang2bohr for coords in coord_list
        ]

        self.interpolated_coords = coord_list


        print(
            "Error in final interpolated vs. product structure (RTRIC%s):" % ic,
            np.linalg.norm(curr_coords - self.prod),
        )


        xyz_dir = os.path.join(self.dir, "interpolated")
        if not os.path.exists(xyz_dir):
            os.makedirs(xyz_dir)
        if self.DistanceIC:
            ic += '_plus'
        self.interpolated_M.write(
            os.path.join(xyz_dir, "interpolated_RTRIC_%s.xyz" % ic)
        )


    def optimize(self, stepsize = 0.1):
        print("Optimizing the interpolated trajectory using TRIC system.")
        str_internals = [x.__repr__() for x in self.PRIMs.Internals]

        for Distance in self.DistanceIC:
            ind = str_internals.index(Distance)
            del self.PRIMs.Internals[ind]
            del str_internals[ind]

        self.PRIMs.add(Handoff(self.atoms_ind[0], self.atoms_ind[1], self.atoms_ind[2]))
        self.PRIMs.add(ReducedDistance(self.atoms_ind[0], self.atoms_ind[1], self.atoms_ind[2]))

        M = self.M[0]
        CoordClass, connect, addcart = self.coordsys_dict['tric']
        def calc_E_F(chain):
            k = 5
            E = []
            F = []
            for i in range(len(chain)):
                M.xyzs = [chain[i].reshape(-1, 3) / ang2bohr]
                IC = CoordClass(
                   M,
                   build=True,
                   Prims=self.PRIMs,
                   connect=connect,
                   addcart=addcart,
                   constraints=None,
                )
                #IC.build_dlc_0(chain[i])
                #IC.Vecs = self.Vecs_list[i]
                #IC.Internals = self.Internals_list[i]
                d1, d2 = 0, 0
                if i > 0:
                    d1 = IC.calcDiff(chain[i-1], chain[i])
                if i < len(chain) -1:
                    d2 = IC.calcDiff(chain[i+1], chain[i])

                d1_E = np.sum(np.square(d1))
                d2_E = np.sum(np.square(d2))
                E.append(k*(d1_E + d2_E))

                d1_F = -d1
                d2_F = d2
                F.append(2*k*(d1_F + d2_F))

            return np.array(E), np.array(F) # (nimages, IC)

        chain_coords = self.interpolated_coords.copy()
        E_array, F_array = calc_E_F(chain_coords)#, IC)
        tot_E = np.sum(E_array)
        mean_F = np.mean(np.abs(F_array))
        max_F = np.max(np.abs(F_array))

        iteration = 0
        stepsize = 0.1
        while True:
            if iteration > 100:
                print("Reached the maximum iteration number")
                break
            new_coords = []
            print("-----------Iteration: %i-------------" %iteration)
            print("Total Energy",tot_E)
            print("Mean Force",mean_F)
            print("Max Force",max_F)

            for i, forces in enumerate(F_array):
                if i == 0 or i == len(F_array)-1:
                    new_coords.append(chain_coords[i])
                else:
                    M.xyzs = [chain_coords[i].reshape(-1, 3) / ang2bohr]
                    forces /= np.linalg.norm(forces)
                    IC = CoordClass(
                       M,
                       build=True,
                       connect=connect,
                       addcart=addcart,
                       constraints=None,
                    )
                    new_coords.append(IC.newCartesian(chain_coords[i], forces*stepsize))

            new_E_array, new_F_array = calc_E_F(new_coords)
            del_E_array = np.abs(new_E_array - E_array)
            new_tot_E = np.sum(new_E_array)
            new_mean_F = np.mean(np.abs(new_F_array))
            new_max_F = np.max(np.abs(new_F_array))

            del_E = new_tot_E-tot_E
            del_mean_F = new_mean_F-mean_F
            del_max_F = new_max_F-max_F
            print("\ndel Total Energy",del_E)
            print("max del E", np.max(del_E_array))
            print("del Energy percentage", del_E/tot_E * 100)
            print("del Mean Force",del_mean_F)
            print("del Max Force",del_max_F)

            if mean_F < 0.025 and max_F < 0.05:
                print("Converged")
                break

            print("Updating..")
            F_array = new_F_array.copy()
            tot_E = new_tot_E.copy()
            mean_F = new_mean_F.copy()
            max_F = new_max_F.copy()
            chain_coords = new_coords.copy()
            iteration += 1

        M.xyzs = [
                coords.reshape(-1, 3) / ang2bohr for coords in chain_coords
            ]

        M.write("interpolated/optimized_%s.xyz" %self.params.coordsys)

    def collect_PRIMs(self):
        print("Collecting Primitive Internal Coordinates...")
        PRIMIC, connect, addcart = self.coordsys_dict["tric-p"]
        M = copy.deepcopy(self.M)

        CoordClass, connect, addcart = self.coordsys_dict['tric']
        IC = CoordClass(
            M,
            build=True,
            connect=connect,
            addcart=addcart,
            constraints=None,
        )

        nDiv = self.params.frames - 1

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
            #print(M.xyzs[0][0])
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

        if self.DistanceIC:
            str_internals = [x.__repr__() for x in PRIM_most_Internals.Internals]

            for Distance in self.DistanceIC:
                ind = str_internals.index(Distance)
                del PRIM_most_Internals.Internals[ind]
                del str_internals[ind]

            PRIM_most_Internals.add(Handoff(self.atoms_ind[0], self.atoms_ind[1], self.atoms_ind[2]))
            PRIM_most_Internals.add(ReducedDistance(self.atoms_ind[0], self.atoms_ind[1], self.atoms_ind[2]))

        #IC.Prims.checkFiniteDifferenceGrad(self.reac)
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

class nullengine:
    def __init__(self, M):
        self.M = M 
    

def main():
    args_dict = parse_interpolate_args(sys.argv[1:])
    args_dict["interpolation"] = True
    M = Molecule(args_dict['coords'])[0]
    args_dict["customengine"] = nullengine(M)
    RTRIC = False

    if args_dict['coordsys'] == 'tric-r':
        RTRIC = True
        args_dict['coordsys'] = 'tric'

    params = IntpParams(**args_dict)
    M, engine = get_molecule_engine(**args_dict)

    TRIC = Interpolate(params, M, engine)

    if RTRIC:
        TRIC.analyze_M()
        TRIC.collect_PRIMs()
        TRIC.RTRIC_interpolate()
    else:
        TRIC.simple_interpolate()

    #TRIC.optimize()
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
