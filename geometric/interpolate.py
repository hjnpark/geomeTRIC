import os, sys, json, copy, tempfile
from .params import parse_interpolate_args, IntpParams
from .prepare import get_molecule_engine
from .nifty import ang2bohr
from .molecule import EqualSpacing, Molecule
from .internal import (
    CartesianCoordinates,
    PrimitiveInternalCoordinates,
    DelocalizedInternalCoordinates, DistanceDifference, ReducedDistance,
)
import numpy as np

def rms_gradient(gradx):
    atomgrad = np.sqrt(np.sum((gradx.reshape(-1,3))**2, axis = 1))
    return np.sqrt(np.mean(atomgrad**2))

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
        self.na = len(self.elem)
        self.params = params

        # Coordinates in Angstrom
        self.M = M

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
                print("Handoff and DistanceDifference IC will be used")
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

        ic = "RTRIC"
        if self.DistanceIC:
            ic += '_plus'

        print(
            "Error in final interpolated vs. product structure (%s):" %ic,
            np.linalg.norm(curr_coords - self.prod),
        )


        xyz_dir = os.path.join(self.dir, "interpolated")
        if not os.path.exists(xyz_dir):
            os.makedirs(xyz_dir)


        self.interpolated_M.write(
            os.path.join(xyz_dir, "interpolated_%s.xyz" %ic)
        )

    def collect_PRIMs(self):
        print("Collecting Primitive Internal Coordinates...")
        PRIMIC, connect, addcart = self.coordsys_dict["tric-p"]
        M = copy.deepcopy(self.M)
        TRIC_p = PRIMIC(
            M,
            connect=connect,
            addcart=addcart,
            warn=False,
        )
        CoordClass, connect, addcart = self.coordsys_dict['tric']
        TRIC = CoordClass(
            M,
            Prims=TRIC_p,
            connect=connect,
            addcart=addcart,
        )

        nDiv = self.params.frames - 1

        curr_coords_f = copy.deepcopy(self.reac)
        curr_coords_b = copy.deepcopy(self.prod)
        reac = copy.deepcopy(self.reac)
        prod = copy.deepcopy(self.prod)

        Initial_PRIM = copy.deepcopy(TRIC_p)
        PRIM_most_Internals = Initial_PRIM

        for i in range(nDiv):
            TRIC.build_dlc_0(curr_coords_f)
            dq_f = TRIC.calcDiff(prod, curr_coords_f)

            new_coords_f = TRIC.newCartesian(curr_coords_f, dq_f / nDiv)

            TRIC.build_dlc_0(curr_coords_b)
            dq_b = TRIC.calcDiff(reac, curr_coords_b)

            new_coords_b = TRIC.newCartesian(curr_coords_b, dq_b / nDiv)

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
                PRIM_most_Internals = new_PRIM
                TRIC.Prims = new_PRIM

            nDiv -= 1

        if self.DistanceIC:
            str_internals = [x.__repr__() for x in PRIM_most_Internals.Internals]

            for Distance in self.DistanceIC:
                ind = str_internals.index(Distance)
                del PRIM_most_Internals.Internals[ind]
                del str_internals[ind]

            PRIM_most_Internals.add(DistanceDifference(self.atoms_ind[0], self.atoms_ind[1], self.atoms_ind[2]))
            PRIM_most_Internals.add(ReducedDistance(self.atoms_ind[0], self.atoms_ind[1], self.atoms_ind[2]))
        # Uncomment the next line to check the analytical derivative values.
        #IC.Prims.checkFiniteDifferenceGrad(self.reac)
        self.PRIMs = PRIM_most_Internals
        print("Primitive Internal Coordinates are ready.")

    def applyCartesianGrad(self, xyz, gradq, n, IC):
        xyz = xyz.reshape(-1, self.na, 3)
        Bmat = IC.wilsonB(xyz[n])
        Gx = np.array(np.matrix(Bmat.T)*np.matrix(gradq).T).flatten()
        return Gx

    def optimize(self):
        print("Optimizing the interpolated trajectory using TRIC system.")
        if not self.PRIMs:
            self.collect_PRIMs()
        M = self.M[0]
        CoordClass, connect, addcart = self.coordsys_dict['tric']
        chain = np.array(self.interpolated_coords.copy())

        TRIC = CoordClass(
            M,
            build=True,
            Prims=self.PRIMs,
            connect=connect,
            addcart=addcart,
            constraints=None,
        )
        iteration=0
        k = 1
        while True:
            if iteration > 500:
                print("Reached the maximum iteration number")
                break

            energy = []
            gradients = np.zeros_like(chain)
            for n in range(1, len(chain)-1):
                TRIC.clearCache()
                TRIC.build_dlc_0(chain[n])
                fplus = 1.0 if n == (len(chain) - 2) else 0.5
                fminus = 1.0 if n == 1 else 0.5

                drplus = TRIC.calcDiff(chain[n+1], chain[n])
                drminus = TRIC.calcDiff(chain[n-1], chain[n])

                force_s_plus = fplus*k*drplus
                force_s_minus = fminus*k*drminus

                IC_disp = force_s_plus + force_s_minus

                gradients[n] += self.applyCartesianGrad(chain, IC_disp, n, TRIC) #np.array(np.matrix(TRIC.wilsonB(chain[n]).T) * np.matrix(IC_disp).T).flatten()
                energy.append(fplus * k * np.dot(drplus, drplus) + fminus * k * np.dot(drminus,drminus))
                #if n > 1 :
                #    gradients[n-1] -= self.applyCartesianGrad(chain, force_s_minus, n-1, TRIC)

                #if n < len(chain) -2:
                #    gradients[n+1] -= self.applyCartesianGrad(chain, force_s_plus, n+1, TRIC)



            print("-----------Iteration: %i-------------" %iteration)
            rmsg = [rms_gradient(gradients[x]) for x in range(1, len(gradients)-1)]
            avgg = np.mean(rmsg)
            maxg = np.max(rmsg)
            print("Energy", sum(energy))
            print("Mean Force",avgg)
            print("Max Force",maxg)

            if avgg < 0.005 and maxg < 0.01:
                print("Converged")
                break

            print("Updating..")
            chain += gradients/np.linalg.norm(gradients)*0.01
            iteration += 1

        M.xyzs = [
                coords.reshape(-1, 3) / ang2bohr for coords in chain
            ]

        M.write("interpolated/optimized_%s.xyz" %self.params.coordsys)


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
    print("Done!")


if __name__ == "__main__":
    main()
