"""
A set of tests for energy minimization
"""

import copy
import numpy as np
import json, os, shutil
from . import addons
import geometric
import pytest
import itertools
import time
import subprocess
from geometric.molecule import Molecule

localizer = addons.in_folder
datad = addons.datad
exampled = addons.exampled

@addons.using_psi4
def test_hcccn_minimize_psi4(localizer):
    """
    Optimize a linear HCCCN molecule
    """
    shutil.copy2(os.path.join(datad, 'hcccn.psi4in'), os.path.join(os.getcwd(), 'hcccn.psi4in'))
    progress = geometric.optimize.run_optimizer(engine='psi4', input='hcccn.psi4in', converge=['gmax', '1.0e-5'], 
                                                nt=4, reset=False, trust=0.1, tmax=0.3)
    e_ref = -167.6136203991
    assert progress.qm_energies[-1] < (e_ref + 1e-5)
    # Check that the optimization converged in less than 10 steps
    assert len(progress) < 10

@addons.using_quick
def test_water2_minimize_quick(localizer):
    """
    Optimize a water dimer 
    """
    shutil.copy2(os.path.join(exampled, "1-simple-examples","water2_quick","Water2.qkin"), os.path.join(os.getcwd(), "Water2.qkin"))
    progress = geometric.optimize.run_optimizer(engine='quick', input='Water2.qkin', converge=['gmax', '1.0e-5'],
                                                nt=4, reset=False, trust=0.1, tmax=0.3)
    e_ref = -149.9412443430
    assert progress.qm_energies[-1] < (e_ref + 1e-5)
    # Check that the optimization converged in less than 25 steps (took 20 steps in LPW's local test.)
    assert len(progress) < 25

@addons.using_quick
def test_water2_minimize_quick_converge_maxiter(localizer):
    """
    Optimize a water dimer, but "converge" when maximum number of iterations (5) is reached.
    """
    shutil.copy2(os.path.join(exampled, "1-simple-examples","water2_quick","Water2.qkin"), os.path.join(os.getcwd(), "Water2.qkin"))
    progress = geometric.optimize.run_optimizer(engine='quick', input='Water2.qkin', converge=['gmax', '1.0e-5', 'maxiter'],
                                                nt=4, reset=False, trust=0.1, tmax=0.3, maxiter=5)
    assert len(progress) == 6
    # e_ref = -149.9412443430
    # assert progress.qm_energies[-1] < (e_ref + 1e-5)
    # # Check that the optimization converged in less than 25 steps (took 20 steps in LPW's local test.)
    # assert len(progress) < 25


# ---------------------------------------------------------------------------
# Two-step MLIP pre-optimization (MACE) then QC (Psi4)
# ---------------------------------------------------------------------------

def _mace_mp_small_path():
    """
    Return path to a local MACE-MP small checkpoint.

    Prefer the standard cache location; if missing, download once via mace_mp.
    """
    cached = os.path.expanduser(
        "~/.cache/mace/20231210mace128L0_energy_epoch249model"
    )
    if os.path.isfile(cached):
        return cached
    from mace.calculators import mace_mp

    mace_mp(model="small", device="cpu", default_dtype="float64")
    if os.path.isfile(cached):
        return cached
    cache_dir = os.path.expanduser("~/.cache/mace")
    if os.path.isdir(cache_dir):
        for name in os.listdir(cache_dir):
            path = os.path.join(cache_dir, name)
            if os.path.isfile(path) and ("mace" in name.lower() or name.endswith("model")):
                return path
    raise RuntimeError("Could not locate a MACE model checkpoint for tests")


def _write_water_psi4(path="water_preopt.psi4in"):
    shutil.copy2(os.path.join(datad, "water_preopt.psi4in"), path)
    return path


def _angle_deg(xyz, i=0, j=1, k=2):
    """Angle j-i-k in degrees from a (N,3) coordinate array in Angstrom."""
    v1 = xyz[j] - xyz[i]
    v2 = xyz[k] - xyz[i]
    c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


@addons.using_psi4
@addons.using_mace
@addons.using_ase
def test_preopt_minimize_psi4(localizer):
    """
    Slightly bent water: MACE pre-opt then Psi4 HF/STO-3G minimization.
    """
    _write_water_psi4("water_preopt.psi4in")
    model = _mace_mp_small_path()

    progress = geometric.optimize.run_optimizer(
        engine="psi4",
        input="water_preopt.psi4in",
        preopt=True,
        model_path=model,
        device="cpu",
        prefix="preopt_min",
        converge=["set", "GAU_LOOSE"],
        maxiter=50,
    )

    assert os.path.isfile("preopt_min_preoptim.xyz")
    assert os.path.isfile("preopt_min_optim.xyz")
    assert os.path.isdir("preopt_min_preopt.tmp")
    assert os.path.isdir("preopt_min.tmp")

    assert len(progress) >= 1
    assert len(progress.qm_energies) == len(progress)

    final = progress.xyzs[-1]
    ang = _angle_deg(final)
    assert 95.0 < ang < 120.0

    assert np.isfinite(progress.qm_energies[-1])
    assert progress.qm_energies[-1] < -70.0  # HF/STO-3G water is ~ -75 Ha


@addons.using_psi4
@addons.using_mace
@addons.using_ase
def test_preopt_constrained_scan_psi4(localizer):
    """
    O-H distance scan with two points: each point uses MACE then Psi4.
    """
    _write_water_psi4("water_preopt.psi4in")
    model = _mace_mp_small_path()

    with open("scan_constraints.txt", "w") as f:
        f.write("$scan\n")
        f.write("distance 1 2 0.95 1.05 2\n")

    progress = geometric.optimize.run_optimizer(
        engine="psi4",
        input="water_preopt.psi4in",
        constraints="scan_constraints.txt",
        preopt=True,
        model_path=model,
        device="cpu",
        prefix="preopt_scan",
        converge=["set", "GAU_LOOSE"],
        maxiter=40,
    )

    assert os.path.isfile("preopt_scan_preoptim-001.xyz")
    assert os.path.isfile("preopt_scan_preoptim-002.xyz")
    assert os.path.isfile("preopt_scan_scan-001.xyz")
    assert os.path.isfile("preopt_scan_scan-002.xyz")
    assert os.path.isfile("scan-final.xyz")

    assert len(progress) >= 2

    m1 = Molecule("preopt_scan_scan-001.xyz")
    m2 = Molecule("preopt_scan_scan-002.xyz")
    d1 = np.linalg.norm(m1.xyzs[-1][0] - m1.xyzs[-1][1])
    d2 = np.linalg.norm(m2.xyzs[-1][0] - m2.xyzs[-1][1])
    assert abs(d1 - 0.95) < 0.05
    assert abs(d2 - 1.05) < 0.05
    assert d2 > d1
