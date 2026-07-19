.. _mace:

MACE machine-learning potentials
================================

geomeTRIC can use `MACE <https://github.com/ACEsuit/mace>`_ machine-learning interatomic potentials (MLIPs)
for energy and gradient evaluations via an ASE calculator backend. This page describes:

* Direct geometry minimization with MACE
* Two-step optimization (MACE pre-optimization, then QC)
* Constrained optimization and constraint scans with MACE pre-opt
* Nudged elastic band (NEB) calculations driven by MACE

Prerequisites
-------------

Install ASE and MACE in the same Python environment as geomeTRIC, for example::

    pip install ase mace-torch

Download a pretrained checkpoint (first use may pull weights into ``~/.cache/mace/``).
Common options:

+------------------+------------------------------------------+----------------------------------+
| Model            | Typical use                              | Example checkpoint name          |
+==================+==========================================+==================================+
| MACE-OMOL        | Molecules, ions, open-shell systems      | ``MACE-omol-0-extra-large-1024.model`` |
+------------------+------------------------------------------+----------------------------------+
| MACE-OFF         | Neutral organic molecules                | ``MACE-OFF23_small.model``       |
+------------------+------------------------------------------+----------------------------------+
| MACE-MP          | Materials / crystals (PBE-like)          | ``20231210mace128L0_energy_epoch249model`` |
+------------------+------------------------------------------+----------------------------------+

For molecular work with net charge or spin multiplicity, **MACE-OMOL** is recommended.
Charge and multiplicity are taken from the QC input (or ASE kwargs) and stored on the ASE
atoms object as total charge and spin multiplicity; see the ASE engine notes in :ref:`engines`.

Command-line overview
---------------------

Two related ways to call MACE:

1. **Dedicated engine** (recommended for MACE)::

       --engine mace --model-path /path/to/checkpoint.model --device cpu

2. **Generic ASE engine** (any ASE calculator, including MACE)::

       --engine ase \
         --ase-class=mace.calculators.mace.MACECalculator \
         --ase-kwargs='{"model_paths":"/path/to/checkpoint.model","device":"cpu","default_dtype":"float64"}'

Shared options:

* ``--model-path`` : Path to a MACE ``.model`` file (required for ``--engine mace`` and for ``--preopt``).
* ``--device [cpu]`` : Device for MLIP evaluation (``cpu`` or ``cuda``).

See also the option index in :ref:`options`.

Direct minimization with MACE
-----------------------------

To minimize using only MACE (no QC step), pass a coordinate file (e.g. ``.xyz``) as the input
and select the MACE engine::

    geometric-optimize --engine mace \
      --model-path $HOME/.cache/mace/MACE-omol-0-extra-large-1024.model \
      --device cpu \
      molecule.xyz

This is equivalent in spirit to ``--engine ase`` with a MACE calculator class, but shorter.

Two-step optimization (MACE then QC)
------------------------------------

Often the goal is a cheap MLIP pre-relaxation followed by optimization at a user-chosen
quantum chemistry level. Enable this with ``--preopt yes`` together with ``--model-path``.
The **positional input remains the QC input file**; the initial geometry and electronic
state (charge / mult) are taken from that file.

Flow:

1. **Stage 1 (MLIP):** Optimize with MACE. Trajectory and final structure are written to
   ``[prefix]_preoptim.xyz`` (scratch under ``[prefix]_preopt.tmp``).
2. **Stage 2 (QC):** Restart optimization from the MACE geometry with the selected
   ``--engine`` (Psi4, TeraChem, etc.) and the same QC input. Final trajectory is
   ``[prefix]_optim.xyz``.

Example (Psi4 HF/STO-3G after MACE-OMOL pre-opt)::

    geometric-optimize --engine psi4 --preopt yes \
      --model-path $HOME/.cache/mace/MACE-omol-0-extra-large-1024.model \
      --device cpu \
      water6.psi4in

**Example:** ``examples/1-simple-examples/water6_preopt_psi4/``

Same system as the standard water hexamer energy-minimization examples (18 atoms), using
two-step optimization: MACE-OMOL pre-optimization followed by Psi4 HF/STO-3G.
See ``command.sh`` in that folder. Reference output in ``output.2026-7-18/``:

* Optimization cycles: ~107 (MACE) + ~48 (Psi4)
* Run time (approx.): ~2.5 minutes (155 s wall time)

.. note::
   ``--preopt`` is supported for energy minimizations and constrained jobs.
   It is **not** supported together with ``--transition`` or ``--irc``.

Constrained optimization and scans
----------------------------------

If a constraints file is provided, the same constraints are applied during the MACE
pre-optimization stage and again during the QC stage.

Single constrained minimum
""""""""""""""""""""""""""

For ``$freeze`` / ``$set`` (one target value), stage 1 and stage 2 both enforce those
constraints::

    geometric-optimize --engine psi4 --preopt yes \
      --model-path $HOME/.cache/mace/MACE-omol-0-extra-large-1024.model \
      --device cpu \
      molecule.psi4in constraints.txt

Constraint scan
"""""""""""""""

For a ``$scan`` grid, **each** scan point is optimized in two stages:

1. MACE pre-opt at that constraint value → ``[prefix]_preoptim-00i.xyz``
2. QC opt at the same value → ``[prefix]_scan-00i.xyz``

The next scan point starts from the QC geometry of the previous point (as in a normal
geomeTRIC scan). The MACE engine is built once and reused for all points.

Example scan file::

    $scan
    distance 1 2 0.95 1.05 5

Example command::

    geometric-optimize --engine psi4 --preopt yes \
      --model-path $HOME/.cache/mace/MACE-omol-0-extra-large-1024.model \
      --device cpu \
      molecule.psi4in scan_constraints.txt

See :ref:`constraints` for constraint file syntax.

NEB with MACE
-------------

The NEB driver (``geometric-neb``) uses the same engine interface for image energies and
gradients. With MACE, no separate QC input is required if the chain is a multi-frame XYZ.

Usage (``--engine mace`` needs only the multi-frame chain XYZ)::

    geometric-neb --engine mace \
      --model-path $HOME/.cache/mace/MACE-omol-0-extra-large-1024.model \
      --device cpu \
      --images 11 --align no \
      chain.xyz

No QC input file is required. For other engines (e.g. Psi4), both a QC input and a
chain XYZ remain required: ``geometric-neb --engine psi4 mol.psi4in chain.xyz``.

Useful NEB options (see :ref:`neb` and :ref:`neb_options`):

* ``--images`` : number of images to take from the input chain
* ``--align yes/no`` : align images to the first frame
* ``--optep yes`` : optimize chain endpoints with the same MACE engine before NEB
* ``--maxg`` / ``--avgg`` : force convergence thresholds (eV/Å)

**Example:** ``examples/1-simple-examples/hcn_hnc_neb_MACE/``

NEB for the HCN ↔ HNC isomerization driven by MACE only (multi-frame chain XYZ;
no QC input). See ``command.sh`` in that folder. Reference output in ``output.2026-7-18/``:

* NEB cycles: 32 chain optimization cycles (converged)
* Run time (approx.): ~1 minute (66 s)

.. note::
   Work Queue and BigChem parallelization of image gradients are intended for QC engines;
   MACE NEB evaluates images sequentially on the selected ``--device``.

Charge and spin
---------------

For MACE-OMOL, total charge and spin multiplicity matter. When using two-step optimization,
they are read from the QC input molecule (e.g. Psi4 ``0 1``). When using
``--engine mace`` with only an XYZ, defaults are charge ``0`` and multiplicity ``1`` unless
set via molecule attributes in a Python driver.

Via ASE kwargs you may also pass ``charge`` and ``mult``; geomeTRIC maps these to
ASE ``atoms.info['charge']`` and ``atoms.info['spin']`` (multiplicity) for OMOL-style models.

Limitations
-----------

* Two-step ``--preopt`` is **not** available for transition-state search or IRC.
* MLIP quality depends on the checkpoint and how close the system is to the training domain.
* Absolute energies from materials-trained models (MACE-MP) are not comparable to hybrid
  molecular DFT; prefer OMOL for organic / ionic molecular problems.

Related pages
-------------

* :ref:`engines` — ASE and other engines
* :ref:`options` — full CLI option list
* :ref:`constraints` — constraint file format
* :ref:`neb` — NEB theory and usage
