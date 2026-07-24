# Symmetrization variance-reduction demonstration

Parked demo (NOT committed, NOT part of the M3′ PR). Shows that imposing the H0-derived point group
on the single-particle G reduces Monte-Carlo variance without shifting the mean, by an amount set by
each entry's symmetry-orbit size. Two claims:

- **(A)** symmetrization reduces variance — control, on `square_lattice<D4>`.
- **(B)** the derivation found symmetry never declared, and it buys real reduction — on **FeAs**
  (declared 2 ops, derived 8).

Both arms are two binaries in **one build**, differing only in the declared point group. Each run
reads G **after `finalize()`** (the real production path where CT-AUX symmetrizes `M_r_w`), and dumps
`data_.G_k_w` to HDF5 tagged with its seed, the k-mesh, and the live imposed op count.

## Files
- `symmetry_variance_setup.hpp` / `ctaux_variance_demo_{on,off}.cpp` — square arms (D4 vs no_symmetry).
- `fe_as_variance_setup.hpp` / `fe_as_variance_demo_{on,off}.cpp` — FeAs arms. OFF subclasses FeAs and
  overrides `DCA_point_group` to `no_symmetry` (FeAsLattice ignores its own template arg), with a
  demo-local `ModelParameters` specialization so the subclass is still recognized as FeAs.
- `variance_demo_main.inc` — shared thin body (byte-identical between arms).
- `*_input.template.json` — Nc, 1 DCA iteration, `__SEED__` placeholder.
- `run_replicas.sh` / `run_replicas_feas.sh` — drive N paired replicas (shared prime-stride seeds).
- `CMakeLists.txt` — builds the four executables (guarded by `DCA_WITH_VARIANCE_DEMO`).

## Build

The demo is a standalone top-level CMake project that pulls DCA in as a subdirectory — **DCA itself
needs no modification**. See `../SETUP.md` for the full command. In short:

```
cmake -S <this repo root> -B <build> -DDCA_SOURCE_DIR=<DCA checkout> [DCA options...]
cmake --build <build> -j --target ctaux_variance_demo_on ctaux_variance_demo_off \
                                  fe_as_variance_demo_on fe_as_variance_demo_off
```

`variance_demo/CMakeLists.txt` + `../patches/cmake-hook.patch` are the **fallback** path only (the
older guarded `add_subdirectory` hook inside DCA's CMakeLists, activated by `DCA_WITH_VARIANCE_DEMO`);
they are unused by the default build.

## Run + analyze
```
BIN_DIR=<build> ./run_replicas.sh       64 8   # square  -> runs/{on,off}/G_<seed>.hdf5
BIN_DIR=<build> ./run_replicas_feas4.sh 64 8   # FeAs 4x4 -> runs_feas4/{on,off}/G_<seed>.hdf5
```
Then the notebooks in `../notebooks/` (`symmetry_variance_demo.ipynb` = square/claim A,
`symmetry_variance_demo_feas.ipynb` = FeAs/claim B), with helpers `variance_demo_lib.py` and
`variance_demo_feas_lib.py` — plain numpy/h5py/matplotlib, no project imports.

## Notes / limits (see ../../variance-demo-plan.md §9)
- CT-AUX only here: CT-INT does not compile in this tree (submatrix-walker header). CT-AUX is the
  plan's primary solver anyway (exercises M3′'s r-space branch).
- One DCA iteration; plain DCA (not DCA+); single-particle only.
- Square 4×4 has max k-orbit 4, and orbit-mate MC noise is strongly correlated (ρ≈0.86–0.94), so the
  measured reduction is ρ-limited (~1.05–1.37) but pinned exactly at 1 for the Γ/(π,π) singletons and
  ω-independent. FeAs adds band-space orbits (a size-8 interband orbit) that the declared group cannot
  reach — the concrete claim-B payoff.
