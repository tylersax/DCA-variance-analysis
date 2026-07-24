# Setup on a new machine

The demo is a **standalone CMake project that pulls DCA in as a subdirectory**, so the DCA
checkout needs **no modification at all** — no patch, no edit to its `CMakeLists.txt`.
Nothing in this repo ever has to live inside the DCA tree.

Paths below are the ones in use on the **current (128-core) machine**; substitute your own if the
layout differs.

| role | path here |
|---|---|
| this repo | `/home/tsax10/dca/analysis` |
| DCA source | `/home/tsax10/dca/source/DCA` |
| build tree | `/home/tsax10/dca/build` |
| dependency prefix | `/home/tsax10/conda/envs/qe` |

## 0. Dependencies

DCA needs HDF5 (C **and** C++ components), FFTW3, LAPACK and MPI. On this machine the system has
none of them in usable form (no MPI, no HDF5 headers, no LAPACK `.so` symlink) and there is no
passwordless sudo — but the `qe` conda env carries the complete set (OpenMPI 4.1.6, HDF5 1.14.3
with C++ bindings, FFTW3, LAPACK). Point CMake at that prefix; the compiler is the **system**
`g++` 13.3, since that env ships none.

MPI is a compile/link requirement only — the demo main constructs a `dca::parallel::MPIConcurrency`
— not a launch one. The replica drivers run each replica as a single serial-but-threaded process
and never invoke `mpirun`.

## 1. Get the DCA source (M3′ branch)

```bash
git clone https://github.com/tylersax/DCA /home/tsax10/dca/source/DCA
cd /home/tsax10/dca/source/DCA && git checkout symm-m3
```

`symm-m3` carries the M3′ derive-authoritative commit on top of `symm-derive-p` (the P-search,
a hard prerequisite). `git status` there should be clean — if it isn't, something drifted.

## 2. Get this repo

```bash
git clone https://github.com/tylersax/DCA-variance-analysis /home/tsax10/dca/analysis
```

## 3. Configure + build

```bash
CONDA_PREFIX=/home/tsax10/conda/envs/qe

cmake -S /home/tsax10/dca/analysis -B /home/tsax10/dca/build \
  -DDCA_SOURCE_DIR=/home/tsax10/dca/source/DCA \
  -DCMAKE_PREFIX_PATH=${CONDA_PREFIX} \
  -DCMAKE_BUILD_TYPE=Release \
  -DDCA_BUILD_DCA=OFF -DDCA_BUILD_ANALYSIS=OFF -DDCA_WITH_TESTS_FAST=OFF \
  -DDCA_CLUSTER_SOLVER=CT-AUX -DDCA_LATTICE=square -DDCA_MODEL=tight-binding \
  -DDCA_POINT_GROUP=D4 -DDCA_RNG=std::mt19937_64 \
  -DDCA_WITH_MPI=ON -DDCA_WITH_THREADED_SOLVER=ON \
  -DTEST_RUNNER=${CONDA_PREFIX}/bin/mpirun -DMPIEXEC_NUMPROC_FLAG=-n

cmake --build /home/tsax10/dca/build -j$(nproc) \
  --target ctaux_variance_demo_on ctaux_variance_demo_off \
           fe_as_variance_demo_on fe_as_variance_demo_off
```

Smoke check — the op counts are the honest witness of which arm is which:

```bash
sed 's/__SEED__/1/' /home/tsax10/dca/analysis/variance_demo/square_variance_input.template.json > /tmp/s.json
/home/tsax10/dca/build/ctaux_variance_demo_on  /tmp/s.json /tmp/on.hdf5  | grep 'live super'   # expect ops=8
/home/tsax10/dca/build/ctaux_variance_demo_off /tmp/s.json /tmp/off.hdf5 | grep 'live super'   # expect ops=1
```

The binaries link against the conda env's shared libraries, but CMake bakes an RPATH into them, so
no `LD_LIBRARY_PATH` is needed to run them — verified with `ldd`. If you ever relocate the conda env,
that breaks and `LD_LIBRARY_PATH=${CONDA_PREFIX}/lib` is the fix.

## 4. Run replicas

`BIN_DIR` must point at the built binaries — it defaults to `/home/tsax10/dca/build`, so on this
machine it can be omitted:

```bash
cd /home/tsax10/dca/analysis/variance_demo
./run_replicas.sh       128 32    # square, N=128/arm, 32 concurrent
./run_replicas_feas4.sh 128 32    # FeAs 4x4 (the claim-B run)
```

Second argument is concurrency. Each run uses ~4 threads (2 walkers + 2 accumulators) and ~80 MB,
so `concurrency ≈ cores/4` — **32 on this 128-core machine**. Outputs land in
`variance_demo/runs*/{on,off}/`.

**Sizing note:** the FeAs 4×4 run is *coarse-graining-bound, not measurement-bound* (~3m50s/run at
only 4000 measurements on a Ryzen 5 3600). On more cores, scale **replicas wide** rather than raising
measurements per run — replica count is what tightens the CI. N=32→128 roughly halves the interval
on the headline (2.45 [2.12, 2.90]).

## 5. Analysis

```bash
python3 -m venv /home/tsax10/dca/analysis/.venv
/home/tsax10/dca/analysis/.venv/bin/pip install numpy h5py matplotlib nbformat jupyter nbconvert
cd /home/tsax10/dca/analysis/notebooks
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace symmetry_variance_demo.ipynb
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace symmetry_variance_demo_feas.ipynb
```

Notebooks default to `../variance_demo/runs*`; override with `VARIANCE_DEMO_RUNS=<path>`.
The committed notebooks still contain the **original results from the first machine** — re-executing
overwrites them, so branch first if you want to keep that record.

## Fallback build path

If the inversion ever fights a future DCA version, `patches/cmake-hook.patch` restores the old
approach (a guarded `add_subdirectory` hook inside DCA's `CMakeLists.txt`, used with
`-DDCA_WITH_VARIANCE_DEMO=ON -DDCA_VARIANCE_DEMO_DIR=<...>/variance_demo`). It dirties a tracked DCA
file, which is why it is not the default.

## Gotchas found the hard way

- **CT-INT does not compile** in this tree (its submatrix-walker header fails to parse with GCC 13).
  CT-AUX only. This is pre-existing, unrelated to the demo.
- **FeAs 2×2 is vacuous for claim B**: interband G is identically 0 there (H0 interband ∝ sin kx·sin ky
  = 0 on that mesh) and the k-orbits are trivial. Use the 4×4 template for claim B.
- **Seeds must be explicit integers spaced well apart** (the drivers use a 1000003 stride). DCA seeds
  streams by `hash(global_id + seed)`, so seeds closer together than ranks×walkers share RNG streams
  and silently deflate variance in both arms. Never use `"seed": "random"` here.
