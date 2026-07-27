// Shared type setup for the symmetrization-variance driver.
//
// NOT part of the DCA++ source tree -- built out-of-tree as a standalone project that pulls DCA in
// as a subdirectory (see CMakeLists.txt). Adapted from the deprecated variance_demo setup, but with
// a single arm per model: we run each model's STOCK (declared, derive-authoritative) point group.
// The raw per-rank sample dumped by the driver is independent of the declared group anyway
// (local_G_k_w(false) skips symmetrization), and the declared group is exactly the one whose
// variance reduction we serialize an orbit table for.
//
// The including .cpp must define SYMM_VARIANCE_LATTICE (the Lattice type) and include the lattice's
// header + its point group before including this file.

#ifndef SYMM_VARIANCE_LATTICE
#error "Define SYMM_VARIANCE_LATTICE (the Lattice type) before including symm_variance_setup.hpp"
#endif

#ifndef SYMM_VARIANCE_SETUP_HPP
#define SYMM_VARIANCE_SETUP_HPP

#include <complex>

#include "dca/config/threading.hpp"
#include "dca/math/random/random.hpp"
#include "dca/parallel/mpi_concurrency/mpi_concurrency.hpp"
#include "dca/phys/dca_data/dca_data.hpp"
#include "dca/phys/dca_loop/dca_loop_data.hpp"
#include "dca/phys/dca_step/cluster_solver/ctaux/ctaux_cluster_solver.hpp"
#include "dca/phys/dca_step/cluster_solver/stdthread_qmci/stdthread_qmci_cluster_solver.hpp"
#include "dca/phys/models/tight_binding_model.hpp"
#include "dca/phys/parameters/parameters.hpp"
#include "dca/profiling/null_profiler.hpp"
#include "dca/util/type_utils.hpp"

namespace dca {
namespace symm_variance {

#ifdef DCA_HAVE_GPU
constexpr dca::linalg::DeviceType device = dca::linalg::GPU;
#else
constexpr dca::linalg::DeviceType device = dca::linalg::CPU;
#endif

using Lattice = SYMM_VARIANCE_LATTICE;
using Model = dca::phys::models::TightBindingModel<Lattice>;
using Concurrency = dca::parallel::MPIConcurrency;
using RandomNumberGenerator = dca::math::random::StdRandomWrapper<std::mt19937_64>;

// Real for a real-G0 lattice (square), complex otherwise; written the production way so the header
// survives being pointed at a complex-G0 lattice.
using DemoScalar = typename dca::util::ScalarSelect<double, Lattice::complex_g0>::type;

using dca::ClusterSolverId;

template <ClusterSolverId CS_NAME>
using ParametersType =
    dca::phys::params::Parameters<Concurrency, Threading, dca::profiling::NullProfiler, Model,
                                  RandomNumberGenerator, CS_NAME,
                                  dca::NumericalTraits<dca::util::RealAlias<DemoScalar>, DemoScalar>>;

template <ClusterSolverId name>
using DcaData = dca::phys::DcaData<ParametersType<name>>;

template <ClusterSolverId name>
struct ClusterSolverSelector;
template <>
struct ClusterSolverSelector<ClusterSolverId::CT_AUX> {
  using type = dca::phys::solver::CtauxClusterSolver<device, ParametersType<ClusterSolverId::CT_AUX>,
                                                     DcaData<ClusterSolverId::CT_AUX>>;
};

template <ClusterSolverId name>
using ThreadedSolver =
    dca::phys::solver::StdThreadQmciClusterSolver<typename ClusterSolverSelector<name>::type>;

}  // namespace symm_variance
}  // namespace dca

#endif  // SYMM_VARIANCE_SETUP_HPP
