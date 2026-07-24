// Variance-reduction demonstration: shared setup for the symmetrization ON/OFF arms.
//
// NOT part of the DCA++ source tree. Parked under .claude/symmetry_project/dev/ per the
// variance-demo plan; built out-of-tree.
//
// The two arms differ in exactly one token: the point group the model declares.
//   ON  : square_lattice<D4>             -> CanDeriveSymmetry true  -> derive-authoritative, 8 ops
//   OFF : square_lattice<no_symmetry<2>> -> CanDeriveSymmetry false -> legacy path, identity only
// Everything below is byte-identical between the arms.

#ifndef VARIANCE_DEMO_POINT_GROUP
#error "Define VARIANCE_DEMO_POINT_GROUP before including symmetry_variance_setup.hpp"
#endif

#ifndef DCA_DEV_VARIANCE_DEMO_SETUP_HPP
#define DCA_DEV_VARIANCE_DEMO_SETUP_HPP

#include <complex>
#include <string>
#include <vector>

#include "dca/config/threading.hpp"
#include "dca/math/random/random.hpp"
#include "dca/parallel/mpi_concurrency/mpi_concurrency.hpp"
#include "dca/phys/dca_data/dca_data.hpp"
#include "dca/phys/dca_loop/dca_loop_data.hpp"
#include "dca/phys/dca_step/cluster_solver/ctaux/ctaux_cluster_solver.hpp"
#include "dca/phys/dca_step/cluster_solver/stdthread_qmci/stdthread_qmci_cluster_solver.hpp"
#include "dca/phys/domains/cluster/cluster_symmetry.hpp"
#include "dca/phys/domains/cluster/symmetries/point_groups/2d/holohedries_2d.hpp"
#include "dca/phys/domains/cluster/symmetries/point_groups/no_symmetry.hpp"
#include "dca/phys/models/analytic_hamiltonians/square_lattice.hpp"
#include "dca/phys/models/tight_binding_model.hpp"
#include "dca/phys/parameters/parameters.hpp"
#include "dca/profiling/null_profiler.hpp"
#include "dca/util/type_utils.hpp"

namespace dca {
namespace variance_demo {
// dca::variance_demo::

#ifdef DCA_HAVE_GPU
constexpr dca::linalg::DeviceType device = dca::linalg::GPU;
#else
constexpr dca::linalg::DeviceType device = dca::linalg::CPU;
#endif

using Lattice = dca::phys::models::square_lattice<VARIANCE_DEMO_POINT_GROUP>;
using Model = dca::phys::models::TightBindingModel<Lattice>;
using Concurrency = dca::parallel::MPIConcurrency;
using RandomNumberGenerator = dca::math::random::StdRandomWrapper<std::mt19937_64>;

// square_lattice has complex_g0 == false, so this is `double`; written the production way so the
// header survives being pointed at a complex-G0 lattice.
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

}  // namespace variance_demo
}  // namespace dca

#endif  // DCA_DEV_VARIANCE_DEMO_SETUP_HPP
