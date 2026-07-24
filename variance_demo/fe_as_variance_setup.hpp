// Variance-demo setup for the FeAs lattice (claim B: derived group is larger than declared).
// Same structure as symmetry_variance_setup.hpp; only the Lattice differs (2-band FeAs). The point
// group is still the one token that separates the arms:
//   ON  : FeAsLattice<FeAsPointGroup>   -> derive-authoritative expands the DECLARED 2 ops to 8
//   OFF : FeAsLattice<no_symmetry<2>>   -> legacy identity-only path, 1 op
#ifndef DCA_DEV_FEAS_VARIANCE_DEMO_SETUP_HPP
#define DCA_DEV_FEAS_VARIANCE_DEMO_SETUP_HPP

#include <complex>
#include <string>

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
#include "dca/phys/models/analytic_hamiltonians/fe_as_lattice.hpp"
#include "dca/phys/models/tight_binding_model.hpp"
#include "dca/phys/parameters/parameters.hpp"
#include "dca/profiling/null_profiler.hpp"
#include "dca/util/type_utils.hpp"

namespace dca {
namespace variance_demo {

#ifdef DCA_HAVE_GPU
constexpr dca::linalg::DeviceType device = dca::linalg::GPU;
#else
constexpr dca::linalg::DeviceType device = dca::linalg::CPU;
#endif

// FeAsLattice ignores its template argument and hardcodes bilayer_lattice<FeAsPointGroup> as its
// base, so DCA_point_group is *always* FeAsPointGroup (2 ops) -- the point-group token cannot switch
// symmetrization off the way it can for square_lattice. To get a genuine OFF arm we subclass and
// override DCA_point_group to no_symmetry, which is what the derive gate (is_no_symmetry) keys on.
using FeAs = dca::phys::models::FeAsLattice<dca::phys::models::FeAsPointGroup>;
#ifdef VARIANCE_DEMO_FEAS_OFF
struct FeAsNoSym : public FeAs {
  using DCA_point_group = dca::phys::domains::no_symmetry<2>;
};
using Lattice = FeAsNoSym;   // OFF: legacy identity-only path, 1 imposed op
#else
using Lattice = FeAs;        // ON:  declared 2 ops -> derive-authoritative expands to 8
#endif
using Model = dca::phys::models::TightBindingModel<Lattice>;
using Concurrency = dca::parallel::MPIConcurrency;
using RandomNumberGenerator = dca::math::random::StdRandomWrapper<std::mt19937_64>;
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

#ifdef VARIANCE_DEMO_FEAS_OFF
// FeAsNoSym is a fresh type, so it does NOT match the ModelParameters<TightBindingModel<FeAsLattice
// <PointGroup>>> specialization keyed on the exact template -- the input-JSON accessors get_t1.. get_J
// would be missing. This demo-local specialization (NOT a production edit) delegates to the FeAs one
// by inheritance, so the OFF arm reads the identical FeAs-model input block.
#include "dca/phys/parameters/model_parameters.hpp"
namespace dca {
namespace phys {
namespace params {
template <>
class ModelParameters<dca::phys::models::TightBindingModel<dca::variance_demo::FeAsNoSym>>
    : public ModelParameters<dca::phys::models::TightBindingModel<
          dca::phys::models::FeAsLattice<dca::phys::models::FeAsPointGroup>>> {};
}  // namespace params
}  // namespace phys
}  // namespace dca
#endif

#endif  // DCA_DEV_FEAS_VARIANCE_DEMO_SETUP_HPP
