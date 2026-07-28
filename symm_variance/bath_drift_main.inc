// Shared driver body for the BATH-DRIFT check (ROADMAP 3e), included by the per-model .cpp files
// after they define SYMM_VARIANCE_LATTICE and SYMM_VARIANCE_MODEL_LABEL.
//
// WHAT THIS IS FOR. Every number in this project is measured at the BARE bath -- iteration 1 of a
// cold-started DCA loop, where the walker samples the non-interacting G0 (see ROADMAP Conventions).
// A production run instead calls the solver ~N times at similar cost each, on a bath that is rebuilt
// from the previous iteration's measured Sigma. A total-compute claim therefore depends on the
// cost-weighted average of R over the iterations actually run, so the error in quoting the bare-bath
// R is roughly (N-1)/N x drift. This driver measures that drift.
//
// HOW IT DIFFERS FROM symm_variance_main.inc. That driver deliberately does NOT instantiate a
// DcaLoop: it calls data.initialize() -> qmc_solver.integrate() directly, which skips cluster
// mapping, the double-counting correction and cluster exclusion -- the steps that turn a measured
// Sigma into the bath the walker samples. This one runs the REAL loop and samples every iteration,
// via the two hooks added to dca_loop.hpp:
//
//   pre_finalize_hook_  -- fires between the solver's integrate() and finalize(), the only gap where
//                          local_G_k_w(false) is legal (it throws once finalize sets averaged_).
//                          Gathers the raw per-rank G + per-rank phase to rank 0.
//   post_finalize_hook_ -- fires just after finalize(), when the averaged+symmetrized data_.G_k_w
//                          for this iteration exists. Rank 0 writes the file.
//
// OUTPUT LAYOUT IS A DELIBERATE REUSE. Each DCA iteration is written as its OWN hdf5 in exactly the
// format symm_variance_main.inc emits (metadata/ + raw/ + functions/ + symmetrization/), named
// <output_stem>_iter<k>.hdf5. So analysis/symm_variance_lib.Run reads a drift iteration with no
// changes at all, and every existing check -- the variance ratios, the per-orbit r law, the
// mean-preservation rung -- applies per iteration for free. The only added metadata key is
// dca_iteration. Nothing downstream had to learn a new schema.

#include <complex>
#include <iostream>
#include <string>
#include <vector>

#include "dca/parallel/mpi_concurrency/dca_mpi.h"

#include "dca/io/hdf5/hdf5_writer.hpp"
#include "dca/io/json/json_reader.hpp"
#include "dca/parallel/util/get_workload.hpp"
#include "dca/phys/dca_loop/dca_loop.hpp"
#include "dca/phys/domains/cluster/cluster_symmetry.hpp"
#include "dca/util/git_version.hpp"

#include "symm_variance_setup.hpp"
#include "orbit_table.hpp"

#ifndef SYMM_VARIANCE_MODEL_LABEL
#define SYMM_VARIANCE_MODEL_LABEL "unknown_model"
#endif

int main(int argc, char** argv) {
  using namespace dca::symm_variance;
  using dca::ClusterSolverId;

  dca::parallel::MPIConcurrency concurrency(argc, argv);

  if (argc < 3) {
    if (concurrency.id() == concurrency.first())
      std::cerr << "usage: " << argv[0] << " <input.json> <output_stem>\n"
                << "  writes <output_stem>_iter<k>.hdf5, one per DCA iteration\n";
    return 1;
  }
  const std::string input_file = argv[1];
  const std::string output_stem = argv[2];

  const int n_ranks = concurrency.number_of_processors();
  const bool is_root = concurrency.id() == concurrency.first();

  using Parameters = ParametersType<ClusterSolverId::CT_AUX>;
  Parameters parameters(dca::util::GitVersion::string(), concurrency);
  parameters.read_input_and_broadcast<dca::io::JSONReader>(input_file);
  parameters.update_model();
  parameters.update_domains();

  const int n_iterations = parameters.get_dca_iterations();

  // DEPTH BOOKKEEPING -- identical semantics to symm_variance_main.inc. "measurements" in the input
  // is the TOTAL over ranks (mci_parameters.hpp: local_meas = measurements / mpi_size), so record
  // both and refuse an indivisible total, which would hand the first (total % n_ranks) ranks one
  // extra measurement and break the "every rank is one iid sample" premise.
  //
  // Note measurements/sweeps-per-measurement are per-DCA-iteration VECTORS. A scalar in the input is
  // broadcast to every iteration by MciParameters::solveDcaIterationConflict (called from
  // Parameters::readWrite), so all N iterations here run at identical depth -- which is what makes
  // iteration k and iteration 1 comparable at all. Indexing [0] below is that same value.
  const int measurements_total = parameters.get_measurements().back();
  const int measurements_per_rank = dca::parallel::util::getWorkload(measurements_total, concurrency);
  if (measurements_total % n_ranks != 0) {
    if (is_root)
      std::cerr << "[bath_drift] ERROR: measurements (" << measurements_total
                << ") is not divisible by n_ranks (" << n_ranks
                << "); ranks would run at unequal depth and the per-rank samples would not be"
                   " identically distributed.\n";
    return 2;
  }
  // A mixing factor of 0 makes mix_self_energy() keep Sigma_cluster and DISCARD the freshly measured
  // Sigma (ctaux_cluster_solver.hpp: Sigma = alpha*Sigma_new + (1-alpha)*Sigma_cluster), so the bath
  // would never move and this experiment would silently measure a flat line -- the exact false
  // negative it exists to rule out. The single-iteration templates set 0. because it is irrelevant
  // there; here it is fatal, so refuse it rather than trust the input.
  if (parameters.get_self_energy_mixing_factor() <= 0.) {
    if (is_root)
      std::cerr << "[bath_drift] ERROR: self-energy-mixing-factor is "
                << parameters.get_self_energy_mixing_factor()
                << "; at <= 0 the measured Sigma is discarded and the bath never drifts, so the"
                   " drift check would return a meaningless flat result.\n";
    return 2;
  }
  if (n_iterations < 2) {
    if (is_root)
      std::cerr << "[bath_drift] ERROR: iterations is " << n_iterations
                << "; the drift check needs at least 2.\n";
    return 2;
  }

  DcaData<ClusterSolverId::CT_AUX> data(parameters);
  data.initialize();

  using LoopType = dca::phys::DcaLoop<Parameters, DcaData<ClusterSolverId::CT_AUX>,
                                      ThreadedSolver<ClusterSolverId::CT_AUX>>;
  LoopType loop(parameters, data, concurrency);

  // The live super-cell op count of the (derived) K-cluster group -- the honest witness of |G|.
  using KClusterType = typename DcaData<ClusterSolverId::CT_AUX>::KClusterType;
  using ClusterSym = dca::phys::domains::cluster_symmetry<KClusterType>;
  const int n_ops = ClusterSym::sym_super_cell_dmn_t::dmn_size();

  // Per-iteration scratch, filled by the pre-finalize hook and consumed by the post-finalize one.
  int local_size = 0;
  std::vector<double> gathered;       // [n_ranks][2*local_size] interleaved re,im
  std::vector<double> gathered_sign;  // [n_ranks][2] interleaved re,im

  loop.pre_finalize_hook_ = [&](int iteration, ThreadedSolver<ClusterSolverId::CT_AUX>& solver) {
    // Raw per-rank sample: no symmetrization of any kind, independent of the declared point group.
    auto G_i = solver.local_G_k_w(/*symmetrize=*/false);
    local_size = static_cast<int>(G_i.size());  // nu*nu*nk*nw complex entries
    const std::complex<double> sign_i = solver.local_accumulated_sign();

    if (is_root) {
      gathered.assign(static_cast<std::size_t>(local_size) * 2 * n_ranks, 0.);
      gathered_sign.assign(static_cast<std::size_t>(2) * n_ranks, 0.);
    }
    MPI_Gather(reinterpret_cast<const double*>(G_i.values()), 2 * local_size, MPI_DOUBLE,
               is_root ? gathered.data() : nullptr, 2 * local_size, MPI_DOUBLE, concurrency.first(),
               concurrency.get());
    MPI_Gather(reinterpret_cast<const double*>(&sign_i), 2, MPI_DOUBLE,
               is_root ? gathered_sign.data() : nullptr, 2, MPI_DOUBLE, concurrency.first(),
               concurrency.get());
    if (is_root)
      std::cout << "[bath_drift] captured raw per-rank G at DCA iteration " << iteration << '\n';
  };

  loop.post_finalize_hook_ = [&](int iteration) {
    if (!is_root)
      return;

    // Split the interleaved gather into per-rank real/imag stacks. Inner buffer is C++ leaf order
    // (b0 fastest ... w slowest); the numpy side reshapes it to (w, k, s1, b1, s0, b0).
    std::vector<std::vector<double>> G_raw_re(n_ranks, std::vector<double>(local_size));
    std::vector<std::vector<double>> G_raw_im(n_ranks, std::vector<double>(local_size));
    for (int r = 0; r < n_ranks; ++r) {
      const double* base = gathered.data() + static_cast<std::size_t>(r) * 2 * local_size;
      for (int i = 0; i < local_size; ++i) {
        G_raw_re[r][i] = base[2 * i];
        G_raw_im[r][i] = base[2 * i + 1];
      }
    }
    std::vector<double> sign_re(n_ranks), sign_im(n_ranks);
    for (int r = 0; r < n_ranks; ++r) {
      sign_re[r] = gathered_sign[2 * r];
      sign_im[r] = gathered_sign[2 * r + 1];
    }

    const auto& k_elements = KClusterType::get_elements();
    const std::string out = output_stem + "_iter" + std::to_string(iteration) + ".hdf5";

    dca::io::HDF5Writer writer;
    writer.open_file(out);

    writer.open_group("metadata");
    writer.execute("model", std::string(SYMM_VARIANCE_MODEL_LABEL));
    writer.execute("seed", parameters.get_seed());
    writer.execute("beta", parameters.get_beta());
    writer.execute("sweeps_per_measurement", parameters.get_sweeps_per_measurement()[0]);
    writer.execute("warm_up_sweeps", parameters.get_warm_up_sweeps());
    writer.execute("n_ops", n_ops);
    writer.execute("n_ranks", n_ranks);
    writer.execute("measurements_total", measurements_total);
    writer.execute("measurements_per_rank", measurements_per_rank);
    writer.execute("local_size", local_size);
    writer.execute("k_elements", k_elements);
    // The drift axis itself. 0-based, matching DcaLoop's dca_iteration_: iteration 0 is the BARE
    // bath and is the run every other number in this project was measured at.
    writer.execute("dca_iteration", iteration);
    writer.execute("dca_iterations_total", n_iterations);
    writer.execute("self_energy_mixing_factor", parameters.get_self_energy_mixing_factor());
    // The loop adjusts mu each iteration (unlike the single-iteration driver, which never
    // instantiates the adjuster), so it is a per-iteration quantity here and has to travel with the
    // file rather than be read off the input.
    writer.execute("chemical_potential", parameters.get_chemical_potential());
    writer.close_group();

    writer.open_group("raw");
    writer.execute("G_raw_re", G_raw_re);
    writer.execute("G_raw_im", G_raw_im);
    writer.execute("sign_re", sign_re);
    writer.execute("sign_im", sign_im);
    writer.close_group();

    // Production symmetrized ensemble mean AT THIS ITERATION, for the mean-preservation rung.
    writer.open_group("functions");
    writer.execute(data.G_k_w);
    // THE BATH ITSELF, so drift is measured rather than inferred from R alone. This is the function
    // the walker actually sampled this iteration (the loop rebuilds it at the START of the next
    // one, so at post-finalize time it still holds this iteration's value).
    //
    // Recording it also exposes a distinction the project's "bare bath" convention hides. At
    // Sigma_cluster = 0 the cluster-exclusion step computes G0_cluster_excluded = G_k_w, the
    // COARSE-GRAINED cluster Green's function (cluster_exclusion.hpp: one_plus_S_G is the identity,
    // so G0_excl = G), whereas DcaData::initialize() sets G0_cluster_excluded = G0_k_w, the bare
    // cluster G0 (dca_data.hpp:599). Those differ by the dispersion's variation within each
    // coarse-graining patch. So iteration 0 of a real loop is NOT the bath the single-iteration
    // driver uses, even though both have Sigma = 0 -- writing all three makes that gap measurable
    // instead of a silent offset between the two drivers.
    writer.execute(data.G0_k_w_cluster_excluded);
    writer.execute(data.G0_k_w);
    writer.execute(data.Sigma);
    writer.close_group();

    writeOrbitTable<Parameters>(writer);

    writer.close_file();
    std::cout << "[bath_drift] wrote " << out << '\n';
  };

  if (is_root) {
    std::cout << "\n[bath_drift] model=" << SYMM_VARIANCE_MODEL_LABEL
              << "  seed=" << parameters.get_seed() << "  beta=" << parameters.get_beta()
              << "  iterations=" << n_iterations << "  ranks=" << n_ranks
              << "  meas/rank=" << measurements_per_rank << " (total " << measurements_total << ")"
              << "  mixing=" << parameters.get_self_energy_mixing_factor() << '\n';
  }

  loop.initialize();
  loop.execute();
  loop.finalize();
  // Closes and renames the loop's own dca.hdf5. We do not read it -- the per-iteration files above
  // are the measurement -- but leaving it unwritten strands a .tmp on disk every run.
  loop.write();

  return 0;
}
