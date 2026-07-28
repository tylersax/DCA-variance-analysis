// Shared driver body for the symmetrization-variance measurement, included by the per-model .cpp
// files after they define SYMM_VARIANCE_LATTICE and SYMM_VARIANCE_MODEL_LABEL.
//
// One invocation = one multi-rank measurement. Each MPI rank runs one QMC iteration and, BEFORE
// finalize (which averages across ranks and would trip local_G_k_w's averaged_ guard), grabs its
// own RAW per-rank Green's function via local_G_k_w(/*symmetrize=*/false). Those raw samples are
// gathered to rank 0 and written as a [n_rank, ...] stack -- the paired-sample input the numpy tier
// needs to form both Var(G) and Var(P.G) from one run. Rank 0 also runs finalize() (the real
// production path, which applies the full symmetrization) and dumps the symmetrized data_.G_k_w plus
// the serialized point-group orbit operator P (orbit_table.hpp).
//
//   read input -> update_model/update_domains -> data.initialize()
//   qmc.initialize(0); qmc.integrate();
//   G_i = qmc.local_G_k_w(false);           // raw per-rank sample  (rank = one sample)
//   gather G_i -> rank 0
//   qmc.finalize(loop_data);                // production symmetrized data_.G_k_w
//   rank 0: write raw stack + finalized G_k_w + orbit table + metadata

#include <complex>
#include <iostream>
#include <string>
#include <vector>

#include "dca/parallel/mpi_concurrency/dca_mpi.h"

#include "dca/io/hdf5/hdf5_writer.hpp"
#include "dca/io/json/json_reader.hpp"
#include "dca/parallel/util/get_workload.hpp"
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
      std::cerr << "usage: " << argv[0] << " <input.json> <output.hdf5>\n";
    return 1;
  }
  const std::string input_file = argv[1];
  const std::string output_file = argv[2];

  const int n_ranks = concurrency.number_of_processors();
  const bool is_root = concurrency.id() == concurrency.first();

  using Parameters = ParametersType<ClusterSolverId::CT_AUX>;
  Parameters parameters(dca::util::GitVersion::string(), concurrency);
  parameters.read_input_and_broadcast<dca::io::JSONReader>(input_file);
  parameters.update_model();
  parameters.update_domains();

  DcaData<ClusterSolverId::CT_AUX> data(parameters);
  data.initialize();

  ThreadedSolver<ClusterSolverId::CT_AUX> qmc_solver(parameters, data, nullptr);
  qmc_solver.initialize(0);
  qmc_solver.integrate();

  // Raw per-rank sample: no symmetrization of any kind, independent of the declared point group.
  auto G_i = qmc_solver.local_G_k_w(/*symmetrize=*/false);
  const int local_size = static_cast<int>(G_i.size());  // nu*nu*nk*nw complex entries
  const std::complex<double> sign_i = qmc_solver.local_accumulated_sign();  // per-rank phase

  // DEPTH BOOKKEEPING. The input's "measurements" is the TOTAL over ranks: every solver takes it
  // through parallel::util::getWorkload (see mci_parameters.hpp, `local_meas = measurements/mpi_size`),
  // so each rank actually runs total/n_ranks. Recording only the input number mislabels the depth by
  // a factor of n_ranks and silently rescales the whole M axis. Record both, taking the per-rank
  // figure from the same getWorkload the solver used rather than re-deriving it.
  const int measurements_total = parameters.get_measurements().back();
  const int measurements_per_rank = dca::parallel::util::getWorkload(measurements_total, concurrency);
  // getWorkload hands the first (total % n_ranks) ranks one extra measurement. Unequal depth breaks
  // the "every rank is one iid sample" premise the variance estimator rests on, so refuse it.
  if (measurements_total % n_ranks != 0) {
    if (is_root)
      std::cerr << "[symm_variance] ERROR: measurements (" << measurements_total
                << ") is not divisible by n_ranks (" << n_ranks
                << "); ranks would run at unequal depth and the per-rank samples would not be"
                   " identically distributed.\n";
    return 2;
  }

  // Gather the raw per-rank buffers to rank 0 as interleaved doubles (re, im contiguous per entry).
  std::vector<double> gathered;
  std::vector<double> gathered_sign;  // per-rank phase, interleaved (re, im)
  if (is_root) {
    gathered.resize(static_cast<std::size_t>(local_size) * 2 * n_ranks);
    gathered_sign.resize(static_cast<std::size_t>(2) * n_ranks);
  }
  MPI_Gather(reinterpret_cast<const double*>(G_i.values()), 2 * local_size, MPI_DOUBLE,
             is_root ? gathered.data() : nullptr, 2 * local_size, MPI_DOUBLE, concurrency.first(),
             concurrency.get());
  MPI_Gather(reinterpret_cast<const double*>(&sign_i), 2, MPI_DOUBLE,
             is_root ? gathered_sign.data() : nullptr, 2, MPI_DOUBLE, concurrency.first(),
             concurrency.get());

  // Production path: averages across ranks and applies the full symmetrization to data_.G_k_w.
  dca::phys::DcaLoopData<Parameters> loop_data;
  qmc_solver.finalize(loop_data);

  // The live super-cell op count of the (derived) K-cluster group -- the honest witness of |G|.
  using KClusterType = typename DcaData<ClusterSolverId::CT_AUX>::KClusterType;
  using ClusterSym = dca::phys::domains::cluster_symmetry<KClusterType>;
  const int n_ops = ClusterSym::sym_super_cell_dmn_t::dmn_size();

  if (is_root) {
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
    // Per-rank phase (sign), for reconstructing the production phase-weighted mean.
    std::vector<double> sign_re(n_ranks), sign_im(n_ranks);
    for (int r = 0; r < n_ranks; ++r) {
      sign_re[r] = gathered_sign[2 * r];
      sign_im[r] = gathered_sign[2 * r + 1];
    }

    const auto& k_elements = KClusterType::get_elements();

    std::cout << "\n[symm_variance] model=" << SYMM_VARIANCE_MODEL_LABEL
              << "  seed=" << parameters.get_seed() << "  live super-cell ops=" << n_ops
              << "  Nk=" << k_elements.size() << "  ranks=" << n_ranks
              << "  meas/rank=" << measurements_per_rank << " (total " << measurements_total << ")"
              << "  local_size=" << local_size << '\n';

    dca::io::HDF5Writer writer;
    writer.open_file(output_file);

    writer.open_group("metadata");
    writer.execute("model", std::string(SYMM_VARIANCE_MODEL_LABEL));
    writer.execute("seed", parameters.get_seed());
    // beta is a swept axis (ROADMAP task 2), and the analysis side takes the file's own metadata as
    // authoritative over the filename -- so it has to be recorded here, not inferred from the name.
    writer.execute("beta", parameters.get_beta());
    // A depth in "measurements per rank" only means something at a stated sweeps-per-measurement,
    // and both the autocorrelation and thermalization times grow with beta -- so the two knobs that
    // decide whether those measurements are independent draws travel with the run, not with a
    // remembered convention. Indexed [0]: these are per-DCA-iteration, and the driver runs one.
    writer.execute("sweeps_per_measurement", parameters.get_sweeps_per_measurement()[0]);
    writer.execute("warm_up_sweeps", parameters.get_warm_up_sweeps());
    // mu is a per-model choice that this driver can never adjust -- it uses DcaLoopData but never
    // DcaLoop, so the chemical-potential adjuster is never instantiated (Gotcha 11) and whatever the
    // input says is what the run sampled. Recording it makes the filling of a new model checkable
    // rather than remembered, and it is the second half of the analytic-G0 oracle below.
    writer.execute("chemical_potential", parameters.get_chemical_potential());
    writer.execute("n_ops", n_ops);
    writer.execute("n_ranks", n_ranks);
    writer.execute("measurements_total", measurements_total);
    writer.execute("measurements_per_rank", measurements_per_rank);
    writer.execute("local_size", local_size);
    writer.execute("k_elements", k_elements);
    writer.close_group();

    // Raw per-rank stack: G_raw_re/im are [n_rank][local_size]; reshape local_size ->
    // (w, k, s1, b1, s0, b0) on the numpy side (reversed C++ leaf order).
    writer.open_group("raw");
    writer.execute("G_raw_re", G_raw_re);
    writer.execute("G_raw_im", G_raw_im);
    writer.execute("sign_re", sign_re);  // per-rank accumulated phase (real part)
    writer.execute("sign_im", sign_im);  // per-rank accumulated phase (imag part)
    writer.close_group();

    // Production symmetrized ensemble mean, for the mean-preservation validation rung.
    writer.open_group("functions");
    writer.execute(data.G_k_w);
    // The non-interacting cluster Hamiltonian, which buys an oracle that depends on NOTHING
    // production does. G0(k, iw) = [(iw + mu) I - H0(k)]^-1 is deterministic and exactly invariant
    // under every symmetry of H0, so `P @ G0_analytic == G0_analytic` tests our derived operator on
    // its own -- the same oracle DCA's symmetrization characterization test uses. That matters for
    // any model where validation rung 2 (agreement with production's Symmetrize::execute) is
    // expected to fail: rung 2 failing then localizes to production's imposition rather than to P.
    // The STORED G0_k_w cannot serve this purpose -- dca_data.hpp:583 runs production Symmetrize
    // over it during initialize(). H_DCA is written by initializeH0 and never symmetrized.
    writer.execute(data.H_DCA);
    writer.close_group();

    // Authoritative point-group orbit operator P over the (b0,b1,k) index.
    writeOrbitTable<Parameters>(writer);

    writer.close_file();
    std::cout << "[symm_variance] wrote " << output_file << '\n';
  }

  return 0;
}
