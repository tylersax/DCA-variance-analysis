// Shared body of the CT-AUX variance-reduction demonstration, included by the two arm .cpp files
// after they define VARIANCE_DEMO_POINT_GROUP. The arms are byte-identical apart from that token.
//
// One invocation = one replica: read input -> update_model/update_domains -> one QMC iteration
// (initialize(0); integrate()) -> finalize() (the REAL production path, which applies the
// symmetrization) -> dump data_.G_k_w to HDF5 tagged with the run's seed, the k-mesh, and the live
// super-cell op count of the K cluster.
//
// The op count is the honest witness of which arm ran: 8 for the ON (D4, derive-authoritative)
// arm, 1 for the OFF (no_symmetry, legacy identity) arm. The notebook asserts on it.

#include <iostream>
#include <string>
#include <vector>

#include "dca/io/hdf5/hdf5_writer.hpp"
#include "dca/io/json/json_reader.hpp"
#include "dca/phys/domains/cluster/cluster_symmetry.hpp"
#include "dca/util/git_version.hpp"

#ifndef VARIANCE_DEMO_SETUP_HEADER
#define VARIANCE_DEMO_SETUP_HEADER "symmetry_variance_setup.hpp"
#endif
#include VARIANCE_DEMO_SETUP_HEADER

int main(int argc, char** argv) {
  using namespace dca::variance_demo;

  dca::parallel::MPIConcurrency concurrency(argc, argv);

  if (argc < 3) {
    if (concurrency.id() == concurrency.first())
      std::cerr << "usage: " << argv[0] << " <input.json> <output.hdf5>\n";
    return 1;
  }
  const std::string input_file = argv[1];
  const std::string output_file = argv[2];

  using Parameters = ParametersType<dca::ClusterSolverId::CT_AUX>;
  Parameters parameters(dca::util::GitVersion::string(), concurrency);
  parameters.read_input_and_broadcast<dca::io::JSONReader>(input_file);
  parameters.update_model();
  parameters.update_domains();

  DcaData<dca::ClusterSolverId::CT_AUX> data(parameters);
  data.initialize();

  // One QMC iteration through the real solver, then finalize() -- which is what applies the
  // symmetrization to the ensemble-mean measurement.
  ThreadedSolver<dca::ClusterSolverId::CT_AUX> qmc_solver(parameters, data, nullptr);
  qmc_solver.initialize(0);
  qmc_solver.integrate();
  dca::phys::DcaLoopData<Parameters> loop_data;
  qmc_solver.finalize(loop_data);

  // The live super-cell op count of the derived (or declared, for OFF) K-cluster group.
  using KClusterType = typename DcaData<dca::ClusterSolverId::CT_AUX>::KClusterType;
  using ClusterSym = dca::phys::domains::cluster_symmetry<KClusterType>;
  const int n_ops = ClusterSym::sym_super_cell_dmn_t::dmn_size();

  if (concurrency.id() == concurrency.first()) {
    const auto& k_elements = KClusterType::get_elements();

    std::cout << "\n[variance_demo] arm=" << VARIANCE_DEMO_ARM_LABEL
              << "  seed=" << parameters.get_seed() << "  live super-cell ops=" << n_ops
              << "  Nk=" << k_elements.size() << "  ranks="
              << concurrency.number_of_processors() << '\n';

    dca::io::HDF5Writer writer;
    writer.open_file(output_file);

    writer.open_group("metadata");
    writer.execute("arm", std::string(VARIANCE_DEMO_ARM_LABEL));
    writer.execute("seed", parameters.get_seed());
    writer.execute("n_ops", n_ops);
    writer.execute("n_ranks", concurrency.number_of_processors());
    writer.execute("measurements_per_node", parameters.get_measurements().back());
    // k-mesh as a flat list of coordinate vectors, one per cluster momentum.
    writer.execute("k_elements", k_elements);
    writer.close_group();

    writer.open_group("functions");
    writer.execute(data.G_k_w);
    writer.close_group();

    writer.close_file();
    std::cout << "[variance_demo] wrote " << output_file << '\n';
  }

  return 0;
}
