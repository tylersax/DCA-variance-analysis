// Sign-aware point-group orbit-table serializer for the symmetrization-variance driver.
//
// Emits the exact linear operator P that the production K-cluster (point-group) symmetrization
// applies to G_k_w at fixed spin and fixed Matsubara frequency -- the operator is spin- and
// omega-independent, so one matrix over the flattened (b0, b1, k) index describes it fully.
//
// This is the single source of truth shared by production Symmetrize and the numpy analysis: it is
// built by replaying the *exact* derive-authoritative orbit-average formula from
// symmetrize_single_particle_function.hpp (the <BDmn,BDmn,k_dmn_t> executeCluster, derive branch)
//     G'(b0,b1,k) = (1/|G|) sum_S sum_{a,b} U_S(b0,a) V(k,a,S) f(a,b, mapped(k,S)) V(k,b,S) U_S(b1,b),
// reading the authoritative records U_S = get_orbital_op(), V = get_fold_phase(),
// mapped(k,S) = get_mapped_point() populated during parameters.update_domains(). It does NOT detect
// orbits from G values (that silently splits signed orbits) -- orbits and per-member signs are
// derived downstream from the support of P.
//
// Scope: the derive-authoritative path (CanDeriveSymmetry == true), which is what the in-scope 2D
// models (square/D4, FeAs) use. For those, U_S and V are signed permutations (+/-1), enforced by the
// records themselves (set_symmetry_matrices rejects non-+/-1 fold phases), so P is a sign-aware
// orbit average with entries in {0, +/- j/|G|}. A static_assert guards the scope.

#ifndef SYMM_VARIANCE_ORBIT_TABLE_HPP
#define SYMM_VARIANCE_ORBIT_TABLE_HPP

#include <array>
#include <vector>

#include "dca/function/domains.hpp"
#include "dca/io/hdf5/hdf5_writer.hpp"
#include "dca/phys/dca_step/symmetrization/derive_point_group.hpp"
#include "dca/phys/domains/cluster/cluster_symmetry.hpp"
#include "dca/phys/domains/quantum/electron_band_domain.hpp"

namespace dca {
namespace symm_variance {

// Flatten (b0, b1, k) -> a single index, b0 outermost, k innermost. The numpy side reads the same
// convention from the emitted flat_labels, so this choice is documented, not assumed.
inline int flatten_bbk(int b0, int b1, int k, int nb, int nk) {
  return (b0 * nb + b1) * nk + k;
}

// Build and serialize P (and its flat labels) under an already-open HDF5 file, in group
// "symmetrization". Templated on the solver Parameters so it can reach the cluster domain + records.
template <class Parameters>
void writeOrbitTable(dca::io::HDF5Writer& writer) {
  using KClusterDmn = typename Parameters::KClusterDmn;
  using KCluster = typename KClusterDmn::parameter_type;
  using BDmn = func::dmn_0<dca::phys::domains::electron_band_domain>;
  using ClusterSym = dca::phys::domains::cluster_symmetry<KCluster>;
  using SymDmn = typename ClusterSym::sym_super_cell_dmn_t;

  static_assert(dca::phys::CanDeriveSymmetry<Parameters>::value,
                "writeOrbitTable only supports the derive-authoritative symmetrization path "
                "(2D model with H0, non-no_symmetry point group).");

  const int nb = BDmn::dmn_size();
  const int nk = KClusterDmn::dmn_size();
  const int n_ops = SymDmn::dmn_size();
  const int E = nb * nb * nk;

  // Authoritative records, populated by parameters.update_domains() / the finalize Symmetrize.
  const auto& u_s = ClusterSym::get_orbital_op();          // (b_row, b_col, S)
  const auto& fold_phase = ClusterSym::get_fold_phase();   // ((k, band), S)
  const auto& mapped_point = ClusterSym::get_mapped_point();  // (k, S)

  // P[out][in], the exact operator: replay the derive-path triple sum verbatim.
  std::vector<std::vector<double>> P(E, std::vector<double>(E, 0.0));
  for (int k = 0; k < nk; ++k) {
    for (int b0 = 0; b0 < nb; ++b0) {
      for (int b1 = 0; b1 < nb; ++b1) {
        const int out = flatten_bbk(b0, b1, k, nb, nk);
        for (int s = 0; s < n_ops; ++s) {
          const int k_new = mapped_point(k, s);
          for (int a = 0; a < nb; ++a) {
            const double wa = u_s(b0, a, s) * fold_phase(k, a, s);
            if (wa == 0.0)
              continue;
            for (int b = 0; b < nb; ++b) {
              const double w = wa * fold_phase(k, b, s) * u_s(b1, b, s);
              if (w != 0.0)
                P[out][flatten_bbk(a, b, k_new, nb, nk)] += w / static_cast<double>(n_ops);
            }
          }
        }
      }
    }
  }

  // (b0, b1, k) label of each flat index -- the numpy contract for interpreting P and G's band/k axes.
  std::vector<std::array<int, 3>> labels(E);
  for (int b0 = 0; b0 < nb; ++b0)
    for (int b1 = 0; b1 < nb; ++b1)
      for (int k = 0; k < nk; ++k)
        labels[flatten_bbk(b0, b1, k, nb, nk)] = {b0, b1, k};

  writer.open_group("symmetrization");
  writer.execute("n_ops", n_ops);  // live |G| of the (derived) point group
  writer.execute("nb", nb);
  writer.execute("nk", nk);
  writer.execute("flat_index_order", std::string("(b0,b1,k) with k innermost, b0 outermost"));
  writer.execute("flat_labels", labels);  // [E] of (b0,b1,k)
  writer.execute("P", P);                  // [E][E] real operator, P[out][in]
  writer.close_group();
}

}  // namespace symm_variance
}  // namespace dca

#endif  // SYMM_VARIANCE_ORBIT_TABLE_HPP
