// Kagome Hubbard model -- ROADMAP task 4, the "how high can R go" point.
//
// Three EQUIVALENT sublattices (flavors() = {0,1,2}, aVectors() the three midpoints) on a hexagonal
// Bravais lattice, giving the largest point group and the largest orbits available anywhere in this
// sweep: |G| = 12 against D4's 8. It is the only model in the tree with three symmetry-EQUIVALENT
// orbitals, which is why it earns its risk -- threeband has only two.
//
// D6, NOT no_symmetry<2>: CanDeriveSymmetry gates on !is_no_symmetry<DCA_point_group>
// (derive_point_group.hpp) and orbit_table.hpp static_asserts on CanDeriveSymmetry, so the
// no_symmetry<2> spelling DCA's own test uses would fail to COMPILE here -- and even without the
// assert, deriveAndPopulateRecord is a compile-time no-op for a no_symmetry declaration, leaving 1
// live op instead of 12. The declared group is only an on/off switch on this path: the derive
// machinery installs holohedry_pool_2D (D4 + D6, geometry-filtered, H0-gated) regardless of what is
// declared. So D6 costs nothing and turns the derive path on.
//
// DCA's characterization test (case KagomeNoSym) asserts expected_num_derived_symmetries = 12 where
// geometry alone derives none: set_symmetry_matrices' position+flavor matching records (-1,-1) for
// every non-trivial op because the three sublattices carry distinct flavors, and the permutation
// search in deriveOrbitalOpForOp recovers all 12. So the ROADMAP's worry that a 3-fold model would
// collapse under the +/-1 signed-permutation gate does not apply here. Gate 1 still checks n_ops.
//
// Kagome_hubbard.hpp includes only no_symmetry.hpp, so holohedries_2d.hpp must be pulled in HERE.
//
// Analysis note: the k-mesh is hexagonal, so noise_diagnostics' real-space machinery
// (_k_to_grid_index assumes an LxL Cartesian grid, d4_shells assumes D4) does NOT apply and returns
// meaningless numbers without raising. model_sweep.square_mesh detects this and skips the mechanism
// block. Everything else -- P, orbits, mate-rho, the reduction map, R, w_null, sign health -- is
// mesh-agnostic and works unchanged.
#include <cmath>  // Cn_2d/Sn_2d use std::cos/M_PI without including it themselves (Gotcha 8).

#include "dca/phys/domains/cluster/symmetries/point_groups/2d/holohedries_2d.hpp"
#include "dca/phys/models/analytic_hamiltonians/Kagome_hubbard.hpp"

#define SYMM_VARIANCE_LATTICE dca::phys::models::KagomeHubbard<dca::phys::domains::D6>
#define SYMM_VARIANCE_MODEL_LABEL "kagome"

#include "symm_variance_main.inc"
