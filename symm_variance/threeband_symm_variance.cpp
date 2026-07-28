// Three-band Emery/CuO2 Hubbard model (d, p_x, p_y) -- ROADMAP task 4, the nb=3 point.
//
// BANDS = 3 despite the header's stale "two-orbital" comment. The orbital structure is what makes
// this model worth a run: flavors() = {0,1,1} and aVectors() = {(0,0), (0.5,0), (0,0.5)}, so the d
// orbital is INEQUIVALENT to the two p orbitals while p_x and p_y are EQUIVALENT to each other
// (related by the axis-exchanging half of D4). One run therefore carries both the nb trend point and
// the "benefit vanishes for inequivalent orbitals" control, separated by the entry-class split --
// and that control is a within-model, genuinely single-axis contrast, which a second whole model
// could not be.
//
// The declared point group must NOT be no_symmetry: CanDeriveSymmetry gates on
// !is_no_symmetry<DCA_point_group> (derive_point_group.hpp) and orbit_table.hpp static_asserts on
// CanDeriveSymmetry, so a no_symmetry declaration fails to COMPILE here. D4 is what DCA's own
// characterization test uses for this model (case ThreebandD4: 8 declared -> 8 derived, every U_S a
// +/-1 signed permutation; the C4 op is diag(1,-1,1) * P(p_x <-> p_y), the d-p bond parity).
//
// WARNING carried from that same test: ThreebandD4 declares
// expectedFailingReps() = {"k_iw", "r_iw", "r_tau"} -- production Symmetrize::execute is not a no-op
// on a deterministic G0 for this model (a known multi-band imposition bug). Validation rung 2 checks
// exactly that agreement and is expected to FAIL here. Rung 1 (including the P^2 = P idempotence
// check) and the analytic-G0 oracle validate our own operator independently; see ROADMAP 4.
#include <cmath>  // Cn_2d/Sn_2d use std::cos/M_PI without including it themselves (Gotcha 8).

#include "dca/phys/domains/cluster/symmetries/point_groups/2d/holohedries_2d.hpp"
#include "dca/phys/models/analytic_hamiltonians/threeband_hubbard.hpp"

#define SYMM_VARIANCE_LATTICE dca::phys::models::ThreebandHubbard<dca::phys::domains::D4>
#define SYMM_VARIANCE_MODEL_LABEL "threeband"

#include "symm_variance_main.inc"
