// FeAs two-band model, stock FeAsPointGroup (declared 2 ops; derive-authoritative expands to the 8
// H0-verified, band-permuting ops). Single arm. FeAs's richer orbits mix band and k indices, which
// is where the more interesting reductions live. Use the 4x4 cluster (2x2 is vacuous -- interband G
// is identically zero there).
#include <cmath>  // some DCA symmetry-operation headers (Cn_2d/Sn_2d) use std::cos/sin/M_PI without
                  // including <cmath> themselves; pull it in before the lattice/point-group headers.

#include "dca/phys/models/analytic_hamiltonians/fe_as_lattice.hpp"

// FeAsLattice ignores its template argument and hardcodes bilayer_lattice<FeAsPointGroup> as its
// base, so DCA_point_group is always FeAsPointGroup regardless of the token passed here.
#define SYMM_VARIANCE_LATTICE \
  dca::phys::models::FeAsLattice<dca::phys::models::FeAsPointGroup>
#define SYMM_VARIANCE_MODEL_LABEL "fe_as"

#include "symm_variance_main.inc"
