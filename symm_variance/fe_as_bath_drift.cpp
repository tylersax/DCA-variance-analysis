// Bath-drift check (ROADMAP 3e) on the FeAs two-band model, stock FeAsPointGroup.
//
// Built alongside the square target but NOT run by default: 3e's decision rule is "square first,
// once; only escalate to FeAs beta=5 (~3 h) if square shows drift". The binary exists so that
// escalation is a run, not a build.
#include <cmath>  // some DCA symmetry-operation headers (Cn_2d/Sn_2d) use std::cos/sin/M_PI without
                  // including <cmath> themselves; pull it in before the lattice/point-group headers.

#include "dca/phys/models/analytic_hamiltonians/fe_as_lattice.hpp"

// FeAsLattice ignores its template argument and hardcodes bilayer_lattice<FeAsPointGroup> as its
// base, so DCA_point_group is always FeAsPointGroup regardless of the token passed here.
#define SYMM_VARIANCE_LATTICE \
  dca::phys::models::FeAsLattice<dca::phys::models::FeAsPointGroup>
#define SYMM_VARIANCE_MODEL_LABEL "fe_as"

#include "bath_drift_main.inc"
