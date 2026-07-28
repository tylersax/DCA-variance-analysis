// Bath-drift check (ROADMAP 3e) on the square single-band Hubbard model, stock D4 point group.
// Same model/group setup as square_symm_variance.cpp -- only the driver body differs: this one runs
// the real DcaLoop for N iterations and dumps the raw per-rank G at every one.
//
// Square is the deliberate first target: it is provably sign-free at any beta, so any drift measured
// here is PURELY the change in noise structure as the bath moves, with the sign channel switched off.
#include <cmath>  // some DCA symmetry-operation headers (Cn_2d/Sn_2d) use std::cos/sin/M_PI without
                  // including <cmath> themselves; pull it in before the point-group headers.

#include "dca/phys/domains/cluster/symmetries/point_groups/2d/holohedries_2d.hpp"
#include "dca/phys/models/analytic_hamiltonians/square_lattice.hpp"

#define SYMM_VARIANCE_LATTICE dca::phys::models::square_lattice<dca::phys::domains::D4>
#define SYMM_VARIANCE_MODEL_LABEL "square_D4"

#include "bath_drift_main.inc"
