// Square single-band Hubbard model, stock D4 point group (derive-authoritative, 8 ops on a 4x4
// cluster). Single arm: the raw per-rank dump is group-independent and the orbit table describes the
// D4 symmetrization whose variance reduction we measure.
#include <cmath>  // some DCA symmetry-operation headers (Cn_2d/Sn_2d) use std::cos/sin/M_PI without
                  // including <cmath> themselves; pull it in before the point-group headers.

#include "dca/phys/domains/cluster/symmetries/point_groups/2d/holohedries_2d.hpp"
#include "dca/phys/models/analytic_hamiltonians/square_lattice.hpp"

#define SYMM_VARIANCE_LATTICE dca::phys::models::square_lattice<dca::phys::domains::D4>
#define SYMM_VARIANCE_MODEL_LABEL "square_D4"

#include "symm_variance_main.inc"
