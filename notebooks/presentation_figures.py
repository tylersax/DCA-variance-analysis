"""Variance-reduction cloud-grid + money-plot for the FeAs symmetrization demo.

Panel A: 4x4 Brillouin-zone grid. Each node = one cluster momentum k; the cloud around it
is 512 replica estimates of band-0 G(k, iw0) in the local complex plane (Re/Im deviation from
the replica mean). OFF = raw estimator (wide), ON = symmetrized (tight). Same visual scale in
both panels, so the contraction is the honest sqrt(Von/Voff). Hero = the (pi/2,pi/2) size-8
band-diagonal orbit that clears the declared 2x ceiling (2.51x).

Panel B: measured Var_off/Von vs orbit size, with the null (y=1), the declared-2op ceiling,
and the per-orbit rho-prediction n/(1+(n-1)rho).
"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                                   # find variance_demo_feas_lib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import variance_demo_feas_lib as L

BASE = os.path.join(_HERE, "..", "variance_demo", "runs_feas4")
OUT  = os.path.join(_HERE, "..", "figures")                 # write beside the other demo figures

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUT = "#0b0b0b", "#52514e", "#b9b8b4"
SURF = "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "font.family": "DejaVu Sans", "svg.fonttype": "none",
    "text.color": INK, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
})

# ---- load -------------------------------------------------------------------
Gon, seeds, kel, nops_on   = L.load_arm(BASE, "on")     # (rep,w,k,b1,b0)
Goff, _,   _,   nops_off   = L.load_arm(BASE, "off")
assert nops_on == {8} and nops_off == {1}
W0 = int(np.argmax(np.abs(Gon).mean(axis=(0, 2, 3, 4))))   # lowest Matsubara
NK = kel.shape[0]

# orbit assignment (up to sign) from the ON arm, for coloring/ratios
groups, labels, signs = L.orbits_from_on(Gon)
Von_all  = L.flatten_entries(Gon)[0].var(axis=0, ddof=1).real   # (w,E)
Voff_all = L.flatten_entries(Goff)[0].var(axis=0, ddof=1).real
lab_index = {lab: i for i, lab in enumerate(labels)}

def entry_orbit(E_idx):
    for gi, mem in enumerate(groups):
        if E_idx in mem:
            return gi
    return -1

# per-node (band-0 diagonal) info
band0 = [(k, 0, 0) for k in range(NK)]           # (k,b1,b0)
node_orbit = np.array([entry_orbit(lab_index[b]) for b in band0])
# hero orbit = band-diagonal size-8 with the largest ratio
def orbit_ratio(gi):
    mem = groups[gi]
    return Voff_all[:, mem].mean() / Von_all[:, mem].mean()
def orbit_char(gi):
    bs = [(labels[m][1], labels[m][2]) for m in groups[gi]]
    return "diag" if all(b1 == b0 for b1, b0 in bs) else "inter"
hero_orbit = max((gi for gi in range(len(groups))
                  if len(groups[gi]) == 8 and orbit_char(gi) == "diag"),
                 key=orbit_ratio)
HERO_RATIO = orbit_ratio(hero_orbit)
hero_k = sorted({labels[m][0] for m in groups[hero_orbit]})
print(f"w0={W0}  hero orbit={hero_orbit} ratio={HERO_RATIO:.3f} k={hero_k}")

# ---- cloud geometry ---------------------------------------------------------
# node positions in the BZ (kx,ky) in [0,2pi); spacing = pi/2
pos = kel.copy()
spacing = np.pi / 2
# band-0 diagonal complex value per (rep,k) at w0
gon0  = Gon[:, W0, :, 0, 0]     # (rep,k)
goff0 = Goff[:, W0, :, 0, 0]
doff = goff0 - goff0.mean(axis=0, keepdims=True)   # (rep,k) fluctuations
don  = gon0  - gon0.mean(axis=0, keepdims=True)
# per-node scale: every OFF cloud drawn to a common radius, so each node visibly
# participates in the collapse. ON reuses the node's OFF scale, so its smaller
# spread renders the honest sqrt(Von/Voff) shrink at this k.
sig_off_k = np.sqrt((np.abs(doff) ** 2).mean(axis=0))     # (k,)
SCALE_K = 0.30 * spacing / (3 * sig_off_k)                # 3-sigma OFF -> 0.30 spacing

def draw_panel(ax, dev, title):
    # orbit connecting lines: real spatial stars only (skip the excluded size-24 null web)
    for gi, mem in enumerate(groups):
        ks = sorted({labels[m][0] for m in mem})
        if len(ks) < 2 or len(ks) > 4:      # >4 == the interband null (12 k's) -> no lines
            continue
        is_hero = (gi == hero_orbit)
        col = ORANGE if is_hero else MUT
        lw  = 1.8 if is_hero else 0.8
        a   = 0.60 if is_hero else 0.35
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                ax.plot([pos[ks[i], 0], pos[ks[j], 0]],
                        [pos[ks[i], 1], pos[ks[j], 1]],
                        color=col, lw=lw, alpha=a, zorder=1,
                        solid_capstyle="round")
    # clouds (hero drawn last / on top, brighter)
    order = sorted(range(NK), key=lambda k: node_orbit[k] == hero_orbit)
    for k in order:
        is_hero = node_orbit[k] == hero_orbit
        col = ORANGE if is_hero else BLUE
        x = pos[k, 0] + SCALE_K[k] * dev[:, k].real
        y = pos[k, 1] + SCALE_K[k] * dev[:, k].imag
        ax.scatter(x, y, s=6 if is_hero else 5, c=col,
                   alpha=0.28 if is_hero else 0.15,
                   linewidths=0, zorder=3 if is_hero else 2, rasterized=True)
        # node ring (the symmetrized center)
        ax.scatter([pos[k, 0]], [pos[k, 1]], s=48, facecolors="none",
                   edgecolors=col if is_hero else INK2, linewidths=1.7, zorder=4)
    ax.set_title(title, fontsize=13, color=INK, pad=10, loc="left", fontweight="bold")
    ax.set_xlim(-spacing * 0.62, 2 * np.pi - spacing + spacing * 0.62)
    ax.set_ylim(-spacing * 0.62, 2 * np.pi - spacing + spacing * 0.62)
    ax.set_aspect("equal")
    ax.set_xticks([0, np.pi, 2 * np.pi - spacing])
    ax.set_xticklabels(["0", "$\\pi$", ""], fontsize=10)
    ax.set_yticks([0, np.pi, 2 * np.pi - spacing])
    ax.set_yticklabels(["0", "$\\pi$", ""], fontsize=10)
    ax.set_xlabel("$k_x$", fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 6.2))
draw_panel(axL, doff, "OFF  ·  raw estimator")
draw_panel(axR, don,  "ON  ·  symmetrized")
axL.set_ylabel("$k_y$", fontsize=11)

# arrow + label between panels
fig.subplots_adjust(left=0.06, right=0.985, top=0.82, bottom=0.10, wspace=0.16)
xa = 0.505
fig.text(xa, 0.50, "symmetrize", ha="center", va="bottom", fontsize=11,
         color=INK2, style="italic")
arr = FancyArrowPatch((xa - 0.028, 0.48), (xa + 0.028, 0.48),
                      transform=fig.transFigure, arrowstyle="-|>",
                      mutation_scale=18, color=INK2, lw=1.6)
fig.patches.append(arr)

fig.suptitle("Symmetrization tightens the Monte-Carlo estimator, orbit by orbit",
             x=0.06, y=0.955, ha="left", fontsize=15.5, fontweight="bold", color=INK)
fig.text(0.06, 0.895,
         "FeAs 4×4 · 512 replicas · band-0 $G(k,\\,i\\omega_0)$ fluctuations in the local complex plane",
         ha="left", fontsize=11, color=INK2)
# hero callout (kept inside the panel to avoid clipping; points to a hero node)
axR.annotate(f"($\\pi$/2, $\\pi$/2) star × band swap\n"
             f"$\\mathrm{{Var}}_{{off}}/\\mathrm{{Var}}_{{on}} = {HERO_RATIO:.2f}$  (> declared 2×)",
             xy=(pos[13, 0], pos[13, 1]),
             xytext=(0.30, 0.035), textcoords="axes fraction",
             fontsize=10.5, color=ORANGE, fontweight="bold", va="bottom", ha="left",
             arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1.2, alpha=0.7))

fig.savefig(f"{OUT}/pres_cloud_grid.png", dpi=200)
fig.savefig(f"{OUT}/pres_cloud_grid.svg")
print("wrote pres_cloud_grid.png / .svg")

# ---- Panel B: money plot ----------------------------------------------------
fig2, ax = plt.subplots(figsize=(7.4, 5.6))
sizes, meas, pred, chars, is_hero_pt = [], [], [], [], []
for gi, mem in enumerate(groups):
    n = len(mem)
    if n == 24:      # symmetry-forced interband null (0/0) -- excluded per plan
        continue
    r = orbit_ratio(gi)
    rho = L.orbit_rho_flat(Goff, mem, signs[gi])
    sizes.append(n); meas.append(r); pred.append(L.predicted_ratio(n, rho))
    chars.append(orbit_char(gi)); is_hero_pt.append(gi == hero_orbit)

sizes = np.array(sizes); meas = np.array(meas); pred = np.array(pred)
is_hero_pt = np.array(is_hero_pt)
xr = np.linspace(1.6, 8.4, 100)
ax.plot(xr, xr, ls=(0, (5, 3)), color=MUT, lw=1.4, zorder=1,
        label="ideal  $y=n$")
ax.axhline(1, color=MUT, lw=1.2, ls=":", zorder=1, label="null  (no reduction)")
ax.axhline(2, color=INK2, lw=1.2, ls="-.", zorder=1,
           label="declared 2-op ceiling")
# jitter x a touch so co-sized orbits separate
rng = np.random.default_rng(0)
jx = sizes + rng.uniform(-0.13, 0.13, len(sizes))
# predicted markers
ax.scatter(jx, pred, s=52, facecolors="none", edgecolors=INK2, linewidths=1.3,
           zorder=3, label="predicted  $n/(1{+}(n{-}1)\\rho)$")
# measured
ax.scatter(jx[~is_hero_pt], meas[~is_hero_pt], s=70, c=BLUE, zorder=4,
           edgecolors="white", linewidths=0.8, label="measured")
ax.scatter(jx[is_hero_pt], meas[is_hero_pt], s=150, c=ORANGE, zorder=5,
           edgecolors="white", linewidths=1.0, marker="D")
ax.annotate(f"{HERO_RATIO:.2f}×  hero orbit",
            xy=(jx[is_hero_pt][0], meas[is_hero_pt][0]),
            xytext=(6.15, 1.55), fontsize=10.5, color=ORANGE, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1.1))
ax.set_xticks([2, 4, 8])
ax.set_xlabel("symmetry-orbit size $n$", fontsize=11.5)
ax.set_ylabel("$\\mathrm{Var}_{off}\\,/\\,\\mathrm{Var}_{on}$", fontsize=11.5)
ax.set_xlim(1.4, 8.7); ax.set_ylim(0.8, 8.7)
ax.set_title("Reduction tracks orbit size, capped by orbit-mate noise correlation",
             fontsize=12.5, fontweight="bold", color=INK, loc="left", pad=10)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(fontsize=9.5, frameon=False, loc="upper left")
fig2.tight_layout()
fig2.savefig(f"{OUT}/pres_money_plot.png", dpi=200)
fig2.savefig(f"{OUT}/pres_money_plot.svg")
print("wrote pres_money_plot.png / .svg")
