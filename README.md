# Symmetrization variance-reduction demo (DCA++)

Standalone harness + analysis for demonstrating that imposing the **H0-derived** point group on the
single-particle Green's function reduces Monte-Carlo variance without shifting the mean.

- **Claim A** — symmetrization reduces variance (control, `square_lattice<D4>`).
- **Claim B** — the derivation found symmetry *never declared*, and it buys real reduction
  (**FeAs**: declared 2 ops → derived 8).

Deliberately kept **outside** the DCA repo: it builds against a DCA checkout without modifying it,
so no dev material can ever end up in a DCA commit or PR.

| | |
|---|---|
| `SETUP.md` | build + run on a new machine — **start here** |
| `variance-demo-plan.md` | the full plan; **§12 = results** from the first run |
| `variance_demo/` | C++ harness (ON/OFF arms), input templates, replica drivers |
| `notebooks/` | analysis notebooks + plain numpy/h5py libs |
| `figures/` | money plots, bias controls, correlation diagnostics |
| `patches/` | fallback build hook (not needed by default) |

## Headline results (first machine, CT-AUX)

- **Square 4×4, N=64/arm:** ratio pinned at exactly 1.0 for the Γ and (π,π) singletons (hard control),
  ω-independent, mean unbiased. Above that it is **ρ-limited** — orbit-mate noise is strongly
  correlated (ρ≈0.86–0.94), so ratios are 1.05–1.37, well below the ideal *n*.
- **FeAs 4×4, N=32/arm:** derivation report proves **2 declared → 8 derived**; the size-8
  band-diagonal orbit measures **Var_off/Var_on = 2.45 [2.12, 2.90]** — CI entirely above the
  declared 2-op ceiling, mean unbiased. Reduction the declaration could never reach.

Requires the DCA branch `symm-m3` (M3′ derive-authoritative, stacked on the `symm-derive-p` P-search).
