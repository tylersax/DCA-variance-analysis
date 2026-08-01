# Working agreement for this repo

## Branching — one branch, no worktrees

**Work directly in this checkout, on `main`. Do not create git worktrees.**

This is deliberate and was decided after worktrees became a nuisance: parallel sessions each created
their own worktree and branch, which left four branches and two extra checkouts to reconcile for work
that was always linear. If you are a background job whose default instructions say to isolate in a
worktree, **that default is overridden here** — commit to `main` in place instead. The guard that
enforces the default is already disabled in `.claude/settings.json`
(`"worktree": {"bgIsolation": "none"}`); if an edit is ever refused for not being isolated, that
setting is what to check rather than creating a worktree.

Corollaries:
- Do not run `EnterWorktree`, `git worktree add`, or create `worktree-*` branches.
- Feature branches are fine when a change genuinely needs review before landing; delete them after
  merging. The default, though, is to commit straight to `main`.
- `.claude/` is local session state and is gitignored. Never commit it.

## Where things live

| file | role |
|---|---|
| `ROADMAP.md` | the single task list, and the state of every milestone |
| `TAKEAWAYS.md` | measured findings — the source of truth for numbers |
| `PAPER-OUTLINE.md` | speculative paper framing; **not** a task list, cites ROADMAP rather than tracking work |
| `symm_variance/analysis/README.md` | the analysis tier: notebooks, modules, and how to regenerate them |

## Things that have cost time before

- **Never hand-edit the working-tier notebooks.** Edit `symm_variance/analysis/build_notebooks.py`,
  regenerate, then re-execute *every* notebook — the builder rewrites all of them and clears outputs.
  Verify by counting `execution_count` per cell; nbconvert can exit 0 having executed nothing.
  **Exception: `hero/symmetrization_variance.ipynb`.** As of 2026-07-31 it is edited directly and is
  its own source; its builder is retired in `hero/attic/` and must not be run. See `hero/README.md`.
- **Check `uptime` before launching runs.** This is a shared 128-core box.
- **Point run output at `/home/tsax10/dca/scratch/`** and commit only the summary JSON. A 64-rank run
  is 17–77 MB.
- The build trees are paired to specific checkouts — see ROADMAP's task sections before rebuilding.
