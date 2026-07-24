# kwimage Quality Improvement Plan

Generated 2026-07-09 from a full-repo audit (5 parallel deep audits: core image
modules, geometry structs, transforms/NMS/CLI, tests/CI, packaging/docs/hygiene)
at commit `c888880` on branch `dev/0.12.0` (version `0.12.0`, unreleased).

## How to execute this plan

Execute the numbered documents **in order**. Each doc is self-contained: goal,
prerequisites, itemized tasks with file:line references, concrete failure
scenarios (which double as regression-test specs), and acceptance criteria.

- **01** — Release blockers for 0.12.0 (past-due deprecation removals, OpenCV 5
  compatibility, uncollectable test file). Do this first; the branch cannot
  ship without it.
- **02** — Bug fixes: core image modules (`kwimage/im_*.py`, `_internal.py`).
- **03** — Bug fixes: geometry structs (`kwimage/structs/`).
- **04** — Bug fixes: transforms, warp, NMS, CLI (`transform.py`,
  `util_warp.py`, `algo/`, `cli/`).
- **05** — Test-suite repair and coverage expansion.
- **06** — CI and tooling: dead CI removal, blocking lint, config
  deduplication, typing-debt burn-down.
- **07** — Repo hygiene and documentation refresh.
- **08** — Systemic refactors (do last; they touch many files and are easier
  after the point fixes and better tests exist).

## Conventions (apply to every task)

1. **Every bug fix gets a regression test.** The "Failure" line in each task is
   the test spec: reproduce it as a failing test first, then fix. Put new
   tests in `tests/` (module-appropriate file; create one if none exists).
2. **Every behavior change gets a CHANGELOG.md entry** under
   `## Version 0.12.0 - Unreleased` (`### Fixed` / `### Changed` / `### Removed`).
3. **Line numbers are as of commit `c888880`.** Always confirm with Grep/Read
   before editing — earlier docs in this plan will have shifted lines.
4. **Verification loop** after each doc:
   ```bash
   python -m pytest tests/ -q
   python -m pytest -p no:doctest --xdoctest kwimage -q
   ```
   Both must be no worse than before your change (see doc 01 for the known
   pre-existing failures and their fixes).
5. **Style**: follow `AGENTS.md` — PEP 8, Google-style docstrings with
   runnable xdoctest examples. New/changed public behavior should get a
   doctest in addition to a regression test. Prefer non-square test images
   (H != W) — the audit found the dominant bug class is (x,y)/(w,h) vs (h,w)
   transposition that square inputs cannot catch.
6. **Journal**: append progress entries to `dev/journals/<agent_name>.md`
   per `AGENTS.md`.
7. **Environment note**: the audit ran on Python 3.12, numpy 2.4.4,
   opencv-python-headless 5.0.0, **without** torch/torchvision, gdal, itk,
   sympy, imgaug, pycocotools, kwplot. Items marked *(needs torch)* etc. were
   verified by code reasoning only — install the dep or re-verify carefully
   before and after fixing.

## Severity/priority key

- **high** — silently wrong results or crashes on valid input.
- **med** — documented features broken, wrong behavior in common configurations.
- **low** — API warts, confusing errors, dead code hazards.

Within each doc, fix high items first, but it is fine to batch related items
in one commit (e.g. all fixes to a single function).
