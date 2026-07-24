# 06 — CI and tooling

**Goal**: one live CI, one linter, one config source of truth, and a typing
setup that actually checks something.
**Prerequisites**: docs 01–05 (CI should go green on the fixed suite).

Context: the live CI is GitLab (`.gitlab-ci.yml`, xcookie-generated,
cp310–cp314 matrix with minimal/full × loose/strict flavors — good structure).
CircleCI and AppVeyor configs are dead since 2019. Note: `setup.py` and parts
of CI are xcookie-generated — prefer changing the `[tool.xcookie]` config in
`pyproject.toml` and regenerating over hand-editing generated files, where
possible.

---

## 1. Delete dead CI

- Delete `.circleci/config.yml` (pins python 3.6.1), `appveyor.yml` (tests
  2.7/3.5), and `.rules.yml` — all frozen since 2019-02-16 and reference a
  nonexistent `optional-requirements.txt`; they cannot pass.
- Remove the `|Appveyor|` badge from `README.rst:8` (full README refresh in
  doc 07).

## 2. Make lint blocking and consolidate to one linter

- `.gitlab-ci.yml` lint job (~line 547) has `allow_failure: true` — flake8
  and `ty` failures gate nothing. Remove `allow_failure` (or split into a
  blocking errors-only job + advisory full job).
- Three lint ecosystems are configured (ruff in pyproject, flake8 in CI,
  mypy config present) but at most one weakly enforced. **Standardize on
  ruff**: wire `ruff check` into `run_linter.sh` and the CI lint job; drop
  the flake8 invocation; delete the vestigial `[tool.mypy]` block if `ty`
  remains the type checker.
- Fix `pyproject.toml:54` — `[tool.ruff] target-version = "py39"` →
  `"py310"` (3.9 was dropped).
- Run `ruff check kwimage/ tests/` and fix or explicitly NOQA the findings
  before flipping lint to blocking.

## 3. Single source of truth for tool configs

`pytest.ini` and `pyproject.toml [tool.pytest.ini_options]` have diverged
(pytest.ini silently wins; it alone has `--xdoctest-verbose=1`,
`--ignore-glob=dev`, `norecursedirs`). Same story for `.coveragerc` vs
`[tool.coverage.*]` (.coveragerc wins with `source = kwimage` + extra omits).

- Merge the *winning* settings into `pyproject.toml`, then delete
  `pytest.ini` and `.coveragerc`.
- Verify: `python -m pytest tests/ -q --collect-only | tail -3` collects the
  same set before/after; coverage still reports `source = kwimage`.

## 4. Typing-debt burn-down

Current state is theater: `pyproject.toml:79-105` blanket-disables `ty` for
21 files (`all = "ignore"`), and commit `f669e99` sprayed 191 blanket
`# type: ignore` comments into `detections.py` (96) and `heatmap.py` (95).
`unused-type-ignore-comment = "ignore"` (`pyproject.toml:73`) even hides
stale ignores.

Plan (incremental; do not attempt in one pass):
1. Treat the `[[tool.ty.overrides]]` list as an explicit burn-down list; add
   a comment in pyproject saying so.
2. Convert `detections.py` / `heatmap.py` from inline-ignore-spray to either
   (a) targeted error-code ignores where the error is a real false positive,
   or (b) entries in the overrides list — consistency with the other 21
   files. Prefer (b) first, then burn down like the rest.
3. Re-enable `unused-type-ignore-comment` detection once inline ignores are
   rationalized.
4. Burn down the overrides list smallest-file-first (`_typing.py`-adjacent
   helpers, `im_stack.py`, `im_filter.py`, ...), one file per PR: remove the
   override, run `ty check`, fix or targeted-ignore each finding.
5. Keep `ty check` in the (now blocking) lint job so files never regress
   after leaving the list.

## 5. CI matrix hardening

- Add an explicit `opencv>=5` test flavor (or verify the "loose" flavor now
  resolves cv2 5.x and stays green after doc 01) so the next cv2 major bump
  is caught early.
- Consider a torch-enabled flavor: 166 doctests and all torch paths
  (util_warp, torch_nms, Detections tensors) are currently never exercised
  in CI if torch isn't in the full-loose env — check `.gitlab-ci.yml`'s
  "full" requirement set and add torch if absent.
- No Windows/macOS coverage exists anywhere (AppVeyor was the Windows leg).
  Decide explicitly: either add a minimal GitHub Actions matrix for
  win/mac wheels-install + smoke test, or record in README that only Linux
  is CI-tested.
- The `--network` flag is never passed in CI; see doc 05 §3 for the
  decision on network tests.

## 6. Repo scripts

- `run_tests.py`, `run_doctests.sh`, `run_linter.sh` are developer
  conveniences not used by live CI (only dead CircleCI referenced
  run_tests.py). Keep them but make them mirror CI exactly (same pytest
  flags, same ruff/ty invocations) and note that in a comment header.

**Acceptance**: GitLab CI pipeline green with lint blocking; `.circleci/`,
`appveyor.yml`, `pytest.ini`, `.coveragerc` deleted; `ruff check` clean;
`ty check` clean for every file not on the overrides list; overrides list
strictly smaller than the starting 21 (+2 inline-sprayed) files.
