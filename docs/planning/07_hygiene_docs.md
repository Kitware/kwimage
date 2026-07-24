# 07 — Repo hygiene and documentation refresh

**Goal**: remove junk, fix stale/incorrect documentation, finish the 3.9-drop
sweep, and tidy packaging.
**Prerequisites**: none strictly, but README/API updates assume doc 01's
removals happened.

---

## 1. Delete untracked junk (confirm with the user if unsure about any item)

At the repo root / docs (all untracked as of the audit):
- `kwimage-source-2026-06-07T191832-5-5fac757eb5da.tar.gz` (2.7 MB snapshot)
  — delete.
- `git-of-theseus/` (1.2 MB analysis output) — delete or move under `dev/`.
- `docs/temp/` (empty images dir + LaTeX byproducts; keep the projective
  maths PDF only if referenced — otherwise delete or move to `dev/`).
- Optionally clean local-only cruft: `htmlcov/` (20 MB), `.coverage`,
  `dist/` (stale 0.11.3 artifacts), `__pycache__/` at repo root.

Update `.gitignore`: add `/*.tar.gz`, `git-of-theseus/`, `docs/temp/`,
`.ruff_cache/`.

## 2. Finish the Python 3.9-drop sweep

- `AGENTS.md:3` — says "Support Python >=3.8"; update to >=3.10.
- `pyproject.toml:54` — ruff `target-version = "py310"` (also in doc 06 §2).
- `requirements/*.txt` — prune dead environment-marker rows for Python
  3.6–3.9 (harmless but noise).
- Delete `requirements/tree.md` (stale 2022 johnnydep dump).

## 3. README.rst refresh

- Remove the `|Appveyor|` badge; fix the ReadTheDocs badge/text mismatch
  (badge targets `?version=release`, text links `/en/main/`); resolve the
  `TODO Get CI services running on gitlab` comment at `README.rst:5`.
- Regenerate the "top-level API" listing from the real
  `kwimage/__init__.py`: it currently omits the whole `im_transform` module
  (misattributes `warp_affine`/`warp_image`/`warp_projective` to `im_cv2`),
  omits `adjust`, `crop_border_by_color`, `imcrop`, and still advertises the
  removed `imscale`. Note the `non_max_supression` typo — list the corrected
  alias once doc 04 adds it.

## 4. CHANGELOG.md

- Ensure everything from docs 01–04 is recorded under
  `## Version 0.12.0 - Unreleased` (`### Fixed` / `### Changed` /
  `### Removed`).
- Backfill or explicitly mark "maintenance release, no notable changes" for
  the empty 0.11.4–0.11.6 entries.

## 5. Sphinx docs

- Regenerate `docs/source/auto/` with sphinx-apidoc: pages are missing for
  `kwimage/im_transform.py`, `kwimage/im_itk.py`, `kwimage/cli/crop_border.py`,
  `kwimage/_backend_info.py`, `kwimage/_common.py` (only 37 auto pages
  committed; `cli.crop_border` absent despite being a 0.11.3 feature).
- Build locally (`make -C docs html`) and fix warnings introduced by the
  plan's code changes.
- Consider trimming the enormous demo-style doctests in `transform.py`
  (:404-499 etc.) into docs/examples pages (see doc 08 §5).

## 6. Packaging polish

- `MANIFEST.in` — add trailing newline.
- `setup.py:213-264` extras — drop the redundant `runtime` extra (duplicates
  install_requires); decide the fate of the `kwimage_ext >= 0.3.1`
  requirement annotated "Not available yet" in `requirements/optional.txt`
  (either it exists on PyPI now — verify — or remove/comment it).
- `pyproject.toml:2` — `setuptools>=41.0.1` is ancient; raise the floor to a
  currently-supported setuptools. Longer-term (out of scope here): migrate
  metadata to `[project]` via xcookie upstream.
- These files are xcookie-generated where noted — prefer regenerating from
  `[tool.xcookie]` config over hand edits.

## 7. Journal & planning upkeep

- Append a `dev/journals/` entry summarizing what was executed from this
  plan, per `AGENTS.md`.
- As docs 01–08 complete, mark them done at the top of each file (a single
  `> STATUS: done YYYY-MM-DD` line) so a future agent can resume mid-plan.

**Acceptance**: `git status` clean except intentional changes; README badges
all resolve to live services; README API listing matches
`kwimage.__all__`; `make -C docs html` succeeds; CHANGELOG complete for
0.12.0.
