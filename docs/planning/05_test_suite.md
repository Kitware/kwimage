# 05 — Test-suite repair and coverage expansion

**Goal**: make every test able to fail, make results deterministic, and add
coverage where the audit found none.
**Prerequisites**: docs 01–04 (their regression tests are assumed present).

Baseline at audit time: `tests/` had 29 tests (19 pass / 8 skip / 2 fail —
failures fixed by doc 01); doctests: 534 collected, 365 pass, 166 skipped
(optional-dep gates), 3 fail (doc 01). The suite leans on doctests; several
core modules have effectively zero enforced coverage.

---

## 1. Repair tests that cannot fail

- `tests/test_import.py:1-2` — body is `pass`; make it `import kwimage` and
  assert a few key attrs exist (`kwimage.imread`, `kwimage.Boxes`, and that
  `__version__` matches a semver pattern).
- `tests/test_bbox_isect.py:16-69` — `test_bbox_isect_failure_case` only
  prints. Add assertions on the expected intersection values.
- `tests/test_geotiff.py:83-162` — no assertions after `imwrite`; ends by
  printing `gdal.Info`. Assert round-trip properties (shape, dtype, CRS
  presence). Give the bare `pytest.skip()` at :90 a reason string.
- `tests/test_warp_affine.py:311-328` — `test_warp_affine_with_many_chans`
  has a bare `warped` expression instead of assertions, plus dead `img`/`M`
  assignments at :322-323. Assert output shape/dtype and content for a known
  transform.
- `tests/test_resize.py:461-488` — `test_imresize_multi_channel` asserts
  nothing and hard-skips if `timerit` is missing though timing is irrelevant.
  Remove the timerit dependency, assert shapes/dtypes, and reinstate the
  commented-out parameter grid (or delete it deliberately).
- `tests/test_draw_on.py:125-137` — the `errors` accumulation + final
  `raise AssertionError` is unreachable because `raise` at :133 re-raises
  first. Decide: fail-fast (delete the accumulator) or collect-then-report
  (remove the inner raise).
- `tests/test_cv2_funcs.py:43-51` — broad `except Exception` records
  failures but only 3 of 8 dtype rows are asserted. Assert the full expected
  support matrix.
- `tests/test_io.py:181,261` — two permanently-skipped "exploration" tests.
  Move to `dev/` or convert into real assertions.

## 2. Determinism: seed all randomness

Pass explicit seeds/rngs everywhere:
- `tests/test_tranform.py:16` — `kwarray.ensure_rng(None)` → `ensure_rng(0)`;
  also `np.random.rand` at :80.
- `tests/test_detections.py:63` — `Detections.random()` → pass `rng=`.
- `tests/test_draw_on.py:87-91,106` — seed `Boxes.random()` / `np.random.rand`.
- `tests/test_cv2_funcs.py:41`, `tests/test_resize.py:498-518` — same.
Sweep: `grep -rn "random(\s*)\|np.random\.\|ensure_rng(None)" tests/` and fix
all hits.

## 3. Housekeeping

- Rename `tests/test_tranform.py` → `tests/test_transform.py` (`git mv`).
- De-duplicate ~40 lines of shared scaffolding in `tests/test_rle.py:190-313`
  into a helper/fixture; delete the permanently-dead `SMALL=False` branches
  (:215-230, :279-294).
- `conftest.py:10-31` — the Python 3.14 TLS/gdal import-order workaround is
  acknowledged tech debt ("I don't quite understand it"). Add a comment
  linking to an issue; periodically retest whether it is still needed.
- `tests/test_demodata_header.py` requires `--network`, which CI never
  passes. Either add a scheduled network-enabled CI job or accept it as a
  manual test and document that in the file's docstring.

## 4. New coverage (prioritized)

Add dedicated, headless-friendly test files (skip-gate optional deps):

1. **`tests/test_stack.py`** (committed in doc 01) — extend: 1 image, N
   images of mixed shapes/dtypes, `axis`, `resize`, grid layouts; empty-list
   behavior.
2. **`tests/test_io_backends.py`** — imread/imwrite backend selection and
   fallback: auto-routing by extension, clear error when gdal missing for
   .nitf/.jp2 (doc 02), PIL handle closing, round-trips for
   png/jpg/tif over dtypes (uint8/uint16/float32) with the cv2 backend;
   gdal-gated round-trips including `nodata_method`/`nodata_value`.
3. **`tests/test_nms.py`** — `non_max_supression` correctness: empty input,
   single box, exact-duplicate boxes, threshold edge cases (0, 1), tie
   scores, zero-area boxes; **backend agreement** across all available impls
   for bias ∈ {0, 1} (skip unavailable backends).
4. **`tests/test_structs_basic.py`** — headless smoke+semantics for the
   structs with zero dedicated tests: `Coords`, `Points`, `Polygon`,
   `Heatmap`, `Segmentation`, `ObjectList`. Cover: non-inplace ops never
   mutate the source (the doc-03 bug class), meta preservation through
   warp/translate/scale/concatenate, empty-struct behavior for every public
   method, int-dtype geometry (translate/rotate/scale with fractional
   params), draw_on with 1/3/4/5-channel canvases.
5. **`tests/test_util_warp.py`** — numpy paths of `util_warp` (subpixel_*
   functions); torch-gated tests for `warp_tensor` (doc 04 items).
6. **`tests/test_im_filter.py`, extend alphablend coverage** — `fourier_mask`
   both backends agree; `overlay_alpha_images` over impls and dtypes.
7. **`tests/test_cli.py`** — smoke tests for `kwimage.cli` crop_border and
   stack_images on tmp images; clean errors on missing args (doc 04).
8. **Property tests for Affine/Projective** (can live in test_transform.py):
   `coerce(concise())` and `coerce(decompose())` round-trips, `inv()`
   composition ≈ identity, over randomized anisotropic params with fixed
   seeds.

## 5. Doctest reliance

166/534 doctests are optional-dep-gated (torch, kwplot, gdal, itk, ...), so
headless environments silently lose ~31% of the coverage. The dedicated tests
in section 4 are the mitigation — keep them free of optional deps wherever
the underlying code allows a numpy path.

**Acceptance**:
- `python -m pytest tests/ -q` — 0 failures, no test lacking assertions
  (spot-check), deterministic across two consecutive runs.
- Coverage: `python -m pytest tests/ --cov=kwimage -q` — im_stack, im_io
  (non-gdal paths), algo_nms, util_warp each ≥ 60% line coverage; overall
  coverage strictly above the pre-plan baseline (record both numbers in the
  journal).
