# 08 — Systemic refactors

**Goal**: eliminate the recurring bug-generating patterns the audit
identified, and remove dead weight. Do this **last** — after docs 01–05 the
point fixes are in and the test suite can catch regressions from these
broader changes.

Each section is an independent unit of work; land them as separate PRs.

---

## 1. One copy-shell helper for all structs

The non-inplace idiom `self.__class__(self.data.copy())` /
`self.__class__(self.data, self.meta)` is re-implemented inconsistently
across Coords/Points/Polygon/Boxes/Detections and produced at least four
distinct bugs (docs 03: Points.round shared dict, Polygon mixin dropped meta,
Coords compress/take/astype shared meta, ObjectList inplace inconsistency).

- Add a single well-tested helper on the shared base (`Spatial` in
  `kwimage/structs/_generic.py`), e.g.
  `_copy_shell(self, data=None, copy_meta=True)` that shallow-copies `data`
  and `meta` dicts (not the arrays) and preserves the concrete class.
- Migrate every non-inplace method in structs to use it. Grep targets:
  `grep -rn "self.__class__(" kwimage/structs/`.
- Tests: for every struct class and every non-inplace method, assert (a) the
  source object is unchanged (data and meta), (b) meta content is preserved
  on the result, (c) mutating the result's meta does not affect the source.
  Write this as one parametrized test over classes/methods.

## 2. Uniform dtype-promotion policy for geometry ops

Integer-dtype geometry currently fails four different ways (silent truncation
in `Coords.translate`, zeros from `Coords.rotate`, UFuncTypeError from
`Boxes.scale(inplace=True)`, AttributeError from `Boxes.intersection`).

- Decide and document the policy: **fractional transform on integer data
  promotes the result to float** (matching `Coords.scale`'s existing
  behavior); `inplace=True` on integer data with a fractional transform
  raises a clear ValueError (cannot promote in place).
- Implement via a shared helper (e.g. in `_generic.py` or the kwarray impl
  layer) used by translate/scale/rotate/warp across Coords, Points, Polygon,
  Boxes.
- Parametrized tests: int8/int32/float32 data × integer/fractional
  parameters × inplace True/False for each op.

## 3. (x,y)/(w,h) vs (h,w) convention guardrails

Transposition was the dominant bug class (imcrop center, 1x1 tile,
morphology kernel, gaussian_patch sigma, heatmap scale indexing, geotiff
overview vars). After doc 02 fixes the instances:

- Write `docs/source/conventions.rst` (or extend an existing page): kwimage
  convention is `dsize=(w, h)`, `shape=(h, w)`, points/coords are `(x, y)`;
  cv2 calls take `(x, y)` / `(w, h)`.
- Sweep every `cv2.` call site that receives a size/point tuple and add a
  short inline comment where a conversion happens
  (`# cv2 expects (w, h)`), verifying correctness as you go:
  `grep -rn "cv2\.\(resize\|warpAffine\|warpPerspective\|getRectSubPix\|getStructuringElement\|putText\|circle\|rectangle\|line\)" kwimage/`.
- Ensure the doc-02/03 regression tests all use non-square inputs; add a
  brief note to `AGENTS.md` recommending non-square inputs in tests.

## 4. Silent-failure elimination pass

The audit found a systematic pattern of silently ignored parameters and
silently divergent fallbacks. After docs 02–04 fix the confirmed instances,
do a sweep:

- **Unknown kwargs**: functions that probe `**kw`/config dicts without
  validating keys (`Affine.random_params` pattern). Raise on unknown keys.
- **Accepted-but-unused parameters**: audit signatures vs bodies (the
  `grab_test_image(space=...)`, itk `border_*`, `isect_area(bias=...)`
  class). A quick heuristic: `ruff check --select ARG` (unused arguments)
  on `kwimage/` and triage.
- **Divergent backend fallbacks**: `_skimage_resize`
  (`kwimage/im_transform.py:875-916`) returns float64 in [0,1] for uint8
  input — set `preserve_range=True` and cast back; NMS backend divergence
  (doc 04 §4.9); pure-python RLE (fixed in doc 02) — add cross-backend
  agreement tests wherever two impls of the same function exist.
- **Broad exception handling**: `except Exception` + print in
  `_cv2_warp_affine` (`im_cv2.py:1903-1916`), `imread` (`im_io.py:517-520`),
  `_NMS_Impls._lazy_init` (`algo_nms.py:281,294`), `detections.py:1104-1116`,
  `heatmap.py:301`, `_generic.ObjectList.dtype`. Narrow the exception types;
  replace prints with `warnings`/logging; never swallow import errors of
  first-party code.
- Pointless `except X: raise` blocks: `transform.py:622-625, 2183-2186`,
  `util_warp.py:703-720, 816-833` (debug prints before re-raise) — delete
  or convert to exception notes.

## 5. Dead-code removal

Delete (verify nothing references them first):
- `transform.py:557-571, 2105-2115` — `if 0:` blocks with alternate cv2 fit
  implementations; `transform.py:2546-2586` — commented-out `_mpmatrix`.
- `util_warp.py:474-537` — `if False:` debug block; `util_warp.py:1188/1243`
  — dead `ndim` parameter of `_padded_slice`.
- `im_io.py:441-462` `USE_FILE_HEADER = 0` block; `_imread_exif`'s
  `USE_PIL_BACKEND = 0` block; duplicate `'.png'` in
  `endswith(('.png', '.png'))` at `im_io.py:2284-2287`; swapped
  `max_x_overviews`/`max_y_overviews` names at `im_io.py:2047-2048`.
- `_common.py:51` bare `dsize` statement; `_common.py:64-82, 97-108`
  `if 0:` blocks.
- `im_color.py:582-587` — unreachable trailing block in `Color.distinct`
  (decide: implement the non-rgb conversion or delete).
- `_generic.py:666-689` — `_handle_color_args_for` returns None and is only
  referenced from commented-out call sites: finish it (and deduplicate the
  color-arg logic in `Boxes.draw_on`/`Polygon.draw_on`) or delete.
- `_generic.py:529-533` — unreachable `return False` in `_isinstance2`.
- `mask.py:795-801` — permanently-disabled integer-offset path in
  `Mask.translate` (`integer_offset = None  # hack`); implement or remove.
- `mask.py:692-706` — duplicated Affine conversion in `Mask.warp` (second is
  dead); consider the noted numpy fallback via `kwimage.warp_image` to drop
  the hard torch requirement.
- `torch_nms.py:87-98, 124, 141-152` — commented-out IoU code, dead `else`
  under `if True:`, unused `n_conflicts`; move `test_class_torch`
  (:164-219, CUDA-only manual test) out of library code into `dev/` or tests.
- `boxes.py:487-489` — dead unpack in `to_xywh`'s `_RCHW` branch.
- `coords.py:464-497` — commented benchmark blocks → `dev/bench/`.
- `algo_nms.py:248-297` — stale numpy>=1.20 gate and, if `kwimage_ext` is
  confirmed unmaintained/unavailable, prune the cython backends from the
  preference tables (keep graceful handling if installed).
- Giant demo doctests in `transform.py:404-499` etc. — move showcase
  material to Sphinx examples; keep short runnable doctests.

## 6. API-consistency pass (behavior-affecting; changelog everything)

- `Boxes.compress/take` have `inplace`; `Detections.compress/take` don't —
  add for symmetry.
- `Detections.translate` doesn't forward `inplace` to keypoints/segmentations
  while `scale`/`warp` do (`detections.py:1493-1500`) — forward it.
- `Coords.concatenate` drops `meta`; `Points.concatenate` keeps
  `first.meta` — unify (keep first.meta).
- `scale(about=...)` exists for Boxes/Coords/Polygon but not Points — add.
- `Segmentation` wrapper lacks `rotate` though both backends have it — add.
- `ObjectList.translate`/`scale` vs `warp` inplace-return inconsistency
  (`_generic.py:222-275`) — harmonize.
- Draw-API inplace story (`im_draw.py`): `draw_boxes_on_image` copies;
  `draw_line_segments_on_image`/`draw_vector_field` mutate;
  `draw_text_on_image` mutates but may return a UMat. Document current
  behavior per function now; consider a uniform `inplace=` kwarg at 1.0.
- `Boxes.round/quantize(inplace=True)` reassign `new.data` rather than
  writing in place (`boxes.py:3139-3176`) — use `out=` semantics or document.
- `imresize`/`morphology` still use `_cv2_input_fixer` (bool-only) while
  `_cv2_input_fixer_v2` exists and handles uint16/float16/int — migrate.
- `gaussian_blur` duplicates `_auto_kernel_sigma` logic and its
  `isinstance(kernel, int)` misses numpy ints — call the helper, accept
  `numbers.Integral`.
- `Detections.dtype` returns dtype-or-set polymorphically
  (`detections.py:1740-1741`) — document or split into `dtypes`.
- `Color.as255` truncates instead of rounding (`im_color.py:311`) and
  `as255`/`ashex` mishandle non-rgb spaces — round, and validate space.
- `stack_images_grid` (`im_stack.py:247`) calls `len()` on a documented
  Iterable — materialize with `list()` first.
- `transform.py:177-192` — identity-`Matrix` matmul returns operands without
  copy and breaks the "prefer LHS type" rule — return a copy of the correct
  class.
- `ensure_alpha_channel` (`im_alphablend.py:276-357`) — validate ndarray
  alpha shape/dtype; don't silently upcast away the requested dtype.
- Import-time costs: `transform.py:2499-2505` imports sympy at module import
  when installed (defer into the function); `coords.py` imports imgaug at
  module import (defer via the `sys.modules.get` pattern used elsewhere).

**Acceptance**: suite + doctests green after each PR; no `if 0:`/`if False:`
blocks remain in `kwimage/` (`grep -rn "if 0:\|if False:" kwimage/`);
parametrized copy-semantics and dtype-promotion tests in place; CHANGELOG
records every behavior change.
