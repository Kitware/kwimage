# 02 — Bug fixes: core image modules

**Goal**: fix confirmed defects in `kwimage/im_*.py` and helpers.
**Prerequisites**: doc 01 complete (test suite green on cv2 5.x).

Every item below was verified by execution unless marked *(plausible)* or
*(needs <dep>)*. Reproduce each "Failure" as a regression test first, then fix.
Use **non-square** images in tests — most of these are (x,y)/(w,h)-vs-(h,w)
transpositions that square inputs cannot detect.

---

## High severity

### 2.1 `imcrop` linear path crops the wrong location
`kwimage/im_cv2.py:531` — in the `interpolation='linear'` path the center is
passed to `cv2.getRectSubPix` as `(cen_h, cen_w)` (y, x) but cv2 expects
(x, y). The `nearest` path is correct, so the two interpolations disagree.
- Failure: `kwimage.imcrop(np.arange(32*32).reshape(32,32).astype(np.float32),
  dsize=(3,3), about=(5.0, 8.0), interpolation='linear')` → center pixel 168
  (row 5, col 8) instead of 261 (row 8, col 5).
- Fix: pass `(cen_w, cen_h)`. Add a test asserting nearest and linear agree
  on integer-aligned `about` for a non-symmetric image.

### 2.2 String `border_value` silently ignored in warp
`kwimage/im_cv2.py:316-325` (`_coerce_border_mode_value`) — `borderMode` is
computed *before* a string `border_value` is reinterpreted as a border mode;
the reassigned mode is never re-coerced, so the documented
"`border_value='replicate'` means border strategy" feature falls back to
constant 0.
- Failure: `kwimage.warp_affine(img, {'offset': (2,0)}, border_value='replicate')`
  → left border is zeros; `border_mode='replicate'` works.
- Fix: check for a string `border_value` before `_coerce_border_mode`, or
  recompute `borderMode` after reassignment.

### 2.3 1x1-input resize transposes output shape
`kwimage/im_cv2.py:969-976` (`_cv2_imresize`/`_patched_resize`) — the 1x1
fallback uses `np.tile(img, (new_w, new_h))` but tile reps are (rows, cols).
- Failure: `kwimage.imresize(np.random.rand(1,1), dsize=(3,2))` returns shape
  `(3,2)` instead of `(2,3)`.
- Fix: `np.tile(img, (new_h, new_w))` (and `(new_h, new_w, 1)` for 3D).

### 2.4 `imresize` drops `border_value` for letterbox
`kwimage/im_transform.py:841-855` — the public wrapper accepts `border_value`
but never forwards it to `_cv2_imresize`.
- Failure: `kwimage.imresize(img, dsize=(10,10), letterbox=True,
  border_value=42)` → padding is 0.
- Fix: forward `border_value=border_value`.

### 2.4b Warp functions dead-end on cv2-less installs under `backend='auto'`
`kwimage/im_transform.py:588-631` (`warp_affine`, same in `warp_projective`
:231-262 and therefore `warp_image`) — with `backend='auto'`,
`_default_backend()` (`kwimage/_backend_info.py:48-57`) returns `'skimage'`
when cv2 is absent, but the warp dispatchers have no skimage branch — they
`raise NotImplementedError('no kwimage backend=skimage for warp_affine')`.
So on a cv2-less install every warp fails with a misleading error, even when
the itk backend is installed and could serve affine warps.
- Fix: make the auto path warp-aware — e.g. try cv2, then itk (affine only),
  else raise a clear "warp_affine requires cv2 (or itk)" ImportError. Keep
  `_default_backend()` for imresize where skimage genuinely exists.

### 2.5 `grab_test_image` ignores `space`
`kwimage/im_demodata.py:365-429` — the documented `space` parameter is never
used; `kwimage.imread(fpath)` is called without it.
- Failure: `kwimage.grab_test_image('astro', space='gray')` returns (512,512,3).
- Fix: pass `space=space` through (handle the `checkerboard` branch too).

### 2.6 Morphology kernel transposed
`kwimage/im_cv2.py:2320-2325` (`_morph_kernel_core`) —
`cv2.getStructuringElement(struct_shape, (h, w))`: cv2 ksize is (width,
height), so the kernel is transposed relative to the documented `kernel=(w,h)`.
- Failure: `morphology(data, 'dilate', kernel=(3,7))` dilates 7 px
  horizontally / 3 px vertically instead of the reverse.
- Fix: pass `(w, h)`. Test with a single-pixel image and a strongly
  anisotropic kernel.

## Medium severity

### 2.7 `connected_components` ltype mapping broken
`kwimage/im_cv2.py:2583-2588` — `elif ltype is np.int16: ltype = cv2.CV_16U`
should be `np.uint16`; the string branch maps `'uint16'` → `np.uint16`, which
then matches neither identity check.
- Failure: `kwimage.connected_components(img, ltype='uint16')` →
  `TypeError: type(ltype) = <class 'type'>`.
- Fix: use `np.uint16` and compare with `==`/`np.dtype`, not `is`.

### 2.8 `imcrop` rejects documented `None` dimension
`kwimage/im_cv2.py:452-462` — `assert isinstance(new_w, numbers.Integral)`
runs before the `if new_w is None` aspect-ratio branch, so `dsize=(5, None)`
always raises AssertionError despite being documented.
- Fix: `assert new_w is None or isinstance(new_w, numbers.Integral)` (both dims).

### 2.9 `make_channels_comparable(atleast3d=True)` broken both ways
`kwimage/im_core.py:191-199` — (a) channel axis inserted at position 1
(`img[:, None]` → (H,1,W)) instead of `img[..., None]`; (b) when both 2D
images share a shape, the outer `if img1.shape != img2.shape` skips the block
so `atleast3d` is silently ignored.
- Fix: use `[..., None]`; hoist atleast3d handling out of the
  shape-inequality branch. Test: different-shape and same-shape 2D pairs.

### 2.10 Unsigned underflow in `crop_border_by_color`
`kwimage/im_core.py:849-859` (`_get_pixel_dist`) — `np.abs(img - pixel)` wraps
for unsigned dtypes (uint8 0−255 → 1).
- Failure: white-bordered uint8 image with
  `fillval=np.array([255,255,255], dtype=np.uint8), thresh=5` → returns
  `(0, 0, 3)` (content treated as fill).
- Fix: `np.abs(img.astype(np.int32) - pixel)` (choose width by dtype).
- While here: `kwimage/im_core.py:806-808` — `np.greater(isfill.shape[1:2],
  [4, 4])` compares a 1-tuple against `[4,4]`; almost certainly meant
  `shape[1:3]`. Fix or delete the dead fix-up.

### 2.11 `draw_vector_field` alpha handling broken
`kwimage/im_draw.py:1225-1251` — `alpha != 1` on an ndarray raises
`ValueError` (ambiguous truth), making the ndarray-alpha code unreachable;
float alpha raises `NotImplementedError` despite being documented.
- Fix: test `isinstance(alpha, np.ndarray)` first; implement or clearly
  reject float alpha.

### 2.12 `make_orimask` NameError on default args
`kwimage/im_draw.py:944-962` — `import kwimage` happens only inside
`if mag is not None:`, but `kwimage.ensure_alpha_channel` is also used when
`mag is None`. *(plausible — needs matplotlib to execute)*
- Failure: `kwimage.make_orimask(radians)` → `NameError: name 'kwimage'`.
- Fix: move the import to the top of the function.

### 2.13 `_alpha_blend_numexpr1` has no return
`kwimage/im_alphablend.py:242-253` — the function computes `rgb3, alpha3` and
returns None. *(needs numexpr to execute)*
- Failure: `overlay_alpha_images(img1, img2, impl='numexpr1')` →
  `TypeError: cannot unpack non-sequence NoneType`.
- Fix: `return rgb3, alpha3`. Add a parametrized test over all impls,
  skipping unavailable ones.

### 2.14 Pure-python RLE decoder drops the final count
`kwimage/im_runlen.py:465-487` (`_rle_bytes_to_array`) — when the byte loop
completes without the `p >= len(s)` break (every count single-byte), the last
count written at index `m` is truncated by `cnts = cnts[:m]`. This is the
**default impl when the cython backend is absent** — silently corrupts
decoded COCO RLEs.
- Failure: `_rle_bytes_to_array(b'11', impl='python')` → `[1]`, expected `[1, 1]`.
- Fix: count writes explicitly (`n += 1` per write; `return cnts[:n]`). Add
  round-trip tests `array → bytes → array` over varied inputs, and an
  agreement test vs the cython impl when available.

### 2.15 `gaussian_patch` sigma axes inverted
`kwimage/im_cv2.py:1410-1423` — with `shape=(h,w)` and `sigma=(sigma_x,
sigma_y)`, sigma_x is applied to rows (vertical), inverting the x=horizontal
convention used by `gaussian_blur` and the rest of kwimage.
- Failure: `gaussian_patch((9,3), sigma=(0.5, 5.0))` → spread is tight along
  rows (sigma_x acted vertically). Verify via marginal sums.
- Fix: apply sigma_x to the width kernel (`shape[1]`). Document row/col
  semantics; note this is a behavior change for non-isotropic callers —
  changelog under `### Changed`.

### 2.16 itk warp silently ignores border args
`kwimage/im_itk.py:1-252` (`_itk_warp_affine`) — `border_mode`/`border_value`
accepted but unused (`default_pixel_value` left commented out). *(needs itk)*
- Fix: wire `border_value` → `default_pixel_value`; raise
  `NotImplementedError` for unsupported `border_mode` values instead of
  silently diverging from the cv2 backend.

## Low severity

- `kwimage/im_color.py:364-379` — `Color._is_base01` is a `@classmethod`
  missing `cls`; any call with an argument raises TypeError. Fix signature.
- `kwimage/im_core.py:121-131` — `ensure_uint255` raises AssertionError for
  out-of-range signed ints where the docstring (and the float branch) promise
  ValueError. Raise ValueError.
- `kwimage/im_io.py:1311-1313` — GDAL-only extensions with gdal missing fall
  through to `KeyError: Unknown imwrite backend='auto'`. Raise a clear
  "gdal is required for .nitf/.jp2..." ImportError instead.
- `kwimage/im_io.py:1425-1438` — failure path replaces the original exception
  with IOError and has an unreachable trailing `raise`. Use
  `raise IOError(msg) from ex`; delete dead code.
- `kwimage/im_io.py:614-631` — `_imread_pil` never closes `Image.open(fpath)`.
  Use a `with` block (see `load_image_shape` for the pattern).
- `kwimage/__init__.py:204-290` — `adjust` and `crop_border_by_color` are in
  the lazy-loader mapping but missing from `__all__`. Add them; consider a
  unit test asserting `submod_attrs` keys ⊆ `__all__`.
- `kwimage/im_demodata.py:740-750` — checkerboard doctest references undefined
  `img3c` (fails under `--show`). Fix the doctest.
- `kwimage/im_transform.py:913` — `_skimage_resize` crashes opaquely on
  `dsize=(6, None)` (TypeError inside skimage; cv2 branch supports
  aspect-preserving None) and on `dsize=None` (`cannot unpack non-iterable
  NoneType`). Validate dsize and raise clear NotImplementedError/ValueError.
  Also its `border_value` parameter is accepted but unused.
- `kwimage/im_transform.py:84-85, 217-218, 347-348` — `origin_convention`
  docstring says `If "center"` twice; the second should be `If "corner"`
  (same copy-paste in all three warp functions).
- `kwimage/im_transform.py` — `warp_image`/`warp_projective` cite
  `[WhereArePixels]_` without a `References:` section (only `warp_affine`
  has one → Sphinx unresolved-citation warnings); the `backend` arg is
  undocumented in all four public functions of this module.
- `kwimage/im_cv2.py:2359-2372` — `morphology` docstring says `input` but the
  parameter is `data`; the "hitmiss requires uint8" claim is unenforced. Fix
  docs; add validation.

**Acceptance for this doc**: all new regression tests pass; full suite +
doctests green; CHANGELOG updated (`### Fixed` entries; 2.15 under
`### Changed`).
