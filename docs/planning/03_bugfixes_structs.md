# 03 — Bug fixes: geometry structs

**Goal**: fix confirmed defects in `kwimage/structs/`.
**Prerequisites**: doc 01 (the `Mask.get_xywh` cv2-5 fix and deprecation
removals in `boxes.py` land there).

Verified by execution unless marked *(plausible)* / *(needs torch)*.
Reproduce each Failure as a regression test first. Three systemic root causes
appear repeatedly — fix the instances here; doc 08 consolidates the patterns:

- **Shallow-copy idiom**: `self.__class__(self.data, self.meta)` variants that
  share dicts between "copies" (items 3.4, 3.19, 3.21, 3.28).
- **Integer-dtype geometry**: ad-hoc dtype promotion causing silent
  truncation or crashes (items 3.3, 3.20, 3.25).
- **Partially-plumbed kwargs** (items 3.6, 3.17).

---

## High severity

### 3.1 `Polygon.fill` fills nothing on ≥4-channel images
`kwimage/structs/polygon.py:2314` — `for bx in enumerate(range(image_.shape[2]))`
binds `bx` to a tuple, extracting a 2-channel slice; cv2 fills only its first
channel and the duplicate fancy-index write-back drops the result.
- Failure: `kwimage.Polygon.random().scale(16).fill(np.zeros((16,16,5),
  np.uint8), value=1)` → all zeros (3-channel works).
- Fix: `for bx in range(image_.shape[2])`. Test 1, 3, 4, 5-channel images.

### 3.2 `Mask.get_xywh` broken on OpenCV 5
Covered in **doc 01 item 2** (`mask.py:1523-1528`). Verify done.

### 3.3 `Coords.translate` truncates fractional offsets on int data
`kwimage/structs/coords.py:927` — the *offset* is cast down to the data dtype,
so translating integer coords by 0.5 is a silent no-op. Inconsistent with
`Coords.scale` (:866-870) which upcasts the data.
- Failure: `kwimage.Coords(np.array([[1,2]])).translate(0.5)` → unchanged.
- Fix: upcast data to float like `scale` does. Propagates to
  `Points.translate` / `Polygon.translate`.

### 3.4 `Points.round(inplace=False)` mutates the original
`kwimage/structs/points.py:393` — the "copy" is built as
`self.__class__(self.data, self.meta)` (same dict), then written into.
- Failure: `pts = Points(xy=np.array([[1.4,2.6]])); pts.round(inplace=False)`
  → `pts.xy` becomes `[[1.,3.]]`.
- Fix: `self.data.copy()` (see `compress`/`take` for the pattern).

### 3.5 `Boxes.intersection` crashes on disjoint int boxes
`kwimage/structs/boxes.py:3594-3596` — calls torch's `.to(float)` on a numpy
array when any pair is disjoint and dtype is integer.
- Failure: `Boxes(np.array([[0,0,1,1]]),'ltrb').intersection(
  Boxes(np.array([[10,10,11,11]]),'ltrb'))` → AttributeError.
- Fix: `ltrb = ltrb.astype(float)` for numpy (keep `.to` for torch via the
  impl layer). The copy-paste in `union_hull` (:3642-3643) is dead code
  (min/max can't produce is_bad) — simplify or fix identically.

## Medium severity

### 3.6 `Boxes.isect_area` ignores `bias`
`kwimage/structs/boxes.py:3550` — `bias` accepted but never forwarded to
`_isect_areas`; also makes `iooas(bias=1)` internally inconsistent.
- Fix: `_isect_areas(..., bias=bias, _impl=_impl)`. Test bias=0 vs bias=1
  differ; extend the doctest beyond default bias.

### 3.7 `Polygon.regular(n)` returns an (n+1)-gon
`kwimage/structs/polygon.py:1321` — calls `cls.circle(resolution=num + 1)`;
after the 0.11.2 circle fix, `resolution` already equals side count.
- Failure: `Polygon.regular(3)` has 4 unique vertices.
- Fix: `resolution=num`.

### 3.8 `Polygon.star` lopsided
`kwimage/structs/polygon.py:1341-1359` — built from `regular(10)` (an 11-gon
per 3.7) with hard-coded indices. Fixing 3.7 fixes this; add a test asserting
the 5 inner radii of `Polygon.star()` are equal.

### 3.9 `_is_clockwise` omits the closing edge
`kwimage/structs/polygon.py:3911-3931` — the shoelace sum skips last→first
vertex; kwimage polygons are stored unclosed, so orientation is misclassified
whenever that edge dominates → `_ensure_vertex_order` produces wrong rings.
- Failure: verts `[(0,3),(1,3),(0.1,3.5)]` — shapely says CCW,
  `_is_clockwise` returns True.
- Fix: include the wrap-around edge (e.g. compute with `np.roll`). Test
  against `shapely ... .exterior.is_ccw` on random triangles.

### 3.10 `Mask.from_text` drops content when padding
`kwimage/structs/mask.py:1244-1248` — a `shape` taller than the text
*replaces* the parsed rows with zero rows and doesn't reach the requested
height.
- Failure: `Mask.from_text('oo', zero_chr='.', shape=(4,2))` → zeros of
  shape (3,2).
- Fix: append padding rows: `data = data + [[0]*max_width for _ in
  range(extra_rows)]`.

### 3.11 `Mask.union` broken for f_mask / array_rle
`kwimage/structs/mask.py:1341-1349` — fallback branch references
`cython_mask` never assigned in scope (NameError with pycocotools installed;
NotImplementedError without).
- Fix: fetch backend via `_lazy_mask_backend()` in that branch, or route
  non-bytes-RLE formats through the c_mask path (numpy bitwise-or works and
  needs no backend).

### 3.12 `Mask.draw_on(show_border=True)` TypeError
`kwimage/structs/mask.py:946,1007` — iterates `poly.data['exterior']` (a
`Coords`, not iterable).
- Failure: `Mask.random().draw_on(img, show_border=True)` → TypeError.
- Fix: iterate `poly.data['exterior'].data`; consider drawing interiors too.

### 3.13 `_coerce_coco_segmentation` inverted shape logic
`kwimage/structs/segmentation.py:287-291` — when an RLE dict lacks shape info
it assigns `data['shape'] = data_shape` (i.e. `None`) instead of `dims`.
- Failure: `_coerce_coco_segmentation({'counts':[...]}, dims=(5,5))` → Mask
  with `shape=None`; `to_c_mask()` raises TypeError.
- Fix: assign `dims` in the None branch.

### 3.14 `Heatmap.random` discards smoothing/renormalization
`kwimage/structs/heatmap.py:1712-1721` — the smoothed/renormalized
`class_probs` is never written back to `self.data`.
- Failure: `smooth_k=3` vs `smooth_k=31` identical; with noise, class probs
  sum to >1 (observed 3.37).
- Fix: `self.data['class_probs'] = class_probs` at the end. Test: probs sum
  to 1 across classes; smooth_k changes output.

### 3.15 `Heatmap.detect` scale indexing wrong
`kwimage/structs/heatmap.py:1323` — `self.tf_data_to_img.scale[::-2]` on a
2-vector yields one element `[sy]`; both H and W thresholds get divided by
the y-scale.
- Fix: `[::-1]`. Test with anisotropic scale (2,4). *(needs torch for
  full `detect`; test the conversion expression directly if torch absent)*

### 3.16 `_dets_to_fcmaps` uses width² as area
`kwimage/structs/detections.py:2124-2125` — `cxywh[..., 2] * cxywh[..., 2]`
→ smaller-on-top draw order wrong for elongated boxes.
- Fix: `cxywh[..., 2] * cxywh[..., 3]`.

### 3.17 `Detections.compress` device check compares flags to itself
`kwimage/structs/detections.py:1616` — `if flags.device != flags.device:` is
always False; should be `flags.device != self.device`. *(needs torch)*
- Fix and add a torch-gated test with CPU data + differently-placed flags if
  feasible; otherwise fix by inspection with a code comment.

### 3.18 `Points.warp` fails for `kwimage.Affine` when `tf_data_to_img` set
`kwimage/structs/points.py:154-169` — the `tf._inv_matrix` update path only
handles skimage transforms/ndarrays; `kwimage.Affine` has no `_inv_matrix`.
- Failure: AttributeError (verified).
- Fix: coerce `kwimage.Affine`/`Matrix` to ndarray (`.matrix`) or use
  `Affine.inv()` for the update.

### 3.19 `Points._warp_imgaug` invalidates meta on the wrong object
`kwimage/structs/points.py:58-65` — copies-and-pops `tf_data_to_img` from
`self.meta` while `new.meta` keeps the stale transform; exactly backwards for
`inplace=False`.
- Fix: operate on `new.meta`. *(needs imgaug to execute; fix by inspection +
  unit test on the meta-handling if imgaug absent)*

### 3.20 `Coords.rotate` returns zeros for int coords
`kwimage/structs/coords.py:995-1002` — rotation matrix built with
`dtype=self.dtype`; cos/sin truncate to 0 for int dtypes.
- Failure: `Coords(np.array([[1,2],[3,4]])).rotate(np.pi/4)` → all zeros.
- Fix: always build the matrix as float; let warp upcast the data (shared
  policy in doc 08).

### 3.21 Polygon warp-family drops `meta`
`kwimage/structs/polygon.py:485,566,638,668,711,2912` — `_PolyWarpMixin`
methods construct results as `self.__class__(self.data.copy())`, dropping
`self.meta` (Points' mixin keeps it).
- Failure: `p.meta['classes']=['a']; p.translate(1).meta == {}`.
- Fix: pass meta through consistently (copy, don't share — see 3.28).

### 3.22 `ObjectList.concatenate([])` IndexError
`kwimage/structs/_generic.py:401-410` — the empty branch assigns `new` but
then unconditionally reads `items[0].meta`.
- Failure: `kwimage.PolygonList.concatenate([])` → IndexError.
- Fix: early-return in the empty branch.

## Low severity

- `polygon.py:2120-2128` — `Polygon.clip` clips only the exterior; interiors
  can extend outside. Clip interiors too (or document the limitation).
- `boxes.py:183` — comment says `_RCHW = (y1, y2, h, w)`; implementation is
  `(y1, x1, h, w)`. Fix the comment.
- `boxes.py:1593-1685` — `scale/warp(inplace=True)` on int boxes with float
  factor raises UFuncTypeError while `inplace=False` upcasts. Unify (doc 08
  dtype policy); at minimum raise a clear error.
- `boxes.py:2795-2800, 3429-3445` — 1-D `Boxes`: `len(boxes) == 4` booby
  trap; `ious` with 1-D self raises IndexError. Either promote 1-D to 2-D in
  the constructor or make `__len__`/`ious` handle it.
- `boxes.py:3665-3670` — `bounding_box()` on empty Boxes raises ValueError;
  return an empty Boxes like sibling methods.
- `coords.py:227,255,270` — `compress`/`take`/`astype` share `meta` with the
  source. Copy meta (see 3.28 pattern).
- `points.py:341-350` — `Points.random(num=(3,4))` misaligns `visible`
  (shape (3,)) with xy (3,4,2); add column-alignment validation (also fix the
  doctest at points.py:1067 which passes 5 category ids for 6 points).
- `detections.py:2106-2119` — `num_kp_classes` unbound when `kp_classes is
  None` but keypoints exist → NameError. Guard the block.
- `detections.py:1115` — if `dset.keypoint_categories()` raises, `kp_classes`
  stays None then `len(kp_classes)` TypeErrors. Handle the None case.
- `heatmap.py:933,991,1027` — conditional local `import warnings` makes other
  branches raise UnboundLocalError. Import `warnings` at module level.
- `heatmap.py:133` — `self.meta['classes']` KeyErrors when absent though the
  code below handles None. Use `.get('classes', None)`.
- `single_box.py:169-198` — `Box.coerce` reshapes torch tensors then feeds
  them to a branch that only accepts ndarray/list → NotImplementedError;
  `**kwargs` silently ignored; multi-box Boxes silently truncated to row 0.
  *(plausible)* Fix coercion; raise on ignored kwargs; assert numel==4.
- `_generic.py:339-368` — `ObjectList.draw_on(fastdraw=True, alpha=None)` →
  `None * alpha` TypeError. *(plausible)* Default alpha before use.

**Acceptance**: all new regression tests pass; suite + doctests green;
CHANGELOG updated.
