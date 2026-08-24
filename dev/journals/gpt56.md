## 2026-07-23 19:34:46 -0400

The user requested a conservative audit overlay and then required every hunk to be justified after an earlier cleanup regressed a valid fix. This entry records the reviewed set of local defects retained in the direct overlay. Changes are limited to explicit copy/paste errors, dropped arguments, documented-but-unreachable branches, backend-specific runtime errors, and geometry calculations whose surrounding comments establish the intended quantity.

The first direct overlay was not fully validated: `Matrix.__imatmul__` returned `self` but still used NumPy's unsupported `ndarray @=` operation. Review also found that the initial `Boxes.intersection` repair was NumPy-only and that the connected-components cleanup dropped the old `np.int16` spelling. The reviewed overlay replaces the matrix rather than applying ndarray `@=`, preserves the tensor conversion branch, and keeps the historical `np.int16` alias while adding documented `np.uint16` forms. All retained hunks and their compatibility implications are itemized in the accompanying review report.

Focused tests run with temporary out-of-tree compatibility stubs for unavailable local dependencies. They are useful regression checks but do not replace the full project suite under the user's supported Python, NumPy, Torch, and OpenCV matrix.

## 2026-08-23 19:04:00 -0400

The user ran the first typing overhaul under their local `ty` environment. The focused regression suite passed 29 tests, but the broader annotations exposed 51 diagnostics. I treated those diagnostics as feedback on the new contracts rather than a reason to restore line-level ignores.

Most failures came from a small set of causes: optional values that `ty` could not narrow through helper predicates, broad container element types, dynamic subclass construction in mixins, and annotations that did not match the actual implementation. The follow-up changes make those boundaries explicit. `Detections` now narrows optional keypoint metadata and imports torch only in paths that require it. `Heatmap` normalizes scalar-or-sequence arguments into local arrays instead of mutating narrowly typed parameters, and its mixin contract includes the spatial properties used by drawing code. `ObjectList` keeps its unconstrained element type for storage but requires a small protocol on methods that actually draw or serialize elements. `Boxes.center_x`, `Boxes.center_y`, and `Boxes.contains` had return annotations that were too broad or incorrect; correcting them removes downstream false errors in `Box` without changing those computations.

I preserved runtime behavior where possible. In particular, dynamic subclass construction remains dynamic and is isolated behind a targeted cast instead of being replaced with a concrete `Heatmap` constructor. `Box.draw_on` keeps passing the same one-element label list, including its existing `None` behavior. The source compiles and `git diff --check` passes in this sandbox. I cannot run the repository's full tests here because `ubelt` and `kwarray` are not installed, and the sandbox cannot install packages from the network. The user's previous run establishes that the pre-follow-up overlay passed `tests/test_audit_regressions.py`; the next local `ty check kwimage tests/` is the important verification for this follow-up.

## 2026-08-23 19:04:30 -0400

Completed the diagnostic-by-diagnostic follow-up for the user's 51-diagnostic `ty` run. The last audit caught a runtime typing hazard in `Heatmap.scale` / `translate`: a `typing.cast` referenced `Sequence`, which is intentionally imported only under `TYPE_CHECKING`. Those casts now use the runtime-imported `Any`, so iterable scale and translation paths cannot raise `NameError` from the typing change. `smooth_prob` uses a narrow `Any` cast around NumPy's scalar `argmin` / `argmax` calls because the reported NumPy stub overload set did not model those no-axis calls consistently.

Validation: `python -m compileall -q kwimage tests/test_audit_regressions.py` and `git diff --check` pass. A direct local pytest attempt cannot reproduce the user's environment: the sandbox lacks the xdoctest pytest plugin, and bypassing repository pytest configuration then fails imports for `ubelt` and `kwarray`. The user's environment remains the authoritative test target for the replacement overlay; rerun `ty check kwimage tests/` first, then the focused regression test file.

## 2026-08-23 19:56:00 -0400

The user asked to continue the typing burn-down after the broader checked surface passed locally. This pass targets `kwimage/transform.py` and `kwimage/structs/boxes.py`, removing both modules from the blanket `ty` override instead of increasing annotations under an ignored module.

`Boxes` now declares the NumPy-or-Torch data contract used by its public properties and overlap methods, while dynamic backend arithmetic is kept behind narrow local `Any` casts where the checker cannot express that two arrays share the same backend. The conversion, property, transform, and drawing mixins explicitly declare the pieces of the concrete `Boxes` interface that their method bodies require. Factory classmethods use `cls` rather than shadowing the `Boxes` type name, and in-place mixin branches are narrowed back to the concrete `Boxes` contract. While reviewing those paths, `union_hull` exposed a runtime defect: invalid integer NumPy boxes used Tensor-only `.to(float)`. The NumPy path now uses `.astype(float)` before assigning NaN.

`transform.py` now has complete public method annotations for `Matrix`, `Projective`, and `Affine`. Symbolic matrix support remains part of the matrix-data contract, and implementation-local casts isolate operations where NumPy and SymPy expose different APIs. Affine helpers materialize the identity representation with `np.asarray(self)` before indexing, so the `matrix=None` identity convention remains valid. Projective and affine factory/fit classmethods construct `cls`, which makes their declared classmethod behavior consistent for subclasses.

Validation in this sandbox includes `compileall`, TOML parsing, AST audits for public annotation completeness and mixin structural dependencies, `git diff --check`, and direct transform/box smoke tests under minimal out-of-tree dependency stubs. The first direct smoke run caught executable `typing.cast(Any, ...)` calls whose `Any` name was imported only under `TYPE_CHECKING`; those calls now use the runtime `typing.Any` spelling and the smoke suite passes. The sandbox still cannot run `ty`: the binary is unavailable and package installation fails because external and internal package indexes cannot be resolved. The repository also cannot be imported normally here because `ubelt` and `kwarray` are unavailable. The user's environment should run `ty check kwimage tests/` as the authoritative check after applying this overlay. Focused regression tests were added for the integer NumPy `union_hull` path, identity `Affine` helpers, and subclass-preserving transform classmethods.

## 2026-08-23 20:09:00 -0400

Followed up on the transform/Boxes typing pass after the user's local `ty` run exposed 67 diagnostics. The main issue was structural: classmethods implemented on `_BoxConversionMixins` were inferred as constructing the mixin itself, so `ty` only knew about `object.__init__`. I kept subclass-preserving runtime behavior and introduced a local concrete `type[Boxes]` view at those constructor boundaries rather than weakening the return types.

The remaining diagnostics mostly came from backend correlation that the type system cannot express directly: a `Boxes` instance holds either NumPy data or Torch data, and masks/assigned values produced from that data use the same backend. I kept the public `ndarray | Tensor` types and isolated `Any` casts at mutation/attribute-dispatch boundaries. I also stopped reusing scalar/optional draw parameters as normalized lists, converted plotting/image-allocation scalars with `.item()`, made dynamic transform parameter dictionaries explicitly `Any`-valued, and gave `Affine.fliprot` an explicit error when flips/rotations are requested without a canvas size.

One compatibility issue in the checked code was also cleaned up: NumPy 2 removed `np.float_`, so the dtype lookup now uses `getattr(np, 'float_', np.float64)`. Focused regressions cover mixin classmethod subclass preservation, NumPy/Torch negative-extent repair, and the `fliprot` canvas requirement. I still cannot execute `ty` in this sandbox because its package cannot be fetched, so the user's environment remains the authoritative checker run.

## 2026-08-23 22:05:00 -0400

Followed up on the host `ty` run after the transform/Boxes unsuppression. The
remaining four diagnostics were all local to `Boxes`: `ty` did not retain the
scalar-versus-sequence narrowing for `draw_on(alpha=...)`, and two casts around
point coordinates had become redundant after the earlier boundary cleanup. I
kept the existing runtime behavior and made only the static distinction explicit
with local `cast` targets. `Sequence` joins the existing runtime import from
`typing` solely so it can be used as the cast target; this does not introduce a
new module import. I removed the redundant casts outright.

This is intentionally a minimal correction rather than another typing expansion.
I am confident it addresses the exact four reported diagnostics without changing
box drawing semantics. The remaining uncertainty is checker verification because
`ty` is not installed in this sandbox; the host run remains authoritative. I
validated syntax, whitespace, and the affected draw path locally.

## 2026-08-23 22:14:00 -0400

The user asked that the next typing phase prioritize useful public contracts rather than merely replacing missing annotations with `Any`. This pass targets `Coords` and `Points` as the next foundational geometry layer. Both are removed from the blanket `ty` override, reducing that list from 19 to 17 modules, but the acceptance criterion is stronger: downstream callers should see array-or-Tensor geometry data, concrete `Coords`/`Points` transformation results, typed Shapely conversions, typed indexing for `PointsList`, and structured COCO conversion results instead of broad `Any` returns.

I added a shared `ArrayData = ndarray | Tensor` typing alias and made `Coords.data` plus `Points.xy` expose it. Public geometry methods now have complete parameter/return annotations, while dynamic backend dispatch remains isolated behind local `Any` casts where NumPy/Torch correlation cannot be expressed directly. `_generic.isinstance_arraytypes` is now a `TypeGuard`, and integer indexing through generic `ObjectList[T]` returns `T`, which makes `PointsList[0]` statically be `Points`. A static contract test module uses `assert_type` under `TYPE_CHECKING` so a future change from a useful type back to `Any` is detectable even if ordinary assignment compatibility would hide it.

The typing audit exposed two runtime/API inconsistencies in `Points`: `to_wkt()` documented a string but returned a Shapely object, and `from_imgaug()` passed a `Coords` instance to the generic `Points(data=...)` path instead of constructing the `{'xy': ...}` data mapping. Both are corrected and the WKT behavior has a focused regression. The v1 COCO parser also had unstable local types and an uninitialized category-index path; it now keeps list-building locals separate from normalized arrays and drops category indices when a category is missing rather than constructing an object array or relying on an unbound local.

The repository metadata says `requires-python = ">=3.10"` and xcookie also declares 3.10, while `AGENTS.md` still said 3.8. I updated the agent guide to 3.10 so future typing work does not optimize for an unsupported language floor. Validation available here includes `compileall`, Python-3.10 grammar parsing via `ast.parse(feature_version=(3, 10))`, TOML parsing, public-annotation audits, and `git diff --check`. The sandbox still cannot execute `ty` because the package is not cached and network resolution is unavailable; the user's local `ty check kwimage tests/` remains the authoritative checker pass.

A final public-API audit found two remaining quality gaps before packaging. The
ImgAug conversion methods still returned bare `Any`, and the shared
`TransformLike` alias included the runtime `SKImageGeometricTransform` symbol,
which is intentionally typed as `Any` for version-compatible `isinstance`
checks. Type-only protocols now describe ImgAug keypoint containers and
augmenters without making ImgAug a required runtime dependency, while
`TransformLike` uses the actual scikit-image geometric base type for static
checking. The audited `Coords`/`Points` public surface now has no bare `Any`
return annotations. The same audit also exposed a real `Points.dtype` bug: it
read `dtype` from the data dictionary instead of the contained `Coords`; it now
delegates to `self.data['xy'].dtype` and has a regression test.

## 2026-08-23 22:37:00 -0400

The user explicitly asked that the typing overhaul not make runtime code less efficient. This is now a first-class constraint for the remaining passes: preserve vectorized NumPy/Torch operations and existing zero-/low-copy behavior, and prefer type-only protocols, annotations, narrowing, or localized dynamic views over Python loops, extra materialization, or checker-driven runtime validation.

The user's v6 `ty` run exposed 19 diagnostics in the newly unsuppressed `Coords` and `Points` modules while the focused regression suite remained green. This cleanup addresses those diagnostics without changing the computational structure. In particular, the alpha normalization in `Coords.draw` and `Points.draw` is restored to the original `ub.iterable` scalar/sequence behavior rather than retaining the `numbers.Number` rewrite from the first typing pass. The NumPy/Torch arithmetic remains vectorized; the `scale`, `translate`, and soft-fill paths only gain static `Any` views around values whose backend-correlated types `ty` cannot express. No new array copies, array materialization, or per-element Python loops are introduced by this follow-up.

The only new runtime branch is an explicit `input_dims is None` error on the optional imgaug warp path, where imgaug already requires image dimensions. This avoids passing `None` into an API that cannot use it and is outside the normal NumPy/Torch transform path. The two-element imgaug dimension coercion now uses direct indexing instead of `tuple(map(...))`, which is at least as cheap and gives the checker a fixed-length tuple. Remaining fixes are import-time version-typing aliases, dictionary typing for matplotlib kwargs, and local dynamic views for optional CategoryTree metadata.

## 2026-08-23 22:41:00 -0400

Followed up on the public-API-first Coords/Points typing pass after the local `ty` run reduced the remaining diagnostics to three. The fixes are deliberately runtime-neutral in geometry paths: `Coords.scale` now uses a type-only cast to expose the already-established array/tensor value to the checker, and `Points.from_coco` uses a type-only protocol view when category IDs require `id_to_idx`. Restored the pre-existing `numbers` import that had been dropped during import cleanup; this returns the draw path to its prior behavior rather than adding new runtime machinery. No vectorized operation was replaced, no new array/tensor copy or materialization was introduced, and no runtime validation was added. The focused regression suite was already green in the user's environment; local validation here is limited to syntax/compile and diff checks because the sandbox lacks the full kwimage dependency environment.

## 2026-08-23 23:02:00 -0400

Continued the public-API-first typing burn-down with `Polygon`, `MultiPolygon`, and `PolygonList`. `kwimage/structs/polygon.py` is removed from the blanket `ty` override, reducing the ignored-module list from 17 to 16. The main public contracts now expose typed polygon ring storage (`PolygonData`), concrete geometry-preserving transform results, concrete Shapely/GeoJSON/COCO conversions, generic `MultiPolygon` / `PolygonList` indexing, and literal-sensitive COCO serialization overloads instead of broad return `Any` types. Static `assert_type` coverage was extended for the common downstream expressions so a future return to `Any` is visible to `ty`.

Runtime efficiency was reviewed explicitly. No vectorized NumPy/Torch operation was replaced with Python iteration, and no checker-driven array/tensor conversion or copy was added. An intermediate draft accidentally made inherited `PolygonList.to_coco()` eager; that change was removed and the existing iterator semantics are preserved with a type-only declaration. Likewise, the ImgAug warp branch uses a static cast for its required dimensions instead of new runtime validation. Existing NumPy/Shapely/OpenCV operations remain structurally the same; local casts isolate backend/stub limitations without changing values.

The source passes `compileall`, Python 3.10 grammar parsing, TOML parsing, public annotation audits, and `git diff --check`. The sandbox still cannot run `ty` or the normal runtime suite because the checker and core dependencies are unavailable, so the user's local `ty check kwimage tests/` remains the authoritative diagnostic pass. I also changed potentially ambiguous Shapely/Matplotlib submodule accesses to explicit imports to avoid the same `possibly-missing-submodule` diagnostics seen in earlier phases.

## 2026-08-24 05:33:17 -0400

Followed up on the first unsuppressed Polygon/MultiPolygon `ty` run. The host checker reported 16 errors plus two redundant-cast warnings, all in the newly exposed polygon surface; the focused regression suite still passed. I kept `polygon.py` out of the override rather than retreating to suppression.

The fixes preserve runtime structure. Mixin construction and backend/stub ambiguities are expressed with local `typing.cast` views, which are runtime no-ops. The polygon fill/draw paths retain their existing OpenCV calls, NumPy allocation behavior, and per-channel fallback; no vectorized operation was replaced and no checker-driven array conversion was added. `draw_on` now uses a separate optional local for its historical `alpha == 1.0 -> None` sentinel instead of assigning `None` back into the public `float` parameter. `morph` likewise uses a separate local sequence instead of mutating the public scalar-or-sequence parameter.

One reported error exposed a real inheritance typing mismatch: `ObjectList.to_coco()` is a generator while `MultiPolygon.to_coco()` has historically returned a list. The base contract now says `Iterable[Any]`, which truthfully admits both existing runtime behaviors without making either eager or lazy path change. `PolygonList` keeps its narrower iterator contract. A focused regression locks this in so future typing work cannot accidentally materialize `PolygonList.to_coco()`. Remaining uncertainty is limited to the unavailable `ty` executable in this sandbox; syntax, diff, and local structural audits are the available validation before the user's next host run.

## 2026-08-24 09:37:00 -0400

The host `ty` run after the polygon cleanup reduced the phase to one diagnostic: `MultiPolygon.to_coco` was still considered an invalid override of `ObjectList.to_coco`. The problem was the base method's explicit `self: ObjectList[_DrawableObject]` annotation. Because `ObjectList` is invariant in its element type, `ObjectList[Polygon]` does not become `ObjectList[_DrawableObject]` merely because `Polygon` satisfies the protocol. The base drawable methods now use a bound type variable (`DrawableObjectT`) so each concrete `ObjectList[T]` keeps its own element type while requiring the drawable protocol. This is annotation-only and does not change generator/list behavior, iteration, allocation, or dispatch at runtime.

## 2026-08-24 09:49:00 -0400

The host `ty` run still reported the same single `MultiPolygon.to_coco` override diagnostic after the bound-element self type change. The remaining conflict is caused by placing any concrete `ObjectList[...]` constraint on the base method's `self` parameter: mutable `ObjectList` is invariant, while subclasses intentionally specialize the element type and may also specialize the concrete iterable return.

`ObjectList.to_coco` now types only its public call/return contract and leaves `self` dynamic (`self: Any`). This keeps the base method callable on heterogeneous object lists while allowing subclasses such as `MultiPolygon` to publish precise COCO overloads. The generator body is byte-for-byte equivalent in behavior: it still iterates `self.data`, yields `None` for missing entries, and otherwise delegates directly to each item's `to_coco`. No casts, helper calls, branches, copies, materialization, or per-element overhead were added to satisfy the checker. The prior bound type variable is removed because it no longer serves a typing purpose.


## 2026-08-24 10:35:00 -0400

The host `ty` run still reported one `invalid-method-override` diagnostic for
`MultiPolygon.to_coco`. The base and implementation signatures are ordinarily
compatible, but the concrete method also carried type-only overloads. Current
`ty` has a known false-positive class around overloaded overrides, so this pass
removes only those `MultiPolygon` overload declarations rather than further
weakening or restructuring `ObjectList`. `MultiPolygon.to_coco` now publishes
`list[CocoPolygon]`, which remains a useful non-`Any` public contract: each
element is either the legacy flat coordinate list or the new-style polygon
dictionary. Runtime code is unchanged.

To keep concrete list APIs strong, `PointsList.to_coco` now has a type-only
`Iterator[CocoKeypoints]` declaration, matching the inherited lazy runtime
implementation. `PolygonList` already had the analogous type-only iterator
contract. The public typing contract test was updated to lock in these concrete
list types. All changes in this follow-up are under `TYPE_CHECKING` or annotation
syntax; no iteration, allocation, array/tensor conversion, copy, validation, or
dispatch behavior is added at runtime.

## 2026-08-24 10:42:00 -0400

Continued the public-API-first typing burn-down with `Mask`, `MaskList`,
`Segmentation`, and `SegmentationList`. Both `kwimage/structs/mask.py` and
`kwimage/structs/segmentation.py` are removed from the blanket `ty` override,
reducing the remaining ignored-module list from 16 to 14. The public mask
surface now exposes a structured `MaskData` union (dense NumPy/Torch data or
RLE dictionaries), typed COCO RLE output, concrete geometry conversions,
backend-preserving mask operations, and `Mask | None` element types for
`MaskList`. Segmentation similarly exposes its actual backend union
(`Mask | Polygon | MultiPolygon`), concrete conversions to `Mask`,
`MultiPolygon`, and `Box`, typed COCO output, and concrete list element types.
Static `assert_type` coverage was extended around those downstream-facing
contracts.

This pass deliberately does not make `Mask` generic over its format. Doing so
could make `to_c_mask().data` statically narrow all the way to a dense array,
but it would be a substantially larger public API design change. Instead,
format-dispatch internals use local dynamic views where a checker cannot infer
the correlation between `format` and `data`, while the public attribute remains
a meaningful representation union rather than `Any`.

Runtime efficiency remains a hard constraint. No vectorized NumPy/OpenCV/Torch
operation is replaced by Python iteration, and no checker-driven array copy,
materialization, device transfer, or format conversion is added. The new local
annotations and dynamic views do not transform values. Existing conversion and
warp behavior is preserved, including the current `Segmentation` wrapper
semantics where transform/numpy/tensor delegation returns the underlying
backend rather than a new wrapper. Validation available in this environment is
Python 3.10 grammar parsing, `compileall`, TOML parsing, public annotation
audits, diff/whitespace checks, and overlay reproduction; the user's local
`ty check kwimage tests/` remains the authoritative checker run.

## 2026-08-24 11:15:00 -0400

Followed up on the first unsuppressed Mask/Segmentation `ty` run. The host
checker reported 12 diagnostics, all in `mask.py`, while the focused runtime
regression suite remained green. The failures were static correlation issues:
mixin `Self` inference on methods that only exist as part of `Mask`, optional
shape flow in `translate`, NumPy ufunc stub acceptance for list inputs, and the
private OpenCV contour accumulator's intentionally staged `None -> ndarray`
construction.

The cleanup keeps the public API strong while leaving runtime algorithms
unchanged. The affected mixin methods use a dynamic annotation only for their
hidden `self` parameter; their public return type remains `Mask`. The NumPy
union/intersection paths still invoke the same `np.bitwise_or.reduce` and
`np.bitwise_and.reduce` operations, but view the ufunc objects dynamically for
stub compatibility. Translation uses the same resolved output-dimension value
through a local dynamic view, and the private contour dictionary is marked
dynamic rather than normalized or copied for the checker.

No array/tensor conversion, copy, device transfer, format conversion, loop,
comprehension, assertion, or validation was added by this follow-up. Python
3.10 parsing and `compileall` pass locally; the user's host `ty check kwimage
tests/` remains the authoritative checker run.

## 2026-08-24 11:13:00 -0400

- Finished the v14/v15 Mask typing cleanup after the local `ty` run exposed three contour-helper diagnostics.
- Kept the existing OpenCV contour algorithm and allocation behavior unchanged: the contour accumulator dictionary is now named `poly_lookup`, and the existing `list(...values())` result is assigned to a distinct `polys` local instead of changing one variable from `dict` to `list`.
- This is a static-flow clarification only; it adds no copies, loops, coercions, validation, or runtime helper calls.
- Validation here: Python 3.10 AST parse, `compileall`, and diff whitespace checks. The authoritative `ty check kwimage tests/` remains the maintainer's local run.

## 2026-08-24 11:16:00 -0400

Refined the already-unsuppressed `Detections` public API after the geometry
primitives became strongly typed. This phase deliberately does not change the
extensible `data`/`meta` storage model; arbitrary custom detection fields are a
supported feature, so those dictionaries remain dynamic internally. Instead,
the standard public access path is strengthened: scores/class indices/
probabilities/weights expose the NumPy-or-Torch array union, NMS and argsort
publish concrete index-array results, device/dtype are non-bare-`Any`, and new
zero-copy `keypoints` / `segmentations` convenience properties expose the
actual supported geometry-list unions. COCO export now publishes a typed
mapping generator rather than `Generator[dict, ...]` with unparameterized
values. The draw alpha annotations were also aligned with the already-supported
per-detection sequence form.

Static `assert_type` coverage now exercises the common downstream Detections
surface: geometry transforms, sort/subset operations, NMS, backend conversion,
device/dtype, COCO export, rasterization, and the standard data/meta
properties. A runtime regression asserts that the new convenience accessors
return the exact objects already stored in `data` / `meta`; they do not
normalize, copy, or materialize values.

Runtime efficiency remains a hard constraint. No NumPy/Torch/OpenCV operation,
array/tensor conversion, copy, device transfer, loop, or comprehension was
added. Formatting-only private code now caches already-existing property
lookups in local variables and one previous runtime `typing.cast` call in the
category-color path was replaced by an annotation-only dynamic view. The
normal Detections compute paths are otherwise unchanged. Python 3.10 parsing,
`compileall`, public annotation auditing, and diff whitespace checks are
available here. The sandbox lacks `ty`, `ubelt`, and `kwarray`; pytest therefore
cannot exercise the repository locally, so the maintainer's `ty check kwimage
tests/` and focused pytest run remain authoritative.

## 2026-08-24 11:23:00 -0400

Follow-up after the maintainer's local `ty` run found one Detections override
diagnostic: `ubelt.NiceRepr.__nice__` is stubbed as returning `str`, while the
longstanding `Detections.__nice__` implementation returns the integer box count.
Changing the body to allocate/return a string would alter runtime behavior solely
to satisfy the checker. Keep that private representation hook typed as `Any`
instead. This does not weaken the public Detections data/geometry contracts from
the preceding phase and adds no runtime work.

## 2026-08-24 11:25:00 -0400

Corrected `Detections.__nice__` to follow the `ubelt.NiceRepr` contract directly. The hook now returns `str(self.num_boxes())` instead of preserving the historical integer return behind an `Any` annotation. This is an intentional repr-hook behavior correction requested by the maintainer; it has no effect on geometry/data-path performance.

## 2026-08-24 11:52:00 -0400

Refined the public `Box` and `Heatmap` APIs and fixed the Detections draw corner-case regression reported by CI. `Box` now exposes concrete array/tensor data, scalar geometry, dtype, Shapely/Polygon/COCO conversion, containment, corners, and drawing return contracts instead of bare `Any`. `Heatmap` now exposes concrete array/tensor channel data, shape/dimension aliases, optional spatial maps, image dimensions, transform/classes metadata, and `detect()` as `Detections`; common transform/backend methods were already concrete and are now covered by static contracts.

The Detections regression came from the typed convenience properties using direct dictionary indexing even though `class_idxs`, `scores`, `probs`, and `weights` are optional by contract. Those properties now use `dict.get(..., None)`, restoring the existing corner-case draw behavior when a field is absent and aligning runtime behavior with the `| None` annotations.

Runtime efficiency remains unchanged in the geometry/image paths. This phase adds no NumPy/Torch/OpenCV coercions, copies, materialization, device transfers, loops, or comprehensions. `Box.draw()` now explicitly returns `None`, matching the existing `kwplot.draw_boxes` behavior. Static `assert_type` coverage was added for the Box and Heatmap public surfaces, plus a focused runtime regression for missing optional Detections fields. Python 3.10 parsing, `compileall`, public return-type auditing, and whitespace checks pass locally. The sandbox lacks `ubelt` / `kwarray`, so repository pytest cannot execute here; the maintainer's local `ty` and pytest runs remain authoritative.

## 2026-08-24 14:20:00 -0400

Follow-up to the Box/Heatmap public API pass after the maintainer's local `ty`
run found three Heatmap body diagnostics. These are representation-correlation
issues inside private alignment/combine code, not public API gaps. Keep the
strong public `Heatmap` property types and expose only local annotation-only
`Any` views where `ty` cannot infer runtime facts: the aligned transform exists
on the path that dereferences `.params`, the root-selection shape is accepted
by the existing NumPy product call, and `Heatmap.numpy()` guarantees NumPy
channel data even though the non-generic `Heatmap` return type still advertises
the broader NumPy-or-Torch union.

No computation was changed. The same `np.linalg.inv`, `np.prod`, NumPy array
construction, and `.astype(dtype)` calls execute on the same objects. The new
locals are simple references to existing objects; there are no casts, copies,
coercions, loops beyond the pre-existing comprehensions, validation branches,
or device transfers. Python 3.10 parsing, `compileall`, and diff whitespace
checks are available here; the maintainer's local `ty check kwimage tests/`
remains the authoritative static validation.
