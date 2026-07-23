## 2026-03-10 19:26:58 +0000
User asked to narrow the scope of the current typing PR. The working target shifted from a generic "fix typing" pass to a more specific constraint: remove `.pyi` files, port useful type information inline, preserve `from __future__ import annotations`, avoid runtime typing workarounds, and keep a separate cv2 workaround intact. The user also pointed out several concrete regressions introduced by the branch: `importlib` replacing ordinary imports, `sys.modules.get('torch', None)` being used as an import avoidance pattern, `Any = object`, `self: Any`, `__docstubs__`, and constructor changes that silently created default empty data structures.

State of mind: the branch clearly accumulated multiple motivations at once, and the hardest part has been separating "typing moved inline" from "behavior changed to placate a checker". The useful distinction here is whether a change exists only to express type information or whether it changes runtime semantics, import behavior, or default object shape. That distinction has been guiding the cleanup. I agree with the user that several of the runtime edits were scope creep. I also think the user is right to protect existing assumptions around `None` data in geometry wrappers, because changing those constructors can hide invalid states and make downstream debugging harder.

What happened: I compared the branch against `origin/main` first and then against `bedde9f33687a355a2ee4b1997511b08613efe59` when the user clarified the real baseline. I reverted the explicit `importlib` conversions in `kwimage/algo/algo_nms.py`, `kwimage/_backend_info.py`, and `kwimage/algo/_nms_backend/torch_nms.py`, while keeping or restoring inline type annotations where requested. I removed `Any = object`, deleted all `__docstubs__` strings, restored `Coords.__init__` so it no longer manufactures empty coordinate arrays, and removed several added constructor defaults in `Detections`, `Heatmap`, `Mask`, `Points`, and `Polygon`. I also normalized several new torch lookups back to guarded imports instead of module-cache probing. I kept the inline typing work where possible and preserved the cv2-related edits because the user explicitly said those should remain.

Tradeoffs and risks: the branch is still not fully audited for every non-typing runtime change after `bedde...`; I removed the most obvious scope-creep categories first because those had the clearest user direction. There is a risk that some remaining type-driven edits still subtly change behavior, especially in larger modules like `heatmap`, `detections`, and `polygon`, where the diff is large and the typing work interleaves with runtime cleanup. Another tradeoff is that restoring standard imports can re-expose optional dependency edge cases, but that is aligned with the user's preference to avoid checker-driven import indirection.

What might break: optional torch / turbojpeg paths could still have latent behavior differences if other files in the branch used the same pattern and I have not reached them yet. Some constructors may still assume dictionary-like input based on older branch edits around them. There is also a risk that removing helper runtime defaults will surface pre-existing invalid call sites; that is acceptable if those defaults were masking real issues.

What I am confident about: the touched modules compile, targeted imports succeeded, and the specific anti-patterns the user called out have been removed from the files I edited. I am confident the journal should help a future maintainer understand that this session was primarily about scope control, not feature work.

Testing notes: ran `python -m compileall` on the touched modules multiple times during the session. Imported `kwimage.algo.algo_nms`, `kwimage.algo._nms_backend.torch_nms`, `kwimage._backend_info`, `kwimage.structs.mask`, `kwimage.structs.coords`, `kwimage.structs.detections`, `kwimage.structs.heatmap`, `kwimage.structs.points`, `kwimage.structs.polygon`, and `kwimage.transform` after the relevant edits. No full test suite run yet.

Open questions / next steps: do a tighter audit against `bedde...` for remaining non-typing runtime edits, especially in `heatmap.py`, `detections.py`, `polygon.py`, and any cv2-adjacent paths. Confirm exactly which util_warp / resize changes constitute the intended cv2 workaround so they are protected while additional scope reductions proceed.

## 2026-03-10 19:51:31 +0000
User then narrowed the request further: the inline-typing commit should be reduced to strictly inline typing, with non-typing edits reverted, while `__docstubs__` removals stay removed. I treated this as a second-pass scope audit focused on the split typing commit rather than the whole branch.

What changed in this pass: I reverted mixed runtime changes in `kwimage/algo/algo_nms.py`, `kwimage/im_color.py`, `kwimage/structs/heatmap.py`, and `kwimage/structs/detections.py` while keeping `from __future__ import annotations` and the inline type signatures. Concretely, I restored old runtime bodies for the NMS implementation registry / torch lookup paths, restored `Color.__init__` and `Color.distinct()` logic, removed the new numpy/torch dual-path behavior that had been added in `Heatmap` warping and drawing helpers, restored the previous `EuclideanTransform` translation-removal construction, and put the older `sys.modules.get('torch', None)` behavior back into `Detections` helper methods. I also kept earlier reversions of non-typing edits in `_common.py`, `util_warp.py`, `boxes.py`, `coords.py`, `mask.py`, `points.py`, `polygon.py`, and `tests/test_resize.py`.

Validation results: `python -m compileall` passed on the touched files. A full `pytest -q` run completed with 4 failures and 437 passes / 122 skips. The four failures appear to be environment-sensitive consequences of restoring older behavior rather than fresh regressions from this cleanup:

- Three `kwimage.structs.heatmap` xdoctests now fail with `ModuleNotFoundError: No module named 'torch'` because the restored pre-typing code again assumes torch is present for those visualization / warp code paths.
- `tests/test_resize.py::test_imresize_multi_channel` now fails with `ModuleNotFoundError: No module named 'timerit'` because the restored pre-typing test body imports `timerit` in this environment.

Interpretation: the local test run no longer points to obvious hidden runtime drift inside the trimmed files; instead it shows that some earlier branch changes had also been compensating for optional dependency availability in this environment. That leaves an open product decision: optimize for strict "typing only" scope, or keep a few non-typing portability changes that make the test suite pass without optional `torch` / `timerit`.

## 2026-06-07 20:34:00 -0400
User asked to fix GitLab CI for kwimage's Python 3.10 and 3.11 runs and to drop Python 3.9. I treated the uploaded CI log as the primary failure signal. The 3.10 full-loose job installed successfully and began pytest, but the log showed a cascade of failures rooted in `kwimage/structs/_generic.py` using PEP 695 generic class syntax (`class _ExperimentalListProxy[T]`), which is only accepted by Python 3.12+.

What changed: I converted `_ExperimentalListProxy` to the older `typing.Generic[T]` spelling so the module remains generic without using 3.12-only syntax. I also updated the packaging metadata to make Python 3.10 the minimum supported version, removed the 3.9 classifier, and removed the cp39 build/test jobs from the generated GitLab CI file.

Tradeoffs and risks: `AGENTS.md` still says runtime support is Python >=3.8, but the user's explicit request and the current `pyproject.toml` already indicated a newer floor. I left the older version-specific requirements markers in place because `python_requires >=3.10` prevents installation on Python 3.9 and older, and removing historical marker branches would create a wider dependency-file churn unrelated to the failing CI. The `.gitlab-ci.yml` file is autogenerated by xcookie, so these changes should be regenerated from `pyproject.toml` in a normal maintainer workflow when convenient.

Testing notes: I verified that the edited Python module parses under Python 3.10, 3.11, and 3.12 grammar using `ast.parse(..., feature_version=...)`. I also ran `python -m compileall` on `kwimage/structs/_generic.py`. I did not run the full dependency-heavy test matrix locally.

Next steps: run the cp310 and cp311 GitLab jobs again. If another failure appears after the syntax fix, it should be a real next-layer test/dependency issue instead of the current import-time cascade.

### 2026-06-08: Restore legacy colormap compatibility

The strict Python 3.10 CI job pins Matplotlib 3.5.0, where the
`LinearSegmentedColormap` returned by `mpl.colormaps[...]` does not provide the
newer `.resampled(...)` method used by `Color.distinct(..., legacy=True)`. Added
a fallback to `mpl.cm.get_cmap(name, lut)` so the legacy color generation path
continues to work with the strict minimum dependency set.

## 2026-07-23 18:43:43 -0400
The user reported fresh-install failures with opencv-python-headless 5.0.0 while requiring continued compatibility with older OpenCV releases. The supplied test log isolated two concrete regressions: the OpenCV 5 text renderer rejects non-uint8 destinations, and Python bindings now flatten some vector<Point> outputs from (N, 1, 2) to (N, 2).

What changed: `draw_text_on_image` now routes `cv2.putText` through a narrow compatibility helper. It preserves the native OpenCV 3/4 path unchanged. When OpenCV raises the specific `img.depth() == CV_8U` assertion on a non-uint8 image, the helper rasterizes text into a binary uint8 mask and writes only glyph pixels into the original array. This avoids quantizing the entire image and preserves dtype, float precision outside the glyphs, NaNs, masked-array masks, and inplace behavior. The binary mask intentionally matches OpenCV 4's behavior, which disabled antialiasing for non-uint8 destinations. `Mask.get_xywh` now reshapes `findNonZero` results to `(-1, 2)`, and the same cheap shape normalization was applied to contour, hierarchy, and convex-hull outputs that can be exposed to the OpenCV 5 vector-array semantic change.

Testing and evidence: added regression tests that monkeypatch OpenCV 4 to emulate OpenCV 5's uint8-only `putText` and flattened point-array returns. `compileall` and `git diff --check` pass. A standalone import of `im_draw.py` verified that the OpenCV 4 native path remains pixel-identical and that the fallback preserves binary value behavior and masked-array masks. The full repository suite could not be run in this container because the project dependencies (`ubelt`, `kwarray`, and `xdoctest`) are not installed; the user's supplied OpenCV 5 run remains the acceptance environment.

Tradeoffs and risks: OpenCV 5 uses a new text rendering engine, so glyph geometry and size can differ from OpenCV 4 even when the legacy API is used; this is upstream behavior and cannot be made pixel-identical without replacing the renderer. The fallback is deliberately triggered only for the known CV_8U assertion so unrelated OpenCV errors are not hidden. The extra contour normalizations are low-risk reshapes that preserve point ordering on both old and new return layouts.

Next step: run the full suite in the user's OpenCV 5.0.0.93 environment, then run at least one older supported OpenCV environment to confirm the native path and strict/minimum dependency matrix remain green.
