## 2026-07-23 19:34:46 -0400

The user requested a conservative audit overlay and then required every hunk to be justified after an earlier cleanup regressed a valid fix. This entry records the reviewed set of local defects retained in the direct overlay. Changes are limited to explicit copy/paste errors, dropped arguments, documented-but-unreachable branches, backend-specific runtime errors, and geometry calculations whose surrounding comments establish the intended quantity.

The first direct overlay was not fully validated: `Matrix.__imatmul__` returned `self` but still used NumPy's unsupported `ndarray @=` operation. Review also found that the initial `Boxes.intersection` repair was NumPy-only and that the connected-components cleanup dropped the old `np.int16` spelling. The reviewed overlay replaces the matrix rather than applying ndarray `@=`, preserves the tensor conversion branch, and keeps the historical `np.int16` alias while adding documented `np.uint16` forms. All retained hunks and their compatibility implications are itemized in the accompanying review report.

Focused tests run with temporary out-of-tree compatibility stubs for unavailable local dependencies. They are useful regression checks but do not replace the full project suite under the user's supported Python, NumPy, Torch, and OpenCV matrix.
