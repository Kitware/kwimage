# 01 — Release blockers for 0.12.0

**Goal**: make the `dev/0.12.0` branch releasable. These items cause hard
failures *today* on fresh installs or are past their scheduled removal date.

**Prerequisites**: none. Do this doc before anything else.

---

## 1. Execute past-due deprecation removals

`ub.schedule_deprecation` raises at/after its `remove` version. The package is
now `0.12.0`, so every site with `remove <= 0.12.0` currently raises
`AssertionError("Forgot to remove deprecated ...")` **on every call**. Verified:
`kwimage.padded_slice(np.arange(10), slice(-2, 5))` raises today.

Physically delete the deprecated code (function bodies, parameters, and their
exports), and remove them from `kwimage/__init__.py` (both the lazy-loader
`submod_attrs` mapping and `__all__`):

| Site | What to delete | Schedule |
|---|---|---|
| `kwimage/im_core.py:388` | `padded_slice` (moved to kwarray) | error 0.11.0, remove 0.12.0 |
| `kwimage/im_core.py:581` | `normalize` (moved to kwarray) | error 0.11.0, remove 0.12.0 |
| `kwimage/im_core.py:603` | `find_robust_normalizers` (moved to kwarray) | error 0.11.0, remove 0.12.0 |
| `kwimage/structs/boxes.py:560` | `Boxes.to_tlbr` (use `to_ltrb`) | error 0.11.0, remove 0.12.0 |
| `kwimage/structs/boxes.py:663` | `to_shapley` (typo method; `to_shapely` stays) | remove 0.11.0 (overdue) |
| `kwimage/structs/boxes.py:3451` | deprecated `impl` argument | remove 0.11.0 (overdue) |
| `kwimage/im_io.py:831` | deprecated `nodata` param (use `nodata_method`) | remove 0.11.0 (overdue) |
| `kwimage/im_io.py:2120` | deprecated `nodata` param (use `nodata_value`) | remove 0.11.0 (overdue) |

Also:
- `kwimage/im_cv2.py:337` — `imscale` has been in error-state since 0.9.5 but
  is still exported from `kwimage/__init__.py` and advertised in README.
  Remove the export (decide: delete the function now, or keep the error stub
  unexported until 1.0). Update the README API listing (covered again in 07).
- Search for any other sites: `grep -rn "remove='0.1[012]" kwimage/`.
- `kwimage/structs/polygon.py:2927-2935` uses `deprecate='now', remove='soon'`
  — replace with real version numbers (e.g. `deprecate='0.12.0',
  error='1.0.0', remove='1.1.0'`).

**Changelog**: add a `### Removed` section listing each removal.

**Acceptance**: `grep -rn "remove='0.1[012]" kwimage/` returns nothing;
`python -c "import kwimage; kwimage.padded_slice"` raises AttributeError;
full test suite + doctests pass.

## 2. OpenCV 5.0 compatibility (breaks fresh installs / loose CI)

The requirement pins are open-ended (`opencv-python-headless>=4.10...`), so
fresh installs get cv2 5.0, where 2 tests + 3 doctests currently fail:

- **`kwimage/im_draw.py:394`** (`draw_text_on_image`): cv2 5 requires uint8
  for `cv2.putText` (`img.depth() == CV_8U` assertion). Currently a float
  canvas is passed. Fix: convert the canvas to uint8 for the putText call and
  convert back (preserving the documented dtype behavior), or restrict putText
  input dtype with an explicit ensure_uint255/ensure_float01 round-trip.
  Failing today: `tests/test_detections.py::test_detections_draw_on_corner_cases`,
  doctests `im_cv2.py::adjust:0`, `im_draw.py::draw_text_on_image:2`,
  `im_draw.py::nodata_checkerboard:0`.
- **`kwimage/structs/mask.py:1523-1528`** (`Mask.get_xywh`):
  `cv2.findNonZero` returned `(N,1,2)` in cv2 4.x but returns `(N,2)` in 5.x;
  `cv2_coords[:, 0, 0]` now raises IndexError for every non-empty c-mask,
  breaking `Mask.box()`, `bounding_box()`, `to_boxes()`, `get_patch()`,
  `get_polygon()`. Fix defensively: `cv2_coords = cv2_coords.reshape(-1, 2)`
  then index `[:, 0]` / `[:, 1]`. Failing today:
  `tests/test_mask.py::test_mask_with_bool_data`.
- **Sweep**: audit all other cv2 result indexing for the same shape change:
  `grep -rn "findNonZero\|findContours\|\[:, 0, 0\]\|\[:, 0, 1\]" kwimage/`.
  (`findContours` still returns `(N,1,2)` in cv2 5.0.0 — but make indexing
  shape-tolerant where cheap.)
- Consider adding a CI job or tox env explicitly on `opencv>=5` (see doc 06).

**Acceptance**: with opencv-python-headless 5.x installed,
`python -m pytest tests/ -q` has 0 failures and
`python -m pytest -p no:doctest --xdoctest kwimage -q` has 0 failures.

## 3. Rescue the uncollectable stack tests

`tests/test_stack.py` is **untracked in git** and its functions are named
`stack_images_empty_list` / `stack_images_grid_empty_list` — no `test_`
prefix, so pytest silently never collects them.

- Rename both functions with the `test_` prefix.
- Verify they pass (`python -m pytest tests/test_stack.py -q`); fix
  `im_stack.py` if they expose real empty-list bugs.
- `git add tests/test_stack.py` and commit.

**Acceptance**: `python -m pytest tests/test_stack.py --collect-only -q` shows
≥ 2 tests; file is tracked by git.

## 4. Fix broken environment-variable parsing (affects users immediately)

`kwimage/_internal.py:14-23` — `FALSY_ENVIRONS` is a copy-paste of the truthy
set `{'true', 'on', 'yes', '1'}`. Falsy strings fall through to the default,
so e.g. `KWIMAGE_DISABLE_TORCHVISION_NMS=0` cannot re-enable torchvision NMS
and `KWIMAGE_DISABLE_WARNINGS=false` is a no-op.

Fix: `FALSY_ENVIRONS = {'false', 'off', 'no', '0'}`. Add a unit test for
`_boolean_environ` covering truthy, falsy, and unset values.

**Acceptance**: regression test passes; grep confirms distinct truthy/falsy sets.
