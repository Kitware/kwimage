# 04 — Bug fixes: transforms, warp, NMS, CLI

**Goal**: fix confirmed defects in `kwimage/transform.py`,
`kwimage/util_warp.py`, `kwimage/algo/`, `kwimage/cli/`.
**Prerequisites**: doc 01.

Verified by execution unless marked *(needs torch)*. The two one-line fixes in
4.1 and 4.2 silently corrupt user data and should be the first commits.

---

## High severity

### 4.1 `Matrix.__imatmul__` returns None
`kwimage/transform.py:135-143` — Python rebinds the target to the return
value of `__imatmul__`; it must `return self`.
- Failure: `m = kwimage.Matrix.random(3); m @= np.eye(3)` leaves `m is None`.
  Affects `Matrix`, `Affine`, `Projective`.
- Fix: `return self`. Test `@=` for all three classes.

### 4.2 `Affine.concise()` drops x-scale
`kwimage/transform.py:1265` — `if math.isclose(sy, 1) and math.isclose(sy, 1)`
checks `sy` twice; should be `sx` and `sy`. Corrupts any serialization
through `concise()`.
- Failure: `Affine.affine(scale=(2.0, 1.0)).concise()` → `{'type': 'affine'}`;
  `Affine.coerce(...)` does not round-trip.
- Fix: `math.isclose(sx, 1) and math.isclose(sy, 1)`. Add a property-style
  round-trip test: `Affine.coerce(A.concise()) ≈ A` over random anisotropic
  params.
- While here: `transform.py:1261/1269` — `math.isclose(x, 0)` with default
  `rel_tol` only matches exact 0.0; add `abs_tol` if tolerance was intended.

### 4.3 `warp_tensor` homogeneous-row bugs *(needs torch)*
`kwimage/util_warp.py:411-415` — the homogeneous row is **prepended**
(`torch.cat([homog_row, mat])`) instead of appended, so 2x3 affine inputs
become a row-permuted matrix whose inverse is wrong (corroborated by the
in-code FIXME at :165).
`kwimage/util_warp.py:420-422` (and duplicate at :1445-1449) — `ishomog`
autodetection is doubly wrong: `mat[-2]` selects the middle row (or a batch
element for Bx3x3), and `not all(x != y)` means "any element equal". Genuine
projective matrices get classified affine (homogeneous division skipped →
quantitatively wrong warps); affine matrices get the slow path.
- Fix: append the row (`torch.cat([mat, homog_row], dim=-2)`); detect with
  `ishomog = not bool((mat[..., -1, :] == homog_row).all())`.
- Also `util_warp.py:401` — shape check tests `mat.shape[-1]` twice; second
  clause should be `mat.shape[-2]`.
- Tests (torch-gated): 2x3 affine agreement vs `kwimage.warp_affine`;
  projective matrix vs cv2 `warpPerspective`.

### 4.4 `Projective` inherits 2x2 default shape
`kwimage/transform.py:388/1204` — `Affine` overrides `shape` to (3,3);
`Projective` does not.
- Failure: `np.asarray(kwimage.Projective(None))` → 2x2 identity;
  `Projective.eye()` is 2x2; `Projective(None).to_skimage()` raises inside
  skimage.
- Fix: move the `shape` property (and `__json__` if applicable) up to
  `Projective`; `Affine` inherits from it.

## Medium severity

### 4.5 `Projective.coerce({'shear': ...})` TypeError
`kwimage/transform.py:801-823` — `'shear'` is in `known_params` but
`Projective.projective()` has no such kwarg.
- Fix: map to `shearx` (like Affine's shim) or remove from `known_params`
  with a clean error.

### 4.6 `Affine.random_params` silently ignores unknown kwargs
`kwimage/transform.py:1607-1701` — only scale/offset/about/theta/shearx are
probed from `**kw`; everything else drops silently. Even the class doctests
(`transform.py:1098`, `:411`) pass `translate=...` which is ignored.
- Failure: `Affine.random(translate=(100, 100))` → offset ~0.17.
- Fix: validate `kw` keys (raise on unknown); accept `translate` as an alias
  for `offset`. Fix the doctests.

### 4.7 `Affine.random_params` range handling inconsistent
`kwimage/transform.py:1641-1655` — `offset=(-20, 20)` raises bare
NotImplementedError while `scale=(0.5, 1.5)` works; and `scale=[1, 2]`
(list) raises while the tuple works due to `not isinstance(tuple)`.
- Fix: support tuple/list ranges uniformly for scale/offset/about/theta, or
  raise a consistent, descriptive error.

### 4.8 `Projective.decompose()` not scale-invariant
`kwimage/transform.py:1039-1072` — assumes `h9 == 1` (assert commented out at
:1040) but never normalizes; equivalent homographies decompose differently.
- Failure: `Projective(P.matrix * 2).decompose()` returns doubled scale/uv.
- Fix: divide by `h9` first (guard `h9 == 0`).

### 4.9 torchvision NMS silently ignores `bias`
`kwimage/algo/algo_nms.py:646-656` — torchvision is top preference for
tensors under `impl='auto'`, but `bias` is not passed and the warning is
commented out; `bias=1` callers silently get bias-0 semantics (the module's
own doctest at :509-510 shows results differ).
- Fix: warn when `bias != 0` and the chosen backend can't honor it, or
  exclude torchvision from auto-preference for `bias != 0`. Add a
  backend-agreement test over available backends with both biases.

### 4.10 Stale numpy warning filter in py_nms
`kwimage/algo/_nms_backend/py_nms.py:98-99` — filter regex
`'invalid value .* true_divide'` no longer matches numpy ≥1.25's message
("...encountered in divide"); zero-area boxes emit an unsuppressed
RuntimeWarning (crashes under `-W error`).
- Fix: better, avoid the invalid divide entirely (mask zero denominators);
  otherwise update the regex to `'invalid value .*divide'`.

### 4.11 CLI: missing required args → raw tracebacks
`kwimage/cli/crop_border.py:49` and `kwimage/cli/stack_images.py` — `src` /
`input_fpaths` default to None with no required check.
- Failure: `python -m kwimage.cli crop_border` → TypeError from
  `ub.Path(None)`.
- Fix: mark positional args required in scriptconfig or validate with a clean
  usage error. Add CLI smoke tests (invoke `main` with tmp files; assert
  non-zero/clean error on missing args).

## Low severity

- `transform.py:1832-1844` — `Affine.decompose()` on rank-deficient matrices
  returns `shearx=nan` with only a RuntimeWarning; the `except TypeError`
  only covers sympy. Detect and raise or document.
- `transform.py:256-261` — `Matrix.det` catches
  `np.core._exceptions.UFuncTypeError` (`np.core` deprecated in numpy 2.x,
  will become AttributeError when removed) and falls back to
  `self.matrix.det()` which only exists on sympy. Catch `TypeError` like
  `Matrix.inv` (:226) does.
- `transform.py:867-872` — `Projective.random` docstring header `Example/`
  (typo) means xdoctest never runs it. Fix to `Example:` and make it pass.
- `transform.py:1586/2060` — `Affine.random`/`Affine.fit` annotated
  `-> Projective`; should be `Affine`. `Matrix.eye` (:263-266) has an unused
  `rng` param.
- `algo_nms.py:627-638` — per-class NMS recursion drops `device_id`. Forward it.
- `algo_nms.py:598-599` + `torch_nms.py:78-79` — empty-input returns `[]`
  while non-empty returns backend-dependent array types. Normalize the
  return type (document + coerce to ndarray of int).
- `algo_nms.py:299` — `_lazy_init` reads the module-global `_impls` instead
  of `self`. Use `self`.
- `algo_nms.py:244-248` — `distutils.version.LooseVersion` fallback raises on
  Python ≥3.12 (distutils removed). Delete the fallback; `packaging` path is
  fine. Also reevaluate the stale "numpy >= 1.20" gate (:248-275).
- `torch_nms.py:22-72` — docstring claims "CURRENTLY NOT WORKING" for the
  registered torch backend; disabled doctests reference undefined names.
  Rewrite the docstring; align the `bias` default (py_nms bias=1 vs torch_nms
  bias=0) or document loudly.
- `cli/__main__.py:47-50` — assert-after-use in the dead `_CLI` branch;
  delete the branch. `cli/__main__.py:40` monkeypatches the shared ModalCLI
  class with a `version` property — set on the instance.
- `cli/stack_images.py:70` — skipped doctest calls undefined `main(...)`;
  should be `StackImagesCLI.main`. Unify `argv=` vs `cmdline=` between the
  two CLIs.
- `kwimage.algo` exports `non_max_supression` (misspelled). Add a correctly
  spelled `non_max_suppression` alias and schedule deprecation of the typo
  for 1.0 (keep both exports until then).

**Acceptance**: all new regression tests pass; suite + doctests green;
CHANGELOG updated. Torch-gated tests use
`# xdoctest: +REQUIRES(module:torch)` / `pytest.importorskip('torch')`.
