# Claude's developer journal

## 2026-07-09 18:49:32 +0000

**What the user requested**: a full repo audit identifying bugs and
improvements, written up as planning docs in `docs/planning` that an
opus-level agent could execute in order.

**What I did**: ran five parallel deep audits (core image modules, geometry
structs, transforms/warp/NMS/CLI, tests/CI, packaging/docs/hygiene) against
commit `c888880` on `dev/0.12.0`, each verifying suspected bugs by executing
minimal repros where the environment allowed (py3.12, numpy 2.4.4, cv2 5.0.0
headless; no torch/gdal/itk/sympy/imgaug/pycocotools). Synthesized ~90
findings into `docs/planning/README.md` + docs 01–08.

**State of mind / reflections**: the codebase is in better shape than the
finding count suggests — the architecture is sound and the doctest culture is
real — but two things worried me. First, the branch is currently
*unreleasable*: `remove='0.12.0'` deprecation shims now hard-raise
("Forgot to remove deprecated"), and cv2 5.0 breaks `putText`-based drawing
and `Mask.get_xywh` on any fresh install. That's why doc 01 exists and is
ordered first. Second, there are silent-corruption bugs that have likely been
shipping for a long time: `Matrix.__imatmul__` returning None,
`Affine.concise()` dropping x-scale, the pure-python RLE decoder truncating
the last run, `Polygon.fill` no-op on ≥4-channel images. None error; all
return wrong data.

**Systemic diagnosis** (this shaped doc 08): (1) (x,y)/(w,h)-vs-(h,w)
transposition is the dominant bug class and survives because nearly all
doctests use square inputs; (2) the structs' hand-rolled shallow-copy idiom
produced at least four distinct mutation/meta-leak bugs; (3) integer-dtype
geometry has no promotion policy and fails four different ways; (4) many
parameters are accepted and silently ignored (`bias`, `space`,
`border_value`, unknown kwargs in `Affine.random`).

**Uncertainties/risks**: line numbers in the plan will drift as docs execute
(each doc says to re-grep). Torch-dependent findings (warp_tensor homog-row
bugs, Detections.compress device check) were verified by reasoning + in-code
FIXMEs, not execution — re-verify with torch installed before fixing. The
`gaussian_patch` sigma-axis fix (doc 02 §2.15) is a behavior change for
existing anisotropic callers; flagged for `### Changed`. I did not check
whether `kwimage_ext` is now on PyPI (affects doc 07 §6 and the NMS backend
pruning in doc 08 §5). setup.py/CI are xcookie-generated, so some fixes
belong upstream in `[tool.xcookie]` — noted in docs 06/07 but an executor
could still hand-edit generated files by mistake.

**Where I might challenge the request**: nothing major; I'd only note the
plan intentionally defers refactors (doc 08) until after point fixes + test
repair, because several refactors (copy-shell helper, dtype policy) rewrite
the exact code the bug fixes touch and need the new tests as a safety net.

**Testing notes**: baseline at audit time — tests/: 19 pass / 8 skip /
2 fail (cv2 5.0); doctests: 365 pass / 166 skip / 3 fail (same root cause).
The two failures + three doctest failures are fixed by doc 01 §2.

**Next steps**: execute `docs/planning/01_release_blockers.md`. Open
questions for Jon: keep or delete the `imscale` error stub; add Windows/mac
CI or declare Linux-only; fate of `kwimage_ext` pin; whether
`docs/temp/projective_maths` PDF is worth keeping.
