# Verification: what each check establishes, and what it does not

All checks are deterministic (fixed seeds). Two layers exist:

* `tests/test_kalman.c` — the `make test` gate (also the CI gate),
  ~1.13M scalar assertions (most are generated inside the statistical
  loops of a dozen test functions — a count of checks, not of
  independent unit tests).
* `examples/run_verification.c` — regenerates the raw evidence in
  `docs/results/` (CSVs + `verification_metrics.json`), exits non-zero if
  any section violates its bounds.

The Python script `examples/run_verification.py` only plots/summarizes
those CSVs; no estimation happens in Python.

## A. Innovation consistency (NIS, 2 dof)

**Claim tested:** the filter's innovation covariance `S` matches the
actual error statistics — i.e. the filter is neither optimistic nor
conservative with respect to the measurements.

**Method:** simulate one trajectory with exactly the filter's model
(truth propagated with the same piecewise-white `sigma_a`, measurements
with the same `sigma_z`, initial state known exactly, `P0 = 0`), run
200,000 steps, and compare the sequence
`NIS_k = y_k^T S_k^{-1} y_k` against chi-square with 2 dof.

**Bounds:** mean must lie in `[1.95, 2.05]` (standard error of the mean
is `2/sqrt(n) = 0.0045`, so this is ~11 sigma); the fraction of samples
inside the two-sided 95% chi-square(2) interval
`[0.0506356, 7.3777589]` must lie in `[0.945, 0.955]`.

**Result:** mean 1.995980, fraction 0.95000.

**Establishes:** filter/model consistency for the innovation sequence in
a model-matched setting. It does *not* establish robustness to model
mismatch (wrong `sigma_a`, `sigma_z`, or a maneuvering target) — under
mismatch NIS is the standard detector for it, but that is a use, not a
guarantee proven here.

## B. Steady-state gain vs the analytic Riccati solution

**Claim tested:** the discrete Riccati iteration converges, and to the
right fixed point: the closed-form (Kalata tracking-index) gain of the
1-D CV filter.

**Method:** iterate predict/update from a diffuse diagonal seed until
the max relative change of `P` per iteration is below `1e-14`; compare
all 8 entries of `K_inf` (2x4, stored measurement-major — the transpose
of the conventional state-by-measurement layout, see
[docs/api.md](api.md)) with the rows `[alpha, beta/dt, 0, 0]` and
`[0, 0, alpha, beta/dt]` from the closed form. Additionally the returned prior covariance is pushed
through one full predict/update cycle and required to reproduce itself
(DARE residual), and the decoupling of the axes (`K[0][2] = 0` etc.) is
checked.

**Bounds:** max gain error over all entries `< 1e-10`; observed
`2.9e-15` (72 iterations, residual 5.5e-15). A second parameter set
(`dt = 0.1`, `sigma_a = 0.3`) and the `sigma_a = 0` degenerate case
(exact `K = 0`) are also checked.

**Establishes:** correctness of the covariance recursion to machine
precision for the CV model — a bug in `F`, `Q`, `S`, `K` or the Joseph
update would shift the fixed point or break the DARE identity. It is a
statement about this linear model only.

## C. Covariance correctness (NEES Monte-Carlo, 4 dof)

**Claim tested:** the posterior covariance `P` is the *actual* error
covariance, not merely a plausible matrix — the filter is consistent.

**Method:** 1000 independent Monte-Carlo runs; each run draws the
initial error from `N(0, P0)` (so the prior is exact), propagates truth
with the filter's `Q` and measures with its `R`. At the final step
(step 240) the NEES of every run is computed; under consistency the
average is distributed as `chi^2(4N)/N`.

**Bounds:** the final-step average must lie in
`[3.8265974, 4.1771911]` (exact 95% interval for N = 1000); the
fraction of individual runs inside the per-run chi-square(4) interval
`[0.4844186, 11.1432868]` is reported (expected 0.95) with a wide
`[0.92, 0.98]` acceptance band.

**Result:** average 3.935397 (in bounds), per-run fraction 0.946,
window-averaged (last 100 steps) NEES 3.978458.

**Establishes:** empirical statistical correctness of the covariance for
this model and these noise levels, at the 95% level. It is not a proof
for all parameter regimes (very small/large tracking indices are covered
only by section B and the unit tests).

## D. Zero-noise limit

**Claim tested:** with `Q = 0` and (near) noise-free measurements the
filter recovers the exact trajectory.

**Method:** exact CV truth, exact position measurements, filter run with
`sigma_a = 0` and `r = 1e-16` (std 1e-8 m; `r` must stay positive by the
API contract). After 400 steps the total position and velocity errors
must be below `1e-8 m` and `1e-6 m/s`.

**Result:** position error 0.0 m (exact), velocity error 1.6e-15 m/s.

**Establishes:** absence of bias/leakage in the update path and sensible
behavior of the Joseph form in the near-singular `R` limit.

## Chi-square bounds provenance

dof 2 has the closed form `-2 ln(1 - q)` (chi-square(2) is exponential):
`[0.0506356159685797, 7.3777589082278722]`. dof 4 and dof 4000 values
are obtained by exact numerical inversion of the regularized incomplete
gamma function (series + Lentz continued fraction, then bisection),
cross-checked at runtime by `examples/run_verification.py`
(`chi2_bounds_crosscheck` in `docs/results/run_summary.json`) and against
standard published tables.

## Limits of the verification (what is NOT claimed)

* **Linear constant-velocity model only.** No EKF, no maneuvering models,
  no nonlinearity anywhere; `F`, `H` are linear and time-invariant per
  fixed `dt`.
* **Tuned noise.** The statistical checks (A, C) require the simulation
  to use exactly the filter's `sigma_a` and `sigma_z`; performance under
  mistuning is not tested.
* **Single 2-D position sensor, diagonal `R`, independent axes.**
  Correlated measurements, other `H` matrices, or data fusion from
  multiple sensors are not covered.
* **Fixed sampling time** `dt > 0`; variable-rate updates are not tested.
* The **demo** (`examples/traj_demo.c`) initializes from the first
  measurement with a heuristic velocity prior, so its early-step NIS/NEES
  are inflated (demo mean NIS 2.26 vs 2 for the matched run); the gated
  consistency results above use exact model-matched initialization.
* Numerical bounds (symmetry tolerance `1e-6`, relative determinant
  threshold `1e-12`, Cholesky pivot tests) are engineering margins, not
  proofs of universal numerical robustness.
