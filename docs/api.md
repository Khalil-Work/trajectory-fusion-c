# C API reference

Everything lives in `src/kalman.h` / `src/kalman.c` (filter plus a
fixed-size dense linear-algebra core) and `src/rng.h` / `src/rng.c`
(deterministic sampling). The library is C99, has no external
dependencies beyond the C standard library (`-lm`), and performs **no
dynamic allocation**.

## Conventions

* Fixed dimensions: state `KF_N = 4`, measurement `KF_M = 2`.
* Matrices are plain `double A[4][4]` arrays (row-major).
* **Error reporting**: the functions that validate inputs and can fail —
  `kf_state_init`, `kf_predict`, `kf_update`, `kf_nees`,
  `kf_steady_state`, `kf_steady_gain_closed_form` — return a `kf_result`
  code and write their outputs only on success. The fixed-size
  linear-algebra and model-matrix helpers (`kf_mat4_*`, `kf_mat2_inverse`
  aside) return `void` and have no validation role; `kf_mat2_inverse`
  and `kf_mat4_chol_solve` return a plain `int` status (0 ok, -1
  singular / not positive definite).
* **Ownership**: the caller owns every buffer. Nothing is allocated,
  retained or freed by the library, and no returned pointer outlives the
  call (`kf_result_str` returns a static string literal).
* **Aliasing**: `kf_mat4_mul/add/sub/scale(D, A, B)` requires `D` distinct
  from `A`/`B`; `kf_mat4_transpose(D, S)` supports `D == S`; all filter
  functions operate in place on their `kf_state`.
* Symmetry: covariance inputs are accepted if symmetric to within
  `1e-6` relative to the diagonal scale; round-off-level asymmetry is
  removed by explicit symmetrization (`(A + A^T)/2`).
* Singularity: a 2x2 matrix is rejected when `|det| <= 1e-12 * scale^2`
  (`scale` = largest magnitude entry); a 4x4 solve uses Cholesky and
  fails on any non-positive pivot.

## Error codes (`kf_result`)

| Code                | Meaning                                     |
|---------------------|---------------------------------------------|
| `KF_OK`             | success                                     |
| `KF_ERR_NULL`       | NULL pointer argument                       |
| `KF_ERR_DT`         | `dt` non-finite or `dt <= 0`                |
| `KF_ERR_SIGMA`      | `sigma_a` non-finite or negative            |
| `KF_ERR_R`          | measurement variance non-finite or `<= 0`   |
| `KF_ERR_Z`          | measurement non-finite                      |
| `KF_ERR_STATE`      | state/covariance non-finite or asymmetric   |
| `KF_ERR_SINGULAR`   | innovation covariance (or `P`) singular     |
| `KF_ERR_MAXITER`    | Riccati iteration did not reach tolerance   |

`kf_result_str(kf_result)` maps codes to human-readable text.

## Linear algebra (fixed size)

```c
void kf_mat4_identity(double A[4][4]);
void kf_mat4_copy(double D[4][4], const double S[4][4]);
void kf_mat4_mul(double D[4][4], const double A[4][4], const double B[4][4]);
void kf_mat4_transpose(double D[4][4], const double S[4][4]);   /* D may alias S */
void kf_mat4_add(double D[4][4], const double A[4][4], const double B[4][4]);
void kf_mat4_sub(double D[4][4], const double A[4][4], const double B[4][4]);
void kf_mat4_scale(double D[4][4], const double S[4][4], double c);
void kf_mat4vec(double y[4], const double A[4][4], const double x[4]);
void kf_mat4_symmetrize(double A[4][4]);
int  kf_mat2_inverse(double inv[2][2], const double A[2][2]);   /* 0 ok, -1 singular */
int  kf_mat4_chol_solve(double x[4], const double A[4][4], const double b[4]);
```

`kf_mat4_chol_solve` solves `A x = b` for symmetric positive-definite
`A` via Cholesky factorization (returns `-1` if `A` is not numerically
positive definite).

## Model matrices

```c
void kf_transition_matrix(double F[4][4], double dt);
void kf_process_noise(double Q[4][4], double dt, double sigma_a);
```

`F` is the constant-velocity transition matrix; `Q` is the
piecewise-white acceleration covariance derived in
[docs/model.md](model.md).

## Filter state and steps

```c
typedef struct {
    double x[4];        /* estimate [px, vx, py, vy] */
    double P[4][4];     /* error covariance */
} kf_state;

kf_result kf_state_init(kf_state *s, const double x0[4], const double P0[4][4]);
kf_result kf_predict(kf_state *s, double dt, double sigma_a);
kf_result kf_update(kf_state *s, const double z[2], const double r[2],
                    double innov_out[2], double S_out[2][2], double *nis_out);
kf_result kf_nees(const kf_state *s, const double x_true[4], double *nees_out);
```

* `kf_state_init` copies the prior mean and covariance (symmetrized).
* `kf_predict` propagates `x <- F x`, `P <- F P F^T + Q`.
* `kf_update` performs the Joseph-form correction (see
  [docs/model.md](model.md)). `innov_out`, `S_out`, `nis_out` are
  optional diagnostics and may each be `NULL`; `r` holds the per-axis
  measurement variances (diagonal `R`).
* `kf_nees` computes the normalized estimation error squared
  `NEES = (x - x_true)^T P^{-1} (x - x_true)` via a Cholesky solve
  (`KF_ERR_SINGULAR` if `P` is not positive definite; the degenerate
  exact-estimate/zero-covariance case returns 0).

## Steady-state analysis

```c
typedef struct {
    double P[4][4];   /* steady-state prior covariance */
    double K[2][4];   /* steady-state gain, measurement-major:
                         the TRANSPOSE of the conventional 4x2
                         state-by-measurement layout; x^+ = x + K^T y */
    int    iters;     /* iterations used */
    double residual;  /* final max relative change of P */
} kf_steady_result;

kf_result kf_steady_state(double dt, double sigma_a, const double r[2],
                          double tol, int max_iter, kf_steady_result *out);
kf_result kf_steady_gain_closed_form(double dt, double sigma_a, double r,
                                     double k_out[2]);
```

`kf_steady_state` iterates predict/update (the discrete Riccati
recursion) from a diffuse diagonal seed until the maximum relative change
of `P` per iteration drops below `tol`, or `max_iter` is exhausted
(`KF_ERR_MAXITER`). The returned gain is stored measurement-major
(`K[2][4]`, `x^+ = x + K^T y` — the transpose of the conventional
state-by-measurement layout). With `sigma_a == 0` it returns the exact
degenerate solution `P = 0, K = 0` directly. `kf_steady_gain_closed_form`
evaluates the Kalata tracking-index solution for one axis (see
[docs/model.md](model.md)); the two axes of the 2-D problem are
identical, and `k_out` is `[alpha, beta/dt]` for the 1-D per-axis state
`[p, v]`.

## Random number generator (`src/rng.h`)

```c
typedef struct { uint64_t state; double spare; int has_spare; } rng64;

void     rng64_seed(rng64 *g, uint64_t seed);
uint64_t rng64_next(rng64 *g);
double   rng64_uniform(rng64 *g);   /* U[0,1), 53 high bits of the LCG */
double   rng64_normal(rng64 *g);    /* N(0,1), Box-Muller with cached spare */
```

64-bit LCG (`s = 6364136223846793005 s + 1442695040888963407 mod 2^64`);
the same seed reproduces the same stream everywhere. Used by the demo
and verification drivers, not by the filter itself.
