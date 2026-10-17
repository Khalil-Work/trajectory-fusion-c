/*
 * kalman.h -- fixed-dimension Kalman filter for a 2-D constant-velocity model.
 *
 * State:      x = [px, vx, py, vy]^T   (KF_N = 4)
 * Measurement: z = [px, py]^T          (KF_M = 2), R diagonal.
 *
 * All matrices are fixed-size stack arrays; the library performs no dynamic
 * allocation. Every function returning kf_result validates its inputs
 * (non-finite values, non-positive measurement variance, ...).
 *
 * See docs/model.md for the state model, Q derivation (piecewise-white
 * acceleration noise) and the Joseph-form measurement update used here.
 */
#ifndef KALMAN_H
#define KALMAN_H

#ifdef __cplusplus
extern "C" {
#endif

#define KF_N 4  /* state dimension: [px, vx, py, vy] */
#define KF_M 2  /* measurement dimension: [px, py]   */

/* Error codes returned by the filter entry points. */
typedef enum {
    KF_OK = 0,        /* success                                        */
    KF_ERR_NULL,      /* NULL pointer argument                          */
    KF_ERR_DT,        /* dt non-finite or dt <= 0                       */
    KF_ERR_SIGMA,     /* sigma_a non-finite or sigma_a < 0              */
    KF_ERR_R,         /* measurement variance non-finite or <= 0        */
    KF_ERR_Z,         /* measurement non-finite                         */
    KF_ERR_STATE,     /* state/covariance non-finite or not symmetric   */
    KF_ERR_SINGULAR,  /* innovation covariance (or P) numerically singular */
    KF_ERR_MAXITER    /* steady-state Riccati iteration did not converge */
} kf_result;

/* Human-readable error string (static storage, never NULL). */
const char *kf_result_str(kf_result r);

/* Filter state: mean estimate and error covariance P (symmetric, PSD). */
typedef struct {
    double x[KF_N];      /* state estimate                                */
    double P[KF_N][KF_N];/* error covariance, P = E[(x-x_true)(.)^T]      */
} kf_state;

/* ------------------------------------------------------------------ */
/* Small dense linear algebra, fixed sizes (no external dependencies). */
/* D may not alias A or B unless noted.                                */
/* ------------------------------------------------------------------ */

void kf_mat4_identity(double A[KF_N][KF_N]);
void kf_mat4_copy(double D[KF_N][KF_N], const double S[KF_N][KF_N]);
void kf_mat4_mul(double D[KF_N][KF_N], const double A[KF_N][KF_N],
                 const double B[KF_N][KF_N]);
void kf_mat4_transpose(double D[KF_N][KF_N], const double S[KF_N][KF_N]); /* D may alias S */
void kf_mat4_add(double D[KF_N][KF_N], const double A[KF_N][KF_N],
                 const double B[KF_N][KF_N]);
void kf_mat4_sub(double D[KF_N][KF_N], const double A[KF_N][KF_N],
                 const double B[KF_N][KF_N]);
void kf_mat4_scale(double D[KF_N][KF_N], const double S[KF_N][KF_N], double c);
void kf_mat4vec(double y[KF_N], const double A[KF_N][KF_N], const double x[KF_N]);
/* Enforce exact symmetry: A <- (A + A^T)/2 (round-off control). */
void kf_mat4_symmetrize(double A[KF_N][KF_N]);
/* Invert a 2x2 matrix; returns 0 on success, -1 if determinant is not
 * positive relative to the matrix scale (numerically singular). */
int kf_mat2_inverse(double inv[KF_M][KF_M], const double A[KF_M][KF_M]);
/* Solve A x = b for symmetric positive-definite 4x4 A via Cholesky.
 * Returns 0 on success, -1 if A is not numerically positive definite. */
int kf_mat4_chol_solve(double x[KF_N], const double A[KF_N][KF_N],
                       const double b[KF_N]);

/* ------------------------------------------------------------------ */
/* Model matrices                                                      */
/* ------------------------------------------------------------------ */

/* F(dt): constant-velocity transition matrix.
 * F = [[1, dt, 0, 0],
 *      [0, 1,  0, 0],
 *      [0, 0, 1, dt],
 *      [0, 0, 0, 1]]                                            */
void kf_transition_matrix(double F[KF_N][KF_N], double dt);

/* Q(dt, sigma_a): process noise from the piecewise-white acceleration
 * model (acceleration constant over one sample, white across samples):
 * per axis G = [dt^2/2, dt]^T, Q_axis = sigma_a^2 * G G^T.           */
void kf_process_noise(double Q[KF_N][KF_N], double dt, double sigma_a);

/* ------------------------------------------------------------------ */
/* Filter steps                                                        */
/* ------------------------------------------------------------------ */

/* Initialize a filter state from a prior mean and covariance.
 * P0 is copied after a symmetry check (KF_ERR_STATE on failure). */
kf_result kf_state_init(kf_state *s, const double x0[KF_N],
                        const double P0[KF_N][KF_N]);

/* Prediction: x <- F x, P <- F P F^T + Q. */
kf_result kf_predict(kf_state *s, double dt, double sigma_a);

/* Measurement update (Joseph form).
 * z: measurement [px, py]; r: measurement variances [var_px, var_py]
 * (diagonal R). On success, s->x and s->P are updated with
 *   K   = P H^T S^{-1},           S = H P H^T + R
 *   x   = x + K (z - H x)
 *   P   = (I - K H) P (I - K H)^T + K R K^T
 * innov_out (optional, may be NULL): z - H x_before.
 * S_out (optional, may be NULL): innovation covariance S.
 * nis_out (optional, may be NULL): innov^T S^{-1} innov (2 dof).     */
kf_result kf_update(kf_state *s, const double z[KF_M], const double r[KF_M],
                    double innov_out[KF_M], double S_out[KF_M][KF_M],
                    double *nis_out);

/* Normalized estimation error squared, nees = e^T P^{-1} e with
 * e = x_est - x_true (4 dof when the filter is consistent). */
kf_result kf_nees(const kf_state *s, const double x_true[KF_N],
                  double *nees_out);

/* ------------------------------------------------------------------ */
/* Steady-state analysis                                               */
/* ------------------------------------------------------------------ */

typedef struct {
    double P[KF_N][KF_N];  /* steady-state *prior* (predicted) covariance */
    double K[KF_M][KF_N];  /* steady-state gain                            */
    int    iters;          /* iterations used                              */
    double residual;       /* final max relative change of P per iteration */
} kf_steady_result;

/* Iterate predict/update (the discrete Riccati recursion) until the max
 * relative change of P falls below tol or max_iter is exhausted.
 * Returns KF_ERR_MAXITER if tol is not reached. */
kf_result kf_steady_state(double dt, double sigma_a, const double r[KF_M],
                          double tol, int max_iter, kf_steady_result *out);

/* Closed-form steady-state gain of the 1-D constant-velocity filter
 * (Kalata tracking-index solution, see docs/model.md):
 *   Lambda = sigma_a dt^2 / sqrt(r)
 *   u      = (4 + Lambda - sqrt(Lambda^2 + 8 Lambda)) / 4
 *   k_out  = [alpha, beta/dt] = [1 - u^2, 2 (1 - u)^2 / dt]
 * For the 2-D filter the axes decouple, so this applies per axis.     */
kf_result kf_steady_gain_closed_form(double dt, double sigma_a, double r,
                                     double k_out[2]);

#ifdef __cplusplus
}
#endif

#endif /* KALMAN_H */
