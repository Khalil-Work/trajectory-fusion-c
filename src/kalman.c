/*
 * kalman.c -- fixed-dimension Kalman filter for a 2-D constant-velocity model.
 * See docs/model.md for derivations and docs/api.md for the API contract.
 */
#include "kalman.h"

#include <math.h>
#include <stddef.h>

/* ------------------------------------------------------------------ */
/* helpers                                                             */
/* ------------------------------------------------------------------ */

static int vec_finite(const double *v, size_t n)
{
    size_t i;
    for (i = 0; i < n; ++i)
        if (!isfinite(v[i]))
            return 0;
    return 1;
}

static int mat4_finite(const double A[KF_N][KF_N])
{
    int i, j;
    for (i = 0; i < KF_N; ++i)
        for (j = 0; j < KF_N; ++j)
            if (!isfinite(A[i][j]))
                return 0;
    return 1;
}

/* Symmetric to round-off tolerance (relative to the diagonal scale). */
static int mat4_symmetric_ok(const double A[KF_N][KF_N])
{
    int i, j;
    for (i = 0; i < KF_N; ++i)
        for (j = i + 1; j < KF_N; ++j) {
            double scale = 1.0 + fabs(A[i][i]) + fabs(A[j][j]);
            if (fabs(A[i][j] - A[j][i]) > 1e-6 * scale)
                return 0;
        }
    return 1;
}

const char *kf_result_str(kf_result r)
{
    switch (r) {
    case KF_OK:           return "ok";
    case KF_ERR_NULL:     return "NULL pointer argument";
    case KF_ERR_DT:       return "invalid sampling time dt";
    case KF_ERR_SIGMA:    return "invalid acceleration noise std sigma_a";
    case KF_ERR_R:        return "invalid measurement variance (need > 0)";
    case KF_ERR_Z:        return "non-finite measurement";
    case KF_ERR_STATE:    return "invalid state estimate or covariance";
    case KF_ERR_SINGULAR: return "numerically singular matrix";
    case KF_ERR_MAXITER:  return "steady-state iteration did not converge";
    }
    return "unknown error code";
}

/* ------------------------------------------------------------------ */
/* fixed-size dense linear algebra                                     */
/* ------------------------------------------------------------------ */

void kf_mat4_identity(double A[KF_N][KF_N])
{
    int i, j;
    for (i = 0; i < KF_N; ++i)
        for (j = 0; j < KF_N; ++j)
            A[i][j] = (i == j) ? 1.0 : 0.0;
}

void kf_mat4_copy(double D[KF_N][KF_N], const double S[KF_N][KF_N])
{
    int i, j;
    for (i = 0; i < KF_N; ++i)
        for (j = 0; j < KF_N; ++j)
            D[i][j] = S[i][j];
}

void kf_mat4_mul(double D[KF_N][KF_N], const double A[KF_N][KF_N],
                 const double B[KF_N][KF_N])
{
    double T[KF_N][KF_N];
    int i, j, k;
    for (i = 0; i < KF_N; ++i)
        for (j = 0; j < KF_N; ++j) {
            double s = 0.0;
            for (k = 0; k < KF_N; ++k)
                s += A[i][k] * B[k][j];
            T[i][j] = s;
        }
    kf_mat4_copy(D, T);
}

void kf_mat4_transpose(double D[KF_N][KF_N], const double S[KF_N][KF_N])
{
    int i, j;
    if (D == S) { /* in-place: swap across the diagonal */
        for (i = 0; i < KF_N; ++i)
            for (j = i + 1; j < KF_N; ++j) {
                double t = S[i][j];
                D[i][j] = S[j][i];
                D[j][i] = t;
            }
    } else {
        for (i = 0; i < KF_N; ++i)
            for (j = 0; j < KF_N; ++j)
                D[i][j] = S[j][i];
    }
}

void kf_mat4_add(double D[KF_N][KF_N], const double A[KF_N][KF_N],
                 const double B[KF_N][KF_N])
{
    int i, j;
    for (i = 0; i < KF_N; ++i)
        for (j = 0; j < KF_N; ++j)
            D[i][j] = A[i][j] + B[i][j];
}

void kf_mat4_sub(double D[KF_N][KF_N], const double A[KF_N][KF_N],
                 const double B[KF_N][KF_N])
{
    int i, j;
    for (i = 0; i < KF_N; ++i)
        for (j = 0; j < KF_N; ++j)
            D[i][j] = A[i][j] - B[i][j];
}

void kf_mat4_scale(double D[KF_N][KF_N], const double S[KF_N][KF_N], double c)
{
    int i, j;
    for (i = 0; i < KF_N; ++i)
        for (j = 0; j < KF_N; ++j)
            D[i][j] = S[i][j] * c;
}

void kf_mat4vec(double y[KF_N], const double A[KF_N][KF_N], const double x[KF_N])
{
    int i, k;
    for (i = 0; i < KF_N; ++i) {
        double s = 0.0;
        for (k = 0; k < KF_N; ++k)
            s += A[i][k] * x[k];
        y[i] = s;
    }
}

void kf_mat4_symmetrize(double A[KF_N][KF_N])
{
    int i, j;
    for (i = 0; i < KF_N; ++i)
        for (j = i + 1; j < KF_N; ++j) {
            double m = 0.5 * (A[i][j] + A[j][i]);
            A[i][j] = m;
            A[j][i] = m;
        }
}

int kf_mat2_inverse(double inv[KF_M][KF_M], const double A[KF_M][KF_M])
{
    double det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    double scale = fabs(A[0][0]);
    double s;
    s = fabs(A[1][1]); if (s > scale) scale = s;
    s = fabs(A[0][1]); if (s > scale) scale = s;
    s = fabs(A[1][0]); if (s > scale) scale = s;
    if (!(det > 1e-12 * scale * scale))
        return -1;
    inv[0][0] =  A[1][1] / det;
    inv[0][1] = -A[0][1] / det;
    inv[1][0] = -A[1][0] / det;
    inv[1][1] =  A[0][0] / det;
    return 0;
}

int kf_mat4_chol_solve(double x[KF_N], const double A[KF_N][KF_N],
                       const double b[KF_N])
{
    double L[KF_N][KF_N] = {{0}};
    double y[KF_N];
    int i, j, k;

    for (j = 0; j < KF_N; ++j) {
        double d = A[j][j];
        for (k = 0; k < j; ++k)
            d -= L[j][k] * L[j][k];
        if (!(d > 0.0) || !isfinite(d))
            return -1; /* not numerically positive definite */
        L[j][j] = sqrt(d);
        for (i = j + 1; i < KF_N; ++i) {
            double s = A[i][j];
            for (k = 0; k < j; ++k)
                s -= L[i][k] * L[j][k];
            L[i][j] = s / L[j][j];
        }
    }
    /* forward solve L y = b */
    for (i = 0; i < KF_N; ++i) {
        double s = b[i];
        for (k = 0; k < i; ++k)
            s -= L[i][k] * y[k];
        y[i] = s / L[i][i];
    }
    /* back solve L^T x = y */
    for (i = KF_N - 1; i >= 0; --i) {
        double s = y[i];
        for (k = i + 1; k < KF_N; ++k)
            s -= L[k][i] * x[k];
        x[i] = s / L[i][i];
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* model matrices                                                      */
/* ------------------------------------------------------------------ */

void kf_transition_matrix(double F[KF_N][KF_N], double dt)
{
    kf_mat4_identity(F);
    F[0][1] = dt;
    F[2][3] = dt;
}

void kf_process_noise(double Q[KF_N][KF_N], double dt, double sigma_a)
{
    double sa2 = sigma_a * sigma_a;
    double q11 = sa2 * dt * dt * dt * dt / 4.0;
    double q12 = sa2 * dt * dt * dt / 2.0;
    double q22 = sa2 * dt * dt;
    int i, j;
    for (i = 0; i < KF_N; ++i)
        for (j = 0; j < KF_N; ++j)
            Q[i][j] = 0.0;
    Q[0][0] = q11; Q[0][1] = q12;
    Q[1][0] = q12; Q[1][1] = q22;
    Q[2][2] = q11; Q[2][3] = q12;
    Q[3][2] = q12; Q[3][3] = q22;
}

/* ------------------------------------------------------------------ */
/* filter                                                              */
/* ------------------------------------------------------------------ */

kf_result kf_state_init(kf_state *s, const double x0[KF_N],
                        const double P0[KF_N][KF_N])
{
    int i;
    if (s == NULL || x0 == NULL || P0 == NULL)
        return KF_ERR_NULL;
    if (!vec_finite(x0, KF_N))
        return KF_ERR_STATE;
    if (!mat4_finite(P0) || !mat4_symmetric_ok(P0))
        return KF_ERR_STATE;
    for (i = 0; i < KF_N; ++i)
        s->x[i] = x0[i];
    kf_mat4_copy(s->P, P0);
    kf_mat4_symmetrize(s->P);
    return KF_OK;
}

kf_result kf_predict(kf_state *s, double dt, double sigma_a)
{
    double F[KF_N][KF_N], Q[KF_N][KF_N];
    double FP[KF_N][KF_N], FPFt[KF_N][KF_N], Fx[KF_N];

    if (s == NULL)
        return KF_ERR_NULL;
    if (!vec_finite(s->x, KF_N) || !mat4_finite(s->P) ||
        !mat4_symmetric_ok(s->P))
        return KF_ERR_STATE;
    if (!isfinite(dt) || dt <= 0.0)
        return KF_ERR_DT;
    if (!isfinite(sigma_a) || sigma_a < 0.0)
        return KF_ERR_SIGMA;

    kf_transition_matrix(F, dt);
    kf_process_noise(Q, dt, sigma_a);

    kf_mat4vec(Fx, F, s->x);
    kf_mat4_mul(FP, F, s->P);
    kf_mat4_transpose(F, F); /* F^T in place (F is used only through here) */
    kf_mat4_mul(FPFt, FP, F);

    {
        int i;
        for (i = 0; i < KF_N; ++i)
            s->x[i] = Fx[i];
    }
    kf_mat4_add(s->P, FPFt, Q);
    kf_mat4_symmetrize(s->P);
    return KF_OK;
}

/*
 * H = [[1, 0, 0, 0],
 *      [0, 0, 1, 0]]  (position measurements in x and y)
 *
 * Gain computation shared by kf_update and kf_steady_state:
 *   PHt = P H^T (4x2), S = H PHt + R (2x2), K = PHt S^{-1} (4x2)
 */
static kf_result compute_gain(const double P[KF_N][KF_N], const double r[KF_M],
                              double K[KF_M][KF_N], double S[KF_M][KF_M],
                              double Sinv[KF_M][KF_M])
{
    double PHt[KF_N][KF_M];
    int i, j, l;

    for (i = 0; i < KF_N; ++i) { /* PHt columns are columns 0 and 2 of P */
        PHt[i][0] = P[i][0];
        PHt[i][1] = P[i][2];
    }
    S[0][0] = P[0][0] + r[0];
    S[0][1] = P[0][2];
    S[1][0] = P[2][0];
    S[1][1] = P[2][2] + r[1];

    if (kf_mat2_inverse(Sinv, S) != 0)
        return KF_ERR_SINGULAR;

    for (i = 0; i < KF_N; ++i)
        for (j = 0; j < KF_M; ++j) {
            double s = 0.0;
            for (l = 0; l < KF_M; ++l)
                s += PHt[i][l] * Sinv[l][j];
            K[j][i] = s;
        }
    return KF_OK;
}

kf_result kf_update(kf_state *s, const double z[KF_M], const double r[KF_M],
                    double innov_out[KF_M], double S_out[KF_M][KF_M],
                    double *nis_out)
{
    double K[KF_M][KF_N], S[KF_M][KF_M], Sinv[KF_M][KF_M];
    double y[KF_M];
    double A[KF_N][KF_N], AP[KF_N][KF_N], APAt[KF_N][KF_N], KRKt[KF_N][KF_N];
    kf_result rc;
    int i, j;

    if (s == NULL || z == NULL || r == NULL)
        return KF_ERR_NULL;
    if (!vec_finite(z, KF_M))
        return KF_ERR_Z;
    if (!isfinite(r[0]) || !isfinite(r[1]) || r[0] <= 0.0 || r[1] <= 0.0)
        return KF_ERR_R;
    if (!vec_finite(s->x, KF_N) || !mat4_finite(s->P) ||
        !mat4_symmetric_ok(s->P))
        return KF_ERR_STATE;

    rc = compute_gain(s->P, r, K, S, Sinv);
    if (rc != KF_OK)
        return rc;

    y[0] = z[0] - s->x[0];
    y[1] = z[1] - s->x[2];

    /* Joseph-form covariance: P = (I-KH) P (I-KH)^T + K R K^T */
    kf_mat4_identity(A);
    for (i = 0; i < KF_N; ++i) {
        A[i][0] -= K[0][i];
        A[i][2] -= K[1][i];
    }
    kf_mat4_mul(AP, A, s->P);
    {
        double At[KF_N][KF_N];
        kf_mat4_transpose(At, A);
        kf_mat4_mul(APAt, AP, At);
    }
    for (i = 0; i < KF_N; ++i)
        for (j = 0; j < KF_N; ++j)
            KRKt[i][j] = r[0] * K[0][i] * K[0][j] + r[1] * K[1][i] * K[1][j];
    kf_mat4_add(s->P, APAt, KRKt);
    kf_mat4_symmetrize(s->P);

    /* state correction */
    for (i = 0; i < KF_N; ++i)
        s->x[i] += K[0][i] * y[0] + K[1][i] * y[1];

    if (innov_out != NULL) {
        innov_out[0] = y[0];
        innov_out[1] = y[1];
    }
    if (S_out != NULL) {
        S_out[0][0] = S[0][0];
        S_out[0][1] = S[0][1];
        S_out[1][0] = S[1][0];
        S_out[1][1] = S[1][1];
    }
    if (nis_out != NULL)
        *nis_out = y[0] * (Sinv[0][0] * y[0] + Sinv[0][1] * y[1]) +
                   y[1] * (Sinv[1][0] * y[0] + Sinv[1][1] * y[1]);
    return KF_OK;
}

kf_result kf_nees(const kf_state *s, const double x_true[KF_N],
                  double *nees_out)
{
    double e[KF_N], w[KF_N];
    int i, all_zero;

    if (s == NULL || x_true == NULL || nees_out == NULL)
        return KF_ERR_NULL;
    if (!vec_finite(s->x, KF_N) || !mat4_finite(s->P))
        return KF_ERR_STATE;
    if (!vec_finite(x_true, KF_N))
        return KF_ERR_STATE;

    all_zero = 1;
    for (i = 0; i < KF_N; ++i) {
        e[i] = s->x[i] - x_true[i];
        if (e[i] != 0.0)
            all_zero = 0;
    }
    if (all_zero) { /* degenerate P = 0 with exact estimate */
        *nees_out = 0.0;
        return KF_OK;
    }
    if (kf_mat4_chol_solve(w, s->P, e) != 0)
        return KF_ERR_SINGULAR;
    {
        double acc = 0.0;
        for (i = 0; i < KF_N; ++i)
            acc += e[i] * w[i];
        *nees_out = acc;
    }
    return KF_OK;
}

/* ------------------------------------------------------------------ */
/* steady state                                                        */
/* ------------------------------------------------------------------ */

kf_result kf_steady_state(double dt, double sigma_a, const double r[KF_M],
                          double tol, int max_iter, kf_steady_result *out)
{
    kf_state st;
    double prev[KF_N][KF_N], Pp[KF_N][KF_N];
    double K[KF_M][KF_N], S[KF_M][KF_M], Sinv[KF_M][KF_M];
    double v0;
    int iter, i, j, done = 0;
    kf_result rc;

    if (out == NULL || r == NULL)
        return KF_ERR_NULL;
    if (!isfinite(dt) || dt <= 0.0)
        return KF_ERR_DT;
    if (!isfinite(sigma_a) || sigma_a < 0.0)
        return KF_ERR_SIGMA;
    if (!isfinite(r[0]) || !isfinite(r[1]) || r[0] <= 0.0 || r[1] <= 0.0)
        return KF_ERR_R;

    /* Seed: diffuse but finite prior. The Riccati fixed point of this
     * stabilizable/detectable model attracts any positive seed. */
    v0 = (sigma_a * dt * 10.0 + 1.0) * (sigma_a * dt * 10.0 + 1.0);
    if (v0 < r[0]) v0 = r[0];
    if (v0 < r[1]) v0 = r[1];
    for (i = 0; i < KF_N; ++i)
        for (j = 0; j < KF_N; ++j)
            st.P[i][j] = 0.0;
    st.P[0][0] = r[0];
    st.P[1][1] = v0;
    st.P[2][2] = r[1];
    st.P[3][3] = v0;
    for (i = 0; i < KF_N; ++i)
        st.x[i] = 0.0;
    kf_mat4_copy(prev, st.P);

    /* Degenerate case: with Q = 0 the unique non-negative DARE solution is
     * the exact fixed point P = 0 (gain 0). Iteration converges to it only
     * algebraically (eigenvalues of the closed loop tend to 1), so return
     * the exact solution directly instead of iterating. */
    if (sigma_a == 0.0) {
        for (i = 0; i < KF_M; ++i)
            for (j = 0; j < KF_N; ++j)
                out->K[i][j] = 0.0;
        for (i = 0; i < KF_N; ++i)
            for (j = 0; j < KF_N; ++j)
                out->P[i][j] = 0.0;
        out->iters = 0;
        out->residual = 0.0;
        return KF_OK;
    }

    for (iter = 1; iter <= max_iter; ++iter) {
        double z[KF_M] = {0.0, 0.0};
        double delta = 0.0;

        rc = kf_predict(&st, dt, sigma_a);
        if (rc != KF_OK)
            return rc;
        kf_mat4_copy(Pp, st.P);
        rc = compute_gain(st.P, r, K, S, Sinv);
        if (rc != KF_OK)
            return rc;
        rc = kf_update(&st, z, r, NULL, NULL, NULL);
        if (rc != KF_OK)
            return rc;

        for (i = 0; i < KF_N; ++i)
            for (j = 0; j < KF_N; ++j) {
                double d = fabs(st.P[i][j] - prev[i][j]) /
                           (1.0 + fabs(prev[i][j]));
                if (d > delta)
                    delta = d;
            }
        kf_mat4_copy(prev, st.P);
        out->iters = iter;
        out->residual = delta;
        if (delta < tol) {
            done = 1;
            break;
        }
    }

    for (i = 0; i < KF_M; ++i)
        for (j = 0; j < KF_N; ++j)
            out->K[i][j] = K[i][j];
    kf_mat4_copy(out->P, Pp);
    return done ? KF_OK : KF_ERR_MAXITER;
}

kf_result kf_steady_gain_closed_form(double dt, double sigma_a, double r,
                                     double k_out[2])
{
    double lam, u, alpha, beta;

    if (k_out == NULL)
        return KF_ERR_NULL;
    if (!isfinite(dt) || dt <= 0.0)
        return KF_ERR_DT;
    if (!isfinite(sigma_a) || sigma_a < 0.0)
        return KF_ERR_SIGMA;
    if (!isfinite(r) || r <= 0.0)
        return KF_ERR_R;

    lam = sigma_a * dt * dt / sqrt(r);
    u = (4.0 + lam - sqrt(lam * lam + 8.0 * lam)) / 4.0;
    alpha = 1.0 - u * u;
    beta = 2.0 * (1.0 - u) * (1.0 - u);
    k_out[0] = alpha;
    k_out[1] = beta / dt;
    return KF_OK;
}
