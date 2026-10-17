/*
 * test_kalman.c -- assert-based verification suite for src/kalman.c.
 *
 * Build & run:  make test     (or see README.md)
 * Exit code 0 iff every check passes; each failure prints file:line,
 * the check message and the offending values.
 *
 * Statistical checks (NIS, NEES) use exact chi-square bounds; the values
 * below match published tables (see docs/verification.md for provenance).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../src/kalman.h"
#include "../src/rng.h"

static int g_checks = 0;
static int g_failures = 0;

static void check_failed(const char *file, int line, const char *msg)
{
    g_failures++;
    printf("FAIL %s:%d: %s\n", file, line, msg);
}

#define CHECK(cond, msg) \
    do { \
        g_checks++; \
        if (!(cond)) \
            check_failed(__FILE__, __LINE__, (msg)); \
    } while (0)

#define CHECK_NEAR(actual, expected, tol, label) \
    do { \
        double a_ = (double)(actual), e_ = (double)(expected); \
        g_checks++; \
        if (!(fabs(a_ - e_) <= (tol))) { \
            g_failures++; \
            printf("FAIL %s:%d: %s: actual %.17g, expected %.17g, " \
                   "tol %g, |diff| %.3g\n", \
                   __FILE__, __LINE__, (label), a_, e_, (double)(tol), \
                   fabs(a_ - e_)); \
        } \
    } while (0)

/* Chi-square two-sided 95% bounds used by the consistency checks.
 * Exact values (dof 2 has the closed form -2*ln(1-q); the others from
 * numerical inversion of the regularized incomplete gamma function,
 * cross-checked against standard tables; see docs/verification.md):
 *   dof 2 : [0.0506356159685797, 7.3777589082278722]
 *   dof 4 : [0.4844185570879299, 11.1432867818777882]
 * Mean of N=1000 iid chi-square(4) variables -> chi-square(4000)/1000:
 *   [3.8265974192512836, 4.1771910562861830]
 */
#define CHI2_DOF2_LO   0.0506356159685797
#define CHI2_DOF2_HI   7.3777589082278722
#define CHI2_DOF4_LO   0.4844185570879299
#define CHI2_DOF4_HI   11.1432867818777882
#define NEES_AVG_LO    3.8265974192512836
#define NEES_AVG_HI    4.1771910562861830

/* ------------------------------------------------------------------ */
/* test: fixed-size linear algebra                                     */
/* ------------------------------------------------------------------ */

static void test_linalg(void)
{
    double A[4][4], B[4][4], D[4][4], y[4], x[4];
    int i;

    kf_mat4_identity(A);
    kf_mat4_mul(D, A, A);
    for (i = 0; i < 4; ++i)
        CHECK_NEAR(D[i][i], 1.0, 0.0, "identity * identity diagonal");
    CHECK_NEAR(D[0][1], 0.0, 0.0, "identity * identity off-diagonal");

    /* A = [[2,1,0,0],[0,2,1,0],[0,0,2,1],[1,0,0,2]] (known product) */
    kf_mat4_identity(A);
    A[0][1] = 1.0; A[1][2] = 1.0; A[2][3] = 1.0; A[3][0] = 1.0;
    kf_mat4_scale(A, A, 2.0);
    kf_mat4_mul(D, A, A);
    /* row 0 of A^2 = [2,2,0,0] . columns of A = [4, 8, 4, 0] */
    CHECK_NEAR(D[0][0], 4.0, 1e-15, "A^2 [0][0]");
    CHECK_NEAR(D[0][1], 8.0, 1e-15, "A^2 [0][1]");
    CHECK_NEAR(D[0][2], 4.0, 1e-15, "A^2 [0][2]");
    CHECK_NEAR(D[0][3], 0.0, 1e-15, "A^2 [0][3]");

    kf_mat4_transpose(B, D);
    CHECK_NEAR(B[2][0], D[0][2], 0.0, "transpose element");
    kf_mat4_transpose(D, D); /* in place */
    CHECK_NEAR(D[2][0], B[2][0], 0.0, "in-place transpose element");
    kf_mat4_transpose(D, D);
    CHECK_NEAR(D[0][1], B[1][0], 0.0, "double transpose restores");

    kf_mat4vec(y, A, A[0]); /* A * (first row of A, as data) */
    CHECK_NEAR(y[0], A[0][0] * A[0][0] + A[0][1] * A[0][1], 1e-15,
               "mat4vec row 0");

    /* 2x2 inverse */
    {
        double M[2][2] = {{4.0, 1.0}, {1.0, 3.0}}, Minv[2][2];
        CHECK(kf_mat2_inverse(Minv, M) == 0, "2x2 inverse succeeds");
        CHECK_NEAR(Minv[0][0], 3.0 / 11.0, 1e-15, "2x2 inverse [0][0]");
        CHECK_NEAR(Minv[1][1], 4.0 / 11.0, 1e-15, "2x2 inverse [1][1]");
        CHECK_NEAR(Minv[0][1], -1.0 / 11.0, 1e-15, "2x2 inverse [0][1]");
        M[0][0] = 1e-30; /* near-singular */
        CHECK(kf_mat2_inverse(Minv, M) != 0, "near-singular 2x2 rejected");
    }

    /* Cholesky solve: SPD system with known solution */
    {
        double Pp[4][4] = {{4.0, 1.0, 0.5, 0.0},
                           {1.0, 3.0, 0.0, 0.25},
                           {0.5, 0.0, 2.0, 0.75},
                           {0.0, 0.25, 0.75, 5.0}};
        double b[4] = {1.0, 2.0, 3.0, 4.0};
        CHECK(kf_mat4_chol_solve(x, Pp, b) == 0, "chol solve succeeds");
        /* verify residual Pp*x = b */
        {
            int i2, j;
            for (i2 = 0; i2 < 4; ++i2) {
                double s = 0.0;
                for (j = 0; j < 4; ++j)
                    s += Pp[i2][j] * x[j];
                CHECK_NEAR(s, b[i2], 1e-12, "chol solve residual");
            }
        }
        Pp[0][0] = -1.0;
        CHECK(kf_mat4_chol_solve(x, Pp, b) != 0, "non-PD chol rejected");
    }
}

/* ------------------------------------------------------------------ */
/* test: model matrices                                                */
/* ------------------------------------------------------------------ */

static void test_model_matrices(void)
{
    double F[4][4], Q[4][4];

    kf_transition_matrix(F, 0.5);
    CHECK_NEAR(F[0][1], 0.5, 0.0, "F[0][1] = dt");
    CHECK_NEAR(F[2][3], 0.5, 0.0, "F[2][3] = dt");
    CHECK_NEAR(F[0][0], 1.0, 0.0, "F[0][0] = 1");
    CHECK_NEAR(F[1][2], 0.0, 0.0, "F[1][2] = 0 (axes decoupled)");

    kf_process_noise(Q, 0.5, 2.0); /* sigma_a^2 = 4 */
    CHECK_NEAR(Q[0][0], 4.0 * 0.5 * 0.5 * 0.5 * 0.5 / 4.0, 1e-15,
               "Q[0][0] = sa^2 dt^4/4");
    CHECK_NEAR(Q[0][1], 4.0 * 0.5 * 0.5 * 0.5 / 2.0, 1e-15,
               "Q[0][1] = sa^2 dt^3/2");
    CHECK_NEAR(Q[1][1], 4.0 * 0.5 * 0.5, 1e-15, "Q[1][1] = sa^2 dt^2");
    CHECK_NEAR(Q[2][2], Q[0][0], 0.0, "Q blocks identical per axis");
    CHECK_NEAR(Q[1][2], 0.0, 0.0, "Q cross-axis zero");
}

/* ------------------------------------------------------------------ */
/* test: predict against hand computation                              */
/* ------------------------------------------------------------------ */

static void test_predict(void)
{
    kf_state s;
    double x0[4] = {1.0, 2.0, 3.0, 4.0};
    double P0[4][4];
    int i, j;

    for (i = 0; i < 4; ++i)
        for (j = 0; j < 4; ++j)
            P0[i][j] = 0.0;
    P0[0][0] = P0[1][1] = P0[2][2] = P0[3][3] = 1.0;

    CHECK(kf_state_init(&s, x0, P0) == KF_OK, "state init");
    CHECK(kf_predict(&s, 0.5, 0.0) == KF_OK, "predict");
    /* x' = F x = [1 + 0.5*2, 2, 3 + 0.5*4, 4] */
    CHECK_NEAR(s.x[0], 2.0, 1e-15, "predict x[0]");
    CHECK_NEAR(s.x[1], 2.0, 1e-15, "predict x[1]");
    CHECK_NEAR(s.x[2], 5.0, 1e-15, "predict x[2]");
    CHECK_NEAR(s.x[3], 4.0, 1e-15, "predict x[3]");
    /* P' = F I F^T = per-axis [[1+dt^2, dt],[dt, 1]] = [[1.25,.5],[.5,1]] */
    CHECK_NEAR(s.P[0][0], 1.25, 1e-15, "predict P[0][0]");
    CHECK_NEAR(s.P[0][1], 0.5, 1e-15, "predict P[0][1]");
    CHECK_NEAR(s.P[1][1], 1.0, 1e-15, "predict P[1][1]");

    /* with process noise sigma_a = 2 (sa^2 = 4): add dt^4/4*4 etc. */
    CHECK(kf_state_init(&s, x0, P0) == KF_OK, "re-init");
    CHECK(kf_predict(&s, 0.5, 2.0) == KF_OK, "predict with Q");
    CHECK_NEAR(s.P[0][0], 1.25 + 4.0 * 0.0625 / 4.0, 1e-15,
               "predict P[0][0] with Q");
    CHECK_NEAR(s.P[0][1], 0.5 + 4.0 * 0.125 / 2.0, 1e-15,
               "predict P[0][1] with Q");
    CHECK_NEAR(s.P[1][1], 1.0 + 4.0 * 0.25, 1e-15, "predict P[1][1] with Q");
}

/* ------------------------------------------------------------------ */
/* test: Joseph-form update                                            */
/* ------------------------------------------------------------------ */

static void test_update(void)
{
    kf_state s;
    double P0[4][4], z[2], r[2];
    double innov[2], S[2][2], nis;
    int i, j;

    for (i = 0; i < 4; ++i)
        for (j = 0; j < 4; ++j)
            P0[i][j] = 0.0;
    P0[0][0] = P0[1][1] = P0[2][2] = P0[3][3] = 4.0;

    {
        double x0[4] = {0.0, 0.0, 0.0, 0.0};
        CHECK(kf_state_init(&s, x0, P0) == KF_OK, "init for update test");
    }
    z[0] = 2.0; z[1] = 0.0;
    r[0] = 4.0; r[1] = 4.0;

    CHECK(kf_update(&s, z, r, innov, S, &nis) == KF_OK, "update runs");
    /* S = P00 + r = 8, K_pos = 4/8 = 0.5 per axis */
    CHECK_NEAR(S[0][0], 8.0, 1e-15, "S[0][0]");
    CHECK_NEAR(S[1][1], 8.0, 1e-15, "S[1][1]");
    CHECK_NEAR(innov[0], 2.0, 1e-15, "innovation[0]");
    /* x' = K z = [1, 0, 0, 0] */
    CHECK_NEAR(s.x[0], 1.0, 1e-15, "post-update x[0]");
    CHECK_NEAR(s.x[1], 0.0, 1e-15, "post-update x[1] (no cross term)");
    /* Joseph: P' = (1-k)^2*4 + k^2*4 = 2, identical to the naive (1-k)*4
     * here; the Joseph form is used for its guaranteed symmetry/PSD. */
    CHECK_NEAR(s.P[0][0], 2.0, 1e-12, "post-update P[0][0] (Joseph)");
    CHECK_NEAR(s.P[1][1], 4.0, 1e-12, "post-update P[1][1]");
    CHECK_NEAR(s.P[2][2], 2.0, 1e-12, "post-update P[2][2]");
    CHECK_NEAR(nis, 2.0 * 2.0 / 8.0, 1e-15, "nis = y^2/S");

    /* Joseph must equal the naive form (I-KH)P(I-KH)^T + KRK^T when both
     * are computed exactly; more telling: it must equal (I-KH)Pprior for a
     * consistent K. Check on a correlated prior. */
    {
        kf_state t;
        double x0[4] = {0.0, 0.0, 0.0, 0.0};
        double Pp[4][4] = {{9.0, 1.5, 0.6, 0.1},
                           {1.5, 4.0, 0.2, 0.3},
                           {0.6, 0.2, 7.0, -0.4},
                           {0.1, 0.3, -0.4, 2.5}};
        double rr[2] = {2.0, 3.0};
        double zz[2] = {0.5, -0.5};
        double PHt[4][2], Sinv[2][2], K[2][4];
        double naive[4][4], tmp[4][4], tmp2[4][4];
        int a, b, l;

        CHECK(kf_state_init(&t, x0, Pp) == KF_OK, "init correlated");
        CHECK(kf_update(&t, zz, rr, NULL, NULL, NULL) == KF_OK,
              "update correlated");

        /* independent naive computation: P_naive = (I-KH) Pp (I-KH)^T */
        for (a = 0; a < 4; ++a) {
            PHt[a][0] = Pp[a][0];
            PHt[a][1] = Pp[a][2];
        }
        {
            double Sm[2][2] = {{Pp[0][0] + rr[0], Pp[0][2]},
                               {Pp[2][0], Pp[2][2] + rr[1]}};
            CHECK(kf_mat2_inverse(Sinv, Sm) == 0, "S inverse");
        }
        for (a = 0; a < 4; ++a)
            for (b = 0; b < 2; ++b) {
                double sum = 0.0;
                for (l = 0; l < 2; ++l)
                    sum += PHt[a][l] * Sinv[l][b];
                K[b][a] = sum;
            }
        /* A = I - K H */
        for (a = 0; a < 4; ++a)
            for (b = 0; b < 4; ++b)
                tmp[a][b] = (a == b ? 1.0 : 0.0) - K[0][a] * (b == 0) -
                            K[1][a] * (b == 2);
        for (a = 0; a < 4; ++a)
            for (b = 0; b < 4; ++b)
                tmp2[a][b] = tmp[a][0] * Pp[0][b] + tmp[a][1] * Pp[1][b] +
                             tmp[a][2] * Pp[2][b] + tmp[a][3] * Pp[3][b];
        for (a = 0; a < 4; ++a)
            for (b = 0; b < 4; ++b)
                naive[a][b] = tmp2[a][0] * tmp[b][0] + tmp2[a][1] * tmp[b][1] +
                              tmp2[a][2] * tmp[b][2] + tmp2[a][3] * tmp[b][3] +
                              K[0][a] * rr[0] * K[0][b] +
                              K[1][a] * rr[1] * K[1][b];
        for (a = 0; a < 4; ++a)
            for (b = 0; b < 4; ++b)
                CHECK_NEAR(t.P[a][b], naive[a][b], 1e-12,
                           "Joseph equals naive form");
    }
}

/* ------------------------------------------------------------------ */
/* test: input validation                                              */
/* ------------------------------------------------------------------ */

static void test_validation(void)
{
    kf_state s;
    double P0[4][4], x0[4] = {0.0, 0.0, 0.0, 0.0};
    double z[2] = {1.0, 1.0}, r[2] = {1.0, 1.0};
    int i, j;

    for (i = 0; i < 4; ++i)
        for (j = 0; j < 4; ++j)
            P0[i][j] = 0.0;
    P0[0][0] = P0[1][1] = P0[2][2] = P0[3][3] = 1.0;
    CHECK(kf_state_init(&s, x0, P0) == KF_OK, "init for validation");

    CHECK(kf_predict(NULL, 1.0, 1.0) == KF_ERR_NULL, "predict NULL state");
    CHECK(kf_predict(&s, 0.0, 1.0) == KF_ERR_DT, "predict dt = 0 rejected");
    CHECK(kf_predict(&s, -0.5, 1.0) == KF_ERR_DT, "predict dt < 0 rejected");
    CHECK(kf_predict(&s, 0.0 / 0.0, 1.0) == KF_ERR_DT,
          "predict NaN dt rejected");
    CHECK(kf_predict(&s, 1.0, -0.1) == KF_ERR_SIGMA,
          "predict negative sigma rejected");
    CHECK(kf_predict(&s, 1.0, 0.0 / 0.0) == KF_ERR_SIGMA,
          "predict NaN sigma rejected");

    CHECK(kf_update(&s, NULL, r, NULL, NULL, NULL) == KF_ERR_NULL,
          "update NULL z");
    CHECK(kf_update(&s, z, NULL, NULL, NULL, NULL) == KF_ERR_NULL,
          "update NULL r");
    CHECK(kf_update(NULL, z, r, NULL, NULL, NULL) == KF_ERR_NULL,
          "update NULL state");
    {
        double rbad[2] = {0.0, 1.0};
        CHECK(kf_update(&s, z, rbad, NULL, NULL, NULL) == KF_ERR_R,
              "update zero variance rejected");
        rbad[0] = -1.0;
        CHECK(kf_update(&s, z, rbad, NULL, NULL, NULL) == KF_ERR_R,
              "update negative variance rejected");
        rbad[0] = 0.0 / 0.0;
        CHECK(kf_update(&s, z, rbad, NULL, NULL, NULL) == KF_ERR_R,
              "update NaN variance rejected");
    }
    {
        double zbad[2] = {0.0 / 0.0, 1.0};
        CHECK(kf_update(&s, zbad, r, NULL, NULL, NULL) == KF_ERR_Z,
              "update NaN measurement rejected");
        zbad[0] = 1.0;
        zbad[1] = 1.0 / 0.0;
        CHECK(kf_update(&s, zbad, r, NULL, NULL, NULL) == KF_ERR_Z,
              "update Inf measurement rejected");
    }
    {
        double Pbad[4][4];
        kf_state t;
        for (i = 0; i < 4; ++i)
            for (j = 0; j < 4; ++j)
                Pbad[i][j] = P0[i][j];
        Pbad[2][0] = 5.0; /* grossly asymmetric */
        CHECK(kf_state_init(&t, x0, Pbad) == KF_ERR_STATE,
              "asymmetric P0 rejected");
        Pbad[2][0] = 0.0;
        Pbad[1][1] = 1.0 / 0.0;
        CHECK(kf_state_init(&t, x0, Pbad) == KF_ERR_STATE,
              "non-finite P0 rejected");
    }
    CHECK(kf_result_str(KF_ERR_R) != NULL, "error strings available");
}

/* ------------------------------------------------------------------ */
/* test: long-run covariance health (symmetry + definiteness)          */
/* ------------------------------------------------------------------ */

static void test_covariance_health(void)
{
    kf_state s;
    rng64 g;
    double dt = 0.5, sigma_a = 1.2, r[2] = {9.0, 9.0};
    double x0[4] = {0.0, 8.0, 0.0, 6.0};
    double P0[4][4] = {{9.0, 0, 0, 0}, {0, 4.0, 0, 0},
                       {0, 0, 9.0, 0}, {0, 0, 0, 4.0}};
    int k, i, j;

    rng64_seed(&g, 7);
    CHECK(kf_state_init(&s, x0, P0) == KF_OK, "init health run");
    for (k = 0; k < 500; ++k) {
        double z[2];
        CHECK(kf_predict(&s, dt, sigma_a) == KF_OK, "predict health");
        z[0] = s.x[0] + 3.0 * rng64_normal(&g); /* pseudo measurements */
        z[1] = s.x[2] + 3.0 * rng64_normal(&g);
        CHECK(kf_update(&s, z, r, NULL, NULL, NULL) == KF_OK,
              "update health");
        for (i = 0; i < 4; ++i)
            for (j = 0; j < 4; ++j)
                CHECK_NEAR(s.P[i][j], s.P[j][i], 1e-9,
                           "covariance stays symmetric");
        {
            double e[4] = {1.0, 0.0, 0.0, 0.0}, w[4];
            CHECK(kf_mat4_chol_solve(w, s.P, e) == 0,
                  "covariance stays positive definite");
        }
    }
}

/* ------------------------------------------------------------------ */
/* test: steady-state gain vs closed form (Kalata)                     */
/* ------------------------------------------------------------------ */

static void test_steady_state(void)
{
    double dt = 0.5, sigma_a = 1.2, r[2] = {9.0, 9.0};
    kf_steady_result st;
    double kan[2];
    int i;

    CHECK(kf_steady_state(dt, sigma_a, r, 1e-14, 100000, &st) == KF_OK,
          "steady-state iteration converges");
    CHECK(st.residual < 1e-14, "steady-state residual below tolerance");
    CHECK(kf_steady_gain_closed_form(dt, sigma_a, r[0], kan) == KF_OK,
          "closed form evaluates");

    CHECK_NEAR(st.K[0][0], kan[0], 1e-10, "steady gain K[0][0] vs analytic");
    CHECK_NEAR(st.K[0][1], kan[1], 1e-10, "steady gain K[0][1] vs analytic");
    CHECK_NEAR(st.K[1][2], kan[0], 1e-10, "steady gain K[1][2] vs analytic");
    CHECK_NEAR(st.K[1][3], kan[1], 1e-10, "steady gain K[1][3] vs analytic");
    CHECK_NEAR(st.K[0][2], 0.0, 1e-12, "x/y axes decoupled (K[0][2])");
    CHECK_NEAR(st.K[0][3], 0.0, 1e-12, "x/y axes decoupled (K[0][3])");
    CHECK_NEAR(st.K[1][0], 0.0, 1e-12, "x/y axes decoupled (K[1][0])");

    /* the returned prior covariance must satisfy the DARE fixed point:
     * Pp = F Pp_post F^T + Q with Pp_post the Joseph update of Pp */
    {
        double F[4][4], Q[4][4];
        double Ppost[4][4], Pnext[4][4];
        kf_state tmp;
        int a, b;
        for (a = 0; a < 4; ++a)
            for (b = 0; b < 4; ++b)
                tmp.P[a][b] = st.P[a][b];
        for (i = 0; i < 4; ++i)
            tmp.x[i] = 0.0;
        {
            double zz[2] = {0.0, 0.0};
            CHECK(kf_update(&tmp, zz, r, NULL, NULL, NULL) == KF_OK,
                  "posterior from steady prior");
        }
        kf_mat4_copy(Ppost, tmp.P);
        kf_transition_matrix(F, dt);
        kf_process_noise(Q, dt, sigma_a);
        {
            double FP[4][4], FPt[4][4], FPFt[4][4];
            kf_mat4_transpose(FPt, F);
            kf_mat4_mul(FP, F, Ppost);
            kf_mat4_mul(FPFt, FP, FPt);
            kf_mat4_add(Pnext, FPFt, Q);
        }
        for (a = 0; a < 4; ++a)
            for (b = 0; b < 4; ++b)
                CHECK_NEAR(Pnext[a][b], st.P[a][b], 1e-9,
                           "DARE fixed point residual");
    }

    /* second parameter set, smaller tracking index */
    CHECK(kf_steady_state(0.1, 0.3, r, 1e-14, 200000, &st) == KF_OK,
          "steady state converges (set 2)");
    CHECK(kf_steady_gain_closed_form(0.1, 0.3, r[0], kan) == KF_OK,
          "closed form (set 2)");
    CHECK_NEAR(st.K[0][0], kan[0], 1e-9, "steady gain set 2 position");
    CHECK_NEAR(st.K[0][1], kan[1], 1e-9, "steady gain set 2 velocity");

    /* sigma_a = 0 must give zero gain (filter trusts the model) */
    CHECK(kf_steady_state(0.5, 0.0, r, 1e-14, 200000, &st) == KF_OK,
          "steady state converges (no process noise)");
    CHECK_NEAR(st.K[0][0], 0.0, 1e-12, "zero process noise -> zero gain");
    CHECK_NEAR(st.K[0][1], 0.0, 1e-12, "zero process noise -> zero gain v");
}

/* ------------------------------------------------------------------ */
/* test: NIS consistency (dof 2) over a long model-matched run         */
/* ------------------------------------------------------------------ */

static void test_nis(void)
{
    rng64 g;
    kf_state s;
    double dt = 0.5, sigma_a = 1.2, r[2] = {9.0, 9.0};
    double x_true[4] = {10.0, 4.0, -5.0, 3.0};
    const long n = 200000L;
    long k;
    double sum = 0.0, in_bounds = 0.0;

    /* exact model-matched conditions: x0 known exactly, P0 = 0 */
    {
        double P0[4][4] = {{0}};
        CHECK(kf_state_init(&s, x_true, P0) == KF_OK, "init NIS run");
    }
    rng64_seed(&g, 42);

    for (k = 0; k < n; ++k) {
        double z[2], nis;
        double ax = sigma_a * rng64_normal(&g);
        double ay = sigma_a * rng64_normal(&g);
        double innov[2];

        /* truth: piecewise-white acceleration (matches filter Q exactly) */
        x_true[0] += dt * x_true[1] + 0.5 * dt * dt * ax;
        x_true[1] += dt * ax;
        x_true[2] += dt * x_true[3] + 0.5 * dt * dt * ay;
        x_true[3] += dt * ay;
        z[0] = x_true[0] + 3.0 * rng64_normal(&g);
        z[1] = x_true[2] + 3.0 * rng64_normal(&g);

        CHECK(kf_predict(&s, dt, sigma_a) == KF_OK, "predict NIS run");
        CHECK(kf_update(&s, z, r, innov, NULL, &nis) == KF_OK,
              "update NIS run");
        (void)innov;
        sum += nis;
        if (nis >= CHI2_DOF2_LO && nis <= CHI2_DOF2_HI)
            in_bounds += 1.0;
    }
    {
        double mean = sum / (double)n;
        double frac = in_bounds / (double)n;
        printf("  NIS  : mean = %.5f (expected 2), in-bounds fraction = "
               "%.5f (expected 0.95), n = %ld\n", mean, frac, n);
        CHECK(mean > 1.95 && mean < 2.05, "NIS mean within [1.95, 2.05]");
        CHECK(frac > 0.945 && frac < 0.955,
              "NIS fraction within chi2_2 bounds in [0.945, 0.955]");
    }
}

/* ------------------------------------------------------------------ */
/* test: NEES Monte-Carlo consistency (dof 4)                          */
/* ------------------------------------------------------------------ */

static void test_nees_mc(void)
{
    const int n_runs = 1000, n_steps = 240;
    const double dt = 0.5, sigma_a = 1.2;
    double r[2] = {9.0, 9.0};
    double sum_final = 0.0, in_bounds = 0.0, sum_window = 0.0;
    int run, k, i;

    for (run = 0; run < n_runs; ++run) {
        rng64 g;
        kf_state s;
        double x_true[4] = {0.0, 8.0, 0.0, 6.0};
        double x0[4];
        double P0[4][4] = {{9.0, 0, 0, 0}, {0, 4.0, 0, 0},
                           {0, 0, 9.0, 0}, {0, 0, 0, 4.0}};
        double nees_final = 0.0, nees_sum = 0.0;

        rng64_seed(&g, (uint64_t)(1234 + run));
        /* draw the initial estimate from N(x_true, P0): exact consistency */
        for (i = 0; i < 4; ++i)
            x0[i] = x_true[i] +
                    sqrt(P0[i][i]) * rng64_normal(&g);
        CHECK(kf_state_init(&s, x0, P0) == KF_OK, "init NEES run");

        for (k = 1; k <= n_steps; ++k) {
            double z[2], nees;
            double ax = sigma_a * rng64_normal(&g);
            double ay = sigma_a * rng64_normal(&g);

            x_true[0] += dt * x_true[1] + 0.5 * dt * dt * ax;
            x_true[1] += dt * ax;
            x_true[2] += dt * x_true[3] + 0.5 * dt * dt * ay;
            x_true[3] += dt * ay;
            z[0] = x_true[0] + 3.0 * rng64_normal(&g);
            z[1] = x_true[2] + 3.0 * rng64_normal(&g);

            CHECK(kf_predict(&s, dt, sigma_a) == KF_OK, "predict NEES MC");
            CHECK(kf_update(&s, z, r, NULL, NULL, NULL) == KF_OK,
                  "update NEES MC");
            CHECK(kf_nees(&s, x_true, &nees) == KF_OK, "nees eval");
            nees_final = nees;
            if (k > n_steps - 100)
                nees_sum += nees;
        }
        sum_final += nees_final;
        sum_window += nees_sum / 100.0;
        if (nees_final >= CHI2_DOF4_LO && nees_final <= CHI2_DOF4_HI)
            in_bounds += 1.0;
    }
    {
        double avg = sum_final / (double)n_runs;
        double frac = in_bounds / (double)n_runs;
        double wavg = sum_window / (double)n_runs;
        printf("  NEES : final-step average = %.5f (95%% bounds for N=%d:"
               " [%.4f, %.4f]), per-run in-bounds fraction = %.4f,"
               " window average = %.5f\n",
               avg, n_runs, NEES_AVG_LO, NEES_AVG_HI, frac, wavg);
        CHECK(avg > NEES_AVG_LO && avg < NEES_AVG_HI,
              "NEES average within chi-square bounds");
        CHECK(frac > 0.92 && frac < 0.98,
              "per-run NEES fraction within [0.92, 0.98]");
    }
}

/* ------------------------------------------------------------------ */
/* test: zero-noise limit recovers the exact trajectory                */
/* ------------------------------------------------------------------ */

static void test_zero_noise(void)
{
    rng64 g;
    kf_state s;
    const double dt = 0.5, sigma_a = 0.0;
    double r[2] = {1e-16, 1e-16}; /* variance 1e-16 -> std 1e-8 */
    double x_true[4] = {3.0, -2.0, 7.0, 1.5};
    double z0[2];
    int k;

    rng64_seed(&g, 99);
    z0[0] = x_true[0]; /* noise-free measurement of the exact position */
    z0[1] = x_true[2];
    {
        double x0[4] = {z0[0], 0.0, z0[1], 0.0};
        double P0[4][4] = {{1e-16, 0, 0, 0}, {0, 1.0, 0, 0},
                           {0, 0, 1e-16, 0}, {0, 0, 0, 1.0}};
        CHECK(kf_state_init(&s, x0, P0) == KF_OK, "init zero-noise");
    }

    for (k = 1; k <= 400; ++k) {
        double z[2];
        /* exact constant-velocity truth, no process noise */
        x_true[0] += dt * x_true[1];
        x_true[2] += dt * x_true[3];
        z[0] = x_true[0];
        z[1] = x_true[2];
        CHECK(kf_predict(&s, dt, sigma_a) == KF_OK, "predict zero-noise");
        CHECK(kf_update(&s, z, r, NULL, NULL, NULL) == KF_OK,
              "update zero-noise");
    }
    {
        double err_p = fabs(s.x[0] - x_true[0]) + fabs(s.x[2] - x_true[2]);
        double err_v = fabs(s.x[1] - x_true[1]) + fabs(s.x[3] - x_true[3]);
        printf("  zero-noise: |pos err| = %.3e, |vel err| = %.3e after 400 "
               "steps\n", err_p, err_v);
        CHECK(err_p < 1e-8, "zero-noise position error < 1e-8");
        CHECK(err_v < 1e-6, "zero-noise velocity error < 1e-6");
    }
}

int main(void)
{
    printf("kalman test suite\n");
    printf("-----------------\n");
    test_linalg();
    printf("linalg ...................... done\n");
    test_model_matrices();
    printf("model matrices .............. done\n");
    test_predict();
    printf("predict ..................... done\n");
    test_update();
    printf("update (Joseph) ............. done\n");
    test_validation();
    printf("input validation ............ done\n");
    test_covariance_health();
    printf("covariance health ........... done\n");
    test_steady_state();
    printf("steady-state Riccati ........ done\n");
    test_nis();
    printf("NIS consistency ............. done\n");
    test_nees_mc();
    printf("NEES Monte-Carlo ............ done\n");
    test_zero_noise();
    printf("zero-noise limit ............ done\n");

    printf("-----------------\n");
    printf("%d checks, %d failures\n", g_checks, g_failures);
    if (g_failures == 0)
        printf("ALL TESTS PASSED\n");
    else
        printf("TESTS FAILED\n");
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
