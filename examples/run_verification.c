/*
 * run_verification.c -- numerical verification suite producing the raw
 * evidence in docs/results/ (CSVs + verification_metrics.json).
 *
 * Sections
 *   A  NIS consistency: one model-matched run of 200k steps; the
 *      normalized innovation squared must behave as chi-square with 2 dof.
 *   B  Steady state: the discrete Riccati iteration must converge to the
 *      closed-form (Kalata tracking-index) gain.
 *   C  NEES consistency: 1000 Monte-Carlo runs; the final-step average
 *      NEES must fall inside the chi-square(4*N)/N interval.
 *   D  Zero-noise limit: with Q = 0 and (near) exact measurements the
 *      filter must recover the exact trajectory.
 *
 * Exit code 0 iff every section is within its bounds. Plots, the GIF and
 * run_summary.json are produced afterwards by examples/run_verification.py.
 *
 * Usage: run_verification [results_dir]   (default docs/results)
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/kalman.h"
#include "../src/rng.h"

/* chi-square two-sided 95% bounds (exact values; dof 2 closed form
 * -2*ln(1-q), others by numerical inversion of the CDF, see
 * docs/verification.md) */
#define CHI2_DOF2_LO 0.0506356159685797
#define CHI2_DOF2_HI 7.3777589082278722
#define CHI2_DOF4_LO 0.4844185570879299
#define CHI2_DOF4_HI 11.1432867818777882
#define NEES_AVG_LO  3.8265974192512836 /* chi2_{4000}(0.025)/1000 */
#define NEES_AVG_HI  4.1771910562861830 /* chi2_{4000}(0.975)/1000 */

/* scenario (same as the C tests and the demo) */
#define DT       0.5
#define SIGMA_A  1.2
#define SIGMA_Z  3.0
#define R_VAL    (SIGMA_Z * SIGMA_Z)
#define NIS_STEPS  200000L
#define NEES_RUNS   1000
#define NEES_STEPS  240
#define NEES_SEED0  1234

typedef struct {
    int pass;
    double nis_mean, nis_frac;
    long ss_iters;
    double ss_residual, ss_max_gain_err, ss_k1, ss_k2, ss_alpha, ss_beta_dt;
    double nees_avg, nees_frac, nees_window;
    double zn_pos_err, zn_vel_err;
} verification_t;

static FILE *open_out(const char *dir, const char *name)
{
    char path[512];
    FILE *f;
    if (snprintf(path, sizeof(path), "%s/%s", dir, name) >= (int)sizeof(path)) {
        fprintf(stderr, "error: path too long for %s\n", name);
        return NULL;
    }
    f = fopen(path, "w");
    if (f == NULL)
        fprintf(stderr, "error: cannot open %s (create the directory "
                "first, e.g. `make results`)\n", path);
    return f;
}

/* ------------------------------------------------------------------ */
/* A: NIS consistency (chi-square, 2 dof)                              */
/* ------------------------------------------------------------------ */

static int section_nis(const char *dir, verification_t *v)
{
    rng64 g;
    kf_state s;
    double x_true[4] = {10.0, 4.0, -5.0, 3.0};
    double r[KF_M] = {R_VAL, R_VAL};
    double P0[4][4] = {{0}};
    FILE *csv = open_out(dir, "nis_long.csv");
    long k, kept = 0;
    double sum = 0.0, in_b = 0.0;
    const long downsample = NIS_STEPS / 1000; /* 1000 rows in the CSV */

    if (csv == NULL)
        return 0;
    fprintf(csv, "k,nis\n");

    /* exact model-matched conditions: known x0, P0 = 0 */
    if (kf_state_init(&s, x_true, P0) != KF_OK)
        return 0;
    rng64_seed(&g, 42);

    for (k = 1; k <= NIS_STEPS; ++k) {
        double z[2], nis;
        double ax = SIGMA_A * rng64_normal(&g);
        double ay = SIGMA_A * rng64_normal(&g);

        x_true[0] += DT * x_true[1] + 0.5 * DT * DT * ax;
        x_true[1] += DT * ax;
        x_true[2] += DT * x_true[3] + 0.5 * DT * DT * ay;
        x_true[3] += DT * ay;
        z[0] = x_true[0] + SIGMA_Z * rng64_normal(&g);
        z[1] = x_true[2] + SIGMA_Z * rng64_normal(&g);

        if (kf_predict(&s, DT, SIGMA_A) != KF_OK ||
            kf_update(&s, z, r, NULL, NULL, &nis) != KF_OK)
            return 0;
        sum += nis;
        if (nis >= CHI2_DOF2_LO && nis <= CHI2_DOF2_HI)
            in_b += 1.0;
        if (k % downsample == 0) {
            fprintf(csv, "%ld,%.10g\n", k, nis);
            kept++;
        }
    }
    fclose(csv);

    v->nis_mean = sum / (double)NIS_STEPS;
    v->nis_frac = in_b / (double)NIS_STEPS;
    printf("[A] NIS   : mean = %.5f (expect 2), in-bounds = %.5f "
           "(expect 0.95), n = %ld, csv rows = %ld\n",
           v->nis_mean, v->nis_frac, NIS_STEPS, kept);
    return (v->nis_mean > 1.95 && v->nis_mean < 2.05 &&
            v->nis_frac > 0.945 && v->nis_frac < 0.955);
}

/* ------------------------------------------------------------------ */
/* B: steady-state Riccati vs closed form                              */
/* ------------------------------------------------------------------ */

static int section_steady(const char *dir, verification_t *v)
{
    double r[KF_M] = {R_VAL, R_VAL};
    kf_steady_result st;
    double kan[2];
    kf_state scratch;
    double prev[4][4];
    FILE *csv = open_out(dir, "riccati.csv");
    int i, j, iter, converged_at = -1;
    double max_err = 0.0;

    if (csv == NULL)
        return 0;
    fprintf(csv, "iter,k1,k2,alpha,beta_over_dt,max_dP\n");

    if (kf_steady_gain_closed_form(DT, SIGMA_A, r[0], kan) != KF_OK)
        return 0;
    v->ss_alpha = kan[0];
    v->ss_beta_dt = kan[1];

    /* manual Riccati iteration for the per-iteration CSV (x-axis gains;
     * the y axis is an identical decoupled problem) */
    for (i = 0; i < 4; ++i)
        for (j = 0; j < 4; ++j)
            scratch.P[i][j] = 0.0;
    scratch.P[0][0] = scratch.P[2][2] = r[0];
    scratch.P[1][1] = scratch.P[3][3] = (SIGMA_A * DT * 10.0 + 1.0) *
                                       (SIGMA_A * DT * 10.0 + 1.0);
    for (i = 0; i < 4; ++i)
        scratch.x[i] = 0.0;
    kf_mat4_copy(prev, scratch.P);

    for (iter = 1; iter <= 400; ++iter) {
        double z[2] = {0.0, 0.0};
        double S, k1, k2, delta = 0.0;

        if (kf_predict(&scratch, DT, SIGMA_A) != KF_OK)
            return 0;
        /* gains of the x axis from the prior covariance (S is diagonal in
         * this decoupled setup, but use the full 2x2 inverse for honesty) */
        {
            double Sm[2][2] = {{scratch.P[0][0] + r[0], scratch.P[0][2]},
                               {scratch.P[2][0], scratch.P[2][2] + r[1]}};
            double Si[2][2], det = Sm[0][0] * Sm[1][1] - Sm[0][1] * Sm[1][0];
            if (kf_mat2_inverse(Si, Sm) != 0)
                return 0;
            (void)det;
            k1 = scratch.P[0][0] * Si[0][0] + scratch.P[0][2] * Si[1][0];
            k2 = scratch.P[1][0] * Si[0][0] + scratch.P[1][2] * Si[1][0];
            S = Sm[0][0];
            (void)S;
        }
        if (kf_update(&scratch, z, r, NULL, NULL, NULL) != KF_OK)
            return 0;
        for (i = 0; i < 4; ++i)
            for (j = 0; j < 4; ++j) {
                double d = fabs(scratch.P[i][j] - prev[i][j]) /
                           (1.0 + fabs(prev[i][j]));
                if (d > delta)
                    delta = d;
            }
        kf_mat4_copy(prev, scratch.P);
        fprintf(csv, "%d,%.12g,%.12g,%.12g,%.12g,%.6g\n",
                iter, k1, k2, kan[0], kan[1], delta);
        if (delta < 1e-14 && converged_at < 0)
            converged_at = iter;
    }
    fclose(csv);

    /* authoritative converged values from the library routine */
    if (kf_steady_state(DT, SIGMA_A, r, 1e-14, 100000, &st) != KF_OK)
        return 0;
    v->ss_iters = st.iters;
    v->ss_residual = st.residual;
    v->ss_k1 = st.K[0][0];
    v->ss_k2 = st.K[0][1];

    {
        double kan_full[KF_M][KF_N] = {{0}};
        kan_full[0][0] = kan[0];
        kan_full[0][1] = kan[1];
        kan_full[1][2] = kan[0];
        kan_full[1][3] = kan[1];
        for (i = 0; i < KF_M; ++i)
            for (j = 0; j < KF_N; ++j) {
                double e = fabs(st.K[i][j] - kan_full[i][j]);
                if (e > max_err)
                    max_err = e;
            }
    }
    v->ss_max_gain_err = max_err;
    printf("[B] Riccati: converged in %d iters (residual %.2e); "
           "K_inf = [%.12f, %.12f] vs closed form [%.12f, %.12f], "
           "max |err| over all 8 gain entries = %.3e\n",
           st.iters, st.residual, st.K[0][0], st.K[0][1], kan[0], kan[1],
           max_err);
    return (converged_at > 0) && (max_err < 1e-10);
}

/* ------------------------------------------------------------------ */
/* C: NEES Monte-Carlo consistency (chi-square, 4 dof)                 */
/* ------------------------------------------------------------------ */

static int section_nees(const char *dir, verification_t *v)
{
    double r[KF_M] = {R_VAL, R_VAL};
    double P0[4][4] = {{9.0, 0, 0, 0}, {0, 4.0, 0, 0},
                       {0, 0, 9.0, 0}, {0, 0, 0, 4.0}};
    FILE *csv = open_out(dir, "nees_mc.csv");
    int run, k, i;
    double sum_final = 0.0, in_b = 0.0, sum_window = 0.0;

    if (csv == NULL)
        return 0;
    fprintf(csv, "run,nees_final,nees_window_mean\n");

    for (run = 0; run < NEES_RUNS; ++run) {
        rng64 g;
        kf_state s;
        double x_true[4] = {0.0, 8.0, 0.0, 6.0};
        double x0[4];
        double nees_final = 0.0, nees_sum = 0.0;

        rng64_seed(&g, (uint64_t)(NEES_SEED0 + run));
        /* initial error drawn from N(0, P0): exact consistency */
        for (i = 0; i < 4; ++i)
            x0[i] = x_true[i] + sqrt(P0[i][i]) * rng64_normal(&g);
        if (kf_state_init(&s, x0, P0) != KF_OK)
            return 0;

        for (k = 1; k <= NEES_STEPS; ++k) {
            double z[2], nees;
            double ax = SIGMA_A * rng64_normal(&g);
            double ay = SIGMA_A * rng64_normal(&g);

            x_true[0] += DT * x_true[1] + 0.5 * DT * DT * ax;
            x_true[1] += DT * ax;
            x_true[2] += DT * x_true[3] + 0.5 * DT * DT * ay;
            x_true[3] += DT * ay;
            z[0] = x_true[0] + SIGMA_Z * rng64_normal(&g);
            z[1] = x_true[2] + SIGMA_Z * rng64_normal(&g);

            if (kf_predict(&s, DT, SIGMA_A) != KF_OK ||
                kf_update(&s, z, r, NULL, NULL, NULL) != KF_OK ||
                kf_nees(&s, x_true, &nees) != KF_OK)
                return 0;
            nees_final = nees;
            if (k > NEES_STEPS - 100)
                nees_sum += nees;
        }
        fprintf(csv, "%d,%.10g,%.10g\n", run, nees_final, nees_sum / 100.0);
        sum_final += nees_final;
        sum_window += nees_sum / 100.0;
        if (nees_final >= CHI2_DOF4_LO && nees_final <= CHI2_DOF4_HI)
            in_b += 1.0;
    }
    fclose(csv);

    v->nees_avg = sum_final / (double)NEES_RUNS;
    v->nees_frac = in_b / (double)NEES_RUNS;
    v->nees_window = sum_window / (double)NEES_RUNS;
    printf("[C] NEES  : final-step average = %.5f (95%% bounds for "
           "N=%d: [%.4f, %.4f]), per-run in-bounds = %.4f, "
           "window average = %.5f\n",
           v->nees_avg, NEES_RUNS, NEES_AVG_LO, NEES_AVG_HI, v->nees_frac,
           v->nees_window);
    return (v->nees_avg > NEES_AVG_LO && v->nees_avg < NEES_AVG_HI &&
            v->nees_frac > 0.92 && v->nees_frac < 0.98);
}

/* ------------------------------------------------------------------ */
/* D: zero-noise limit                                                 */
/* ------------------------------------------------------------------ */

static int section_zero_noise(verification_t *v)
{
    kf_state s;
    double r[KF_M] = {1e-16, 1e-16};
    double x_true[4] = {3.0, -2.0, 7.0, 1.5};
    double x0[4], P0[4][4] = {{0}};
    int k;

    x0[0] = x_true[0];
    x0[1] = 0.0;
    x0[2] = x_true[2];
    x0[3] = 0.0;
    P0[0][0] = P0[2][2] = 1e-16;
    P0[1][1] = P0[3][3] = 1.0;
    if (kf_state_init(&s, x0, P0) != KF_OK)
        return 0;

    for (k = 1; k <= 400; ++k) {
        double z[2];
        x_true[0] += DT * x_true[1];
        x_true[2] += DT * x_true[3];
        z[0] = x_true[0];
        z[1] = x_true[2];
        if (kf_predict(&s, DT, 0.0) != KF_OK ||
            kf_update(&s, z, r, NULL, NULL, NULL) != KF_OK)
            return 0;
    }
    v->zn_pos_err = fabs(s.x[0] - x_true[0]) + fabs(s.x[2] - x_true[2]);
    v->zn_vel_err = fabs(s.x[1] - x_true[1]) + fabs(s.x[3] - x_true[3]);
    printf("[D] zero-noise: |pos err| = %.3e m, |vel err| = %.3e m/s "
           "after 400 steps (sigma_a = 0, sigma_z = 1e-8)\n",
           v->zn_pos_err, v->zn_vel_err);
    return (v->zn_pos_err < 1e-8 && v->zn_vel_err < 1e-6);
}

/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    const char *dir = "docs/results";
    verification_t v;
    FILE *j;
    int ok_a, ok_b, ok_c, ok_d;

    if (argc > 1)
        dir = argv[1];
    memset(&v, 0, sizeof(v));

    printf("run_verification -> %s\n", dir);
    ok_a = section_nis(dir, &v);
    ok_b = section_steady(dir, &v);
    ok_c = section_nees(dir, &v);
    ok_d = section_zero_noise(&v);
    v.pass = ok_a && ok_b && ok_c && ok_d;

    j = open_out(dir, "verification_metrics.json");
    if (j == NULL)
        return EXIT_FAILURE;
    fprintf(j,
        "{\n"
        "  \"config\": {\n"
        "    \"dt\": %g, \"sigma_a\": %g, \"sigma_z\": %g, \"r\": %g,\n"
        "    \"nis_steps\": %ld, \"nees_runs\": %d, \"nees_steps\": %d,\n"
        "    \"seed_nis\": 42, \"seed_base_nees\": %d\n"
        "  },\n"
        "  \"nis\": {\n"
        "    \"mean\": %.10g, \"frac_in_bounds\": %.10g,\n"
        "    \"lo\": %.10g, \"hi\": %.10g, \"dof\": 2, \"pass\": %s\n"
        "  },\n"
        "  \"steady_state\": {\n"
        "    \"iters\": %ld, \"residual\": %.6g,\n"
        "    \"k1\": %.12g, \"k2\": %.12g,\n"
        "    \"alpha\": %.12g, \"beta_over_dt\": %.12g,\n"
        "    \"max_gain_err\": %.6g, \"pass\": %s\n"
        "  },\n"
        "  \"nees\": {\n"
        "    \"avg_final\": %.10g, \"lo\": %.8g, \"hi\": %.8g,\n"
        "    \"frac_in_bounds\": %.10g, \"window_avg\": %.10g,\n"
        "    \"runs\": %d, \"dof\": 4, \"pass\": %s\n"
        "  },\n"
        "  \"zero_noise\": {\n"
        "    \"pos_err\": %.6g, \"vel_err\": %.6g, \"steps\": 400,\n"
        "    \"pass\": %s\n"
        "  },\n"
        "  \"all_pass\": %s\n"
        "}\n",
        DT, SIGMA_A, SIGMA_Z, R_VAL, NIS_STEPS, NEES_RUNS, NEES_STEPS,
        NEES_SEED0,
        v.nis_mean, v.nis_frac, CHI2_DOF2_LO, CHI2_DOF2_HI,
        ok_a ? "true" : "false",
        (long)v.ss_iters, v.ss_residual, v.ss_k1, v.ss_k2, v.ss_alpha,
        v.ss_beta_dt, v.ss_max_gain_err, ok_b ? "true" : "false",
        v.nees_avg, NEES_AVG_LO, NEES_AVG_HI, v.nees_frac, v.nees_window,
        NEES_RUNS, ok_c ? "true" : "false",
        v.zn_pos_err, v.zn_vel_err, ok_d ? "true" : "false",
        v.pass ? "true" : "false");
    fclose(j);

    printf("overall: %s\n", v.pass ? "ALL VERIFICATION SECTIONS PASSED"
                                    : "VERIFICATION FAILED");
    return v.pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
