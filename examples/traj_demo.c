/*
 * traj_demo.c -- simulate a noisy 2-D constant-velocity trajectory, track it
 * with the Kalman filter from src/kalman.c and write the full run to CSV.
 *
 * Output columns (one row per time step):
 *   t, px_true, py_true, vx_true, vy_true, z_x, z_y,
 *   px_est, vx_est, py_est, vy_est,
 *   innov_x, innov_y, S00, S01, S11, nis,
 *   nees, nees_px, nees_vx, nees_py, nees_vy,
 *   P_pxpx, P_pxpy, P_pypy
 *
 * nees_* columns are diagonal-normalized error contributions e_i^2/P_ii;
 * the total `nees` uses the full inverse P^-1. Row 0 holds the initial
 * estimate (innovation/NIS undefined -> nan).
 *
 * Usage: traj_demo [output_csv] [seed]
 * Defaults: docs/results/trajectory.csv, seed 20261018.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../src/kalman.h"
#include "../src/rng.h"

int main(int argc, char **argv)
{
    /* scenario parameters (units: m, s, m/s, m/s^2) */
    const double dt = 0.5;      /* sampling time [s]                    */
    const int    n_steps = 240; /* 120 s of data                        */
    const double sigma_a = 1.2; /* acceleration noise std [m/s^2]       */
    const double sigma_z = 3.0; /* measurement noise std [m]            */
    double r[KF_M] = {sigma_z * sigma_z, sigma_z * sigma_z};
    double x_true[4] = {0.0, 8.0, 0.0, 6.0}; /* [px, vx, py, vy]        */
    uint64_t seed = 20261018ULL;
    const char *out_path = "docs/results/trajectory.csv";

    rng64 g;
    kf_state s;
    FILE *f;
    int k;
    double sum_nis = 0.0, cnt_nis = 0.0, sum_nees = 0.0, cnt_nees = 0.0;
    double sq_err = 0.0;

    if (argc > 1)
        out_path = argv[1];
    if (argc > 2)
        seed = (uint64_t)strtoull(argv[2], NULL, 10);

    f = fopen(out_path, "w");
    if (f == NULL) {
        fprintf(stderr, "error: cannot open '%s' for writing "
                "(create docs/results first, e.g. via `make results`)\n",
                out_path);
        return EXIT_FAILURE;
    }

    rng64_seed(&g, seed);

    /* first measurement initializes position; velocity prior std 5 m/s */
    {
        double z0[2], x0[4];
        double P0[4][4] = {{0}};
        z0[0] = x_true[0] + sigma_z * rng64_normal(&g);
        z0[1] = x_true[2] + sigma_z * rng64_normal(&g);
        x0[0] = z0[0];
        x0[1] = 0.0;
        x0[2] = z0[1];
        x0[3] = 0.0;
        P0[0][0] = P0[2][2] = sigma_z * sigma_z;
        P0[1][1] = P0[3][3] = 25.0;
        if (kf_state_init(&s, x0, P0) != KF_OK) {
            fprintf(stderr, "error: state init failed\n");
            fclose(f);
            return EXIT_FAILURE;
        }
        fprintf(f, "t,px_true,py_true,vx_true,vy_true,z_x,z_y,"
                   "px_est,vx_est,py_est,vy_est,"
                   "innov_x,innov_y,S00,S01,S11,nis,"
                   "nees,nees_px,nees_vx,nees_py,nees_vy,"
                   "P_pxpx,P_pxpy,P_pypy\n");
        {
            double nees0;
            if (kf_nees(&s, x_true, &nees0) != KF_OK)
                nees0 = 0.0 / 0.0;
            fprintf(f, "%.6g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,"
                       "%.9g,%.9g,%.9g,%.9g,"
                       "nan,nan,nan,nan,nan,nan,"
                       "%.9g,%.9g,%.9g,%.9g,%.9g,"
                       "%.9g,%.9g,%.9g\n",
                    0.0, x_true[0], x_true[2], x_true[1], x_true[3],
                    z0[0], z0[1],
                    s.x[0], s.x[1], s.x[2], s.x[3],
                    nees0,
                    (s.x[0] - x_true[0]) * (s.x[0] - x_true[0]) / s.P[0][0],
                    (s.x[1] - x_true[1]) * (s.x[1] - x_true[1]) / s.P[1][1],
                    (s.x[2] - x_true[2]) * (s.x[2] - x_true[2]) / s.P[2][2],
                    (s.x[3] - x_true[3]) * (s.x[3] - x_true[3]) / s.P[3][3],
                    s.P[0][0], s.P[0][2], s.P[2][2]);
        }
    }

    for (k = 1; k <= n_steps; ++k) {
        double z[2], innov[2], Sm[2][2], nis, nees;
        double ax = sigma_a * rng64_normal(&g);
        double ay = sigma_a * rng64_normal(&g);

        /* truth: constant velocity perturbed by piecewise-white accel */
        x_true[0] += dt * x_true[1] + 0.5 * dt * dt * ax;
        x_true[1] += dt * ax;
        x_true[2] += dt * x_true[3] + 0.5 * dt * dt * ay;
        x_true[3] += dt * ay;
        z[0] = x_true[0] + sigma_z * rng64_normal(&g);
        z[1] = x_true[2] + sigma_z * rng64_normal(&g);

        if (kf_predict(&s, dt, sigma_a) != KF_OK ||
            kf_update(&s, z, r, innov, Sm, &nis) != KF_OK ||
            kf_nees(&s, x_true, &nees) != KF_OK) {
            fprintf(stderr, "error: filter step %d failed\n", k);
            fclose(f);
            return EXIT_FAILURE;
        }

        sum_nis += nis;
        cnt_nis += 1.0;
        if (k > n_steps / 2) {
            sum_nees += nees;
            cnt_nees += 1.0;
            sq_err += (s.x[0] - x_true[0]) * (s.x[0] - x_true[0]) +
                      (s.x[2] - x_true[2]) * (s.x[2] - x_true[2]);
        }

        fprintf(f, "%.6g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,"
                   "%.9g,%.9g,%.9g,%.9g,"
                   "%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,"
                   "%.9g,%.9g,%.9g,%.9g,%.9g,"
                   "%.9g,%.9g,%.9g\n",
                (double)k * dt, x_true[0], x_true[2], x_true[1], x_true[3],
                z[0], z[1],
                s.x[0], s.x[1], s.x[2], s.x[3],
                innov[0], innov[1], Sm[0][0], Sm[0][1], Sm[1][1], nis,
                nees,
                (s.x[0] - x_true[0]) * (s.x[0] - x_true[0]) / s.P[0][0],
                (s.x[1] - x_true[1]) * (s.x[1] - x_true[1]) / s.P[1][1],
                (s.x[2] - x_true[2]) * (s.x[2] - x_true[2]) / s.P[2][2],
                (s.x[3] - x_true[3]) * (s.x[3] - x_true[3]) / s.P[3][3],
                s.P[0][0], s.P[0][2], s.P[2][2]);
    }
    fclose(f);

    printf("traj_demo: %d steps, dt=%.3g s, sigma_a=%.3g m/s^2, "
           "sigma_z=%.3g m, seed=%llu\n",
           n_steps, dt, sigma_a, sigma_z, (unsigned long long)seed);
    printf("  wrote %s\n", out_path);
    printf("  mean NIS  = %.4f (2 dof, expected 2)\n", sum_nis / cnt_nis);
    printf("  mean NEES = %.4f over last %d steps (4 dof, expected 4; "
           "demo init is heuristic, see run_verification for the gated "
           "Monte-Carlo check)\n",
           sum_nees / cnt_nees, n_steps / 2);
    printf("  position RMSE (2nd half) = %.3f m\n",
           sqrt(sq_err / cnt_nees));
    return EXIT_SUCCESS;
}
