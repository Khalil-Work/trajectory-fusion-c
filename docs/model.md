# State model and filter equations

This document defines the motion/measurement model, derives the process
noise used by the filter, states the Joseph-form measurement update, and
collects every symbol with its unit. Implementation: `src/kalman.c`.

## State, transition, measurement

The target state stacks the two constant-velocity (CV) axes:

```
        [ px ]                [ 1  dt  0  0 ]
  x  =  [ vx ]      F(dt) =  [ 0  1   0  0 ]
        [ py ]                [ 0  0   1  dt ]
        [ vy ]                [ 0  0   0  1  ]
```

A measurement observes both positions with independent noise of variance
`r` per axis (diagonal `R = diag(r, r)`):

```
  z = H x + v,   H = [ 1  0  0  0 ]     v ~ N(0, R)
                   [ 0  0  1  0 ]
```

Because `F`, `Q` and `R` are block-diagonal per axis, the x/y subproblems
are two decoupled 1-D CV filters; the implementation nevertheless carries
the full 4-state covariance (cross terms stay exactly zero, which is
itself checked by the tests).

## Process noise: piecewise-white acceleration

The acceleration is modeled as constant over each sampling interval and
white (independent, zero-mean) from interval to interval with standard
deviation `sigma_a`:

```
  a_k ~ N(0, sigma_a^2),   a(t) = a_k  for t in [t_k, t_k + dt)
```

Integrating `x'' = a` over one step gives, per axis,

```
  [ p_{k+1} ]   [ 1  dt ] [ p_k ]   [ dt^2 / 2 ]
  [ v_{k+1} ] = [ 0   1  ] [ v_k ] + [   dt     ] a_k  =  F_1d x_k + G a_k
```

so the discrete process noise is

```
  Q_1d = sigma_a^2 * G G^T
       = sigma_a^2 * [ dt^4/4   dt^3/2 ]
                     [ dt^3/2   dt^2   ]
```

and the 4-state `Q = diag_block(Q_1d, Q_1d)`. This is the
*piecewise-constant white acceleration* model of Bar-Shalom et al.
(the alternative *continuous white noise acceleration* model would give
`dt^3/3, dt^2/2, dt` entries instead; the two must not be mixed when
comparing steady-state formulas).

## Prediction

```
  x^- = F x                          (KF: x is the previous posterior mean)
  P^- = F P F^T + Q
```

After propagation the implementation symmetrizes `P <- (P + P^T)/2` to
suppress round-off asymmetry.

## Measurement update (Joseph form)

With the innovation `y = z - H x^-` and innovation covariance

```
  S = H P^- H^T + R
```

the gain and update are

```
  K    = P^- H^T S^{-1}
  x^+  = x^- + K y
  P^+  = (I - K H) P^- (I - K H)^T + K R K^T        (Joseph form)
```

The Joseph form is used because it preserves symmetry and positive
semidefiniteness of `P^+` under round-off for any gain `K`, whereas the
compact form `P^+ = (I - K H) P^-` is only valid for the exact optimal
gain and can lose definiteness numerically. At these dimensions the extra
matrix products are negligible; the tests verify that both forms agree to
1e-12 on correlated priors and that `P` stays symmetric and
Cholesky-factorizable over long noisy runs.

## Consistency statistics

* **NIS** (normalized innovation squared), 2 degrees of freedom:

  ```
  NIS = y^T S^{-1} y
  ```

  If the model matches the truth (including `Q` and `R`), `NIS ~ chi^2(2)`
  at every step, including transients.

* **NEES** (normalized estimation error squared), 4 degrees of freedom:

  ```
  NEES = (x^+ - x_true)^T (P^+)^{-1} (x^+ - x_true)
  ```

  Averaged over independent Monte-Carlo runs at a fixed time index, the
  average is distributed as `chi^2(4N)/N`.

## Steady-state solution (alpha-beta / Kalata tracking index)

For the 1-D CV filter with position measurements the discrete algebraic
Riccati equation (DARE) has a closed-form solution in terms of the
tracking index

```
  Lambda = sigma_a * dt^2 / sqrt(r)
```

namely, with `u = (4 + Lambda - sqrt(Lambda^2 + 8 Lambda)) / 4`,

```
  K_inf = [ alpha ]        alpha = 1 - u^2
          [ beta/dt]       beta  = 2 (1 - u)^2
```

(equivalently `beta^2 = Lambda^2 (1 - alpha)` and
`beta = 2(2 - alpha) - 4 sqrt(1 - alpha)`, the classical alpha-beta
filter relations; see the reference by Kalata in the README). `u` is the
smaller root, chosen so `K_inf -> 0` as `Lambda -> 0` and
`K_inf -> [1, 2/dt]` as `Lambda -> infinity`. The verification suite
compares the numerically iterated Riccati gain against this expression
and finds agreement to ~3e-15 (machine precision).

With `Q = 0` (no process noise) the unique non-negative DARE solution is
`P = 0`, `K = 0`; the code returns that exact solution directly instead
of iterating, because convergence of the iteration is only algebraic in
that degenerate case.

## Symbols and units

| Symbol    | Meaning                                        | Unit      |
|-----------|------------------------------------------------|-----------|
| `dt`      | sampling interval                              | s         |
| `x`       | state `[px, vx, py, vy]`                       | m, m/s    |
| `P`       | state error covariance (4x4, symmetric, PSD)   | mixed     |
| `P[i][i]` | variance of state component `i`               | m^2, (m/s)^2 |
| `F`       | constant-velocity transition matrix            | -         |
| `Q`       | process noise covariance (piecewise-white acc.)| mixed     |
| `sigma_a` | acceleration noise std per axis                | m/s^2     |
| `z`       | position measurement `[zx, zy]`                | m         |
| `r`       | measurement noise variance per axis            | m^2       |
| `sigma_z` | measurement noise std per axis, `sqrt(r)`      | m         |
| `S`       | innovation covariance                          | m^2       |
| `K`       | Kalman gain (2x4)                              | -         |
| `NIS`     | normalized innovation squared                  | - (chi^2_2) |
| `NEES`    | normalized estimation error squared            | - (chi^2_4) |
| `Lambda`  | tracking index                                 | -         |
