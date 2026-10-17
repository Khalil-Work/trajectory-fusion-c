#!/usr/bin/env python3
"""Post-processing for the Trajectory Fusion verification run.

Reads the raw evidence produced by the C programs (examples/traj_demo and
examples/run_verification) from docs/results/ and renders:

  trajectory.png          true path vs measurements vs estimate (cover)
  nis_nees.png            NIS / NEES time series with chi-square bounds
  riccati_convergence.png gain convergence to the closed-form solution
  covariance_ellipses.gif 2-sigma position covariance shrinking along the
                          trajectory (one frame per 4th filter step)
  run_summary.json        compiler/tool versions + all metrics

All estimation is done in C; this script only plots and summarizes.

Usage (from the repository root):
    make results                # runs the C binaries, then this script
    python examples/run_verification.py
"""

from __future__ import annotations

import csv
import datetime as _dt
import io
import json
import math
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import PIL  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "docs", "results")

# Chi-square two-sided 95% bounds (must match the constants compiled into
# tests/test_kalman.c and examples/run_verification.c). dof 2 has the exact
# closed form -2*ln(1-q); the others come from numerical inversion of the
# regularized incomplete gamma (verified below by crosscheck_chi2_constants).
CHI2_DOF2_LO, CHI2_DOF2_HI = 0.0506356159685797, 7.3777589082278722
CHI2_DOF4_LO, CHI2_DOF4_HI = 0.4844185570879299, 11.1432867818777882
NEES_AVG_LO, NEES_AVG_HI = 3.8265974192512836, 4.1771910562861830

GIF_STRIDE = 4  # one frame per 4th step -> 60 frames for 240 steps
GIF_DURATION_MS = 130


# --------------------------------------------------------------------- #
# chi-square machinery (exact incomplete-gamma inversion), used only to
# cross-check the constants above and to annotate plots
# --------------------------------------------------------------------- #

def _gammainc_p(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a, x)."""
    if x <= 0.0:
        return 0.0
    if x < a + 1.0:  # series
        ap, s, term = a, 1.0 / a, 1.0 / a
        for _ in range(2000):
            ap += 1.0
            term *= x / ap
            s += term
            if abs(term) < abs(s) * 1e-16:
                break
        return s * math.exp(-x + a * math.log(x) - math.lgamma(a))
    b = x + 1.0 - a  # continued fraction (Lentz)
    c = 1e300
    d = 1.0 / b
    h = d
    for i in range(1, 2000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return 1.0 - math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chi2_ppf(q: float, dof: int) -> float:
    """Quantile of the chi-square distribution (bisection)."""
    lo, hi = 0.0, 1.0
    while _gammainc_p(dof / 2.0, hi / 2.0) < q:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _gammainc_p(dof / 2.0, mid / 2.0) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def crosscheck_chi2_constants() -> dict:
    """Compare the compiled-in constants with fresh exact inversion."""
    checks = {
        "chi2_dof2": (chi2_ppf(0.025, 2), chi2_ppf(0.975, 2),
                      CHI2_DOF2_LO, CHI2_DOF2_HI),
        "chi2_dof4": (chi2_ppf(0.025, 4), chi2_ppf(0.975, 4),
                      CHI2_DOF4_LO, CHI2_DOF4_HI),
        "nees_avg_1000": (chi2_ppf(0.025, 4000) / 1000.0,
                          chi2_ppf(0.975, 4000) / 1000.0,
                          NEES_AVG_LO, NEES_AVG_HI),
    }
    out = {}
    ok = True
    for name, (lo, hi, lo_ref, hi_ref) in checks.items():
        good = abs(lo - lo_ref) < 1e-6 and abs(hi - hi_ref) < 1e-6
        ok = ok and good
        out[name] = {"computed": [lo, hi], "compiled": [lo_ref, hi_ref],
                     "match": good}
    out["all_match"] = ok
    return out


# --------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------- #

def read_csv(name: str) -> list[dict]:
    path = os.path.join(RES, name)
    if not os.path.isfile(path):
        sys.exit(f"missing {path}; run the C binaries first (`make results`)")
    with open(path, newline="") as fh:
        return [row for row in csv.DictReader(fh)]


def style_axes(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(False)


def gcc_version() -> str:
    for cc in ("gcc", "cc", "clang"):
        try:
            out = subprocess.run([cc, "--version"], capture_output=True,
                                 text=True, check=True)
            return out.stdout.splitlines()[0]
        except (OSError, subprocess.CalledProcessError):
            continue
    return "unavailable"


# --------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------- #

def plot_trajectory(rows: list[dict]) -> dict:
    tx = [float(r["px_true"]) for r in rows]
    ty = [float(r["py_true"]) for r in rows]
    zx = [float(r["z_x"]) for r in rows]
    zy = [float(r["z_y"]) for r in rows]
    ex = [float(r["px_est"]) for r in rows]
    ey = [float(r["py_est"]) for r in rows]

    fig, ax = plt.subplots(figsize=(9.6, 6.8), dpi=110)
    fig.patch.set_facecolor("white")
    style_axes(ax)
    ax.plot(tx, ty, color="tab:blue", lw=1.8, label="true trajectory")
    ax.scatter(zx, zy, s=12, color="tab:orange", alpha=0.55,
               label="measurements $z_k$ ($\\sigma_z$ = 3 m)")
    ax.plot(ex, ey, color="tab:green", lw=1.4, ls="--",
            label="filter estimate")
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.set_title("Trajectory Fusion: constant-velocity Kalman filter "
                 "(dt = 0.5 s, 240 steps)")
    ax.legend(loc="best", framealpha=0.9)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    out = os.path.join(RES, "trajectory.png")
    fig.savefig(out)
    plt.close(fig)

    # demo statistics from the run itself
    half = len(rows) // 2
    nis = [float(r["nis"]) for r in rows[1:]]
    nees = [float(r["nees"]) for r in rows[half:]]
    sq = [(float(r["px_est"]) - float(r["px_true"])) ** 2 +
          (float(r["py_est"]) - float(r["py_true"])) ** 2 for r in rows[half:]]
    return {
        "n_rows": len(rows),
        "mean_nis": sum(nis) / len(nis),
        "mean_nees_second_half": sum(nees) / len(nees),
        "position_rmse_second_half": math.sqrt(sum(sq) / len(sq)),
        "file": out,
    }


def plot_nis_nees(rows: list[dict]) -> None:
    t = [float(r["t"]) for r in rows]
    nis = [float(r["nis"]) for r in rows]
    nees = [float(r["nees"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.6, 6.8), dpi=110,
                                   sharex=True)
    fig.patch.set_facecolor("white")
    style_axes(ax1)
    style_axes(ax2)

    ax1.semilogy(t[1:], nis[1:], color="tab:blue", lw=0.9)
    ax1.axhline(CHI2_DOF2_HI, color="tab:red", ls="--", lw=1.0,
                label="95% bounds, $\\chi^2_2$: [0.0506, 7.3778]")
    ax1.axhline(CHI2_DOF2_LO, color="tab:red", ls="--", lw=1.0)
    ax1.set_ylabel("NIS (2 dof)")
    ax1.set_title("Normalized innovation squared (NIS) and estimation "
                  "error squared (NEES)")
    ax1.legend(loc="upper right", framealpha=0.9)

    ax2.plot(t, nees, color="tab:purple", lw=1.0)
    ax2.axhline(CHI2_DOF4_HI, color="tab:red", ls="--", lw=1.0,
                label="95% bounds, $\\chi^2_4$: [0.4844, 11.1433]")
    ax2.axhline(CHI2_DOF4_LO, color="tab:red", ls="--", lw=1.0)
    ax2.axhline(4.0, color="tab:gray", ls=":", lw=1.0, label="mean (4)")
    ax2.set_ylabel("NEES (4 dof)")
    ax2.set_xlabel("time (s)")
    ax2.set_ylim(bottom=min(0.0, min(nees) - 0.5))
    ax2.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "nis_nees.png"))
    plt.close(fig)


def plot_riccati(rows: list[dict]) -> None:
    it = [int(r["iter"]) for r in rows]
    k1 = [float(r["k1"]) for r in rows]
    k2 = [float(r["k2"]) for r in rows]
    alpha = float(rows[-1]["alpha"])
    beta_dt = float(rows[-1]["beta_over_dt"])
    dp = [float(r["max_dP"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.6, 6.8), dpi=110,
                                   sharex=True)
    fig.patch.set_facecolor("white")
    style_axes(ax1)
    style_axes(ax2)

    ax1.plot(it, k1, color="tab:blue", lw=1.5, label="$K_{pos}$ (Riccati)")
    ax1.plot(it, k2, color="tab:green", lw=1.5, label="$K_{vel}$ (Riccati)")
    ax1.axhline(alpha, color="tab:blue", ls="--", lw=1.0,
                label=f"closed form $\\alpha$ = {alpha:.6f}")
    ax1.axhline(beta_dt, color="tab:green", ls="--", lw=1.0,
                label=f"closed form $\\beta/\\Delta t$ = {beta_dt:.6f}")
    ax1.set_ylabel("steady-state gain entries")
    ax1.set_title("Discrete Riccati iteration vs closed-form (Kalata) "
                  "solution, $\\Lambda$ = 0.1")
    ax1.legend(loc="center right", framealpha=0.9)

    ax2.semilogy(it, dp, color="tab:red", lw=1.2)
    ax2.axhline(1e-14, color="tab:gray", ls=":", lw=1.0, label="tol = 1e-14")
    ax2.set_xlabel("Riccati iteration")
    ax2.set_ylabel("max relative $\\Delta P$ per step")
    ax2.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "riccati_convergence.png"))
    plt.close(fig)


# --------------------------------------------------------------------- #
# animated GIF: 2-sigma covariance ellipses along the trajectory
# --------------------------------------------------------------------- #

def ellipse_params(a: float, b: float, c: float) -> tuple[float, float, float]:
    """Axis lengths (1-sigma) and angle [deg] of the 2x2 covariance
    [[a, b], [b, c]] (analytic eigendecomposition)."""
    tr_half = 0.5 * (a + c)
    disc = max(0.0, tr_half * tr_half - (a * c - b * b))
    lam1 = tr_half + math.sqrt(disc)
    lam2 = max(0.0, tr_half - math.sqrt(disc))
    angle = math.degrees(0.5 * math.atan2(2.0 * b, a - c))
    return math.sqrt(lam1), math.sqrt(lam2), angle


def make_gif(rows: list[dict]) -> dict:
    tx = [float(r["px_true"]) for r in rows]
    ty = [float(r["py_true"]) for r in rows]
    zx = [float(r["z_x"]) for r in rows]
    zy = [float(r["z_y"]) for r in rows]
    ex = [float(r["px_est"]) for r in rows]
    ey = [float(r["py_est"]) for r in rows]
    tt = [float(r["t"]) for r in rows]

    xlo = min(min(tx), min(zx)) - 12
    xhi = max(max(tx), max(zx)) + 12
    ylo = min(min(ty), min(zy)) - 12
    yhi = max(max(ty), max(zy)) + 12

    frames: list[Image.Image] = []
    n_frames = 0
    for k in range(GIF_STRIDE, len(rows) + 1, GIF_STRIDE):
        r = rows[k - 1]
        s1, s2, ang = ellipse_params(float(r["P_pxpx"]), float(r["P_pxpy"]),
                                     float(r["P_pypy"]))

        fig, ax = plt.subplots(figsize=(10.0, 6.8), dpi=105)
        fig.patch.set_facecolor("white")
        style_axes(ax)
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        ax.set_aspect("equal", adjustable="box")

        ax.plot(tx, ty, color="0.82", lw=1.0, zorder=1)
        ax.scatter(zx[:k], zy[:k], s=14, color="tab:orange", alpha=0.6,
                   zorder=2, label="measurements so far")
        ax.plot(ex[:k], ey[:k], color="tab:green", lw=1.4, ls="--", zorder=3,
                label="estimate so far")
        ax.plot(tx[k - 1], ty[k - 1], marker="o", ms=6, color="tab:blue",
                zorder=4, label="true position")
        ell = matplotlib.patches.Ellipse(
            (ex[k - 1], ey[k - 1]), width=4.0 * s1, height=4.0 * s2,
            angle=ang, fill=False, edgecolor="tab:red", lw=1.8, zorder=5,
            label="2$\\sigma$ position covariance")
        ax.add_patch(ell)
        ax.set_xlabel("x position (m)")
        ax.set_ylabel("y position (m)")
        ax.set_title(f"Covariance fusion: step {k - 1} of {len(rows) - 1} "
                     f"(t = {tt[k - 1]:.1f} s)")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        frame = Image.open(buf).convert("P", palette=Image.ADAPTIVE)
        # force the (near-)white palette entry to pure white: adaptive
        # quantization otherwise represents white as (254,254,254)
        pal = frame.getpalette()
        if pal:
            for i in range(len(pal) // 3):
                r, g, b = pal[3 * i:3 * i + 3]
                if r >= 251 and g >= 251 and b >= 251:
                    pal[3 * i:3 * i + 3] = [255, 255, 255]
            frame.putpalette(pal)
        frames.append(frame)
        n_frames += 1

    out = os.path.join(RES, "covariance_ellipses.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=GIF_DURATION_MS, loop=0, optimize=False)
    w, h = frames[0].size
    return {"file": out, "frames": n_frames, "size_px": [w, h],
            "duration_ms": GIF_DURATION_MS}


# --------------------------------------------------------------------- #

def rel(path: str) -> str:
    """Repo-relative path with forward slashes (portable JSON output)."""
    return os.path.relpath(path, ROOT).replace("\\", "/")


def main() -> int:
    if not os.path.isdir(RES):
        sys.exit(f"missing {RES}; run the C binaries first (`make results`)")

    traj = read_csv("trajectory.csv")
    riccati = read_csv("riccati.csv")
    with open(os.path.join(RES, "verification_metrics.json")) as fh:
        metrics = json.load(fh)

    demo_stats = plot_trajectory(traj)
    plot_nis_nees(traj)
    plot_riccati(riccati)
    gif_info = make_gif(traj)

    summary = {
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="seconds"),
        "compiler": {"version": gcc_version(),
                     "flags": "-std=c99 -Wall -Wextra -O2"},
        "tools": {
            "python": sys.version.split()[0],
            "matplotlib": matplotlib.__version__,
            "pillow": PIL.__version__,
        },
        "scenario": {
            "dt_s": 0.5,
            "steps": len(traj) - 1,
            "sigma_a_m_s2": 1.2,
            "sigma_z_m": 3.0,
            "seed": 20261018,
        },
        "demo_stats": demo_stats,
        "verification": metrics,
        "chi2_bounds_crosscheck": crosscheck_chi2_constants(),
        "artifacts": {
            "cover_plot": rel(demo_stats["file"]),
            "nis_nees_plot": "docs/results/nis_nees.png",
            "riccati_plot": "docs/results/riccati_convergence.png",
            "gif": {
                "path": rel(gif_info["file"]),
                "frames": gif_info["frames"],
                "size_px": gif_info["size_px"],
                "duration_ms": gif_info["duration_ms"],
            },
            "csv": ["docs/results/trajectory.csv",
                    "docs/results/nis_long.csv",
                    "docs/results/riccati.csv",
                    "docs/results/nees_mc.csv"],
        },
    }
    with open(os.path.join(RES, "run_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")

    print(f"plots + gif + run_summary.json written to {RES}")
    print(f"  gif: {gif_info['frames']} frames, "
          f"{gif_info['size_px'][0]}x{gif_info['size_px'][1]} px")
    print(f"  verification all_pass = {metrics.get('all_pass')}")
    return 0 if metrics.get("all_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
