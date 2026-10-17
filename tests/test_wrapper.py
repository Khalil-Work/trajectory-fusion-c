"""Thin pytest wrapper around the compiled C test binary.

The Kalman filter, its tests and every numerical result live in C; this
file only locates the compiled binary, runs it and asserts on the exit
code and the final banner line. Skips (rather than fails) when the
binary has not been built yet, so `pytest` on a fresh clone is quiet and
`make test` remains the authoritative gate.

Usage:
    make test                 # builds and runs the C suite directly
    pytest tests/test_wrapper.py
"""

from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINARY = os.path.join(ROOT, "build", "test_kalman.exe" if os.name == "nt" else "test_kalman")


def _run() -> subprocess.CompletedProcess:
    return subprocess.run([BINARY], capture_output=True, text=True, check=False)


def test_c_suite_passes() -> None:
    if not os.path.isfile(BINARY):
        import pytest

        pytest.skip(f"C test binary not built: {BINARY} (run `make test` first)")
    result = _run()
    if result.returncode != 0:
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
    assert result.returncode == 0, "C test suite exited with failures"
    assert "ALL TESTS PASSED" in result.stdout


if __name__ == "__main__":
    if not os.path.isfile(BINARY):
        print(f"C test binary not built: {BINARY} (run `make test` first)")
        sys.exit(2)
    res = _run()
    sys.stdout.write(res.stdout)
    sys.exit(res.returncode)
