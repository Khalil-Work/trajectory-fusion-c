# Trajectory Fusion -- native C Kalman filter with covariance verification.
#
# Primary entry points:
#   make            build everything found in src/ and examples/
#   make test       build and run the C test suite (the CI gate)
#   make demo       run examples/traj_demo  -> docs/results/trajectory.csv
#   make verify     run examples/run_verification -> docs/results/*.csv + metrics json
#   make results    demo + verify + plots/GIF/summary (needs python + matplotlib)
#   make clean      remove build artifacts
#
# Notes
# - CC/CFLAGS are overridable:  make CC=clang CFLAGS="-std=c99 -O2 -g"
# - On Windows use mingw32-make (recipes fall back to cmd.exe built-ins).

# GNU make predefines CC=cc; use gcc unless the caller overrides it.
ifeq ($(origin CC),default)
CC := gcc
endif
CFLAGS  ?= -std=c99 -Wall -Wextra -O2
PYTHON  ?= python
LDLIBS  ?= -lm

ifeq ($(OS),Windows_NT)
EXE   := .exe
MKDIR := if not exist build mkdir build
MKDIR_RES := if not exist docs\results mkdir docs\results
RMDIR := if exist build rmdir /s /q build
else
EXE   :=
MKDIR := mkdir -p build
MKDIR_RES := mkdir -p docs/results
RMDIR := rm -rf build
endif

SRCS    := src/kalman.c src/rng.c
HDRS    := src/kalman.h src/rng.h
EXAMPLES:= $(wildcard examples/*.c)

BIN_TEST := build/test_kalman$(EXE)
BIN_DEMO := build/traj_demo$(EXE)
BIN_VER  := build/run_verification$(EXE)

.PHONY: all test demo verify results clean

all: $(BIN_TEST) $(EXAMPLES:examples/%.c=build/%$(EXE))

build:
	@$(MKDIR)

$(BIN_TEST): tests/test_kalman.c $(SRCS) $(HDRS) | build
	$(CC) $(CFLAGS) -Isrc tests/test_kalman.c $(SRCS) $(LDLIBS) -o $@

$(BIN_DEMO): examples/traj_demo.c $(SRCS) $(HDRS) | build
	$(CC) $(CFLAGS) -Isrc examples/traj_demo.c $(SRCS) $(LDLIBS) -o $@

$(BIN_VER): examples/run_verification.c $(SRCS) $(HDRS) | build
	$(CC) $(CFLAGS) -Isrc examples/run_verification.c $(SRCS) $(LDLIBS) -o $@

test: $(BIN_TEST)
	$(BIN_TEST)

demo: $(BIN_DEMO)
	@$(MKDIR_RES)
	$(BIN_DEMO)

verify: $(BIN_VER)
	@$(MKDIR_RES)
	$(BIN_VER)

results: demo verify
	$(PYTHON) examples/run_verification.py

clean:
	@$(RMDIR)
