"""Runs a deepsocflow script with Hardware.simulate() redirected into the pinned-
Verilator container.

Everything except the simulator itself runs on the macOS host, where it already
works: this only replaces the one step that cannot (verilator 5.024 does not build
against Apple clang's libc++ (tested: clang 21), and 5.050 breaks firebridge's
re-entrant eval() - see CLAUDE.md's "Verilator version" Known Issue for details).

Usage:  python deepsocflow/sim/docker_sim.py run/example.py
        (from the worktree root; the target script runs with its own directory as
         cwd, matching how these scripts expect to be invoked)
"""
import os
import runpy
import subprocess
import sys

# Worktree root = two directories up from this file (deepsocflow/sim/docker_sim.py).
# Derived rather than hardcoded so this works from any checkout/worktree, not just
# the one it was written in.
WORKTREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNNER = os.path.join(WORKTREE, "deepsocflow/sim/run-sim-in-docker.sh")


def _docker_simulate(self, SIM='verilator', SIM_PATH='', TRACE=False):
    """Drop-in for Hardware.simulate(). SIM/SIM_PATH/TRACE are accepted for
    signature compatibility and ignored - the container pins the simulator."""
    if SIM != 'verilator':
        raise NotImplementedError(
            f"the container only provides verilator, not {SIM!r}")
    if TRACE:
        raise NotImplementedError(
            "waveform tracing is not wired through the container runner yet")

    run_dir = os.path.relpath(os.getcwd(), WORKTREE)
    print(f"\n[docker_sim] simulating '{run_dir}' in deepsocflow-sim:v5.024\n")
    subprocess.run([RUNNER, run_dir], check=True)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)

    target = os.path.abspath(sys.argv[1])

    sys.path.insert(0, WORKTREE)  # beat site-packages' deepsocflow.pth, which
                                  # points at a different checkout (see CLAUDE.md's
                                  # ".pth points at a different checkout" Known Issue)
    from deepsocflow.py.hardware import Hardware
    Hardware.simulate = _docker_simulate

    # The brevitas backend has its own, separate, non-inheriting Hardware class
    # (deepsocflow/py/brevitas/hardware/hardware.py) - kept separate on purpose
    # so that backend doesn't pull in the legacy TensorFlow/qkeras stack just by
    # importing Hardware. Patch it too, explicitly and unconditionally, rather
    # than leaving it to a guard in the target script that can never actually
    # tell whether this bridging is needed: importing deepsocflow.py.brevitas.
    # export already imports deepsocflow.py.hardware (and therefore
    # tensorflow) transitively via deepsocflow/__init__.py, so any such guard
    # in the target script would always fire and its "only under docker_sim.py"
    # framing would be false.
    from deepsocflow.py.brevitas.hardware.hardware import Hardware as BrevitasHardware
    BrevitasHardware.simulate = _docker_simulate

    sys.argv = sys.argv[1:]
    os.chdir(os.path.dirname(target))
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
