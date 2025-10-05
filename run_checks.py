# -*- coding: utf-8 -*-
import os
import subprocess
import sys
from pathlib import Path

CHECKS = [
    "examples/check_axisymmetric_geometry.py",
    "examples/check_bc.py",
    "examples/check_correlations.py",
    "examples/check_discretization.py",
    "examples/check_geometry_consistency.py",
    "examples/check_kinetics.py",
    "examples/check_properties.py",
    "examples/check_system_ready.py",
    "examples/test_fvm_validation.py",
    "examples/test_grid_visualization.py",
]

def main():
    root = Path(__file__).resolve().parent
    # чутка защиты от проблем кодировки/GUI
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("MPLBACKEND", "Agg")

    for check in CHECKS:
        script = root / check
        print("\n" + "="*100)
        print(f"RUNNING: {script}")
        print("="*100)
        result = subprocess.run([sys.executable, str(script)], env=env)
        if result.returncode != 0:
            print(f"FAILED: {script} (exit {result.returncode})")
            sys.exit(result.returncode)
    print("\nAll checks completed successfully.")

if __name__ == "__main__":
    main()
