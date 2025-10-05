# -*- coding: utf-8 -*-
import subprocess
import sys
from pathlib import Path

STEPS = [
    "examples/step01_energy_only.py",
    "examples/step02_momentum_brinkman.py",
    "examples/step03_brinkman_radial.py",
    "examples/step04_energy_with_brinkman.py",
    "examples/step05_simplec_stokes.py",
    "examples/step06_coupled_energy_brinkman.py",
    "examples/step07_coupled_energy_brinkman_radial.py",
    "examples/step08_porosity_kinetics.py",
    "examples/step09_two_temp_energy.py",
]

def main():
    root = Path(__file__).resolve().parent
    for step in STEPS:
        script = root / step
        print("\n" + "="*100)
        print(f"RUNNING: {script}")
        print("="*100)
        result = subprocess.run([sys.executable, str(script)])
        if result.returncode != 0:
            print(f"FAILED: {script} (exit {result.returncode})")
            sys.exit(result.returncode)
    print("\nAll steps completed successfully.")

if __name__ == "__main__":
    main()
