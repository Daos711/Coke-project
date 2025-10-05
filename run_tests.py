"""Скрипт для запуска тестов."""

import subprocess
import sys
from pathlib import Path


def run_tests():
    """Запуск всех тестов."""

    # Путь к директории с тестами
    tests_dir = Path(__file__).parent / "tests"

    # Запускаем pytest
    result = subprocess.run([
        sys.executable, "-m", "pytest",
        str(tests_dir),
        "-v",  # Подробный вывод
        "--tb=short"  # Короткий формат traceback
    ])

    return result.returncode


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)