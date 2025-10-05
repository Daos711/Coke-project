from pathlib import Path
import sys

def main():
    # позволим импортировать examples/*
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))
    from examples.check_system_ready import *  # noqa
    # просто запустим скрипт, как если бы его вызвали напрямую
    # (в нём всё печатается в stdout)
    # Ничего делать не нужно — импорт уже выполняет логику

if __name__ == "__main__":
    main()