"""Настройка логирования (цвет в консоли опционально, файл — по желанию)."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

# colorlog не обязателен
try:
    import colorlog  # type: ignore
    _HAS_COLORLOG = True
except Exception:  # noqa: BLE001
    _HAS_COLORLOG = False


def default_logfile(log_dir: Union[str, Path] = "logs") -> Path:
    """
    Вернуть путь к файлу логов вида logs/run_YYYYmmdd_HHMMSS.log.
    Директория создаётся при необходимости.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"run_{stamp}.log"


def setup_logging(
    log_level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    logger_name: str = "",  # "" = корневой логгер
    propagate: bool = False,
) -> logging.Logger:
    """
    Настроить логирование.

    log_level  — уровень логов (logging.INFO и т.д.)
    log_file   — путь к .log файлу; если None — файл не создаётся;
                 если строка 'auto' — создаётся в ./logs/run_*.log
    logger_name — имя логгера (по умолчанию корневой)
    propagate — прокидывать сообщения вверх по иерархии
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.propagate = propagate

    # удалить старые хендлеры, чтобы не дублировать вывод при повторном вызове
    for h in list(logger.handlers):
        logger.removeHandler(h)

    # ---- консоль ----
    console = logging.StreamHandler()
    if _HAS_COLORLOG:
        console.setFormatter(
            colorlog.ColoredFormatter(
                "%(log_color)s%(levelname)-8s%(reset)s %(cyan)s%(name)s%(reset)s: %(message)s",
                reset=True,
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "red,bg_white",
                },
                style="%",
            )
        )
    else:
        console.setFormatter(logging.Formatter("%(levelname)-8s %(name)s: %(message)s"))
    logger.addHandler(console)

    # ---- файл (опционально) ----
    if isinstance(log_file, (str, Path)) or (isinstance(log_file, str) and log_file == "auto"):
        path = default_logfile() if str(log_file) == "auto" else Path(log_file)  # type: ignore[arg-type]
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s %(name)s: %(message)s")
        )
        logger.addHandler(file_handler)

    return logger
