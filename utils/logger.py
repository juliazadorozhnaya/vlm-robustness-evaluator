import logging
import os
from pathlib import Path
from typing import Optional


def setup_logger(name: str, log_dir: Optional[str] = None, level=logging.INFO) -> logging.Logger:
    """
    Создаёт и возвращает логгер с указанным именем.

    Args:
        name (str): имя логгера
        log_dir (str, optional): если указано — лог будет также писаться в файл
        level (int): уровень логирования (по умолчанию INFO)

    Returns:
        logging.Logger: настроенный логгер
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Предотвращаем повторное добавление хендлеров
    if logger.hasHandlers():
        return logger

    # Формат логов
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")

    # === Консольный вывод ===
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # === Файловый лог ===
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_path = Path(log_dir) / f"{name}.log"
        file_handler = logging.FileHandler(file_path, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
