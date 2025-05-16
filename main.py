import torch
import argparse
from pathlib import Path

from configs.config_loader import load_config
from hub.load_model import load_model
from data.data_loader import get_dataset_loader
from controller.main_controller import run_all_tasks
from utils.logger import setup_logger


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    # === Парсинг CLI-параметров ===
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Путь к YAML конфигу")
    parser.add_argument("--output_dir", type=str, default=None, help="Папка для сохранения результатов")
    parser.add_argument("--log_level", type=str, default="info", help="Уровень логирования: debug/info/warning")
    parser.add_argument("--attack_only", action="store_true", help="Запуск только этапа атаки")
    parser.add_argument("--inference_only", action="store_true", help="Запуск только инференса")
    parser.add_argument("--eval_only", action="store_true", help="Запуск только подсчёта метрик")
    args = parser.parse_args()

    # === Загрузка и настройка конфигурации ===
    config = load_config(args.config)

    if args.output_dir:
        config["output"]["output_dir"] = Path(args.output_dir)

    config["output"]["log_level"] = args.log_level.lower()

    # Флаги выполнени
    config["execution"] = {
        "attack_only": args.attack_only,
        "inference_only": args.inference_only,
        "eval_only": args.eval_only,
    }

    # === Инициализация логгера ===
    logger = setup_logger("main", level=args.log_level.upper())
    logger.info("=== Старт VLM Robustness Evaluator ===")

    # === Загрузка моделей из конфигурации ===
    logger.info("Загрузка моделей...")
    model_registry = {
        model_name: load_model(model_cfg)
        for model_name, model_cfg in config["models"].items()
    }

    # === Загрузка датасетов ===
    logger.info("Загрузка датасетов...")
    datasets = {
        task_name: get_dataset_loader(task_cfg, config["processing"])
        for task_name, task_cfg in config["tasks"].items()
    }

    # === Запуск основного контроллера ===
    run_all_tasks(
        config=config,
        model_registry=model_registry,
        datasets=datasets
    )
