import yaml
from pathlib import Path

class DotDict(dict):
    """
    Расширенный словарь с доступом к полям через точку (cfg.models.llava).
    """
    def __getattr__(self, name):
        value = self.get(name)
        if isinstance(value, dict):
            return DotDict(value)
        return value

    def __setattr__(self, name, value):
        self[name] = value

    def __getitem__(self, key):
        value = super().__getitem__(key)
        if isinstance(value, dict):
            return DotDict(value)
        return value


def load_config(config_path: str = "config.yaml") -> DotDict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Файл конфигурации '{config_path}' не найден.")

    with open(config_path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    cfg = {}

    # === МОДЕЛИ ===
    cfg["models"] = {
        model["name"]: model
        for model in raw_cfg.get("models", [])
        if model.get("enabled", True)
    }

    # === АТАКИ ===
    cfg["attacks"] = {
        attack["name"]: attack
        for attack in raw_cfg.get("attacks", [])
        if attack.get("enabled", True)
    }

    # === ОПТИМИЗАТОРЫ ===
    cfg["optimizers"] = {
        opt["name"]: opt
        for opt in raw_cfg.get("optimizers", [])
        if opt.get("enabled", True)
    }

    # === ЗАДАЧИ ===
    cfg["tasks"] = {
        task["name"]: task
        for task in raw_cfg.get("tasks", [])
        if task.get("enabled", True)
    }

    # === НАСТРОЙКИ ОБРАБОТКИ ===
    cfg["processing"] = {
        "batch_size": raw_cfg.get("batch_size", 16),
        "device": raw_cfg.get("device", "cuda:0"),
        "seed": raw_cfg.get("seed", 42),
        "num_workers": raw_cfg.get("num_workers", 4),
        "precision": raw_cfg.get("precision", "fp16"),
        "pin_memory": raw_cfg.get("pin_memory", True)
    }

    # === НАСТРОЙКИ ВЫВОДА ===
    output_cfg = raw_cfg.get("output", {})
    cfg["output"] = {
        "enabled": output_cfg.get("enabled", True),
        "save_images": output_cfg.get("save_images", True),
        "save_logits": output_cfg.get("save_logits", False),
        "save_predictions": output_cfg.get("save_predictions", True),
        "save_metrics": output_cfg.get("save_metrics", True),
        "save_config_snapshot": output_cfg.get("save_config_snapshot", True),
        "output_dir": Path(output_cfg.get("output_dir", "outputs/benchmark_run/")),
        "export_format": output_cfg.get("export_format", ["csv", "json"]),
        "overwrite": output_cfg.get("overwrite", False),
        "log_level": output_cfg.get("log_level", "info"),
        "verbose_console": output_cfg.get("verbose_console", True),
    }

    # === Дополнительный вывод в консоль (если включено) ===
    if cfg["output"]["verbose_console"]:
        print(f"[Конфигурация] Загружены модели: {list(cfg['models'].keys())}")
        print(f"[Конфигурация] Загружены атаки: {list(cfg['attacks'].keys())}")
        print(f"[Конфигурация] Загружены оптимизаторы: {list(cfg['optimizers'].keys())}")
        print(f"[Конфигурация] Загружены задачи: {list(cfg['tasks'].keys())}")

    return DotDict(cfg)