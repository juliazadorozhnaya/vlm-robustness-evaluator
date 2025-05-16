from pathlib import Path
from typing import Dict, Any

from hub.wrapper import ModelWrapper
from models import MODEL_REGISTRY


def load_model(model_cfg: Dict[str, Any]) -> ModelWrapper:
    """
    Загружает и оборачивает модель по конфигурации.

    Аргументы:
        model_cfg: словарь конфигурации модели:
            - name: имя модели
            - type: ключ в MODEL_REGISTRY
            - path: путь или HuggingFace repo
            - device: cuda:0 / cpu

    Возвращает:
        ModelWrapper — обёртка над моделью.
    """
    model_name = model_cfg["name"]
    model_type = model_cfg["type"]
    model_path = model_cfg["path"]
    device = model_cfg.get("device", "cuda:0")

    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"[load_model] Неизвестный тип модели: {model_type}")

    path_obj = Path(model_path)
    if not path_obj.exists():
        print(f"[load_model] Используем модель из HuggingFace: {model_path}")
    else:
        model_path = str(path_obj.resolve())

    model_class = MODEL_REGISTRY[model_type]
    model_instance = model_class(model_path=model_path, device=device)
    return ModelWrapper(name=model_name, model=model_instance)
