import os
from pathlib import Path
from typing import Dict, Any
from data.datasets import captioning, refcoco, scienceqa, vqa

# Словарь: имя задачи → модуль с загрузкой
DATASET_LOADERS = {
    "image_captioning": captioning,
    "referring_expression": refcoco,
    "visual_commonsense_reasoning": scienceqa,
    "vqa": vqa
}


def get_dataset_loader(task_cfg: Dict[str, Any], processing_cfg: Dict[str, Any]):
    """
    Загружает датасет по имени из конфигурации. Если датасета нет локально — скачивает его.

    Args:
        task_cfg: конфигурация задачи из YAML (dataset, max_samples и т.п.)
        processing_cfg: общие параметры обработки (batch_size, device и т.д.)

    Returns:
        torch.utils.data.DataLoader
    """
    task_name = task_cfg["name"]
    dataset_name = task_cfg["dataset"]

    if task_name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset task: {task_name}")

    # Путь к корню данных
    root_dir = Path("datasets") / dataset_name
    if not root_dir.exists():
        print(f"Dataset '{dataset_name}' not found locally. Attempting to download...")
        os.makedirs(root_dir, exist_ok=True)
        # вызываем download внутри соответствующего модуля
        DATASET_LOADERS[task_name].download(root_dir)

    # загружаем готовый датасет через модуль
    dataset = DATASET_LOADERS[task_name].load_dataset(root_dir, task_cfg)

    from torch.utils.data import DataLoader
    return DataLoader(
        dataset,
        batch_size=processing_cfg.get("batch_size", 16),
        num_workers=processing_cfg.get("num_workers", 4),
        pin_memory=processing_cfg.get("pin_memory", True),
        shuffle=False
    )
