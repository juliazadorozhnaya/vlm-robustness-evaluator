from pathlib import Path
import os
import json
import csv
from typing import List, Dict, Any

import torch
import pandas as pd
from metrics import METRIC_REGISTRY


def compute_metrics(metric_names: List[str], clean_outputs, adv_outputs, targets, **kwargs) -> Dict[str, float]:
    """
    Вычисляет все указанные метрики по именам из METRIC_REGISTRY.
    """
    results = {}
    for name in metric_names:
        metric_fn = METRIC_REGISTRY[name]
        try:
            results[name] = metric_fn(clean_outputs, adv_outputs, targets, **kwargs)
        except TypeError:
            results[name] = metric_fn(clean_outputs, adv_outputs)
    return results


def save_config_snapshot(config: dict, path: Path):
    """
    Сохраняет полный YAML-конфиг в JSON-формате.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def save_metrics(metrics: Dict[str, float], task_name: str, attack_name: str, config: dict):
    """
    Сохраняет метрики в JSON и CSV, а также снапшот конфига при необходимости.
    """
    if not config["output"].get("enabled", True) or not config["output"].get("save_metrics", True):
        return
    if not metrics:
        print(f"No metrics to save for {task_name}/{attack_name}")
        return

    output_dir = Path(config["output"]["output_dir"]) / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    row = {
        "task": task_name,
        "attack": attack_name,
        **metrics
    }

    if config["output"].get("overwrite", False):
        for fmt in config["output"].get("export_format", []):
            path = output_dir / f"{task_name}__{attack_name}.{fmt}"
            if path.exists():
                path.unlink()

    if config["output"].get("save_config_snapshot", False):
        snapshot_path = output_dir / f"{task_name}__{attack_name}__config.json"
        save_config_snapshot(config, snapshot_path)

    for fmt in config["output"].get("export_format", []):
        if fmt == "json":
            json_path = output_dir / f"{task_name}__{attack_name}.json"
            with open(json_path, "w") as f:
                json.dump(row, f, indent=2)

        elif fmt == "csv":
            csv_path = output_dir / f"{task_name}__{attack_name}.csv"
            write_header = not csv_path.exists() or config["output"].get("overwrite", False)
            with open(csv_path, "a", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if write_header:
                    writer.writeheader()
                writer.writerow(row)

    if config["output"].get("verbose_console", False):
        print(f"\nMetrics saved for task '{task_name}' and attack '{attack_name}':")
        for k, v in metrics.items():
            print(f"  - {k}: {v:.4f}")

def save_predictions(
    output_cfg: Dict[str, Any],
    model_name: str,
    task_name: str,
    data: List[Dict[str, Any]],
    is_adv: bool = False
):
    """
    Сохраняет предсказания модели (clean / adv) в JSON и CSV.

    Args:
        output_cfg: конфигурация вывода из YAML
        model_name: имя модели
        task_name: имя задачи
        data: список словарей с предсказаниями
        is_adv: если True — сохраняем атакованные (adv) предсказания
    """
    if not output_cfg.get("enabled", True):
        return

    if not output_cfg.get("save_predictions", True):
        return

    if not data or not isinstance(data, list) or not isinstance(data[0], dict):
        print(f"Invalid prediction data format for {model_name}/{task_name} ({'adv' if is_adv else 'clean'})")
        return

    subfolder = "adv" if is_adv else "clean"
    output_dir = os.path.join(output_cfg["output_dir"], task_name, model_name, subfolder)
    os.makedirs(output_dir, exist_ok=True)

    # === Сохраняем JSON ===
    json_path = os.path.join(output_dir, "predictions.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # === Сохраняем CSV ===
    csv_path = os.path.join(output_dir, "predictions.csv")
    fieldnames = sorted(data[0].keys())
    with open(csv_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    if output_cfg.get("verbose_console", False):
        print(f"Predictions saved to:")
        print(f"    ├─ JSON: {json_path}")
        print(f"    └─ CSV : {csv_path}")


def save_logits(output_cfg, model_name, task_name, logits, is_adv=False):
    """
    Сохраняет логиты модели в формате .pt
    """
    if not output_cfg.get("enabled", True) or not output_cfg.get("save_logits", False):
        return

    subfolder = "adv" if is_adv else "clean"
    output_dir = os.path.join(output_cfg["output_dir"], task_name, model_name, subfolder)
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(output_dir, "logits.pt")
    torch.save(logits, path)


def save_embeddings(output_cfg, model_name, task_name, embeddings, is_adv=False):
    """
    Сохраняет эмбеддинги в формате .pt
    """
    if not output_cfg.get("enabled", True) or not output_cfg.get("save_predictions", True):
        return

    subfolder = "adv" if is_adv else "clean"
    output_dir = os.path.join(output_cfg["output_dir"], task_name, model_name, subfolder)
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(output_dir, "embeddings.pt")
    torch.save(embeddings, path)


def save_embedding_metadata(output_cfg, model_name, task_name, metadata: List[dict], is_adv=False):
    """
    Сохраняет метаинформацию эмбеддингов (например, id, вопрос, bbox и т.д.).
    """
    if not output_cfg.get("enabled", True):
        return

    subfolder = "adv" if is_adv else "clean"
    output_dir = os.path.join(output_cfg["output_dir"], task_name, model_name, subfolder)
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(output_dir, "embedding_meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def aggregate_all_metrics(output_dir: str, filename: str = "all_results.csv"):
    """
    Сканирует директорию metrics/ и агрегирует все CSV-файлы в единый.
    """
    metrics_dir = Path(output_dir) / "metrics"
    if not metrics_dir.exists():
        print(f"Metrics directory not found: {metrics_dir}")
        return

    all_rows = []
    for file in metrics_dir.glob("*.csv"):
        try:
            df = pd.read_csv(file)
            all_rows.append(df)
        except Exception as e:
            print(f"Failed to read {file}: {e}")

    if not all_rows:
        print("No metric files found for aggregation.")
        return

    merged = pd.concat(all_rows, ignore_index=True)
    merged_path = metrics_dir / filename
    merged.to_csv(merged_path, index=False)
    print(f"Aggregated metrics saved to {merged_path}")

def save_flat_adv_predictions(
    output_root: str,
    optimizer: str,
    model_name: str,
    task_name: str,
    attack_name: str,
    prediction_data: dict
):
    """
    Сохраняет атакованные предсказания в плоскую структуру:
    <output_root>/<optimizer>/<model>/<task>/predictions_adv_<attack>.json
    """
    out_path = Path(output_root) / optimizer / model_name / task_name
    out_path.mkdir(parents=True, exist_ok=True)

    file_path = out_path / f"predictions_adv_{attack_name}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(prediction_data, f, indent=2, ensure_ascii=False)

    print(f"[✓] Saved: {file_path}")
