from pathlib import Path
import json
from typing import List, Dict, Any
from classification import compute_classification_metrics
from generation import compute_generation_metrics
from localization import compute_localization_metrics
from semantic import compute_semantic_metrics

class MetricsEvaluator:
    def __init__(self, output_cfg: Dict[str, Any], output_path: Path):
        self.output_cfg = output_cfg
        self.output_path = output_path
        self.results = {}

    def evaluate(
            self,
            task_name: str,
            task_cfg: Dict[str, Any],
            predictions_clean: List[Dict],
            predictions_adv: List[Dict]
    ) -> Dict[str, Dict[str, float]]:
        metrics = task_cfg["metrics"]
        task_type = task_cfg["task_type"]

        if task_type == "classification":
            clean_scores = compute_classification_metrics(predictions_clean, metrics)
            adv_scores = compute_classification_metrics(predictions_adv, metrics)
        elif task_type == "generation":
            clean_scores = compute_generation_metrics(predictions_clean, metrics)
            adv_scores = compute_generation_metrics(predictions_adv, metrics)
        elif task_type == "localization":
            clean_scores = compute_localization_metrics(predictions_clean, metrics)
            adv_scores = compute_localization_metrics(predictions_adv, metrics)
        elif task_type == "semantic":
            clean_scores = compute_semantic_metrics(predictions_clean, metrics)
            adv_scores = compute_semantic_metrics(predictions_adv, metrics)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        self.results = {
            "clean": clean_scores,
            "adv": adv_scores
        }

        if self.output_cfg.get("save_metrics", True):
            self._save_metrics(task_name)

        return self.results

    def _save_metrics(self, task_name: str):
        output_dir = Path(self.output_cfg.get("output_dir", "outputs/benchmark_run/"))
        export_formats = self.output_cfg.get("export_format", ["json"])
        metrics_path = output_dir / f"{task_name}_metrics"

        output_dir.mkdir(parents=True, exist_ok=True)

        if "json" in export_formats:
            with open(metrics_path.with_suffix(".json"), "w") as f:
                json.dump(self.results, f, indent=2)

        if "csv" in export_formats:
            import csv
            with open(metrics_path.with_suffix(".csv"), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Metric", "Clean", "Adversarial"])
                for k in self.results["clean"]:
                    writer.writerow([k, self.results["clean"][k], self.results["adv"].get(k, None)])
