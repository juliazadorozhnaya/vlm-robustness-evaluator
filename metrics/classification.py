import numpy as np
import Levenshtein

def compute_classification_metrics(predictions, metric_names):
    preds = [p["pred"] for p in predictions]
    labels = [p["label"] for p in predictions]

    results = {}
    if "accuracy" in metric_names:
        results["accuracy"] = accuracy(preds, labels)
    if "asr" in metric_names:
        results["asr"] = asr(preds, labels)
    if "levenshtein" in metric_names:
        results["levenshtein"] = levenshtein_distance_list(preds, labels)

    return results

def accuracy(preds: list[str], labels: list[str]) -> float:
    """
    Вычисляет точность: доля предсказаний, совпадающих с метками.

    Args:
        preds: список строк — предсказания модели
        labels: список строк — правильные ответы

    Returns:
        float: значение точности от 0 до 1
    """
    if not preds or not labels:
        return 0.0
    return float(np.mean([p == l for p, l in zip(preds, labels)]))


def asr(clean_preds: list[str], adv_preds: list[str]) -> float:
    """
    Attack Success Rate — насколько часто атака изменила результат.

    Args:
        clean_preds: предсказания модели на чистых данных
        adv_preds: предсказания модели на атакованных данных

    Returns:
        float: доля изменений между clean и adv
    """
    if not clean_preds or not adv_preds:
        return 0.0
    return float(np.mean([c != a for c, a in zip(clean_preds, adv_preds)]))


def levenshtein_distance_list(strs1: list[str], strs2: list[str]) -> float:
    """
    Средняя редакционная (Левенштейна) дистанция между строками.

    Args:
        strs1: список строк 1
        strs2: список строк 2

    Returns:
        float: средняя Levenshtein-дистанция
    """
    if not strs1 or not strs2:
        return 0.0
    return float(np.mean([Levenshtein.distance(s1, s2) for s1, s2 in zip(strs1, strs2)]))
