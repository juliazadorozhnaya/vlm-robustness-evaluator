import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

def compute_semantic_metrics(predictions, metric_names):
    half = len(predictions) // 2
    clean_preds = predictions[:half]
    adv_preds = predictions[half:]

    results = {}

    if "semantic_stability" in metric_names:
        clean_embeds = [p["embedding"] for p in clean_preds]
        adv_embeds = [p["embedding"] for p in adv_preds]
        results["semantic_stability"] = semantic_stability(clean_embeds, adv_embeds)

    if "embedding_distance" in metric_names:
        clean_embeds = [p["embedding"] for p in clean_preds]
        adv_embeds = [p["embedding"] for p in adv_preds]
        results["embedding_distance"] = embedding_distance(clean_embeds, adv_embeds)

    return results

def semantic_stability(clean_embeds, adv_embeds):
    """
    Возвращает среднее косинусное сходство между чистыми и атакованными эмбеддингами.

    Args:
        clean_embeds: список или массив эмбеддингов оригинальных объектов
        adv_embeds: список или массив эмбеддингов после атаки

    Returns:
        float: среднее косинусное сходство (от -1 до 1)
    """
    assert len(clean_embeds) == len(adv_embeds), "Количество эмбеддингов не совпадает"
    sims = [cosine_similarity(c.reshape(1, -1), a.reshape(1, -1))[0, 0]
            for c, a in zip(clean_embeds, adv_embeds)]
    return float(np.mean(sims))

def embedding_distance(clean_embeds, adv_embeds):
    """
    Возвращает среднее L2-расстояние между эмбеддингами до и после атаки.

    Args:
        clean_embeds: список или массив эмбеддингов оригинальных объектов
        adv_embeds: список или массив эмбеддингов после атаки

    Returns:
        float: средняя евклидова дистанция
    """
    assert len(clean_embeds) == len(adv_embeds), "Количество эмбеддингов не совпадает"
    diffs = [np.linalg.norm(c - a) for c, a in zip(clean_embeds, adv_embeds)]
    return float(np.mean(diffs))
