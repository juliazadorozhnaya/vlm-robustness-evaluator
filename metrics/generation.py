from bert_score import score as bert_score_func
from nltk.translate.bleu_score import sentence_bleu
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
import Levenshtein
from pycocoevalcap.cider.cider import Cider

def compute_generation_metrics(predictions, metric_names):
    preds = [p["pred"] for p in predictions]
    refs = [p["label"] for p in predictions]
    images = [p.get("image") for p in predictions]
    clip_model = predictions[0].get("clip_model") if predictions and "clipscore" in metric_names else None

    results = {}
    if "levenshtein" in metric_names:
        results["levenshtein"] = levenshtein_distance_list(preds, refs)
    if "cider" in metric_names:
        results["cider"] = cider(preds, refs)
    if "bertscore" in metric_names:
        results["bertscore"] = bertscore(preds, refs)
    if "clipscore" in metric_names and clip_model is not None:
        results["clipscore"] = clipscore(preds, images, clip_model)

    return results

def cider(preds, refs):
    """
    Вычисляет CIDER-метрику между предсказаниями и референсами.
    """
    gts = {i: [r] for i, r in enumerate(refs)}  # ground truths
    res = {i: [p] for i, p in enumerate(preds)}  # model outputs
    scorer = Cider()
    score, _ = scorer.compute_score(gts, res)
    return score

def bertscore(preds, refs, lang='en'):
    """
    BERTScore — семантическое сходство между строками.
    """
    P, R, F1 = bert_score_func(preds, refs, lang=lang, rescale_with_baseline=True)
    return float(F1.mean())

def clipscore(preds, images, clip_model):
    """
    CLIPScore — косинусное сходство между текстом и изображением.
    Требует clip_model с методами .get_text_features() и .get_image_features()
    """
    with torch.no_grad():
        text_feats = clip_model.get_text_features(preds)  # ожидается batch обработка строк
        image_feats = clip_model.get_image_features(images)
        sim = torch.cosine_similarity(text_feats, image_feats, dim=-1)
        return float(sim.mean() * 100)

def levenshtein_distance_list(strs1, strs2):
    """
    Средняя редакционная дистанция (Левенштейна) между парами строк.
    """
    if not strs1 or not strs2:
        return 0.0
    return np.mean([Levenshtein.distance(a, b) for a, b in zip(strs1, strs2)])
