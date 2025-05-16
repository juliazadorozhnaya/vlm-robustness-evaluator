def asr(clean_outputs, adv_outputs, *args, **kwargs):
    """
    Attack Success Rate (ASR): измеряет долю примеров, для которых
    предсказание изменилось после атаки.

    Args:
        clean_outputs (List[str] или List[int]): Предсказания модели на чистых примерах
        adv_outputs (List[str] или List[int]): Предсказания на атакованных примерах

    Returns:
        float: значение ASR ∈ [0, 1]
    """
    assert len(clean_outputs) == len(adv_outputs), "Clean and adversarial lists must have same length"
    changed = [c != a for c, a in zip(clean_outputs, adv_outputs)]
    return sum(changed) / len(changed) if changed else 0.0
