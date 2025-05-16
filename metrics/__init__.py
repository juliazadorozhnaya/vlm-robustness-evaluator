from .classification import accuracy, asr, levenshtein_distance_list as levenshtein
from .localization import iou
from .generation import cider, bertscore, clipscore
from .semantic import semantic_stability, embedding_distance

METRIC_REGISTRY = {
    # Classification
    "accuracy": accuracy,
    "asr": asr,
    "levenshtein": levenshtein,

    # Localization
    "iou": iou,

    # Generation
    "cider": cider,
    "bertscore": bertscore,
    "clipscore": clipscore,

    # Semantic
    "semantic_stability": semantic_stability,
    "embedding_distance": embedding_distance,
}
