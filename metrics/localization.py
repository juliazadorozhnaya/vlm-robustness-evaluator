import numpy as np

def compute_localization_metrics(predictions, metric_names):
    boxes1 = [p["pred"] for p in predictions]
    boxes2 = [p["label"] for p in predictions]

    results = {}
    if "iou" in metric_names:
        results["iou"] = iou(boxes1, boxes2)

    return results

def iou(boxes1, boxes2):
    """
    Вычисляет среднее IoU (intersection over union) между парами ограничивающих рамок.

    Args:
        boxes1, boxes2: списки строк в формате '{<x1><y1><x2><y2>}' или списков/кортежей [x1, y1, x2, y2]

    Returns:
        float: среднее значение IoU по всем парам
    """

    def parse(box):
        # Если bbox — уже список или кортеж из 4 чисел
        if isinstance(box, (list, tuple)) and len(box) == 4:
            return list(map(int, box))

        # Если строка — пытаемся извлечь координаты
        if isinstance(box, str):
            box = box.replace('{', '').replace('}', '')    # Удаляем внешние фигурные скобки
            box = box.replace('<', '').replace('>', ' ')   # Убираем < > и добавим пробелы
            parts = box.strip().split()
            if len(parts) != 4:
                raise ValueError(f"Неверный формат строки: '{box}'")
            return list(map(int, parts))

        raise TypeError(f"Неподдерживаемый формат bbox: {box}")

    def iou_single(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0

    # Применяем IoU к каждой паре (после парсинга)
    parsed = [(parse(a), parse(b)) for a, b in zip(boxes1, boxes2)]
    return float(np.mean([iou_single(a, b) for a, b in parsed]))
