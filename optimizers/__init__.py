from .pgd import PGDOptimizer
from .apgd import APGDOptimizer
from .cw import CWOptimizer


def get_optimizer(name: str):
    """
    Возвращает класс оптимизатора по имени из конфигурации.

    Args:
        name: строковое имя оптимизатора ("pgd", "apgd", "cw")

    Returns:
        Класс соответствующего оптимизатора

    Raises:
        ValueError: если имя не распознано
    """
    name = name.lower()
    if name == "pgd":
        return PGDOptimizer
    elif name == "apgd":
        return APGDOptimizer
    elif name == "cw":
        return CWOptimizer
    else:
        raise ValueError(f"Неизвестное имя оптимизатора: '{name}'. Допустимые: pgd, apgd, cw")