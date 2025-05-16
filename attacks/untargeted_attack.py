import torch
from typing import Dict
from optimizers import get_optimizer

class UntargetedAttack:
    def __init__(self, model, tokenizer, attack_cfg: Dict, optimizer_cfg: Dict):
        """
        Args:
            model: мультимодальная модель с .forward() и .generate()
            tokenizer: токенизатор (если требуется — например, для генерации целевых токенов)
            attack_cfg: словарь конфигурации атаки (из cfg["attacks"]["untargeted_attack"])
            optimizer_cfg: словарь конфигурации оптимизатора (из cfg["optimizers"][...])
        """
        self.model = model
        self.tokenizer = tokenizer

        self.attack_scope = attack_cfg.get("attack_scope", "full_image")
        self.projection = optimizer_cfg.get("projection", "linf")
        self.random_start = optimizer_cfg.get("random_start", False)

        self.optimizer_class = get_optimizer(optimizer_cfg["name"])
        self.optimizer_params = {
            k: v for k, v in optimizer_cfg.items() if k not in ("name", "enabled")
        }

        # Создание оптимизатора с дополнительными параметрами
        self.optimizer = self.optimizer_class(
            model,
            targeted=False,
            projection=self.projection,
            random_start=self.random_start,
            **self.optimizer_params
        )

    def run(self, image: torch.Tensor, prompt: str, clean_answer: str) -> torch.Tensor:
        """
        Запускает нецеленаправленную (untargeted) атаку на изображение.

        Args:
            image: torch.Tensor — одно изображение в форме [1, C, H, W]
            prompt: str — текстовый запрос (например, вопрос в задаче VQA)
            clean_answer: str — правильный (оригинальный) ответ, от которого нужно отклонить модель

        Returns:
            torch.Tensor — атакованное изображение, провоцирующее неверный ответ
        """
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("UntargetedAttack expects a single image with shape [1, C, H, W].")

        return self.optimizer.optimize(image, prompt, clean_answer)