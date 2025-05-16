import torch
from optimizers import get_optimizer
from typing import Dict


class TargetedAttack:
    def __init__(self, model, tokenizer, attack_cfg: Dict, optimizer_cfg: Dict):
        """
        Args:
            model: модель с методами .forward() и .generate()
            tokenizer: токенизатор (если нужно для совместимости)
            attack_cfg: блок из cfg["attacks"]["targeted_attack"]
            optimizer_cfg: блок из cfg["optimizers"][...]
        """
        self.model = model
        self.tokenizer = tokenizer
        self.target_answer = attack_cfg["target_answer"]

        # Новые опции из конфигов
        self.attack_scope = attack_cfg.get("attack_scope", "full_image")
        self.projection = optimizer_cfg.get("projection", "linf")
        self.random_start = optimizer_cfg.get("random_start", False)

        self.optimizer_class = get_optimizer(optimizer_cfg["name"])
        self.optimizer_params = {
            k: v for k, v in optimizer_cfg.items() if k not in ("name", "enabled")
        }

        # Создание оптимизатора с расширенными параметрами
        self.optimizer = self.optimizer_class(
            model,
            targeted=True,
            projection=self.projection,
            random_start=self.random_start,
            **self.optimizer_params
        )

    def run(self, image: torch.Tensor, prompt: str) -> torch.Tensor:
        """
        Запускает целенаправленную (targeted) атаку на одно изображение.

        Args:
            image: torch.Tensor — [1, C, H, W]
            prompt: str — текстовый запрос

        Returns:
            torch.Tensor — атакованное изображение
        """
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("TargetedAttack expects a single image with shape [1, C, H, W].")

        return self.optimizer.optimize(image, prompt, self.target_answer)