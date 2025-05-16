import torch
from typing import List
from optimizers import get_optimizer


class ExclusiveTargetedRecAttack:
    def __init__(self, model, attack_cfg: dict, optimizer_cfg: dict):
        self.model = model
        self.attack_cfg = attack_cfg
        self.optimizer_cfg = optimizer_cfg
        self.optimizer_class = get_optimizer(optimizer_cfg["name"])

        self.target_bbox_text = attack_cfg["target_box_text"]
        self.epsilon = optimizer_cfg["epsilon"]
        self.max_steps = optimizer_cfg["max_steps"]
        self.attack_scope = attack_cfg.get("attack_scope", "full_image")
        self.projection = optimizer_cfg.get("projection", "linf")
        self.random_start = optimizer_cfg.get("random_start", False)

        self.optimizer_params = {
            k: v for k, v in optimizer_cfg.items() if k not in ("name", "enabled")
        }

    def _project_l2(self, delta: torch.Tensor, epsilon: float) -> torch.Tensor:
        """
        Проецирует delta в L2 радиуса epsilon
        """
        flat = delta.view(delta.size(0), -1)
        norm = flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
        factor = torch.minimum(torch.ones_like(norm), epsilon / norm)
        projected = flat * factor
        return projected.view_as(delta)

    def run(self, image_batch: torch.Tensor, prompts: List[str]) -> torch.Tensor:
        B = image_batch.shape[0]
        if B == 0:
            raise ValueError("Empty image batch passed to attack.")
        device = image_batch.device

        # Инициализация delta
        delta = torch.zeros_like(image_batch, requires_grad=True).to(device)
        if self.random_start:
            delta.data.uniform_(-self.epsilon, self.epsilon)
            if self.projection == "l2":
                delta.data = self._project_l2(delta.data, self.epsilon)

        optimizer = self.optimizer_class([delta], **self.optimizer_params)

        for step in range(self.max_steps):
            optimizer.zero_grad()
            losses = []

            for i in range(B):
                # Применение возмущения к изображению
                image_i = torch.clamp(image_batch[i:i+1] + delta[i:i+1], 0, 1)

                # Создание входа для модели
                item = {
                    "image": image_i,
                    "instruction_input": [f"<Img><ImageHere></Img> {prompts[i]}"],
                    "answer": [self.target_bbox_text],
                }

                # Получение значения loss
                out = self.model.forward(item)
                losses.append(out["loss"])

            total_loss = torch.stack(losses).mean()
            total_loss.backward()
            optimizer.step()

            # Проекция delta в зависимости от выбранной нормы
            if self.projection == "l2":
                delta.data = self._project_l2(delta.data, self.epsilon)
            else:  # linf
                delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)

            if step % 10 == 0:
                print(f"[Step {step}] Loss: {total_loss.item():.4f}")

        # Финальное применение возмущения
        with torch.inference_mode():
            return torch.clamp(image_batch + delta, 0, 1)