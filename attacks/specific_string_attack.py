import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple

from optimizers.random_patch import RandomPatchImageProcessor
from transformers import PreTrainedTokenizer


class SpecificStringAttack:
    def __init__(
        self,
        model,
        tokenizer: PreTrainedTokenizer,
        processor: RandomPatchImageProcessor,
        attack_cfg: dict,
        optimizer_cfg: dict,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor

        self.target_string = attack_cfg["target_string"]
        self.epsilon = optimizer_cfg["epsilon"]
        self.iterations = optimizer_cfg["max_steps"]
        self.lr = optimizer_cfg["step_size"]

        # Новое: поддержка дополнительных параметров
        self.projection = optimizer_cfg.get("projection", "linf")
        self.random_start = optimizer_cfg.get("random_start", False)
        self.attack_scope = attack_cfg.get("attack_scope", "patch_based")

        # Инициализация патча (random start)
        if self.random_start:
            self.processor.learned_patch.data.uniform_(
                self.processor.init_patch - self.epsilon,
                self.processor.init_patch + self.epsilon,
            )
            if self.projection == "l2":
                self.processor.learned_patch.data = self._project_l2(
                    self.processor.learned_patch.data, self.epsilon
                )

        self.optimizer = torch.optim.SGD([self.processor.learned_patch], lr=self.lr)

        with torch.no_grad():
            self.target_ids = tokenizer(self.target_string, return_tensors="pt")["input_ids"].to(
                self.processor.learned_patch.device
            )

    def _project_l2(self, patch: torch.Tensor, epsilon: float) -> torch.Tensor:
        """
        Проекция L2-нормы относительно исходного патча.
        """
        delta = patch - self.processor.init_patch
        flat = delta.view(-1)
        norm = torch.norm(flat, p=2).clamp(min=1e-8)
        factor = min(1.0, epsilon / norm.item())
        projected = self.processor.init_patch + delta * factor
        return projected.clamp(0, 1)

    def run(self, clean_images: Tensor, prompts: List[str]) -> Tuple[Tensor, List[str]]:
        B = clean_images.shape[0]

        for step in range(self.iterations):
            self.optimizer.zero_grad()

            patched_images = self.processor(clean_images)
            total_loss = 0.0

            for i in range(B):
                image_i = patched_images[i:i+1]
                prompt_i = prompts[i]

                logits = self.model(image_i, prompt_i)
                log_probs = F.log_softmax(logits, dim=-1)

                loss = F.nll_loss(
                    log_probs[:, : self.target_ids.size(1)].reshape(-1, log_probs.size(-1)),
                    self.target_ids.view(-1),
                )
                total_loss += loss

            avg_loss = total_loss / B

            if step % 10 == 0:
                print(f"[Step {step}] Avg Loss: {avg_loss.item():.4f}")

            avg_loss.backward()
            self.optimizer.step()

            # Проекция патча в допустимую область по выбранной норме
            if self.projection == "l2":
                self.processor.learned_patch.data = self._project_l2(
                    self.processor.learned_patch.data, self.epsilon
                )
            else:  # linf
                self.processor.learned_patch.data = torch.clamp(
                    self.processor.learned_patch.data,
                    min=self.processor.init_patch - self.epsilon,
                    max=self.processor.init_patch + self.epsilon,
                ).clamp_(0, 1)

        # Генерация вывода после атаки
        final_images = self.processor(clean_images)
        outputs = []
        for i in range(B):
            result = self.model.generate(final_images[i:i+1], prompts[i])
            outputs.append(result)

        return final_images, outputs