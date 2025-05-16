import torch
import torch.nn.functional as F


class CWOptimizer:
    def __init__(self, model, targeted=True, c=1.0, lr=5e-3, max_steps=100):
        """
        Упрощённая реализация атаки Карлини-Вагнера (CW) с L2-нормой и фиксированным параметром c.

        Args:
            model: атакуемая мультимодальная модель (обёрнутая с методом forward и generate)
            targeted: тип атаки (True — целевая, False — нецелевая)
            c: коэффициент важности основной цели в функции потерь (эквивалент c_init в конфиге)
            lr: шаг оптимизации
            max_steps: число итераций градиентного спуска
        """
        self.model = model
        self.targeted = targeted
        self.c = c
        self.lr = lr
        self.max_steps = max_steps

    def _loss_fn(self, logits, target_ids):
        """
        Стандартная cross-entropy потеря между логитами и целевыми токенами.
        """
        log_probs = F.log_softmax(logits, dim=-1)
        loss = F.nll_loss(
            log_probs[:, :target_ids.size(1)].reshape(-1, log_probs.size(-1)),
            target_ids.view(-1),
        )
        return loss

    def optimize(self, image: torch.Tensor, prompt: str, target_text: str):
        """
        Запуск CW-атаки на одно изображение.

        Args:
            image: [C, H, W] — входное изображение
            prompt: str — текстовый запрос к модели
            target_text: str — желаемый целевой ответ

        Returns:
            adv_image: атакованное изображение
            generated_output: текст, сгенерированный моделью по результату
        """
        image = image.clone().detach().unsqueeze(0).to(next(self.model.model.parameters()).device)
        delta = torch.zeros_like(image, requires_grad=True)

        target_ids = self.model.tokenize(target_text).to(image.device)
        optimizer = torch.optim.Adam([delta], lr=self.lr)

        for step in range(self.max_steps):
            adv_image = torch.clamp(image + delta, 0, 1)

            # Получаем логиты от модели
            logits = self.model.model(adv_image, prompt)
            ce_loss = self._loss_fn(logits, target_ids)

            # Инвертируем знак, если атака нецелевая
            if not self.targeted:
                ce_loss = -ce_loss

            # Основной loss = цель + регуляризация на L2-норму
            loss = self.c * ce_loss + torch.norm(delta.view(delta.size(0), -1), p=2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_image = torch.clamp(image + delta.detach(), 0, 1)
        return final_image, self.model.generate(final_image, prompt)

    def optimize_rec(self, image_batch: torch.Tensor, prompts: list[str], target_text: str):
        """
        CW-атака на батч изображений (например, bbox-атака с фиксированным target_text).

        Args:
            image_batch: [B, C, H, W] — батч изображений
            prompts: список текстовых запросов
            target_text: фиксированный строковый целевой ответ (одинаковый для всех)

        Returns:
            torch.Tensor: батч атакованных изображений
        """
        device = next(self.model.model.parameters()).device
        image_batch = image_batch.to(device)

        B = len(prompts)
        delta = torch.zeros_like(image_batch, requires_grad=True).to(device)
        optimizer = torch.optim.Adam([delta], lr=self.lr)

        for step in range(self.max_steps):
            losses = []

            for i in range(B):
                # Создаём input для forward() модели
                item = {
                    "image": torch.clamp(image_batch[i:i + 1] + delta[i:i + 1], 0, 1),
                    "instruction_input": ["<Img><ImageHere></Img> {}".format(prompts[i])],
                    "answer": [target_text],
                }
                loss = self.model.forward(item)['loss']
                losses.append(loss)

            # Средний loss по батчу + регуляризация
            total_loss = torch.stack(losses).mean()
            loss = self.c * total_loss + torch.norm(delta.view(delta.size(0), -1), p=2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return torch.clamp(image_batch + delta.detach(), 0, 1)