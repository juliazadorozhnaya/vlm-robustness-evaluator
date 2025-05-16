import torch

class APGDOptimizer:
    """
    Оптимизатор Auto-PGD (APGD) для целевых и нецелевых атак.
    Ссылка: Croce & Hein (2020) - https://arxiv.org/abs/2003.01690
    """

    def __init__(self, model, targeted: bool, epsilon: float, step_size: float, max_steps: int,
                 projection: str = "linf", norm_decay: float = 0.75):
        self.model = model                      # атакуемая модель
        self.targeted = targeted                # тип атаки: targeted или нет
        self.epsilon = epsilon                  # допустимое возмущение
        self.step_size = step_size              # шаг оптимизации
        self.max_steps = max_steps              # максимальное количество итераций
        self.projection = projection            # способ проекции (linf или l2)
        self.norm_decay = norm_decay            # уменьшение шага при стагнации

    def _project(self, delta, original):
        """
        Ограничение возмущения delta в пределах epsilon по выбранной норме
        """
        if self.projection == "linf":
            delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)
        elif self.projection == "l2":
            delta_flat = delta.view(delta.size(0), -1)
            norm = delta_flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
            factor = (norm / self.epsilon).clamp(min=1.0)
            delta.data = (delta_flat / factor).view_as(delta)
        else:
            raise ValueError(f"Неизвестная проекция: {self.projection}")
        return delta

    def optimize(self, image: torch.Tensor, prompt: str, target_text: str):
        """
        Атака одного изображения: заставить модель выдать target_text
        """
        device = image.device
        delta = torch.zeros_like(image, requires_grad=True).to(device)
        optimizer = torch.optim.SGD([delta], lr=self.step_size)

        best_loss = None
        best_delta = delta.data.clone()
        stagnation_counter = 0

        for step in range(self.max_steps):
            optimizer.zero_grad()

            # Подготавливаем вход с добавленным возмущением
            item = {
                "image": torch.clamp(image + delta, 0, 1),
                "instruction_input": [f"<Img><ImageHere></Img> {prompt}"],
                "answer": [target_text],
            }

            # Вычисляем loss
            loss = self.model.forward(item)['loss']
            if not self.targeted:
                loss = -loss  # инвертируем loss для untargeted атаки

            loss.backward()
            optimizer.step()
            self._project(delta, image)  # ограничиваем возмущение

            # Обновление лучшего delta и контроль стагнации
            with torch.no_grad():
                if best_loss is None or loss.item() < best_loss:
                    best_loss = loss.item()
                    best_delta = delta.data.clone()
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

            # Уменьшаем шаг при стагнации
            if stagnation_counter >= 10:
                for g in optimizer.param_groups:
                    g['lr'] *= self.norm_decay
                stagnation_counter = 0

        # Возвращаем итоговое изображение с наилучшим возмущением
        return torch.clamp(image + best_delta, 0, 1)

    def optimize_rec(self, image_batch: torch.Tensor, prompts, targets):
        """
        Вариант APGD для батча
        """
        B = len(prompts)
        delta = torch.zeros_like(image_batch, requires_grad=True).to(image_batch.device)
        optimizer = torch.optim.SGD([delta], lr=self.step_size)

        best_loss = None
        best_delta = delta.data.clone()
        stagnation_counter = 0

        for step in range(self.max_steps):
            optimizer.zero_grad()
            losses = []

            for i in range(B):
                item = {
                    "image": torch.clamp(image_batch[i:i+1] + delta[i:i+1], 0, 1),
                    "instruction_input": [f"<Img><ImageHere></Img> {prompts[i]}"],
                    "answer": [targets[i]],
                }
                loss = self.model.forward(item)['loss']
                if not self.targeted:
                    loss = -loss
                losses.append(loss)

            total_loss = torch.stack(losses).mean()
            total_loss.backward()
            optimizer.step()
            self._project(delta, image_batch)

            with torch.no_grad():
                if best_loss is None or total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    best_delta = delta.data.clone()
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

            if stagnation_counter >= 10:
                for g in optimizer.param_groups:
                    g['lr'] *= self.norm_decay
                stagnation_counter = 0

        return torch.clamp(image_batch + best_delta, 0, 1)