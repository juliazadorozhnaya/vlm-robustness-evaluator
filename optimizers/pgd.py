import torch
import torch.nn.functional as F

class PGDOptimizer:
    def __init__(
        self,
        model,
        targeted=True,
        epsilon=8 / 255,
        step_size=1 / 255,
        max_steps=40,
        projection="linf",
        random_start=False,
    ):
        """
        Реализация PGD-атаки с поддержкой L∞ и L2 норм, и random start.
        """
        self.model = model
        self.targeted = targeted
        self.epsilon = epsilon
        self.step_size = step_size
        self.max_steps = max_steps
        self.projection = projection
        self.random_start = random_start

    def _project(self, delta: torch.Tensor) -> torch.Tensor:
        """
        Проекция delta в допустимую область по заданной норме.
        """
        if self.projection == "linf":
            return torch.clamp(delta, -self.epsilon, self.epsilon)
        elif self.projection == "l2":
            flat = delta.view(delta.size(0), -1)
            norm = flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
            factor = torch.minimum(torch.ones_like(norm), self.epsilon / norm)
            projected = flat * factor
            return projected.view_as(delta)
        else:
            raise ValueError(f"Неизвестная проекция: {self.projection}")

    def optimize(self, image: torch.Tensor, prompt: str, target_text: str):
        image = image.clone().detach().unsqueeze(0).cuda()
        delta = torch.zeros_like(image, requires_grad=True).cuda()

        if self.random_start:
            delta.data.uniform_(-self.epsilon, self.epsilon)
            delta.data = self._project(delta.data)

        target_ids = self.model.tokenize(target_text).to(image.device)

        for step in range(self.max_steps):
            logits = self.model(image + delta, prompt)
            log_probs = F.log_softmax(logits, dim=-1)
            loss = F.nll_loss(
                log_probs[:, :target_ids.size(1)].reshape(-1, log_probs.size(-1)),
                target_ids.view(-1),
            )
            if not self.targeted:
                loss = -loss

            self.model.zero_grad()
            loss.backward()
            grad = delta.grad.detach()

            # Обновление delta и проекция
            delta.data = delta + self.step_size * torch.sign(grad)
            delta.data = self._project(delta.data)
            delta.grad.zero_()

        return torch.clamp(image + delta.detach(), 0, 1), self.model.generate(image + delta, prompt)

    def optimize_rec(self, image_batch: torch.Tensor, prompts: list[str], target_text: str):
        B = len(prompts)
        delta = torch.zeros_like(image_batch.detach(), requires_grad=True).cuda()

        if self.random_start:
            delta.data.uniform_(-self.epsilon, self.epsilon)
            delta.data = self._project(delta.data)

        for step in range(self.max_steps):
            losses = []
            for i in range(B):
                item = {
                    "image": torch.clamp(image_batch[i:i + 1] + delta[i:i + 1], 0, 1),
                    "instruction_input": ["<Img><ImageHere></Img> {}".format(prompts[i])],
                    "answer": [target_text],
                }
                loss = self.model.forward(item)['loss']
                losses.append(-loss if not self.targeted else loss)

            total_loss = torch.stack(losses).mean()
            self.model.zero_grad()
            total_loss.backward()
            grad = delta.grad.detach()

            # Обновление delta и проекция
            delta.data = delta + self.step_size * torch.sign(grad)
            delta.data = self._project(delta.data)
            delta.grad.zero_()

        return torch.clamp(image_batch + delta.detach(), 0, 1)