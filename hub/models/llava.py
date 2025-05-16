from pathlib import Path
import torch
from transformers import LlavaProcessor, LlavaForConditionalGeneration

class LlavaModel:
    def __init__(self, model_path: str, device: str = "cuda:0"):
        """
        Загрузка модели LLaVA. Поддерживает HuggingFace repo и локальный путь.
        """
        self.device = device

        if Path(model_path).exists():
            model_dir = Path(model_path)
        else:
            print(f"[LlavaModel] Downloading model from HuggingFace repo: {model_path}")
            model_dir = model_path

        self.processor = LlavaProcessor.from_pretrained(model_dir)
        self.model = LlavaForConditionalGeneration.from_pretrained(model_dir).to(self.device).eval()

    def forward(self, batch):
        """
        Вычисление loss модели на батче изображений и инструкций.
        Args:
            batch: {
                "instruction_input": List[str],
                "image": Tensor [B, 3, H, W],
                "answer": List[str]
            }
        Returns:
            dict: {"loss": float}
        """
        inputs = self.processor(
            text=batch["instruction_input"],
            images=batch["image"],
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        labels = self.processor.tokenizer(
            batch["answer"], return_tensors="pt", padding=True
        ).input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, labels=labels)

        return {"loss": outputs.loss}

    def generate(self, image: torch.Tensor, prompt: str) -> str:
        """
        Генерация ответа моделью LLaVA по изображению и текстовому запросу.
        Args:
            image: torch.Tensor [1, 3, H, W]
            prompt: str
        Returns:
            str: сгенерированный текст
        """
        inputs = self.processor(
            text=[prompt],
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=64)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return generated_text.strip()
