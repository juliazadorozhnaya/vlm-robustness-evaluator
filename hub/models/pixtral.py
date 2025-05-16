from pathlib import Path
import torch
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration

class PixtralModel:
    def __init__(self, model_path: str, device: str = "cuda:0"):
        """
        Загрузка модели PixTral (Pix2Struct). Поддержка HuggingFace repo и локального пути.
        """
        self.device = device

        if Path(model_path).exists():
            model_dir = Path(model_path)
        else:
            print(f"[PixtralModel] Downloading model from HuggingFace repo: {model_path}")
            model_dir = model_path

        self.processor = Pix2StructProcessor.from_pretrained(model_dir)
        self.model = Pix2StructForConditionalGeneration.from_pretrained(model_dir).to(self.device).eval()

    def forward(self, batch):
        """
        Вычисление потерь (loss) модели PixTral по батчу.
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
            batch["answer"],
            return_tensors="pt",
            padding=True
        ).input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, labels=labels)

        return {"loss": outputs.loss}

    def generate(self, image: torch.Tensor, prompt: str) -> str:
        """
        Генерация текста моделью PixTral по изображению и инструкции.
        Args:
            image: torch.Tensor [1, 3, H, W]
            prompt: str
        Returns:
            str: сгенерированный ответ
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
