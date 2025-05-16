from pathlib import Path
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

class DeepseekModel:
    def __init__(self, model_path: str, device: str = "cuda:0"):
        """
        Загрузка DeepSeek-VL модели. Поддерживает как локальный путь, так и HuggingFace репозиторий.
        """
        self.device = device

        # Определяем, нужно ли скачивать модель или использовать локальную директорию
        if Path(model_path).exists():
            model_dir = Path(model_path)
        else:
            print(f"[DeepseekModel] Downloading model from HuggingFace repo: {model_path}")
            model_dir = model_path

        # Загрузка процессора и модели
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.model = AutoModelForVision2Seq.from_pretrained(model_dir).to(self.device).eval()

    def forward(self, batch):
        """
        Получение loss от модели. Используется в обучении или при оценке.
        Аргументы:
            batch: словарь с ключами 'instruction_input', 'image', 'answer'
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
        Генерация текста по изображению и текстовому запросу.
        Аргументы:
            image: torch.Tensor — тензор изображения [1, 3, H, W]
            prompt: str — текстовая инструкция

        Возвращает:
            str — сгенерированный текст
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
