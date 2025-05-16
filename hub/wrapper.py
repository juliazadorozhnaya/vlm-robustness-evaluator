import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Dict, Any
from models import MODEL_REGISTRY


class ModelWrapper:
    """
    Унифицированный обёрточный класс для мультимодальных моделей.
    Позволяет единообразно вызывать forward(), generate(), tokenize(), preprocess_image().
    """

    def __init__(self, model_cfg: Dict[str, Any]):
        """
        Инициализация модели по конфигурации.
        model_cfg должен содержать:
            - 'name': имя модели
            - 'type': тип модели (ключ из MODEL_REGISTRY)
            - 'path': путь к модели (локальный или HuggingFace repo)
            - 'device': cuda:0 / cpu
        """
        self.name = model_cfg["name"]
        self.device = model_cfg.get("device", "cuda:0")
        model_type = model_cfg["type"]
        model_path = model_cfg["path"]

        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model type: {model_type}")

        model_class = MODEL_REGISTRY[model_type]
        self.model = model_class(model_path, self.device)
        self.tokenizer = getattr(self.model, "tokenizer", None)

    def forward(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward-вызов модели.
        item: dict с ключами:
            - image
            - instruction_input
            - answer (опционально)
        """
        return self.model.forward(item)

    def generate(self, image: torch.Tensor, prompt: str, **kwargs) -> str:
        """
        Генерация текста по изображению и текстовому запросу.
        """
        if hasattr(self.model, "generate"):
            return self.model.generate(image.to(self.device), prompt, **kwargs)
        raise RuntimeError(f"Model '{self.name}' does not support generate().")

    def preprocess_image(self, pil_image) -> torch.Tensor:
        """
        Преобразование PIL-изображения в тензор.
        """
        if hasattr(self.model, "preprocess"):
            return self.model.preprocess(pil_image)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        return transform(pil_image).unsqueeze(0).to(self.device)

    def tokenize(self, text: str) -> torch.Tensor:
        """
        Токенизация текста (если tokenizer доступен).
        """
        if self.tokenizer:
            tokens = self.tokenizer(text, return_tensors="pt")["input_ids"]
            return tokens.to(self.device)
        raise RuntimeError(f"Tokenizer not set for model '{self.name}'.")

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def make_dataloader(
        self,
        dataset,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True
    ) -> DataLoader:
        """
        Унифицированный DataLoader для inference/атак.
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False
        )
