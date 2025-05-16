import json
from pathlib import Path
from typing import Optional
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from data.loader_utils import download_and_extract


class RefCOCODataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str = "val",
        max_samples: Optional[int] = None,
        prompt_template: str = "Locate: {expression}",
        download: bool = True,
        to_tensor: bool = True,
    ):
        """
        Args:
            root: путь к папке с датасетом (datasets/refcoco)
            split: 'train' / 'val' / 'test'
            max_samples: максимум примеров
            prompt_template: шаблон текста запроса
            download: скачивать ли датасет при необходимости
            to_tensor: преобразовать изображение в torch.Tensor
        """
        self.root = root
        self.split = split
        self.max_samples = max_samples
        self.prompt_template = prompt_template
        self.to_tensor = to_tensor

        self.annotations_file = root / "refcoco" / f"{split}.json"
        self.image_root = root / "images"

        if download:
            self._download_if_needed()

        if not self.annotations_file.exists():
            raise FileNotFoundError(f"Missing annotations file: {self.annotations_file}")
        if not self.image_root.exists():
            raise FileNotFoundError(f"Missing image folder: {self.image_root}")

        with open(self.annotations_file) as f:
            self.items = json.load(f)

        if self.max_samples:
            self.items = self.items[:self.max_samples]

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]) if self.to_tensor else None

    def _download_if_needed(self):
        url = "https://huggingface.co/datasets/GeorgiaTechResearchInstitute/refcoco/resolve/main/refcoco_data.zip"
        print(f"[RefCOCO] Downloading dataset from: {url}")
        download_and_extract(url, self.root)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image_path = self.image_root / item["image"]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        prompt = self.prompt_template.format(expression=item["expression"])

        return {
            "image": image,
            "text": prompt,
            "label": item["bbox"],
            "meta": {
                "image": item["image"],
                "expression": item["expression"],
                "bbox": item["bbox"],
            },
        }