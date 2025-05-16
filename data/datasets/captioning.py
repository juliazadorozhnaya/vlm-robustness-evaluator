import os
import json
from pathlib import Path
from typing import Optional
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from data.loader_utils import download_and_extract


class MSCOCODataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str = "val",
        max_samples: Optional[int] = None,
        prompt_template: str = "",
        download: bool = True,
        to_tensor: bool = True,
    ):
        """
        Args:
            root: путь к папке с датасетом (например, datasets/ms_coco)
            split: "train" или "val"
            max_samples: максимум примеров
            prompt_template: шаблон (например, "Describe: {caption}")
            download: если True, скачает датасет при необходимости
            to_tensor: преобразовывать изображения в torch.Tensor
        """
        self.root = root
        self.split = split
        self.max_samples = max_samples
        self.prompt_template = prompt_template
        self.download = download
        self.to_tensor = to_tensor

        self.annotation_file = root / "annotations" / f"captions_{split}2014.json"
        self.image_dir = root / f"{split}2014"

        if self.download:
            self._download_if_needed()

        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Missing annotation file: {self.annotation_file}")
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Missing image directory: {self.image_dir}")

        with open(self.annotation_file) as f:
            annotations = json.load(f)["annotations"]

        if self.max_samples:
            annotations = annotations[:self.max_samples]

        self.samples = annotations

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]) if self.to_tensor else None

    def _download_if_needed(self):
        ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
        img_url = f"http://images.cocodataset.org/zips/{self.split}2014.zip"
        download_and_extract(ann_url, self.root)
        download_and_extract(img_url, self.root)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_id = sample["image_id"]
        caption = sample["caption"]
        img_filename = f"COCO_{self.split}2014_{image_id:012d}.jpg"
        img_path = self.image_dir / img_filename
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Если задан шаблон — подставим caption, иначе используем caption как prompt
        prompt = self.prompt_template.format(caption=caption) if self.prompt_template else caption

        return {
            "image": image,
            "text": prompt,          # используется как prompt
            "label": caption,        # используется как ground truth
            "meta": {
                "image_id": image_id,
                "caption_id": sample.get("id", None),
            },
        }
