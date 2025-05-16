import json
from pathlib import Path
from typing import Optional
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from data.loader_utils import download_and_extract

class ScienceQADataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str = "val",
        max_samples: Optional[int] = None,
        prompt_template: str = "Q: {question} A:",
        download: bool = True,
        to_tensor: bool = True,
    ):
        self.root = root
        self.split = split
        self.max_samples = max_samples
        self.prompt_template = prompt_template
        self.to_tensor = to_tensor

        self.data_file = root / "scienceqa" / f"{split}.json"
        self.image_dir = root / "images"

        if download:
            self._download_if_needed()

        if not self.data_file.exists():
            raise FileNotFoundError(f"Missing data file: {self.data_file}")
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Missing image folder: {self.image_dir}")

        with open(self.data_file) as f:
            self.items = json.load(f)

        if self.max_samples:
            self.items = self.items[:self.max_samples]

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]) if self.to_tensor else None

    def _download_if_needed(self):
        url = "https://huggingface.co/datasets/GeorgiaTechResearchInstitute/scienceqa/resolve/main/scienceqa.zip"
        print(f"[ScienceQA] Downloading dataset from: {url}")
        download_and_extract(url, self.root)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image_path = self.image_dir / item["image"]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        prompt = self.prompt_template.format(question=item["question"])

        return {
            "image": image,
            "text": prompt,
            "label": item["answer"],
            "meta": {
                "id": item["id"],
                "question": item["question"],
                "answer": item["answer"],
            },
        }