import json
from pathlib import Path
from typing import Optional
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from data.loader_utils import download_and_extract

class VQADataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str = "val",
        max_samples: Optional[int] = None,
        prompt_template: str = "Question: {question} Answer:",
        download: bool = True,
        to_tensor: bool = True,
    ):
        self.root = root
        self.split = split
        self.max_samples = max_samples
        self.prompt_template = prompt_template
        self.to_tensor = to_tensor

        self.annotations_file = root / "vqa/v2_mscoco_val_annotations.json"
        self.questions_file = root / "vqa/v2_OpenEnded_mscoco_val_questions.json"
        self.image_dir = root / "vqa/images"

        if download:
            self._download_if_needed()

        if not self.annotations_file.exists():
            raise FileNotFoundError(f"Missing annotations file: {self.annotations_file}")
        if not self.questions_file.exists():
            raise FileNotFoundError(f"Missing questions file: {self.questions_file}")
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Missing image folder: {self.image_dir}")

        with open(self.questions_file) as f:
            questions = json.load(f)["questions"]
        with open(self.annotations_file) as f:
            annotations = json.load(f)["annotations"]

        merged = [
            {
                "image_id": q["image_id"],
                "question": q["question"],
                "answer": a["multiple_choice_answer"],
            }
            for q, a in zip(questions, annotations)
        ]

        self.items = merged[:max_samples] if max_samples else merged

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]) if self.to_tensor else None

    def _download_if_needed(self):
        q_url = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_OpenEnded_mscoco_val_questions.zip"
        a_url = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip"
        img_url = "http://images.cocodataset.org/zips/val2014.zip"
        download_and_extract(q_url, self.root / "vqa")
        download_and_extract(a_url, self.root / "vqa")
        download_and_extract(img_url, self.root / "vqa")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_file = f"COCO_val2014_{item['image_id']:012d}.jpg"
        image = Image.open(self.image_dir / img_file).convert("RGB")
        if self.transform:
            image = self.transform(image)

        prompt = self.prompt_template.format(question=item["question"])

        return {
            "image": image,
            "text": prompt,
            "label": item["answer"],
            "meta": {
                "image_id": item["image_id"],
                "question": item["question"],
                "answer": item["answer"],
            },
        }