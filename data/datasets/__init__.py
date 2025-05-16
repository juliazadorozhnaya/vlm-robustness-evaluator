from .captioning import MSCOCODataset
from .refcoco import RefCOCODataset
from .scienceqa import ScienceQADataset
from .vqa import VQADataset

from pathlib import Path

def download_refcoco(root: Path):
    RefCOCODataset(root=root, download=True)

def load_refcoco(root: Path, task_cfg: dict):
    return RefCOCODataset(
        root=root,
        split="val",
        max_samples=task_cfg.get("max_samples"),
        prompt_template=task_cfg.get("prompt_template", ""),
        download=task_cfg.get("download", False)
    )

def download_scienceqa(root: Path):
    ScienceQADataset(root=root, download=True)

def load_scienceqa(root: Path, task_cfg: dict):
    return ScienceQADataset(
        root=root,
        split="val",
        max_samples=task_cfg.get("max_samples"),
        prompt_template=task_cfg.get("prompt_template", ""),
        download=task_cfg.get("download", False)
    )

def download_vqa(root: Path):
    VQADataset(root=root, download=True)

def load_vqa(root: Path, task_cfg: dict):
    return VQADataset(
        root=root,
        split="val",
        max_samples=task_cfg.get("max_samples"),
        prompt_template=task_cfg.get("prompt_template", ""),
        download=task_cfg.get("download", False)
    )

DATASET_LOADERS = {
    "referring_expression": {
        "download": download_refcoco,
        "load_dataset": load_refcoco,
    },
    "visual_commonsense_reasoning": {
        "download": download_scienceqa,
        "load_dataset": load_scienceqa,
    },
    "vqa": {
        "download": download_vqa,
        "load_dataset": load_vqa,
    },
}
