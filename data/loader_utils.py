from pathlib import Path
import zipfile
import requests
from tqdm import tqdm

def download_file(url: str, dest: Path) -> Path:
    """Скачивает файл по URL в указанную директорию."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest

    response = requests.get(url, stream=True)
    total = int(response.headers.get("content-length", 0))

    with open(dest, "wb") as file, tqdm(
        desc=dest.name,
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

    return dest

def extract_zip(zip_path: Path, extract_to: Path):
    """Распаковывает zip-архив."""
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

def download_and_extract(url: str, target_dir: Path):
    """Скачивает и распаковывает архив, если нужно."""
    filename = url.split("/")[-1]
    archive_path = target_dir / filename
    if not archive_path.exists():
        download_file(url, archive_path)
    extract_zip(archive_path, target_dir)
