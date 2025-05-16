FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# === Устанавливаем режим установки по умолчанию
ARG INSTALL_MODE=noninteractive
ENV DEBIAN_FRONTEND=${INSTALL_MODE}

# === Отключаем создание .pyc файлов и __pycache__ директорий
ENV DISABLE_PYC_CACHE=1
ENV PYTHONDONTWRITEBYTECODE=${DISABLE_PYC_CACHE}

# === Включаем немедленный вывод stdout/stderr
ENV FORCE_STDOUT_FLUSH=1
ENV PYTHONUNBUFFERED=${FORCE_STDOUT_FLUSH}

RUN apt-get update && \
    apt-get install -y python3.10 python3-pip git wget unzip ffmpeg libglib2.0-0 libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

WORKDIR /app
COPY . /app

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -r requirements.txt

RUN mkdir -p /app/outputs /app/datasets

CMD ["python3", "main.py"]
