FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir \
        "torch==2.8.0+cu128" \
        "torchaudio==2.8.0+cu128" \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
    && python -m pip install --no-cache-dir \
        "git+https://github.com/k2-fsa/OmniVoice.git" \
        runpod

COPY handler.py /app/handler.py

RUN python -m py_compile /app/handler.py

RUN python -c "from omnivoice import OmniVoice; OmniVoice.from_pretrained('k2-fsa/OmniVoice')"

CMD ["python", "-u", "handler.py"]
