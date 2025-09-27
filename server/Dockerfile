FROM python:3.11-slim

# Optional: faster installs & smaller image
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps for torch wheels & sentencepiece tokenizers
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY main.py .

# Model name overrideable at runtime
ENV MODEL_NAME=sentence-transformers/paraphrase-MiniLM-L3-v2

EXPOSE 8000
CMD ["uvicorn", "main:main", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
