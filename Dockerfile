FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and EasyOCR
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Use CPU-only PyTorch wheels; the API does not require CUDA in this image.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision
RUN pip install --no-cache-dir -r requirements.txt

# Copy only files required at runtime.
COPY api ./api
COPY models ./models

# Hugging Face Spaces uses port 7860 by default. The host port can still be
# mapped to 8000 when running locally.
EXPOSE 7860

# Start FastAPI application
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
