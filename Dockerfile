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

# Expose the FastAPI port.
EXPOSE 8000

# Start FastAPI application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
