# Medicine Authenticity Checker

Medicine Authenticity Checker is a FastAPI application that analyzes a medicine-package image using OCR, public medicine databases, and a PyTorch image-classification model.

## What It Does

1. Accepts a JPG, JPEG, or PNG medicine image.
2. Extracts visible text with EasyOCR.
3. Checks detected medicine names against the FDA OpenFDA label API and RxNorm.
4. Uses the trained MobileNetV2 model as an image-based fallback.
5. Optionally returns a short medicine-use description through Groq.

This tool is for experimentation and education. It is not a substitute for a pharmacist, doctor, or regulatory verification service.

## Features

- FastAPI REST API with an integrated browser UI.
- EasyOCR text extraction.
- FDA and RxNorm verification with retry and caching logic.
- PyTorch MobileNetV2 classifier for `Fake` and `Real` images.
- CPU and CUDA inference support where the installed PyTorch build supports it.
- Docker support.
- Optional Groq-generated medicine usage information.

## Requirements

- Python 3.10 or newer.
- pip.
- Internet access for EasyOCR model downloads and FDA/RxNorm requests.
- Optional NVIDIA GPU with a compatible CUDA-enabled PyTorch installation.

## Run Locally

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn api.main:app --reload
```

Open the web interface at `http://127.0.0.1:8000/`.

API documentation is available at `http://127.0.0.1:8000/docs`.

On Windows, `run.bat` can also be used after the virtual environment and dependencies have been configured. The batch file may require updating if the repository is moved.

## A-to-Z Workflow

### 1. Prepare the Dataset

Training images are organized by class:

```text
data/
├── Fake/
└── Real/
```

The training pipeline reads these folders, resizes images to `150 x 150`, converts them to tensors, normalizes them with ImageNet values, and creates training and validation sets.

### 2. Train the Model

Run the training script from the repository root:

```powershell
python src/train.py
```

The script trains a MobileNetV2 classifier and writes:

```text
models/medicine_model.pt
models/preprocessing.pkl
```

The `.pt` file contains the model weights. The preprocessing file contains class metadata such as `Fake` and `Real`.

### 3. Start the API

When the API starts, `api/main.py` loads the model, preprocessing metadata, EasyOCR, and API verifier. The browser interface is served from `api/static/index.html`.

### 4. Process an Uploaded Image

The prediction flow is:

```text
Image upload
    -> Image validation
    -> EasyOCR text extraction
    -> Medicine-name cleaning
    -> FDA and RxNorm verification
    -> MobileNetV2 Fake/Real prediction
    -> Optional Groq usage description
    -> JSON response
```

### 5. Configure Secrets

Create `.env` locally. Never commit it:

```env
HF_API_KEY=your_huggingface_key
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
```

### 6. Run Locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000/`. Check the service with:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

### 7. Test the API

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict" `
  -F "file=@data/Real/images01.jpg"
```

### 8. Build and Run with Docker

The Dockerfile installs CPU-only PyTorch and copies only runtime files. The port is configurable through `PORT`.

```powershell
docker build -t fake-medicine-api:slim .
docker run -d --name fake-medicine-api -e PORT=8000 -p 8000:8000 --env-file .env fake-medicine-api:slim
```

Open `http://localhost:8000` and view logs with `docker logs -f fake-medicine-api`.

### 9. Publish to GitHub

1. Keep API keys in a local `.env` file only. The `.env` file is ignored by Git:

  ```env
  HF_API_KEY=your_huggingface_key
  GROQ_API_KEY=your_groq_key
  TAVILY_API_KEY=your_tavily_key
  ```

2. Commit and push the source code:

  ```powershell
  git add .
  git commit -m "Prepare application for deployment"
  git push origin main
  ```

### 10. Deploy

The application requires more memory than Render's free 512 MB plan because PyTorch and EasyOCR load at startup. Use a deployment plan with at least 2 GB RAM, such as Google Cloud Run, and add the three secrets through the provider's environment settings.

For Render, select **New > Web Service**, connect the GitHub repository, select branch `main`, choose **Docker**, leave **Root Directory** blank, add the environment variables, set the health check path to `/health`, and deploy.

Save the public URL here after deployment:

  ```text
  https://YOUR-SERVICE-NAME.onrender.com
  ```

### 11. Verify the Deployment

  ```text
  https://YOUR-SERVICE-NAME.onrender.com/health
  ```

  The response should contain `"status": "healthy"`, `"model_loaded": true`, and `"ocr_loaded": true`.

The response should contain `"status": "healthy"`, `"model_loaded": true`, and `"ocr_loaded": true`.

Free or low-cost services may sleep after inactivity, so the first request can be slow. Never upload `.env` or exposed API keys to GitHub.

## API Endpoints

### `GET /health`

Returns application and resource status:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "ocr_loaded": true
}
```

### `POST /predict`

Upload a medicine image using the `file` form field:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict" `
  -F "file=@data/Real/images01.jpg"
```

The response can contain `detected_text`, `api_verification`, `ml_analysis`, and optional `usage_info` fields. Supported upload extensions are `.jpg`, `.jpeg`, and `.png`.

## Optional Groq Configuration

Create a `.env` file in the repository root:

```env
GROQ_API_KEY=your_api_key_here
```

The API still starts without this variable. Only the optional `usage_info` response is disabled. Do not commit `.env` or API keys.

## Project Structure

```text
Fake-Mediceine-detector/
├── api/
│   ├── main.py              # FastAPI app, OCR, model inference, endpoints
│   ├── api_verifier.py      # FDA and RxNorm verification
│   ├── config.py            # API and model configuration
│   └── static/index.html    # Browser interface
├── data/
│   ├── Real/                # Real medicine images
│   ├── Fake/                # Fake medicine images
│   └── *.csv                # Dataset and OCR exports
├── models/
│   └── medicine_model.pt   # PyTorch model weights
├── notebooks/
│   └── train_model.ipynb    # OCR and text-classification experiments
├── src/
│   └── train.py             # MobileNetV2 image-model training script
├── Dockerfile
├── requirements.txt
└── README.md
```

## Training the Image Model

The training script expects this folder structure:

```text
data/
├── Fake/
└── Real/
```

Run it from the repository root:

```powershell
python src/train.py
```

The script saves trained weights to `models/medicine_model.pt` and class metadata to `models/preprocessing.pkl`. It also writes `confusion_matrix.png` in the repository root.

The notebook is for OCR and TF-IDF experiments. Its first cell installs notebook dependencies with `%pip`; restart the notebook kernel after installation if imports were previously failing. Run the OCR-processing cell before the training cell and confirm that the generated `Text` column contains detected text.

## Limitations

- OCR quality depends on image clarity, lighting, orientation, and packaging language.
- FDA and RxNorm do not cover every country, brand, supplement, or traditional medicine.
- The image model is only as reliable as its training data and evaluation results.
- The application does not verify physical packaging, batch numbers, seals, or supply-chain provenance.
- External API responses may be unavailable or rate-limited.