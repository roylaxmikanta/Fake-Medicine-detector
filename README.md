# Medicine Authenticity Checker

Medicine Authenticity Checker is an educational FastAPI application that analyzes medicine-package images with OCR, public medicine databases, and a PyTorch image classifier. It helps identify warning signs such as unreadable packaging, expired dates, and missing database matches.

> This project is not a medical device and does not prove that a medicine is safe or authentic. Always confirm medicine details with a licensed pharmacist, doctor, manufacturer, or local regulator.

## Contents

- [How It Works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Run the Application](#run-the-application)
- [Use the Web Interface](#use-the-web-interface)
- [Train the Image Model](#train-the-image-model)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Docker](#docker)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations-and-safety)

## How It Works

The application processes an uploaded JPG, JPEG, or PNG image in this order:

1. Validates the uploaded file.
2. Extracts visible text with multilingual EasyOCR.
3. Stops with `OCR_EMPTY` when no readable text is found and asks for a clearer image.
4. Detects expiry labels such as `EXP 09/2026` or `EXP 09/13/2026`. A month-only expiry date is treated as valid through the final day of that month.
5. Returns `EXPIRED` when the detected expiry date has passed. Users are advised not to use the medicine.
6. Extracts medicine names and available identifiers, including licence/application numbers, batch numbers, and NDC values.
7. Identifies the dominant language of the OCR text and checks structured identifiers and medicine names against FDA OpenFDA and RxNorm.
8. Runs the MobileNetV2 image classifier as a visual analysis fallback.
9. Optionally generates general usage information through Groq.
10. Provides a Google search link for an additional manual cross-check.

Database verification and image classification are supporting signals, not a guarantee of authenticity.

## Prerequisites

- Windows, Linux, or macOS.
- Python 3.10 or newer. Python 3.11 is used by the Docker image.
- Git, if cloning the repository.
- Internet access for EasyOCR model downloads and FDA/RxNorm requests.
- At least 2 GB of available memory for local inference.
- Docker Desktop, if using Docker.
- An NVIDIA GPU is optional. The application automatically uses CUDA when the installed PyTorch build supports it.

## Installation

Open PowerShell in the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation for the current session, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

Then activate the environment again. Keep the terminal activated for the remaining commands.

## Run the Application

Start the FastAPI server from the repository root:

```powershell
uvicorn api.main:app --reload
```

Open the application in a browser at `http://127.0.0.1:8000/`.

Useful development URLs:

- Web interface: `http://127.0.0.1:8000/`
- Swagger API documentation: `http://127.0.0.1:8000/docs`
- ReDoc API documentation: `http://127.0.0.1:8000/redoc`
- Health check: `http://127.0.0.1:8000/health`

The first startup may take longer because EasyOCR can download its language model.

The default OCR configuration loads English, Hindi, French, German, Spanish, Italian, Portuguese, Russian, and Arabic. These languages are selected in [api/config.py](api/config.py). EasyOCR must support a language before it can recognize that script; add a supported language code to `OCR_LANGUAGES` and restart the API when expanding coverage. Loading many languages increases startup time and memory use.

## Use the Web Interface

1. Start the server.
2. Open `http://127.0.0.1:8000/`.
3. Upload a sharp, well-lit image of the medicine package.
4. Ensure the medicine name, expiry date, and package identifiers are visible.
5. Review the OCR text, expiry result, database result, image-model result, usage information, and Google cross-check link.

For the best OCR result, photograph the package straight on, avoid glare, and do not crop out the expiry or licence area.

## Train the Image Model

Training images must be arranged as follows:

```text
data/
├── Fake/
└── Real/
```

Place representative images in both directories. The training script uses `ImageFolder`, so the directory names become the class labels.

Run training from the repository root:

```powershell
python src/train.py
```

The script resizes images to `150 x 150`, applies training augmentation, splits data into training/validation/test sets, trains MobileNetV2 for five epochs, and saves the best model by validation accuracy.

Generated files:

```text
models/medicine_model.pt
models/preprocessing.pkl
confusion_matrix.png
```

Restart the API after training so it loads the new model files.

### Notebook Workflow

The notebook at [notebooks/train_model.ipynb](notebooks/train_model.ipynb) contains OCR and TF-IDF experiments. It is separate from the production API model-training script.

Run notebook cells in order. The OCR-processing cell must run before the text-classification cells, and the generated dataset must contain non-empty `Text` and `Label` columns. Notebook dependencies can be installed from its first cell.

## API Reference

### `GET /health`

Check whether the model and OCR resources loaded successfully:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "ocr_loaded": true
}
```

### `POST /predict`

Upload an image using the `file` form field:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict" `
  -F "file=@data/Real/example.jpg"
```

Important response statuses:

- `OK`: OCR and analysis completed without an expired date.
- `OCR_EMPTY`: no readable text was extracted; upload a cleaner image.
- `EXPIRED`: the detected expiry date has passed; do not use the medicine.

Relevant response fields include `detected_text`, `detected_language`, `ocr_languages`, `expiry`, `api_verification`, `ml_analysis`, `usage_info`, and `google_search_url`.

## Configuration

Create a `.env` file in the repository root only when optional Groq responses are required:

```env
GROQ_API_KEY=your_groq_api_key
```

The application works without this variable. Without it, the API still performs OCR, expiry detection, database checks, image analysis, and Google-link generation. Never commit `.env` or expose API keys.

Key settings are defined in [api/config.py](api/config.py), including API timeouts, OCR confidence, supported labels, and ignored packaging terms.

## Docker

The Dockerfile uses Python 3.11 and CPU-only PyTorch. Build the image from the repository root:

```powershell
docker build -t fake-medicine-api:latest .
```

Run the container:

```powershell
docker run --rm --name fake-medicine-api -p 8000:8000 --env-file .env fake-medicine-api:latest
```

If `.env` does not exist, omit `--env-file .env`:

```powershell
docker run --rm --name fake-medicine-api -p 8000:8000 fake-medicine-api:latest
```

Open `http://localhost:8000/`. The container copies `api/` and `models/`, so `models/medicine_model.pt` must exist before building the image. The training dataset is not included in the runtime image.

## Project Structure

```text
Fake-Mediceine-detector/
├── api/
│   ├── main.py              # FastAPI routes, OCR, expiry checks, inference
│   ├── api_verifier.py      # FDA/RxNorm and identifier verification
│   ├── config.py            # API, OCR, and verification settings
│   └── static/index.html    # Browser interface
├── data/
│   ├── Fake/                # Training images for the Fake class
│   ├── Real/                # Training images for the Real class
│   └── *.csv                # OCR and dataset exports
├── models/
│   ├── medicine_model.pt    # MobileNetV2 weights
│   └── preprocessing.pkl    # Saved class-name metadata
├── notebooks/
│   └── train_model.ipynb    # OCR and text-model experiments
├── src/
│   └── train.py             # Image-model training script
├── Dockerfile
├── requirements.txt
└── README.md
```

## Troubleshooting

### The server cannot load the model

Confirm that `models/medicine_model.pt` and `models/preprocessing.pkl` exist. If they are missing, arrange the dataset and run `python src/train.py`.

### EasyOCR fails or returns no text

Check the internet connection during the first startup, then upload a sharper image with better lighting and less glare. The UI intentionally asks for a clean photo when no text is detected.

### FDA or RxNorm returns no match

These services do not contain every country-specific product or brand. Check the medicine name, licence number, batch number, and NDC manually using the supplied Google link, then confirm with a pharmacist or the manufacturer.

### Port 8000 is already in use

Run the server on another port:

```powershell
uvicorn api.main:app --reload --port 8001
```

## Limitations and Safety

- OCR can misread small, rotated, reflective, or low-quality text.
- Automatic language identification describes the dominant language in extracted text; mixed-language packaging may produce an approximate result.
- Language support depends on EasyOCR's available models. "All languages" cannot be guaranteed by a single OCR engine.
- FDA and RxNorm coverage varies by country, product type, and brand.
- A database match does not prove that the photographed package is genuine.
- The image classifier is only as reliable as its training data and evaluation quality.
- The application cannot confirm seals, tampering, storage conditions, supply-chain provenance, or physical contents.
- Batch-number verification is limited by public database fields available for a product.
- Google results are provided for manual investigation and are not an authenticity decision.
- Do not use this result to replace professional medical advice.