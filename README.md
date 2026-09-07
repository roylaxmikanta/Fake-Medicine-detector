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

## Docker

Build and run the API:

```powershell
docker build -t medicine-authenticity-checker .
docker run --rm -p 8000:8000 medicine-authenticity-checker
```

Then open `http://127.0.0.1:8000/`.

## Verification Flow

```text
Image upload
    -> EasyOCR text extraction
    -> FDA lookup
    -> RxNorm lookup
    -> MobileNetV2 image prediction
    -> Optional Groq usage information
```

Database verification depends on the detected medicine name and network access. A failed database lookup does not by itself prove that a medicine is fake.

## Limitations

- OCR quality depends on image clarity, lighting, orientation, and packaging language.
- FDA and RxNorm do not cover every country, brand, supplement, or traditional medicine.
- The image model is only as reliable as its training data and evaluation results.
- The application does not verify physical packaging, batch numbers, seals, or supply-chain provenance.
- External API responses may be unavailable or rate-limited.