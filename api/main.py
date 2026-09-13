import os
import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import easyocr
from langdetect import DetectorFactory, detect_langs
import pickle
import numpy as np
import re
import calendar
from datetime import date, datetime
from urllib.parse import quote_plus
from dotenv import load_dotenv
import groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = groq.Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Adjust imports from local module
from .api_verifier import MedicineAPIVerifier
from .config import OCR_LANGUAGES, OCR_MIN_CONFIDENCE, OCR_USE_GPU

DetectorFactory.seed = 42

app = FastAPI(title="Medicine Authenticity Checker API")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Mount static files for UI
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "api", "static")), name="static")

@app.get("/", response_class=HTMLResponse)
async def root_ui():
    with open(os.path.join(BASE_DIR, "api", "static", "index.html"), "r", encoding="utf-8") as f:
        return f.read()

MODEL_PATH = os.path.join(BASE_DIR, 'models', 'medicine_model.pt')
PREPROCESSING_PATH = os.path.join(BASE_DIR, 'models', 'preprocessing.pkl')
IMG_SIZE = 150

# Global resources
model = None
reader = None
ocr_languages_loaded = []
api_verifier = None
class_names = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_expiry_date(text):
    """Return the expiry date and whether it has passed, using the package text."""
    patterns = [
        r'(?i)\b(?:exp(?:iry)?|use\s*before)\s*[:./-]?\s*(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})\b',
        r'(?i)\b(?:exp(?:iry)?|use\s*before)\s*[:./-]?\s*(\d{1,2})[./-](\d{2,4})\b',
        r'(?i)\b(?:exp(?:iry)?|use\s*before)\s*[:./-]?\s*(\d{4})[./-](\d{1,2})\b',
    ]
    for index, pattern in enumerate(patterns):
        match = re.search(pattern, text)
        if not match:
            continue
        values = [int(value) for value in match.groups()]
        try:
            if index == 2:
                year, month = values
                expiry = date(year, month, calendar.monthrange(year, month)[1])
                display = f"{year:04d}-{month:02d}"
            elif index == 0:
                month, day, year = values
                year += 2000 if year < 100 else 0
                expiry = date(year, month, day)
                display = expiry.isoformat()
            else:
                month, year = values
                year += 2000 if year < 100 else 0
                expiry = date(year, month, calendar.monthrange(year, month)[1])
                display = f"{year:04d}-{month:02d}"
            return {
                'value': display,
                'expired': expiry < date.today(),
                'checked_on': date.today().isoformat(),
            }
        except ValueError:
            continue
    return None


def build_google_search_url(text, api_result):
    medicine_name = api_result.get('medicine_name', '') if api_result else ''
    identifiers = api_result.get('identifiers', {}) if api_result else {}
    terms = [medicine_name]
    terms.extend(value for values in identifiers.values() for value in values)
    query = ' '.join(term for term in terms if term).strip() or text
    return f"https://www.google.com/search?q={quote_plus(query + ' medicine verification')}"


def build_evidence_summary(detected_text, expiry_info, api_result, ml_result):
    """Build user-facing evidence without treating one signal as proof of fraud."""
    identifiers = api_result.get('identifiers', {}) if api_result else {}
    licence_values = identifiers.get('licence_numbers', [])
    ml_prediction = (ml_result or {}).get('prediction', '').lower()
    api_status = (api_result or {}).get('status', 'NOT_CHECKED')
    reasons = []

    if ml_prediction == 'fake':
        reasons.append('The image model detected visual patterns associated with the Fake training class.')
    if api_status == 'UNVERIFIED':
        reasons.append('The extracted medicine information was not found in the checked FDA or RxNorm databases.')
    if not expiry_info:
        reasons.append('No readable expiry date was detected, so expiry could not be confirmed.')
    if not licence_values:
        reasons.append('No readable licence or application number was detected.')

    likely_fake = ml_prediction == 'fake'
    if likely_fake and api_status == 'UNVERIFIED':
        conclusion = 'Likely counterfeit based on visual analysis and no database match.'
    elif likely_fake:
        conclusion = 'Potentially counterfeit based on visual analysis; verify with a pharmacist or manufacturer.'
    elif api_status == 'UNVERIFIED':
        conclusion = 'Not verified. A database miss is not proof that the medicine is fake.'
    else:
        conclusion = 'No counterfeit signal was identified by the available checks.'

    return {
        'conclusion': conclusion,
        'fake_reasons': reasons if likely_fake else [],
        'extracted_text': detected_text,
        'expiry_visible': expiry_info is not None,
        'licence_visible': bool(licence_values),
        'licence_numbers': licence_values,
        'database_status': api_status,
        'google_cross_check_required': True,
    }


def detect_text_language(text):
    """Identify the dominant language after OCR, without changing the OCR text."""
    try:
        candidates = detect_langs(text)
        if candidates:
            best = candidates[0]
            return {
                'code': best.lang,
                'confidence': round(float(best.prob), 3),
            }
    except Exception:
        pass
    return {'code': 'unknown', 'confidence': 0.0}

test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.on_event("startup")
async def startup_event():
    global model, reader, api_verifier, class_names, ocr_languages_loaded
    print("Loading resources...")
    
    # Load preprocessing metadata
    try:
        with open(PREPROCESSING_PATH, 'rb') as f:
            mapping = pickle.load(f)
            class_names = mapping['class_names']
    except Exception as e:
        print(f"Warning: Could not load preprocessing metadata: {e}")
        class_names = ['Fake', 'Real']  # Fallback

    # Load ML Model
    try:
        model_instance = models.mobilenet_v2(pretrained=False)
        model_instance.classifier[1] = nn.Linear(model_instance.last_channel, len(class_names))
        model_instance.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model_instance.to(device)
        model_instance.eval()
        model = model_instance
    except Exception as e:
        print(f"Error loading model: {e}")

    # Load EasyOCR
    try:
        reader = easyocr.Reader(
            OCR_LANGUAGES,
            gpu=OCR_USE_GPU and torch.cuda.is_available()
        )
        ocr_languages_loaded = OCR_LANGUAGES
    except Exception as e:
        print(f"Multilingual EasyOCR setup failed: {e}")
        try:
            reader = easyocr.Reader(['en'], gpu=False)
            ocr_languages_loaded = ['en']
            print("Falling back to English OCR.")
        except Exception as fallback_error:
            print(f"Error loading EasyOCR fallback: {fallback_error}")

    # Load API Verifier
    api_verifier = MedicineAPIVerifier()
    print("Resources loaded successfully.")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "ocr_loaded": reader is not None,
        "ocr_languages": ocr_languages_loaded
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="Invalid image format. Supported: JPG, PNG.")
    
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    # 1. OCR Text Extraction
    detected_text = None
    if reader is not None:
        try:
            img_array = np.array(image)
            ocr_results = reader.readtext(img_array)
            texts = [det[1] for det in ocr_results if det[2] > OCR_MIN_CONFIDENCE]
            if texts:
                detected_text = ' '.join(texts)
        except Exception as e:
            print(f"OCR Error: {e}")

    if not detected_text:
        return JSONResponse(content={
            "status": "OCR_EMPTY",
            "message": "No readable text was extracted. Please upload a clean, well-lit photo showing the medicine name and expiry date.",
            "detected_text": None,
            "detected_language": None,
            "ocr_languages": ocr_languages_loaded,
                "evidence": None,
            "api_verification": None,
            "ml_analysis": None,
            "usage_info": None,
        })

    expiry_info = extract_expiry_date(detected_text)
    detected_language = detect_text_language(detected_text)

    # 2. API Database Check
    api_result = None
    if detected_text and api_verifier is not None:
        api_result = api_verifier.verify_medicine(detected_text)
    
    # 3. ML Fallback
    ml_result = None
    if model is not None:
        try:
            input_tensor = test_transforms(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                pred_idx = predicted.item()
                ml_result = {
                    "prediction": class_names[pred_idx],
                    "confidence": float(confidence.item()),
                    "class_index": pred_idx
                }
        except Exception as e:
            ml_result = {"error": str(e)}

    google_search_url = build_google_search_url(detected_text, api_result)
    is_expired = expiry_info and expiry_info['expired']
    evidence = build_evidence_summary(detected_text, expiry_info, api_result, ml_result)
    is_suspected_fake = bool(ml_result and ml_result.get('prediction', '').lower() == 'fake')

    # 4. Fetch Usage Information (Groq)
    usage_info = None
    is_real_api = api_result and api_result.get('status') == 'VERIFIED'
    is_real_ml = ml_result and ml_result.get('prediction', '').lower() == 'real'
    
    if (is_real_api or is_real_ml) and not is_expired and not is_suspected_fake and groq_client:
        med_name = ""
        if is_real_api:
            med_name = api_result.get('medicine_name', '')
        elif is_real_ml and detected_text:
            # use the first few words of the detected text
            med_name = " ".join(detected_text.split()[:2])
            
        if med_name:
            try:
                chat_completion = groq_client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a careful medical information assistant. Explain what this medicine is commonly used for and give brief, general instructions for how it is usually taken or applied. Reply in the detected language when practical. Do not prescribe, do not invent a dose, and advise following the package label or a pharmacist. Keep it to 2 or 3 concise sentences."
                        },
                        {
                            "role": "user",
                            "content": f"Detected language: {detected_language['code']}. What is {med_name} used for, and how is it generally used?"
                        }
                    ],
                    model="llama3-8b-8192",
                    temperature=0.3,
                    max_tokens=150
                )
                usage_info = chat_completion.choices[0].message.content
            except Exception as e:
                print(f"Groq API Error: {e}")

    if not usage_info and is_real_api and not is_expired and not is_suspected_fake:
        usage_info = api_result.get('indications_and_usage') or api_result.get('dosage_and_administration')

    return JSONResponse(content={
        "status": "EXPIRED" if is_expired else ("SUSPECTED_FAKE" if is_suspected_fake else "OK"),
        "message": "This medicine is expired. Do not use it; consult a pharmacist for safe disposal and replacement." if is_expired else ("This medicine is suspected to be counterfeit. Do not use it until a pharmacist, manufacturer, or regulator verifies it." if is_suspected_fake else None),
        "detected_text": detected_text,
        "detected_language": detected_language,
        "ocr_languages": ocr_languages_loaded,
        "expiry": expiry_info,
        "evidence": evidence,
        "api_verification": api_result,
        "ml_analysis": ml_result,
        "usage_info": usage_info if not is_expired and not is_suspected_fake else None,
        "google_search_url": google_search_url
    })

