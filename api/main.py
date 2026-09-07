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
import pickle
import numpy as np
from dotenv import load_dotenv
import groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = groq.Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Adjust imports from local module
from .api_verifier import MedicineAPIVerifier

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
api_verifier = None
class_names = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.on_event("startup")
async def startup_event():
    global model, reader, api_verifier, class_names
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
        reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    except Exception as e:
        print(f"Error loading EasyOCR: {e}")

    # Load API Verifier
    api_verifier = MedicineAPIVerifier()
    print("Resources loaded successfully.")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "ocr_loaded": reader is not None
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
            texts = [det[1] for det in ocr_results if det[2] > 0.5]
            if texts:
                detected_text = ' '.join(texts)
        except Exception as e:
            print(f"OCR Error: {e}")

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

    # 4. Fetch Usage Information (Groq)
    usage_info = None
    is_real_api = api_result and api_result.get('status') == 'VERIFIED'
    is_real_ml = ml_result and ml_result.get('prediction', '').lower() == 'real'
    
    if (is_real_api or is_real_ml) and groq_client:
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
                            "content": "You are a helpful medical assistant. Describe what the given medicine is used for in 1 or 2 concise, easy-to-understand sentences."
                        },
                        {
                            "role": "user",
                            "content": f"What is the medicine {med_name} used for?"
                        }
                    ],
                    model="llama3-8b-8192",
                    temperature=0.3,
                    max_tokens=150
                )
                usage_info = chat_completion.choices[0].message.content
            except Exception as e:
                print(f"Groq API Error: {e}")

    return JSONResponse(content={
        "detected_text": detected_text,
        "api_verification": api_result,
        "ml_analysis": ml_result,
        "usage_info": usage_info
    })
