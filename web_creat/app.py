# web_creat/app.py

import streamlit as st
import pickle
import cv2
import numpy as np
from PIL import Image
import easyocr
from api_verifier import MedicineAPIVerifier
import os

# Page config
st.set_page_config(
    page_title="Medicine Authenticity Checker",
    page_icon="💊",
    layout="wide"
)

# Configuration
MODEL_PATH = 'medicine_model.pkl'
IMAGE_SIZE = (150, 150)  # Change this based on your model
CONFIDENCE_THRESHOLD = 0.7

# Initialize with error handling
@st.cache_resource
def load_model():
    """Load ML model with error handling"""
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Model file not found: {MODEL_PATH}")
            return None
        
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

@st.cache_resource
def load_ocr():
    """Load EasyOCR with error handling"""
    try:
        return easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have GPU
    except Exception as e:
        st.error(f"❌ Error loading OCR: {str(e)}")
        return None

@st.cache_resource
def load_api_verifier():
    """Load API verifier"""
    try:
        return MedicineAPIVerifier()
    except Exception as e:
        st.error(f"❌ Error loading API verifier: {str(e)}")
        return None

def preprocess_image(image, target_size=IMAGE_SIZE):
    """Preprocess image for ML model"""
    try:
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Convert RGB to BGR for CV2 (if needed by your model)
        # img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Resize
        img_resized = cv2.resize(img_array, target_size)
        
        # Normalize
        img_normalized = img_resized / 255.0
        
        # Add batch dimension
        img_input = np.expand_dims(img_normalized, axis=0)
        
        return img_input
    except Exception as e:
        st.error(f"❌ Image preprocessing error: {str(e)}")
        return None

def extract_text_from_image(image, reader):
    """Extract text using OCR"""
    try:
        img_array = np.array(image)
        ocr_results = reader.readtext(img_array)
        
        if not ocr_results:
            return None
        
        # Extract text with confidence
        texts = []
        for detection in ocr_results:
            text = detection[1]
            confidence = detection[2]
            if confidence > 0.5:  # Only keep high-confidence text
                texts.append(text)
        
        return ' '.join(texts) if texts else None
        
    except Exception as e:
        st.error(f"❌ OCR error: {str(e)}")
        return None

def predict_with_model(image, model):
    """Run ML model prediction"""
    try:
        img_input = preprocess_image(image)
        if img_input is None:
            return None
        
        prediction = model.predict(img_input, verbose=0)[0][0]
        return float(prediction)
        
    except Exception as e:
        st.error(f"❌ Model prediction error: {str(e)}")
        return None

# Load resources
model = load_model()
reader = load_ocr()
api_verifier = load_api_verifier()

# Main UI
st.title("🔍 Advanced Medicine Authenticity Checker")
st.markdown("**Supports:** Human Medicine, Veterinary Medicine, and International Medicines")

# Sidebar
with st.sidebar:
    st.title("ℹ️ About")
    st.markdown("""
    ### Verification Methods:
    1. **🌐 FDA Database** - US medicines
    2. **💊 RxNorm** - Generic names  
    3. **🤖 ML Model** - Image-based (fallback)
    
    ### Supported Medicines:
    - ✅ Human medicines
    - ✅ Veterinary medicines
    - ✅ International medicines
    
    ### Confidence Levels:
    - **HIGH**: Verified in databases
    - **MEDIUM**: Found but limited info
    - **LOW**: ML model prediction only
    """)
    
    st.write("---")
    
    # Settings
    st.subheader("⚙️ Settings")
    threshold = st.slider(
        "ML Model Threshold", 
        min_value=0.5, 
        max_value=0.95, 
        value=CONFIDENCE_THRESHOLD,
        help="Higher = More strict"
    )

# File uploader
uploaded_file = st.file_uploader(
    "📤 Upload Medicine Image", 
    type=['jpg', 'jpeg', 'png'],
    help="Upload a clear photo of the medicine packaging"
)

if uploaded_file:
    # Create columns for better layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Medicine", use_column_width=True)
    
    with col2:
        # Check if resources are loaded
        if reader is None or api_verifier is None:
            st.error("❌ System not ready. Please refresh the page.")
            st.stop()
        
        # Step 1: OCR
        st.subheader("📝 Step 1: Text Extraction")
        with st.spinner("🔎 Reading text from image..."):
            detected_text = extract_text_from_image(image, reader)
            
            if detected_text:
                st.success(f"✅ Detected: **{detected_text}**")
            else:
                st.warning("⚠️ No clear text detected")
        
        # Step 2: API Verification
        st.subheader("🌐 Step 2: Database Check")
        
        if detected_text:
            with st.spinner("Checking international databases..."):
                api_result = api_verifier.verify_medicine(detected_text)
                
                if api_result['status'] == 'VERIFIED':
                    st.success(f"✅ **REAL MEDICINE**")
                    
                    # Display details in a nice format
                    with st.container():
                        st.markdown(f"**Source:** {api_result['source']}")
                        st.markdown(f"**Medicine:** {api_result['medicine_name']}")
                        
                        if 'manufacturer' in api_result:
                            st.markdown(f"**Manufacturer:** {api_result['manufacturer']}")
                        if 'type' in api_result:
                            st.markdown(f"**Type:** {api_result['type']}")
                        
                        st.markdown(f"**Confidence:** :green[{api_result['confidence']}]")
                    
                elif api_result['status'] == 'UNVERIFIED':
                    st.warning("⚠️ **NOT FOUND in Databases**")
                    st.caption(api_result.get('message', 'No match found'))
                    
                    # Step 3: ML Model Fallback
                    if model is not None:
                        st.subheader("🤖 Step 3: ML Model Analysis")
                        
                        with st.spinner("Analyzing image..."):
                            prediction = predict_with_model(image, model)
                            
                            if prediction is not None:
                                # Display result
                                if prediction > threshold:
                                    st.success(f"✅ **LIKELY REAL**")
                                    st.metric(
                                        "Confidence", 
                                        f"{prediction*100:.1f}%",
                                        delta="Above threshold"
                                    )
                                else:
                                    st.error(f"❌ **LIKELY FAKE**")
                                    st.metric(
                                        "Confidence", 
                                        f"{(1-prediction)*100:.1f}%",
                                        delta="Below threshold",
                                        delta_color="inverse"
                                    )
                                
                                # Progress bar for confidence
                                st.progress(prediction)
                            else:
                                st.error("❌ Could not analyze image")
                    else:
                        st.error("❌ ML model not available")
                else:
                    st.error("❌ Invalid input")
        else:
            # If no text detected, try model directly
            st.info("ℹ️ Skipping database check (no text found)")
            
            if model is not None:
                st.subheader("🤖 ML Model Analysis")
                
                with st.spinner("Analyzing image..."):
                    prediction = predict_with_model(image, model)
                    
                    if prediction is not None:
                        if prediction > threshold:
                            st.success(f"✅ **LIKELY REAL** ({prediction*100:.1f}%)")
                        else:
                            st.error(f"❌ **LIKELY FAKE** ({(1-prediction)*100:.1f}%)")
                        
                        st.progress(prediction)

# Footer
st.markdown("---")
st.caption("⚠️ This tool is for reference only. Always consult a pharmacist for verification.")