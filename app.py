import streamlit as st
import joblib
import cv2
import easyocr
import numpy as np
from PIL import Image
import os


# Page Setup
st.set_page_config(page_title="Fake Medicine Detector", page_icon="💊", layout="centered")


# Title and Design
st.title("💊 Fake Medicine Detector")
st.write("Dawai ki packaging ki photo upload karein aur check karein ki wo Asli hai ya Nakli.")


# Load Model & OCR (Cache use karenge taaki fast chale)
@st.cache_resource
def load_tools():
    """Load the trained model and OCR reader"""
    try:
        model = joblib.load('medicine_model.pkl')
        reader = easyocr.Reader(['en'], gpu=False)
        return model, reader
    except FileNotFoundError:
        st.error("❌ Model file 'medicine_model.pkl' not found! Please ensure it's in the same directory.")
        st.stop()


st.write("⏳ Loading AI Model... (Please wait)")
model, reader = load_tools()
st.success("✅ System Ready!")


# Upload Section
st.markdown("---")
st.subheader("📸 Upload Medicine Image")
uploaded_file = st.file_uploader("Choose a medicine packaging image", type=["jpg", "png", "jpeg"])


if uploaded_file is not None:
    # Image Display
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with col2:
        st.write("**Image Details:**")
        st.write(f"📁 File Name: {uploaded_file.name}")
        st.write(f"📏 File Size: {uploaded_file.size / 1024:.2f} KB")
        st.write(f"📐 Image Size: {image.size}")
   
    if st.button('🔍 SCAN MEDICINE', key='scan_button', use_container_width=True):
        with st.spinner('🔄 Scanning text details... This may take a moment...'):
            # Image process
            image_np = np.array(image)
           
            # Convert BGR to RGB if needed
            if len(image_np.shape) == 3:
                image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR) if image_np.dtype == np.uint8 else image_np
            
            # OCR Extraction
            try:
                results = reader.readtext(image_np, detail=0)
                full_text = " ".join(results)
               
                st.markdown("---")
                st.subheader("📝 Extracted Text:")
                st.text_area("OCR Output", full_text, height=100, disabled=True)
               
                if len(full_text.strip()) < 3:
                    st.warning("⚠️ Photo mein text saaf nahi dikh raha. Kripya clear aur proper tarike se photo lein.")
                else:
                    # Prediction
                    prediction = model.predict([full_text])[0]
                    probability = model.predict_proba([full_text]).max() * 100
                   
                    st.markdown("---")
                    st.subheader("💡 Final Result:")
                   
                    if prediction == 'Fake':
                        st.error(f"🚨 FAKE MEDICINE DETECTED!")
                        st.metric("Confidence Level", f"{probability:.1f}%", delta="High Risk")
                        st.warning("⚠️ **WARNING**: Ye medicine NAKLI (Fake) hai! Ise use na karein. Foran apne doctor ya chemist se consultation karein.")
                    else:
                        st.success(f"✅ REAL MEDICINE DETECTED!")
                        st.metric("Confidence Level", f"{probability:.1f}%", delta="Verified")
                        st.info("✔️ **VERIFIED**: Ye medicine ASLI (Real) lagti hai. Lekin fir bhi apne doctor ke instructions follow karein.")
                       
            except Exception as e:
                st.error(f"❌ Error during scanning: {e}")
                st.info("Please try with a clearer image or a different medicine packaging photo.")


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>⚠️ Important Disclaimer:</strong></p>
    <p>Ye AI model ka accuracy approximately 82% hai. Ye sirf preliminary screening ke liye hai.</p>
    <p>Final verification ke liye hamesha apne chemist ya authorized medical representative se check karwayein.</p>
    <p><strong>Always verify with official sources before using any medicine.</strong></p>
</div>
""", unsafe_allow_html=True)
