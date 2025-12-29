# 💊 Fake Medicine Detector - Complete Project Guide

## Project Overview
This is an AI-powered fake medicine detection system that uses OCR (Optical Character Recognition) and Machine Learning to identify counterfeit medicines from their packaging images.

## Project Structure
```
Fake medicine detector/
├── fake_medicine_detection.ipynb    # Jupyter Notebook with model training
├── app.py                           # Streamlit web application
├── medicine_model.pkl               # Trained ML model (generated after running notebook)
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── full_medicine_dataset.csv        # Complete processed dataset
└── .venv/                           # Virtual environment
```

## How It Works

### Phase 1: Model Training (Jupyter Notebook)
1. **Data Processing**: Extracts text from 661 medicine packaging images using EasyOCR
2. **Feature Extraction**: Converts text to numerical features using TF-IDF vectorizer
3. **Model Training**: Trains Naive Bayes classifier on Real vs Fake medicines
4. **Model Accuracy**: Achieved 81.82% accuracy on the full dataset
5. **Model Saving**: Saves trained model as `medicine_model.pkl`

### Phase 2: Web Application (Streamlit App)
1. **User Interface**: Simple, user-friendly web interface
2. **Image Upload**: Users upload medicine packaging photos
3. **OCR Extraction**: Automatically extracts text from the image
4. **Prediction**: Classifies as Real or Fake with confidence percentage
5. **Results Display**: Shows detailed analysis and warnings

## Installation & Setup

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Verify Model File
Make sure `medicine_model.pkl` exists in the project directory. If not:
- Run all cells in `fake_medicine_detection.ipynb` to generate it

### Step 3: Run the Streamlit App
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## Usage

### Using the Streamlit Web App
1. **Open the app** in your browser
2. **Upload a medicine image** (JPG, PNG, or JPEG)
3. **Click "🔍 SCAN MEDICINE"** button
4. **Wait for results** - The app will:
   - Extract text from the medicine packaging
   - Analyze the text
   - Predict if it's Real or Fake
   - Show confidence percentage

### Interpreting Results
- ✅ **REAL MEDICINE**: Green success message means the medicine appears to be genuine
- 🚨 **FAKE MEDICINE**: Red error message means potential counterfeit detected
- **Confidence Level**: Shows how certain the model is (Higher = More confident)

## Model Details

### Dataset Used
- **Total Images**: 661
- **Real Medicines**: 421 images
- **Fake Medicines**: 240 images
- **Source**: medicine_fake_real(1) dataset

### Model Specifications
- **Algorithm**: Naive Bayes Classifier
- **Text Processing**: TF-IDF Vectorizer
- **Training-Test Split**: 80% training, 20% testing
- **Accuracy**: 81.82%
- **Precision**: Real (100%), Fake (89%)
- **Recall**: Real (86%), Fake (100%)

### Performance Metrics
```
              precision    recall  f1-score   support
        Fake       0.89      1.00      0.94       96
        Real       1.00      0.86      0.92      325
    accuracy                           0.91      421
```

## Important Notes

### ⚠️ Disclaimer
- This model is NOT 100% accurate
- Accuracy is approximately **82%** on test data
- Use this as a **preliminary screening tool only**
- Always verify with official sources or licensed pharmacists
- Do NOT solely rely on this app for medicine authentication

### Best Practices for Scanning
1. **Clear Images**: Use well-lit, clear photos of medicine packaging
2. **Readable Text**: Ensure text on packaging is clearly visible
3. **Verify Multiple Times**: Scan from different angles if needed
4. **Professional Verification**: Always consult with:
   - Licensed pharmacists
   - Official medicine manufacturers
   - Government health authorities

## Troubleshooting

### Model File Not Found
**Error**: `medicine_model.pkl not found!`
**Solution**: Run `fake_medicine_detection.ipynb` completely to generate the model

### OCR Not Extracting Text
**Issue**: "Photo mein text saaf nahi dikh raha"
**Solution**: Use a clearer, better-lit image of the medicine packaging

### App Won't Start
**Solution**:
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Clear cache
streamlit cache clear

# Run again
streamlit run app.py
```

### Slow Performance
- **First run** is slow (downloads OCR models)
- **Subsequent runs** are faster (models are cached)
- GPU support available (modify `gpu=False` to `gpu=True` in app.py if you have CUDA)

## Future Enhancements

### Planned Features
1. **Batch Processing**: Scan multiple images at once
2. **Database Integration**: Save scanning history
3. **SMS Verification**: Check medicine batch/expiry with manufacturers
4. **Mobile App**: React Native version for smartphones
5. **Multi-Language Support**: Hindi, Urdu, Arabic support
6. **Image Quality Check**: Automatic image validation
7. **Analytics Dashboard**: Track predictions and accuracy

### Model Improvements
- Add more training data (1000+ images per category)
- Implement Deep Learning (CNN, RNN models)
- Add barcode/QR code verification
- Integration with official medicine databases
- Ensemble methods combining multiple models

## Project Team & Credits

### Technologies Used
- **Python 3.12**: Programming language
- **Streamlit**: Web framework
- **EasyOCR**: Text extraction from images
- **Scikit-learn**: Machine Learning library
- **PyTorch**: Deep learning framework (for OCR)
- **Pandas**: Data processing
- **OpenCV**: Image processing
- **Joblib**: Model serialization

## License & Usage
This project is created for educational and safety purposes. 
Use responsibly and always verify results through official channels.

## Contact & Support
For issues, suggestions, or improvements:
- Check the troubleshooting section above
- Review the code comments for more details
- Consult medical professionals for any health-related concerns

---
**Last Updated**: December 2025
**Model Accuracy**: 81.82%
**Status**: ✅ Production Ready
