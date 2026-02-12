# 🔍 Advanced Medicine Authenticity Checker

An AI-powered medicine verification system that combines **FDA/RxNorm API databases** with **Deep Learning** to detect fake medicines. Supports human medicines, veterinary medicines, and international brands.

![Medicine Checker](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)

---

## 🎯 **Features**

✅ **Multi-Source Verification:**
- FDA Database (US-approved medicines)
- RxNorm API (Generic medicine names)
- Deep Learning Model (Image-based fallback)

✅ **Advanced OCR:**
- EasyOCR for text extraction
- Confidence-based filtering
- Multi-language support

✅ **Smart Detection:**
- Extracts medicine names from packaging
- Handles dosage variations (500mg, 250ml, etc.)
- Supports brand and generic names

✅ **User-Friendly Interface:**
- Interactive Streamlit web app
- Two-column layout for easy viewing
- Adjustable confidence threshold
- Real-time verification results

---

## 🚀 **Quick Start**

### **Prerequisites**
- Python 3.8 or higher
- pip (Python package manager)

### **Installation**

1. **Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/Medicine-Authenticity-Checker.git
cd Medicine-Authenticity-Checker
```

2. **Create virtual environment (recommended):**
```bash
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate

# Mac/Linux:
source .venv/bin/activate
```

3. **Install dependencies:**
```bash
cd web_creat
pip install -r requirements.txt
```

4. **Run the application:**
```bash
streamlit run app.py
```

5. **Open in browser:**
```
http://localhost:8501
```

---

## 📂 **Project Structure**
```
Medicine-Authenticity-Checker/
│
├── web_creat/                      # Main application folder
│   ├── app.py                      # Streamlit web app
│   ├── api_verifier.py             # API verification logic
│   ├── config.py                   # Configuration settings
│   ├── medicine_model.pkl          # Trained ML model
│   └── requirements.txt            # Python dependencies
│
├── data/                           # Dataset folder (not in repo)
│   ├── full_medicine_dataset/
│   └── medicine_dataset.csv
│
├── models/                         # Model files (optional)
│   └── medicine_model.pkl
│
├── notebooks/                      # Jupyter notebooks (training)
│   └── model_training.ipynb
│
├── .gitignore                      # Git ignore file
├── README.md                       # This file
└── run.bat                         # Windows run script
```

---

## 🛠️ **Technology Stack**

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **OCR** | EasyOCR |
| **APIs** | FDA openAPI, RxNorm |
| **Deep Learning** | TensorFlow/Keras |
| **Image Processing** | OpenCV, PIL |
| **HTTP Requests** | Requests library |

---

## 📖 **How It Works**

### **Verification Process:**
```
1. Upload Medicine Image
         ↓
2. OCR Text Extraction (EasyOCR)
         ↓
3. Extract Medicine Name (Clean & Parse)
         ↓
4. API Verification (Priority Order):
   ├─→ FDA Database (Brand/Generic names)
   ├─→ RxNorm API (Alternative names)
   └─→ ML Model (If API fails)
         ↓
5. Display Verification Result
   ├─→ VERIFIED (High Confidence)
   ├─→ UNVERIFIED (Not in databases)
   └─→ FAKE/REAL (ML prediction)
```

### **API Verification Logic:**
```python
# Multiple search strategies
1. Search by brand name (e.g., "Crocin")
2. Search by generic name (e.g., "Paracetamol")
3. Search by active ingredient
4. Fallback to ML model if APIs fail
```

---

## 🎮 **Usage Guide**

### **Step 1: Upload Image**
- Click "Upload Medicine Image"
- Select clear photo of medicine packaging
- Supported formats: JPG, PNG, JPEG

### **Step 2: View OCR Results**
- App extracts text from image
- Shows detected medicine name

### **Step 3: API Verification**
- Automatically checks FDA database
- Falls back to RxNorm if not found
- Shows verification status

### **Step 4: ML Model Analysis**
- If not found in databases
- Uses trained deep learning model
- Shows confidence percentage

### **Step 5: Adjust Settings**
- Use sidebar slider for threshold
- Higher threshold = stricter verification

---

## ⚙️ **Configuration**

Edit `web_creat/config.py` to customize:
```python
# Image size (match your trained model)
IMAGE_SIZE = (150, 150)

# Confidence threshold
MODEL_CONFIDENCE_THRESHOLD = 0.7

# OCR settings
OCR_LANGUAGES = ['en']  # Add more: ['en', 'hi', 'bn']
OCR_USE_GPU = False     # Set True if you have GPU

# API timeout
API_TIMEOUT = 10  # seconds
```

---

## 📊 **Model Training**

The ML model was trained on:
- **Dataset:** Kaggle Medicine Dataset
- **Classes:** REAL vs FAKE
- **Architecture:** CNN (Convolutional Neural Network)
- **Input Size:** 150x150 RGB images
- **Accuracy:** ~85% (on test set)

To retrain:
```bash
# See notebooks/model_training.ipynb
jupyter notebook notebooks/model_training.ipynb
```

---

## 🧪 **Testing**

### **Test API Verifier:**
```bash
cd web_creat
python api_verifier.py
```

### **Test with Sample Images:**
```bash
streamlit run app.py
# Upload images from data/test_images/
```

---

## 📈 **Supported Medicines**

✅ **US-approved medicines** (FDA database)
✅ **Generic medicines** (RxNorm database)
✅ **Veterinary medicines** (FDA includes animal drugs)
✅ **International brands** (if in databases)

⚠️ **Limited support:**
- Very local/regional medicines not in US databases
- Traditional/Ayurvedic medicines (use ML model)

---

## 🚨 **Limitations**

1. **API Coverage:** Only medicines in FDA/RxNorm databases
2. **OCR Accuracy:** Depends on image quality
3. **ML Model:** Trained only on specific dataset
4. **Internet Required:** For API verification

---

## 🤝 **Contributing**

Contributions are welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📝 **To-Do List**

- [ ] Add more international medicine databases
- [ ] Improve ML model accuracy
- [ ] Add barcode/QR code scanning
- [ ] Multi-language OCR support
- [ ] Mobile app version
- [ ] Batch processing for multiple images

---

## ⚠️ **Disclaimer**

This tool is for **reference purposes only**. It should **NOT** replace professional medical advice or official verification by pharmacists/regulatory authorities.

**Always consult a licensed pharmacist or healthcare provider for medicine verification.**

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 **Author**

**Your Name**
- GitHub: [@your-username](https://github.com/your-username)
- LinkedIn: [Your Profile](https://linkedin.com/in/your-profile)
- Email: your.email@example.com

---

## 🙏 **Acknowledgments**

- **FDA openAPI** - For medicine database access
- **RxNorm** - For generic medicine names
- **EasyOCR** - For text extraction
- **Streamlit** - For web framework
- **Kaggle** - For medicine dataset

---

## 📞 **Support**

If you encounter issues:
1. Check [Issues](https://github.com/your-username/repo/issues) page
2. Create new issue with error details
3. Contact via email

---

## 📸 **Screenshots**

### Main Interface
![Main Interface](screenshots/main_interface.png)

### Verification Result
![Verification](screenshots/verification_result.png)

### Settings Panel
![Settings](screenshots/settings_panel.png)

---

## 🔗 **Useful Links**

- [FDA API Documentation](https://open.fda.gov/apis/)
- [RxNorm API Guide](https://rxnav.nlm.nih.gov/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [EasyOCR GitHub](https://github.com/JaidedAI/EasyOCR)

---

**⭐ If this project helped you, please give it a star!**

---

*Last Updated: February 2026*