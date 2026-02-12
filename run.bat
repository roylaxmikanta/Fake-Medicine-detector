@echo off
echo Starting Medicine Detector App...

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Go to web_creat folder
cd web_creat

REM Install/update dependencies
pip install -r requirements.txt

REM Run Streamlit app
streamlit run app.py

pause
```

---

### **Step 5: Create `.gitignore` (For GitHub)**
```
# web_creat/.gitignore

# Virtual Environment
.venv/
venv/
env/

# Python cache
__pycache__/
*.pyc
*.pyo

# Model files (large)
*.pkl
*.h5

# Data files (large)
*.csv
full_medicine_dataset/

# OS files
.DS_Store
Thumbs.db

# Streamlit cache
.streamlit/