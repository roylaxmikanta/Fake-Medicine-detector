# 🔧 Fake Medicine Detector - Configuration Guide

## System Requirements

### Minimum Requirements
- **OS**: Windows 10+, macOS 10.13+, Linux (Ubuntu 18.04+)
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 2GB free space
- **Python**: 3.8 - 3.12

### Recommended Setup
- **RAM**: 8GB+
- **GPU**: NVIDIA GPU with CUDA support (optional, for faster OCR)
- **Storage**: 5GB free space
- **Internet**: Required for first run (to download models)

## Installation Steps

### 1️⃣ **Install Python**
Download from: https://www.python.org/downloads/
- ✅ Check "Add Python to PATH" during installation
- Choose version 3.8 to 3.12

### 2️⃣ **Setup Project**

**Option A: Using Batch Script (Windows - Easiest)**
```bash
# Just double-click run.bat
run.bat
```

**Option B: Using PowerShell (Windows - Advanced)**
```powershell
# Run as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\run.ps1
```

**Option C: Manual Setup (All Platforms)**
```bash
# Navigate to project directory
cd "C:\Users\royla\OneDrive\Desktop\Fake medicine detector"

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3️⃣ **Run the Application**
```bash
# Make sure virtual environment is activated
streamlit run app.py
```

## Configuration Options

### Enable GPU Acceleration (Optional)

If you have an NVIDIA GPU with CUDA:

1. **Install CUDA Toolkit**: https://developer.nvidia.com/cuda-downloads
2. **Install cuDNN**: https://developer.nvidia.com/cudnn
3. **Modify app.py** - Change line 21:
   ```python
   # Change from:
   reader = easyocr.Reader(['en'], gpu=False)
   
   # To:
   reader = easyocr.Reader(['en'], gpu=True)
   ```
4. **Install GPU PyTorch**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### Adjust Model Performance

**For Faster But Less Accurate Results:**
1. Open `fake_medicine_detection.ipynb`
2. In data processing, reduce `limit=50` parameter
3. Retrain the model

**For Better Accuracy:**
1. Increase limit to 100+ images
2. Use more training data
3. Adjust model parameters in training cell

### OCR Language Support

To add more languages (currently supports English):

Edit `app.py` line 21:
```python
# Single language
reader = easyocr.Reader(['en'], gpu=False)

# Multiple languages
reader = easyocr.Reader(['en', 'hi', 'ur'], gpu=False)  # English, Hindi, Urdu
```

Available language codes:
- `en`: English
- `hi`: Hindi
- `ur`: Urdu
- `ar`: Arabic
- `fr`: French
- `es`: Spanish
- `de`: German
- `zh`: Chinese

## Project File Descriptions

| File | Purpose |
|------|---------|
| `fake_medicine_detection.ipynb` | Jupyter notebook for model training |
| `app.py` | Main Streamlit application |
| `medicine_model.pkl` | Trained ML model (binary format) |
| `requirements.txt` | Python dependencies list |
| `run.bat` | Windows quick start script |
| `run.ps1` | PowerShell quick start script |
| `README.md` | Complete project documentation |
| `CONFIG.md` | This configuration guide |
| `.venv/` | Virtual environment folder |
| `full_medicine_dataset.csv` | Complete processed dataset |
| `medicine_dataset.csv` | Sample dataset (50 samples) |

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named..."

**Solution:**
```bash
# Activate virtual environment first
.venv\Scripts\activate

# Reinstall all dependencies
pip install --upgrade -r requirements.txt
```

### Issue: "medicine_model.pkl not found"

**Solution:**
1. Open `fake_medicine_detection.ipynb` in Jupyter
2. Run all cells from top to bottom
3. This will generate `medicine_model.pkl`
4. Then run `app.py`

### Issue: Streamlit not found

**Solution:**
```bash
# Deactivate and reactivate virtual environment
deactivate
.venv\Scripts\activate

# Install Streamlit specifically
pip install streamlit
```

### Issue: OCR is very slow

**Solution:**
- First run downloads OCR models (takes 5-10 minutes)
- Subsequent runs will be much faster
- If you have GPU, enable it (see GPU section above)

### Issue: "CUDA out of memory"

**Solution:**
```python
# In app.py, change gpu=True back to gpu=False
reader = easyocr.Reader(['en'], gpu=False)
```

### Issue: Port 8501 already in use

**Solution:**
```bash
# Run on different port
streamlit run app.py --server.port 8502
```

## Performance Optimization

### Reduce Startup Time
```python
# app.py - Clear cache on startup
streamlit cache clear
```

### Reduce Memory Usage
- Use `gpu=False` if you have limited RAM
- Process images one at a time
- Clear browser cache

### Improve OCR Accuracy
- Use high-quality images (300+ DPI)
- Ensure good lighting
- Keep text in focus
- Avoid tilted images

## Monitoring and Logs

### View Streamlit Logs
```bash
# Logs appear in terminal where you ran streamlit run app.py
# Check for warnings and errors

# More detailed logging:
streamlit run app.py --logger.level=debug
```

### Model Performance Tracking
- Check terminal output during app startup
- Monitor OCR processing time
- Track prediction confidence scores

## Security Considerations

1. **Image Privacy**: Images are NOT stored (processed in-memory only)
2. **Model Security**: Keep `medicine_model.pkl` safe
3. **API Keys**: None required (fully local)
4. **Data**: No data sent to external servers

## Backup & Recovery

### Backup Important Files
```bash
# Files to backup:
# - medicine_model.pkl (trained model)
# - full_medicine_dataset.csv (training data)
# - app.py (application code)
# - fake_medicine_detection.ipynb (training notebook)
```

### Restore from Backup
```bash
# If medicine_model.pkl is corrupted:
# 1. Restore backup
# 2. Or rerun fake_medicine_detection.ipynb to regenerate
```

## Performance Metrics

### Typical Performance
- **First Load**: 30-60 seconds (OCR model download)
- **Subsequent Loads**: 5-10 seconds
- **Image Processing**: 10-20 seconds per image
- **Prediction**: <1 second

### System Impact
- **Idle Memory**: 500MB - 1GB
- **During Processing**: 2-4GB
- **CPU Usage**: 30-80% during OCR

## Advanced Configuration

### Custom Model Training
See `fake_medicine_detection.ipynb` for:
- Changing dataset paths
- Adjusting train/test split
- Tuning model parameters
- Adding new features

### API Integration (Future)
For external integration:
```python
# Can be refactored to expose API endpoints
# Possible integrations:
# - Pharmacy management systems
# - E-commerce platforms
# - Healthcare applications
```

## Contact & Support

For detailed help:
1. Check the main [README.md](README.md)
2. Review code comments in `app.py`
3. Check Streamlit documentation: https://docs.streamlit.io/
4. EasyOCR docs: https://github.com/JaidedAI/EasyOCR

---

**Last Updated**: December 2025
**Config Version**: 1.0
**Status**: ✅ Verified & Tested
