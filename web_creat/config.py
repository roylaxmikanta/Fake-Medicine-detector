# web_creat/config.py

"""
Configuration file for Medicine Authenticity Checker
Contains API settings, file paths, and application constants
"""

import os

# ==========================================
# FILE PATHS
# ==========================================

# Base directory (web_creat folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model file path
MODEL_PATH = os.path.join(BASE_DIR, 'medicine_model.pkl')

# Dataset paths (if needed)
DATASET_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data')

# ==========================================
# API ENDPOINTS (FREE APIs)
# ==========================================

# FDA API - US Food and Drug Administration
# Documentation: https://open.fda.gov/apis/
FDA_API_BASE = "https://api.fda.gov/drug/label.json"
FDA_NDC_API = "https://api.fda.gov/drug/ndc.json"  # National Drug Code

# RxNorm API - National Library of Medicine
# Documentation: https://rxnav.nlm.nih.gov/
RXNORM_API_BASE = "https://rxnav.nlm.nih.gov/REST"

# OpenFDA Drug Event API (for adverse events check)
FDA_EVENT_API = "https://api.fda.gov/drug/event.json"

# ==========================================
# API SETTINGS
# ==========================================

# Request timeout (seconds)
API_TIMEOUT = 10

# Maximum retries for failed requests
API_MAX_RETRIES = 2

# Rate limiting (requests per minute)
API_RATE_LIMIT = 60

# ==========================================
# MODEL SETTINGS
# ==========================================

# Image preprocessing
IMAGE_SIZE = (150, 150)  # (width, height) - CHANGE this based on your trained model
IMAGE_CHANNELS = 3  # RGB

# Model prediction threshold
MODEL_CONFIDENCE_THRESHOLD = 0.7  # 70% confidence

# Prediction classes
CLASS_LABELS = {
    0: "FAKE",
    1: "REAL"
}

# ==========================================
# OCR SETTINGS
# ==========================================

# Languages for EasyOCR
OCR_LANGUAGES = ['en']  # Add more: ['en', 'hi', 'bn'] for Hindi, Bengali

# Minimum OCR confidence to accept text
OCR_MIN_CONFIDENCE = 0.5  # 50%

# Use GPU for OCR (faster but needs CUDA)
OCR_USE_GPU = False  # Set to True if you have NVIDIA GPU

# ==========================================
# VERIFICATION LOGIC
# ==========================================

# Confidence levels
CONFIDENCE_LEVELS = {
    'HIGH': 0.9,      # 90%+ - Database verified
    'MEDIUM': 0.7,    # 70-90% - Partial match
    'LOW': 0.5        # 50-70% - ML model only
}

# Verification priority order
VERIFICATION_PRIORITY = [
    'FDA',       # Try FDA first (most reliable)
    'RxNorm',    # Then RxNorm (generic names)
    'ML_Model'   # Finally ML model (fallback)
]

# ==========================================
# TEXT PROCESSING
# ==========================================

# Common medicine name patterns to extract
MEDICINE_NAME_PATTERNS = [
    r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b',  # Capitalized words
    r'\b\d+\s*mg\b',                        # Dosage (e.g., 500 mg)
    r'\b\d+\s*ml\b',                        # Volume (e.g., 100 ml)
]

# Words to ignore (common packaging terms)
IGNORE_WORDS = [
    'tablets', 'capsules', 'syrup', 'cream', 'ointment',
    'injection', 'bottle', 'pack', 'strip', 'box',
    'use', 'before', 'expiry', 'mfg', 'exp', 'date'
]

# ==========================================
# UI SETTINGS
# ==========================================

# Page configuration
PAGE_TITLE = "Medicine Authenticity Checker"
PAGE_ICON = "💊"
PAGE_LAYOUT = "wide"

# Maximum file upload size (MB)
MAX_FILE_SIZE_MB = 10

# Allowed image formats
ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'webp']

# ==========================================
# LOGGING & DEBUGGING
# ==========================================

# Enable debug mode
DEBUG_MODE = False  # Set to True for detailed logs

# Log file path
LOG_FILE = os.path.join(BASE_DIR, 'app.log')

# ==========================================
# CACHE SETTINGS
# ==========================================

# Cache duration for API results (seconds)
CACHE_DURATION = 3600  # 1 hour

# Enable caching
ENABLE_CACHE = True

# ==========================================
# SECURITY & PRIVACY
# ==========================================

# Do not log uploaded images
LOG_IMAGES = False

# Clear uploaded files after processing
AUTO_CLEANUP = True

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_api_url(api_name, endpoint=''):
    """
    Get full API URL
    
    Args:
        api_name (str): 'FDA', 'RxNorm', etc.
        endpoint (str): Additional endpoint path
    
    Returns:
        str: Complete URL
    """
    api_bases = {
        'FDA': FDA_API_BASE,
        'FDA_NDC': FDA_NDC_API,
        'RxNorm': RXNORM_API_BASE,
        'FDA_EVENT': FDA_EVENT_API
    }
    
    base = api_bases.get(api_name, '')
    if endpoint:
        return f"{base}/{endpoint}"
    return base

def validate_config():
    """
    Validate configuration settings
    
    Returns:
        tuple: (is_valid, error_message)
    """
    errors = []
    
    # Check model file exists
    if not os.path.exists(MODEL_PATH):
        errors.append(f"Model file not found: {MODEL_PATH}")
    
    # Check image size
    if len(IMAGE_SIZE) != 2:
        errors.append("IMAGE_SIZE must be (width, height)")
    
    # Check threshold range
    if not 0 < MODEL_CONFIDENCE_THRESHOLD < 1:
        errors.append("MODEL_CONFIDENCE_THRESHOLD must be between 0 and 1")
    
    if errors:
        return False, "\n".join(errors)
    
    return True, "Configuration valid"

# ==========================================
# EXPORT IMPORTANT SETTINGS
# ==========================================

__all__ = [
    'FDA_API_BASE',
    'RXNORM_API_BASE',
    'API_TIMEOUT',
    'MODEL_PATH',
    'IMAGE_SIZE',
    'MODEL_CONFIDENCE_THRESHOLD',
    'OCR_LANGUAGES',
    'OCR_MIN_CONFIDENCE',
    'OCR_USE_GPU',
    'CLASS_LABELS',
    'CONFIDENCE_LEVELS',
    'get_api_url',
    'validate_config'
]

# ==========================================
# AUTO-VALIDATION ON IMPORT
# ==========================================

if __name__ != "__main__":
    # Auto-validate when imported (not when run directly)
    is_valid, message = validate_config()
    if not is_valid and DEBUG_MODE:
        print(f"⚠️ Configuration Warning: {message}")