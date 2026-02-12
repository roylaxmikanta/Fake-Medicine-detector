# web_creat/api_verifier.py

import requests
import re
from typing import Dict, Optional, List
from functools import lru_cache
import time

class MedicineAPIVerifier:
    """
    Multi-source medicine verification using free APIs
    Supports: FDA, RxNorm with caching and retry logic
    """
    
    def __init__(self):
        from config import (
            FDA_API_BASE, 
            RXNORM_API_BASE, 
            API_TIMEOUT,
            API_MAX_RETRIES,
            IGNORE_WORDS,
            DEBUG_MODE
        )
        self.fda_base = FDA_API_BASE
        self.rxnorm_base = RXNORM_API_BASE
        self.timeout = API_TIMEOUT
        self.max_retries = API_MAX_RETRIES
        self.ignore_words = [w.lower() for w in IGNORE_WORDS]
        self.debug = DEBUG_MODE
        
        # Cache for repeated searches
        self._cache = {}
    
    def _log(self, message: str):
        """Internal logging (only if debug mode)"""
        if self.debug:
            print(f"[MedicineAPIVerifier] {message}")
    
    def _make_request(self, url: str, retries: int = None) -> Optional[requests.Response]:
        """
        Make HTTP request with retry logic
        
        Args:
            url: API endpoint URL
            retries: Number of retry attempts (default from config)
        
        Returns:
            Response object or None
        """
        if retries is None:
            retries = self.max_retries
        
        for attempt in range(retries + 1):
            try:
                self._log(f"Requesting: {url} (Attempt {attempt + 1})")
                response = requests.get(url, timeout=self.timeout)
                
                if response.status_code == 200:
                    return response
                elif response.status_code == 404:
                    self._log(f"Not found (404): {url}")
                    return None
                else:
                    self._log(f"HTTP {response.status_code}: {url}")
                    
            except requests.Timeout:
                self._log(f"Timeout on attempt {attempt + 1}")
            except requests.RequestException as e:
                self._log(f"Request error: {str(e)}")
            
            # Wait before retry (exponential backoff)
            if attempt < retries:
                wait_time = 2 ** attempt  # 1s, 2s, 4s...
                self._log(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
        
        return None
    
    def clean_medicine_name(self, text: str) -> str:
        """
        Extract clean medicine name from OCR text
        
        Examples:
            "Paracetamol 500mg Tablets" → "Paracetamol"
            "Amoxicillin 250 MG/5ML" → "Amoxicillin"
            "CROCIN ADVANCE" → "Crocin Advance"
        """
        # Remove special characters but keep spaces
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove dosage patterns (500mg, 10ml, etc.)
        cleaned = re.sub(r'\d+\s*(mg|ml|g|mcg|iu)\b', '', cleaned, flags=re.IGNORECASE)
        
        # Remove numbers
        cleaned = re.sub(r'\b\d+\b', '', cleaned)
        
        # Split into words
        words = cleaned.split()
        
        # Filter out common packaging terms
        filtered_words = [
            word for word in words 
            if word.lower() not in self.ignore_words and len(word) > 2
        ]
        
        # Take first 2 meaningful words (handles "Crocin Advance", "Dolo 650", etc.)
        medicine_name = ' '.join(filtered_words[:2]) if filtered_words else text
        
        return medicine_name.strip().title()
    
    def extract_all_possible_names(self, text: str) -> List[str]:
        """
        Extract multiple possible medicine names from text
        
        Returns:
            List of potential medicine names to search
        """
        names = []
        
        # Original cleaned name
        main_name = self.clean_medicine_name(text)
        names.append(main_name)
        
        # First word only (brand name)
        first_word = main_name.split()[0] if main_name else ''
        if first_word and first_word not in names:
            names.append(first_word)
        
        # Capitalized words (often brand names)
        caps_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        caps_matches = re.findall(caps_pattern, text)
        for match in caps_matches:
            if match not in names and len(match) > 3:
                names.append(match)
        
        return names
    
    @lru_cache(maxsize=100)
    def verify_with_fda(self, medicine_name: str) -> Optional[Dict]:
        """
        Verify medicine using FDA API
        Supports: Brand name, generic name search
        """
        self._log(f"Checking FDA for: {medicine_name}")
        
        # Try multiple search strategies
        search_fields = [
            f"openfda.brand_name:{medicine_name}",           # Brand name
            f"openfda.generic_name:{medicine_name}",         # Generic name
            f"openfda.substance_name:{medicine_name}",       # Active ingredient
        ]
        
        for search_query in search_fields:
            try:
                url = f"{self.fda_base}?search={search_query}&limit=1"
                response = self._make_request(url)
                
                if response and response.status_code == 200:
                    data = response.json()
                    
                    if data.get('results'):
                        result = data['results'][0]
                        openfda = result.get('openfda', {})
                        
                        return {
                            'status': 'VERIFIED',
                            'source': 'FDA Database',
                            'medicine_name': medicine_name,
                            'brand_name': openfda.get('brand_name', ['N/A'])[0],
                            'generic_name': openfda.get('generic_name', ['N/A'])[0],
                            'manufacturer': openfda.get('manufacturer_name', ['Unknown'])[0],
                            'type': openfda.get('product_type', ['Unknown'])[0],
                            'route': openfda.get('route', ['N/A'])[0],
                            'confidence': 'HIGH'
                        }
            except Exception as e:
                self._log(f"FDA API Error: {str(e)}")
        
        return None
    
    @lru_cache(maxsize=100)
    def verify_with_rxnorm(self, medicine_name: str) -> Optional[Dict]:
        """
        Verify medicine using RxNorm API
        Supports: Generic and brand names
        """
        self._log(f"Checking RxNorm for: {medicine_name}")
        
        try:
            # RxNorm approximate match (handles typos)
            url = f"{self.rxnorm_base}/approximateTerm.json?term={medicine_name}"
            response = self._make_request(url)
            
            if response and response.status_code == 200:
                data = response.json()
                candidates = data.get('approximateGroup', {}).get('candidate')
                
                if candidates:
                    # Take best match (first result)
                    best_match = candidates[0] if isinstance(candidates, list) else candidates
                    
                    return {
                        'status': 'VERIFIED',
                        'source': 'RxNorm Database',
                        'medicine_name': medicine_name,
                        'matched_name': best_match.get('name', medicine_name),
                        'rxcui': best_match.get('rxcui', 'N/A'),
                        'confidence': 'MEDIUM'
                    }
        except Exception as e:
            self._log(f"RxNorm API Error: {str(e)}")
        
        return None
    
    def verify_medicine(self, text: str) -> Dict:
        """
        Main verification function - tries multiple sources and names
        
        Args:
            text: Raw OCR text from medicine image
        
        Returns:
            Dict with verification result
        """
        # Validate input
        if not text or len(text.strip()) < 3:
            return {
                'status': 'INVALID', 
                'message': 'Text too short or empty',
                'confidence': 'NONE'
            }
        
        # Check cache first
        cache_key = text.lower().strip()
        if cache_key in self._cache:
            self._log(f"Cache hit for: {text}")
            return self._cache[cache_key]
        
        # Extract all possible medicine names
        possible_names = self.extract_all_possible_names(text)
        self._log(f"Searching for: {possible_names}")
        
        # Try each name with FDA first (most reliable)
        for name in possible_names:
            fda_result = self.verify_with_fda(name)
            if fda_result:
                self._cache[cache_key] = fda_result
                return fda_result
        
        # Try RxNorm as backup
        for name in possible_names:
            rxnorm_result = self.verify_with_rxnorm(name)
            if rxnorm_result:
                self._cache[cache_key] = rxnorm_result
                return rxnorm_result
        
        # If all fail, return unverified
        result = {
            'status': 'UNVERIFIED',
            'source': 'No API Match',
            'medicine_name': possible_names[0] if possible_names else text,
            'searched_names': possible_names,
            'message': 'Not found in FDA/RxNorm databases. May be non-US medicine or local brand.',
            'confidence': 'UNKNOWN'
        }
        
        self._cache[cache_key] = result
        return result
    
    def clear_cache(self):
        """Clear the verification cache"""
        self._cache.clear()
        self.verify_with_fda.cache_clear()
        self.verify_with_rxnorm.cache_clear()
        self._log("Cache cleared")


# Standalone testing
if __name__ == "__main__":
    verifier = MedicineAPIVerifier()
    
    # Test cases
    test_medicines = [
        "Paracetamol 500mg",
        "CROCIN ADVANCE",
        "Amoxicillin 250mg",
        "Aspirin",
        "RandomFakeMedicine123"
    ]
    
    print("=" * 60)
    print("Testing Medicine API Verifier")
    print("=" * 60)
    
    for med in test_medicines:
        print(f"\n🔍 Testing: {med}")
        result = verifier.verify_medicine(med)
        print(f"   Status: {result['status']}")
        print(f"   Source: {result.get('source', 'N/A')}")
        print(f"   Confidence: {result.get('confidence', 'N/A')}")
