# utils/ocr_handler.py

import os
from google.cloud import vision
import pytesseract
from PIL import Image

# --- OCR Usage Tracker ---
# 간단한 인메모리 카운터. 서버가 재시작되면 초기화됩니다.
# 더 정확한 추적을 위해서는 DB나 외부 서비스 연동이 필요합니다.
VISION_API_CALL_COUNT = 0
MONTHLY_FREE_LIMIT = 1000

# --- OCR Implementations ---

def ocr_with_google_vision(image_path: str):
    """Google Cloud Vision AI를 사용하여 이미지에서 텍스트를 추출합니다."""
    global VISION_API_CALL_COUNT
    VISION_API_CALL_COUNT += 1
    print(f"    (OCR Handler) - Using Google Cloud Vision (Call #{VISION_API_CALL_COUNT})")
    
    client = vision.ImageAnnotatorClient()
    with open(image_path, "rb") as image_file:
        content = image_file.read()
    
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if texts:
        return texts[0].description
    return "Google Vision: No text found."

def ocr_with_tesseract(image_path: str):
    """Tesseract OCR 엔진을 사용하여 이미지에서 텍스트를 추출합니다."""
    print("    (OCR Handler) - Using Tesseract (Fallback)")
    try:
        return pytesseract.image_to_string(Image.open(image_path), lang='kor+eng')
    except Exception as e:
        print(f"    [ERROR] Tesseract failed: {e}")
        return "Tesseract: OCR failed."

# --- Main Hybrid OCR Function ---

def perform_hybrid_ocr(image_path: str) -> str:
    """
    하이브리드 OCR을 수행합니다. 무료 사용량 내에서는 Google Vision을,
    초과 시에는 Tesseract를 사용합니다.
    """
    if VISION_API_CALL_COUNT < MONTHLY_FREE_LIMIT:
        try:
            return ocr_with_google_vision(image_path)
        except Exception as e:
            print(f"    [ERROR] Google Vision API failed: {e}. Falling back to Tesseract.")
            return ocr_with_tesseract(image_path)
    else:
        print(f"    (OCR Handler) - Google Vision API monthly limit reached. Using Tesseract.")
        return ocr_with_tesseract(image_path)