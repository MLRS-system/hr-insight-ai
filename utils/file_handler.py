import os
import pandas as pd
import PyPDF2
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
import docx
from pptx import Presentation
import easyocr
from PIL import Image

# --- New Imports for Fallback Logic ---
from pdf2image import convert_from_path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

# --- Global Initializations for Performance ---
# OCR 리더는 한 번만 로드하여 재사용
ocr_reader = easyocr.Reader(['ko', 'en'], gpu=False)

# --- Fallback Helper Functions ---

def ocr_image(image_path: str):
    """주어진 이미지 경로에 대해 OCR을 수행하고 텍스트를 반환합니다."""
    try:
        print(f"    (OCR Fallback) - Performing OCR on: {image_path}")
        if not os.path.exists(image_path): return ""
        ocr_results = ocr_reader.readtext(image_path, paragraph=True)
        return "\n".join([res[1] for res in ocr_results])
    except Exception as e:
        print(f"    [ERROR] OCR process failed: {e}")
        return ""

def get_content_from_screenshot(url: str):
    """Selenium으로 URL의 스크린샷을 찍고 OCR을 수행하여 텍스트를 추출합니다."""
    print(f"    (Crawl Fallback) - Attempting screenshot OCR for URL: {url}")
    screenshot_path = "temp_screenshot.png"
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless") # UI 없이 백그라운드에서 실행
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("window-size=1920,1080")
        
        # Cloud Run 환경에서는 chromedriver 경로가 고정될 수 있습니다.
        # 로컬 테스트 시에는 chromedriver.exe 경로를 지정해야 할 수 있습니다.
        service = Service() 
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        driver.get(url)
        driver.save_screenshot(screenshot_path)
        driver.quit()
        
        print(f"    (Crawl Fallback) - Screenshot saved to {screenshot_path}")
        return ocr_image(screenshot_path), "text"
    except Exception as e:
        print(f"    [ERROR] Screenshot OCR failed: {e}")
        return None, None
    finally:
        if os.path.exists(screenshot_path):
            os.remove(screenshot_path) # 임시 스크린샷 파일 삭제

# --- Original Extractors ---
def get_text_from_article_url(url: str):
    print(f"    (File Handler) - Extracting text from article: {url}")
    try:
        headers = {'User-Agent': '...'} # User-Agent는 기존과 동일
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        paragraphs = soup.find_all('p')
        body_text = "\n".join([p.get_text() for p in paragraphs])

        # 크롤링 실패 시 (텍스트가 거의 없을 때) 플랜 B 실행
        if len(body_text.strip()) < 100:
            print("    (File Handler) - Low text content from crawl, attempting screenshot OCR.")
            return get_content_from_screenshot(url)
            
        return f"Article Body:\n{body_text}", "text"
    except Exception as e:
        print(f"    [ERROR] Failed to process article link: {e}. Attempting screenshot OCR.")
        return get_content_from_screenshot(url)

# ... (get_transcript_from_youtube_url 함수는 이전과 동일)

# --- Main Content Getter (수정된 버전) ---
def get_content_from_input(path_or_url: str):
    # ... (URL 처리 로직은 이전과 동일)

    file_path = path_or_url
    # ... (파일 경로 및 디렉터리 처리 로직은 이전과 동일)

    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        
        # 1. 이미지 파일 처리
        if file_extension in ['.jpg', '.jpeg', '.png']:
            return file_path, "file_path"
        
        # 2. 텍스트 추출 시도 (PDF, DOCX, PPTX 등)
        content, content_type = "", "text"
        if file_extension == '.pdf':
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f, strict=False)
                for page in reader.pages: content += page.extract_text() or ""
            
            # PDF 텍스트 추출 실패 시, 이미지로 변환하여 OCR 시도
            if len(content.strip()) < 20:
                print("    (PDF Fallback) - Low text from PDF, attempting OCR.")
                # Poppler 경로를 환경 변수나 직접 경로로 지정해야 합니다.
                # 로컬 테스트: poppler_path=r"C:\path\to\poppler-ver\bin"
                images = convert_from_path(file_path)
                content = ""
                for img in images:
                    img_path = "temp_pdf_page.png"
                    img.save(img_path)
                    content += ocr_image(img_path) + "\n"
                    os.remove(img_path)
        
        # ... (DOCX, PPTX, TXT 등 다른 텍스트 파일 처리 로직은 이전과 동일)

        return content, content_type

    except Exception as e:
        print(f"    [ERROR] Failed to process file {file_path}: {e}")
        return None, None