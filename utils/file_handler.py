import os
import pandas as pd
import PyPDF2
from PIL import Image
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi

# --- Web Content Extractors --- #

def get_text_from_article_url(url: str):
    """Extracts the main text content from an article URL."""
    print(f"    (File Handler) - Extracting text from article: {url}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        title = soup.find('h1').get_text(strip=True) if soup.find('h1') else 'No Title Found'
        article_body = soup.find('article') or soup.find('main') or soup.find('body')
        paragraphs = article_body.find_all('p') if article_body else []
        body_text = "\n".join([p.get_text() for p in paragraphs])
        
        full_content = f"Article Title: {title}\n\nBody:\n{body_text}"
        return full_content, "text"
    except Exception as e:
        print(f"    [ERROR] Failed to process article link: {e}")
        return None, None

def get_transcript_from_youtube_url(url: str):
    """Extracts the transcript from a YouTube URL."""
    print(f"    (File Handler) - Extracting transcript from YouTube: {url}")
    try:
        video_id = url.split('v=')[-1].split('&')[0]
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko', 'en'])
        transcript = " ".join([item['text'] for item in transcript_list])
        return transcript, "text"
    except Exception as e:
        print(f"    [ERROR] Failed to process YouTube link: {e}")
        return None, None

# --- Main Content Getter --- #

def get_content_from_input(path_or_url: str):
    """
    Reads content from a URL (article/YouTube), local file, or directory.
    """
    print(f"    (File Handler) - Reading content from: {path_or_url}")

    # 1. Handle URL input
    if path_or_url.startswith(('http://', 'https://')):
        if "youtube.com" in path_or_url or "youtu.be" in path_or_url:
            return get_transcript_from_youtube_url(path_or_url)
        else:
            return get_text_from_article_url(path_or_url)

    # 2. Handle local directory path
    file_path = path_or_url
    if os.path.isdir(file_path):
        print(f"    (File Handler) - Path is a directory. Searching for files inside.")
        try:
            files_in_dir = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
            if not files_in_dir:
                print(f"    [ERROR] No files found in directory: {file_path}")
                return None, None
            file_path = os.path.join(file_path, files_in_dir[0])
            print(f"    (File Handler) - Processing first file found: {file_path}")
        except Exception as e:
            print(f"    [ERROR] Failed to list files in directory {file_path}: {e}")
            return None, None

    # 3. Handle local file path
    if not os.path.exists(file_path):
        print(f"    [ERROR] File not found: {file_path}")
        return None, None

    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        print(f"    (File Handler) - Detected file extension: '{file_extension}'")

        if file_extension == '.csv':
            df = pd.read_csv(file_path, encoding='utf-8')
            content = "\n".join(df.iloc[:, 0].astype(str).tolist())
            return content, "text"

        elif file_extension == '.xlsx':
            df = pd.read_excel(file_path)
            content = "\n".join(df.iloc[:, 0].astype(str).tolist())
            return content, "text"

        elif file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read(), "text"

        elif file_extension == '.pdf':
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                content = ""
                for page_num in range(len(reader.pages)):
                    content += reader.pages[page_num].extract_text() or ""
                return content, "text"

        elif file_extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
            return file_path, "file_path"

        else:
            print(f"    [WARNING] Unsupported file type: {file_extension}. Skipping.")
            return None, None

    except Exception as e:
        print(f"    [ERROR] Failed to read file {file_path}: {e}")
        return None, None