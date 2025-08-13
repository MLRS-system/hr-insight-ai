import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
import os

def get_content_from_input(path_or_url: str):
    """입력 경로가 로컬 파일인지 URL인지 판단하여 적절한 추출기를 호출합니다."""
    if os.path.exists(path_or_url):
        # 로컬 파일 처리 로직은 기존 file_handler에 있다고 가정
        # 여기서는 간단히 텍스트 파일만 읽는다고 가정
        # 실제로는 utils.file_handler.get_content_from_input를 호출해야 함
        try:
            with open(path_or_url, 'r', encoding='utf-8') as f:
                return f.read(), "file"
        except:
             # 텍스트 파일이 아닐 경우(이미지 등) 경로 자체를 반환
            return path_or_url, "file_path"

    elif "youtube.com" in path_or_url or "youtu.be" in path_or_url:
        return get_transcript_from_youtube_url(path_or_url), "youtube"
    elif path_or_url.startswith('http'):
        return get_text_from_article_url(path_or_url), "article"
    else:
        return None, None

def get_text_from_article_url(url: str) -> str:
    """기사 URL에서 본문 텍스트를 추출합니다."""
    print(f"    (Extractor) - 기사 링크에서 텍스트 추출 중: {url}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        title = soup.find('h1').get_text() if soup.find('h1') else '제목 없음'
        article_body = soup.find('article') or soup.find('main') or soup.find('body')
        paragraphs = article_body.find_all('p') if article_body else []
        body_text = "\n".join([p.get_text() for p in paragraphs])
        
        return f"기사 제목: {title}\n\n본문:\n{body_text}"
    except Exception as e:
        print(f"    [오류] 기사 링크 처리 실패: {e}")
        return None

def get_transcript_from_youtube_url(url: str) -> str:
    """유튜브 URL에서 스크립트(자막)를 텍스트로 추출합니다."""
    print(f"    (Extractor) - 유튜브 링크에서 스크립트 추출 중: {url}")
    try:
        video_id = url.split('v=')[-1].split('&')[0]
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko', 'en'])
        return " ".join([item['text'] for item in transcript_list])
    except Exception as e:
        print(f"    [오류] 유튜브 스크립트 처리 실패: {e}")
        return None