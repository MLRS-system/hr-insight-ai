import os
from typing import Optional
import requests

# config.py에서 API 키와 검색엔진 ID를 가져옵니다.
try:
    from config import GOOGLE_API_KEY, GOOGLE_CSE_ID, BRAVE_API_KEY
except ImportError:
    print("[CRITICAL] config.py 파일을 찾을 수 없거나 API 키/검색엔진 ID 설정이 누락되었습니다.")
    GOOGLE_API_KEY = None
    GOOGLE_CSE_ID = None
    BRAVE_API_KEY = None

# google-api-python-client 라이브러리가 필요합니다.
try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except ImportError:
    print("    (Web Search) [WARNING] 'google-api-python-client' 라이브러리가 설치되지 않았습니다. Google 검색을 비활성화합니다.")
    build = None

# Google 검색 할당량 초과를 식별하기 위한 별도의 에러 클래스를 정의합니다.
class QuotaExceededError(Exception):
    """Google Search API 할당량 초과 시 발생하는 에러"""
    pass

def _perform_google_search_internal(query: str, num_results: int = 5) -> Optional[str]:
    """내부적으로 사용되는 Google 검색 함수. 할당량 초과(HTTP 429) 시 QuotaExceededError를 발생시킵니다."""
    if not build: return None
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY_HERE": return None
    if not GOOGLE_CSE_ID or GOOGLE_CSE_ID == "YOUR_SEARCH_ENGINE_ID_HERE": return None

    try:
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        res = service.cse().list(q=query, cx=GOOGLE_CSE_ID, num=num_results).execute()
        
        if 'items' in res and res['items']:
            snippets = [item.get('snippet', '') for item in res['items']]
            formatted_results = "\n".join(f"- {s}" for s in snippets if s)
            return formatted_results
        else:
            return "웹 검색 결과가 없습니다."

    except HttpError as e:
        if e.resp.status == 429:
            raise QuotaExceededError("Google Search API daily quota exceeded.")
        else:
            print(f"    (Web Search) [ERROR] Google 웹 검색 중 HTTP 오류 발생: {e}")
            return None
    except Exception as e:
        print(f"    (Web Search) [ERROR] Google 웹 검색 중 알 수 없는 오류 발생: {e}")
        return None

def perform_brave_data_search(query: str, count: int = 5) -> Optional[str]:
    """Brave Search의 'Data for AI' API를 호출하여 검색 결과 데이터를 가져옵니다."""
    if not BRAVE_API_KEY or BRAVE_API_KEY == "YOUR_BRAVE_API_KEY_HERE":
        print("    (Web Search) [WARNING] Brave API Key가 설정되지 않았습니다. Brave 검색을 건너뜁니다.")
        return None

    print(f"    (Web Search) - Brave 'Data for AI' API 호출 중: {query}")
    headers = {"X-Subscription-Token": BRAVE_API_KEY, "Accept": "application/json"}
    params = {"q": query, "count": count, "country": "KR", "search_lang": "ko"}

    try:
        response = requests.get("https://api.search.brave.com/res/v1/web/search", params=params, headers=headers, timeout=10)
        response.raise_for_status()
        results = response.json()
        print("    (Web Search) - Brave API 호출 성공.")
        if not results.get("web") or not results["web"].get("results"):
            return "웹 검색 결과가 없습니다."
        snippets = [f"출처 {i+1}:\n- 제목: {res.get('title', 'N/A')}\n- 요약: {res.get('description', 'N/A')}\n- URL: {res.get('url', 'N/A')}" for i, res in enumerate(results["web"]["results"])]
        return "\n\n".join(snippets)
    except requests.exceptions.RequestException as e:
        print(f"    (Web Search) [ERROR] Brave API 호출 중 오류 발생: {e}")
        return None

def perform_web_search(query: str) -> Optional[str]:
    """
    하이브리드 웹 검색 함수. Google을 먼저 시도하고, 할당량 초과 시 Brave로 자동 전환합니다.
    """
    try:
        print("    (Web Search) - 1순위: Google 검색 시도...")
        google_results = _perform_google_search_internal(query)
        if google_results:
            print("    (Web Search) - Google 검색 성공.")
            return google_results
        
        # Google 검색이 실패했거나(None) 결과가 없을 때 Brave 시도
        print("    (Web Search) - Google 검색에 실패했거나 결과가 없습니다. 2순위: Brave 검색으로 전환합니다.")
        return perform_brave_data_search(query)
        
    except QuotaExceededError:
        print("    (Web Search) [INFO] Google 할당량 초과. 2순위: Brave 검색으로 자동 전환합니다.")
        return perform_brave_data_search(query)
    except Exception as e:
        print(f"    (Web Search) [ERROR] 웹 검색 중 예상치 못한 심각한 오류 발생: {e}. Brave로 대체합니다.")
        return perform_brave_data_search(query)