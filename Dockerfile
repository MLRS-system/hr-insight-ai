# Dockerfile

# 1. 베이스 이미지 선택 (파이썬 3.10 버전)
FROM python:3.10-slim

# 2. 시스템 프로그램 설치 (Poppler & Chrome)
# apt-key 대신 새로운 방식으로 Chrome 인증키를 추가합니다.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-kor \
    poppler-utils \
    wget \
    gnupg \
    unzip \
    ca-certificates \
    && wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-chrome-archive-keyring.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome-archive-keyring.gpg] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update && apt-get install -y --no-install-recommends google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. requirements.txt 파일을 먼저 복사
COPY requirements.txt .

# 5. CPU 버전의 PyTorch를 먼저 설치 (메모리 절약)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 6. requirements.txt에 명시된 나머지 라이브러리 설치
RUN pip install --no-cache-dir -r requirements.txt

# 7. 프로젝트의 나머지 모든 코드를 복사
COPY . .

# 8. 서버 시작 명령어 설정
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "10000"]