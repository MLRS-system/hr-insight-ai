# Dockerfile

# 1. 베이스 이미지 선택 (파이썬 3.10 버전)
FROM python:3.10-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. requirements.txt 파일을 먼저 복사
COPY requirements.txt .

# 4. CPU 버전의 PyTorch를 먼저 설치 (메모리 절약)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 5. 나머지 라이브러리 설치
RUN pip install -r requirements.txt

# 6. 프로젝트의 나머지 모든 코드를 복사
COPY . .

# 7. 서버 시작 명령어 설정
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "10000"]