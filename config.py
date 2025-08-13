import os
import torch
from dotenv import load_dotenv

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# ----------------- Model Paths -----------------
# 이 부분은 민감 정보가 아니므로 그대로 둡니다.
CAPTIONING_MODEL_PATH = "Salesforce/blip-image-captioning-large"
SYNTHESIS_LLM_PATH = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
HR_ANALYZER_MODEL_PATH = "yanolja/EEVE-Korean-Instruct-7B-v2.0-Preview"
PROGRAM_GENERATOR_MODEL_PATH = "kakaocorp/kanana-1.5-8b-instruct-2505"

# ----------------- System Settings -----------------
# .env 파일에서 시스템 경로를 읽어옵니다.
HF_CACHE_DIR = os.getenv("HF_CACHE_DIR")
TEMP_DIR = os.getenv("TEMP_DIR")
if TEMP_DIR:
    os.environ["TEMP"] = TEMP_DIR
    os.environ["TMP"] = TEMP_DIR

USE_4BIT_QUANTIZATION = os.getenv("RUNNING_IN_CLOUD", "false").lower() != "true"
MAX_SEQUENCE_LENGTH = 4096
BATCH_SIZE = 4

# -----------------------------------------------------------------------------
# API Keys & Cloud Settings
# -----------------------------------------------------------------------------
# os.getenv()를 사용하여 .env 파일에서 민감 정보를 안전하게 읽어옵니다.
GCLOUD_PROJECT_ID = os.getenv("GCLOUD_PROJECT_ID")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
