import os
import uuid
import shutil
from fastapi import FastAPI, BackgroundTasks, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# --- AI 시스템의 핵심 로직 임포트 ---
from core.shared_context import SharedContext
from agents.orchestrator import OrchestratorAgent
from utils.file_handler import get_content_from_input

app = FastAPI()

# --- 'uploads' 임시 폴더 생성 ---
UPLOADS_DIR = "uploads"
if not os.path.exists(UPLOADS_DIR):
    os.makedirs(UPLOADS_DIR)

# '/static' 경로로 오는 요청은 'static' 폴더의 파일을 제공하도록 설정
app.mount("/static", StaticFiles(directory="static"), name="static")

# 여러 분석 작업을 관리하기 위한 작업 보관소
jobs = {}

# --- 백그라운드에서 AI 분석을 수행하는 메인 함수 ---
def run_analysis_pipeline(job_id: str, user_request: str, path_or_url: str):
    """
    main.py의 핵심 로직을 가져와 백그라운드에서 실행 가능한 함수로 만듭니다.
    """
    try:
        # 1. 작업 상태를 '시작'으로 업데이트
        jobs[job_id]["status"] = "콘텐츠 원본을 읽는 중..."
        
        content, content_type = get_content_from_input(path_or_url)
        if content is None:
            raise ValueError(f"'{path_or_url}'에서 콘텐츠를 처리할 수 없습니다.")

        # 2. 공유 컨텍스트 및 오케스트레이터 초기화
        shared_context = SharedContext(initial_goal=user_request)
        shared_context.set("input_source_path_or_url", path_or_url)
        
        if content_type == "file_path":
            shared_context.set("content_file_path", content)
            shared_context.set("content_text", None)
        else:
            shared_context.set("content_file_path", None)
            shared_context.set("content_text", content)

        # 3. 오케스트레이터 실행 (가장 오래 걸리는 부분)
        jobs[job_id]["context"] = shared_context
        orchestrator = OrchestratorAgent(shared_context)
        orchestrator.run_workflow()

        # 4. 최종 결과 저장 및 상태를 '완료'로 업데이트
        final_report = shared_context.get("final_report")
        jobs[job_id]["status"] = "완료"
        jobs[job_id]["result"] = final_report

    except Exception as e:
        print(f"[ERROR] Job {job_id} failed: {e}")
        jobs[job_id]["status"] = "오류 발생"
        jobs[job_id]["result"] = str(e)
    finally:
        # 5. 분석이 끝나면 서버에 업로드된 임시 파일 삭제
        if os.path.exists(path_or_url) and UPLOADS_DIR in path_or_url:
             os.remove(path_or_url)
             print(f"    (Cleanup) - 임시 파일 삭제: {path_or_url}")

# --- API 라우트(규칙) 정의 ---

@app.get("/")
async def read_root():
    """메인 페이지인 index.html을 보여줍니다."""
    return FileResponse('static/index.html')

@app.post("/run-analysis")
async def start_analysis(
    background_tasks: BackgroundTasks,
    user_request: str = Form(...),
    url: str = Form(None),
    file: UploadFile = File(None)
):
    """파일 또는 URL을 폼 데이터로 받아 분석을 시작합니다."""
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "대기 중", "result": None}

    path_or_url = ""
    if file and file.filename:
        # 파일이 첨부된 경우, 'uploads' 폴더에 고유한 이름으로 저장
        temp_file_path = os.path.join(UPLOADS_DIR, f"{job_id}_{file.filename}")
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        path_or_url = temp_file_path
    elif url:
        # URL이 입력된 경우
        path_or_url = url
    else:
        return {"error": "파일 또는 URL 중 하나는 반드시 제공되어야 합니다."}

    # AI 파이프라인을 백그라운드에서 실행하도록 예약
    background_tasks.add_task(run_analysis_pipeline, job_id, user_request, path_or_url)
    
    return {"message": "분석 작업이 시작되었습니다.", "job_id": job_id}

@app.get("/status/{job_id}")
def get_status(job_id: str):
    """Job ID를 이용해 현재 작업 상태와 중간 로그를 확인합니다."""
    job = jobs.get(job_id)
    if not job:
        return {"status": "오류", "message": "해당 Job ID를 찾을 수 없습니다."}
    
    # SharedContext에서 최신 로그(history) 가져오기
    history = []
    if "context" in job and hasattr(job["context"], "history"):
        history = job["context"].history

    return {"job_id": job_id, "status": job["status"], "history": history}

@app.get("/result/{job_id}")
def get_result(job_id: str):
    """Job ID를 이용해 최종 분석 결과를 가져옵니다."""
    job = jobs.get(job_id)
    if not job:
        return {"status": "오류", "message": "해당 Job ID를 찾을 수 없습니다."}
    
    return {"job_id": job_id, "status": job["status"], "result": job.get("result")}