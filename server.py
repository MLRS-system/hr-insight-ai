# server.py

import os
import uuid
import shutil
from fastapi import FastAPI, BackgroundTasks, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# --- AI 시스템의 핵심 로직 임포트 ---
from core.shared_context import SharedContext
from agents.orchestrator import OrchestratorAgent
# file_handler는 이제 에이전트 내부에서 사용되므로 여기서 직접 호출하지 않습니다.

app = FastAPI()

# --- 'uploads' 임시 폴더 생성 ---
UPLOADS_DIR = "uploads"
if not os.path.exists(UPLOADS_DIR):
    os.makedirs(UPLOADS_DIR)

app.mount("/static", StaticFiles(directory="static"), name="static")
jobs = {}

# --- 백그라운드에서 AI 분석을 수행하는 메인 함수 ---
def run_analysis_pipeline(job_id: str, user_request: str, path_or_url: str):
    """
    SharedContext에 원본 경로만 저장하고, 모든 처리를 Orchestrator에게 위임하는 최종 버전
    """
    try:
        jobs[job_id]["status"] = "AI 에이전트 초기화 중..."
        
        # 1. 공유 컨텍스트 생성 및 원본 경로 저장 (가장 중요)
        # 여기서 get_content_from_input을 호출하지 않습니다.
        shared_context = SharedContext(initial_goal=user_request)
        shared_context.set("input_source_path_or_url", path_or_url)
        
        # 2. 오케스트레이터 실행
        jobs[job_id]["context"] = shared_context
        orchestrator = OrchestratorAgent(shared_context)
        orchestrator.run_workflow()

        # 3. 최종 결과 저장 및 상태 업데이트
        final_report = shared_context.get("final_report")
        jobs[job_id]["status"] = "완료"
        jobs[job_id]["result"] = final_report

    except Exception as e:
        print(f"[ERROR] Job {job_id} failed: {e}")
        jobs[job_id]["status"] = "오류 발생"
        jobs[job_id]["result"] = str(e)
    finally:
        # 4. 분석이 끝나면 서버에 업로드된 임시 파일 삭제
        if os.path.exists(path_or_url) and UPLOADS_DIR in path_or_url:
             os.remove(path_or_url)
             print(f"    (Cleanup) - 임시 파일 삭제: {path_or_url}")

# --- API 라우트(규칙) 정의 ---

@app.get("/")
async def read_root():
    return FileResponse('static/index.html')

@app.post("/run-analysis")
async def start_analysis(
    background_tasks: BackgroundTasks,
    user_request: str = Form(...),
    url: str = Form(None),
    file: UploadFile = File(None)
):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "대기 중", "result": None}

    path_or_url = ""
    if file and file.filename:
        temp_file_path = os.path.join(UPLOADS_DIR, f"{job_id}_{file.filename}")
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        path_or_url = temp_file_path
    elif url:
        path_or_url = url
    else:
        return {"error": "파일 또는 URL 중 하나는 반드시 제공되어야 합니다."}

    background_tasks.add_task(run_analysis_pipeline, job_id, user_request, path_or_url)
    return {"message": "분석 작업이 시작되었습니다.", "job_id": job_id}

@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = jobs.get(job_id)
    if not job: return {"status": "오류", "message": "해당 Job ID를 찾을 수 없습니다."}
    
    history = []
    if "context" in job and hasattr(job["context"], "history"):
        history = job["context"].history

    return {"job_id": job_id, "status": job["status"], "history": history}

@app.get("/result/{job_id}")
def get_result(job_id: str):
    job = jobs.get(job_id)
    if not job: return {"status": "오류", "message": "해당 Job ID를 찾을 수 없습니다."}
    
    return {"job_id": job_id, "status": job["status"], "result": job.get("result")}