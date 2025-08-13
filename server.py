import os
import uuid
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# --- AI 시스템의 핵심 로직 임포트 ---
from core.shared_context import SharedContext
from agents.orchestrator import OrchestratorAgent
from utils.file_handler import get_content_from_input

app = FastAPI()

# 'static' 폴더에 있는 파일들을 웹에서 접근 가능하게 설정
app.mount("/static", StaticFiles(directory="static"), name="static")

jobs = {}

def run_analysis_pipeline(job_id: str, user_request: str, path_or_url: str):
    try:
        jobs[job_id]["status"] = "콘텐츠 분석 시작..."

        content, content_type = get_content_from_input(path_or_url)
        if content is None:
            raise ValueError(f"'{path_or_url}'에서 콘텐츠를 처리할 수 없습니다.")

        shared_context = SharedContext(initial_goal=user_request)
        shared_context.set("input_source_path_or_url", path_or_url)

        if content_type == "file_path":
            shared_context.set("content_file_path", content)
        else:
            shared_context.set("content_file_path", None)
            shared_context.set("content_text", content)

        jobs[job_id]["context"] = shared_context
        orchestrator = OrchestratorAgent(shared_context)
        orchestrator.run_workflow()

        final_report = shared_context.get("final_report")
        jobs[job_id]["status"] = "완료"
        jobs[job_id]["result"] = final_report

    except Exception as e:
        print(f"[ERROR] Job {job_id} failed: {e}")
        jobs[job_id]["status"] = "오류 발생"
        jobs[job_id]["result"] = str(e)

class AnalysisRequest(BaseModel):
    user_request: str
    path_or_url: str

# --- API 라우트 정의 ---

@app.get("/")
async def read_index():
    """메인 페이지(index.html)를 보여줍니다."""
    return FileResponse('static/index.html')

@app.post("/run-analysis")
def start_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "대기 중", "result": None}
    background_tasks.add_task(run_analysis_pipeline, job_id, request.user_request, request.path_or_url)
    return {"message": "분석 작업이 시작되었습니다.", "job_id": job_id}

@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return {"status": "오류", "message": "해당 Job ID를 찾을 수 없습니다."}

    history = []
    if "context" in job and hasattr(job["context"], "history"):
        history = job["context"].history

    return {"job_id": job_id, "status": job["status"], "history": history}

@app.get("/result/{job_id}")
def get_result(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return {"status": "오류", "message": "해당 Job ID를 찾을 수 없습니다."}

    return {"job_id": job_id, "status": job["status"], "result": job.get("result")}