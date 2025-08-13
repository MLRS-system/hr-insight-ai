import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import torch

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(project_root))


from agents.content_processor import ContentProcessorAgent
from agents.hr_analyzer import HRApplicabilityAgent
from agents.program_generator import ProgramGeneratorAgent
from core.shared_context import SharedContext
from agents.orchestrator import OrchestratorAgent
from PIL import Image
from utils.file_handler import get_content_from_input

def display_header():
    """Displays the application header."""
    print("=" * 60)
    print(" HR Insight: AI-Powered HRD Program Generation System")
    print("=" * 60)

def save_output(filename, content):
    """Saves the generated content to a file in the output directory."""
    output_dir = os.path.join(project_root, 'output')
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"\n[알림] 결과가 다음 파일에 저장되었습니다: {file_path}")

def main():
    display_header()

    # 1. 상호작용형 인터페이스로 변경
    user_request = input("▶ 무엇을 도와드릴까요 (예: 이 기사를 요약하고 교육 프로그램을 제안해줘): ")
    path_or_url = input("▶ 분석할 파일 경로 또는 URL을 주세요 (Enter를 누르시면 기본 폴더로 설정됩니다): ")
    if not path_or_url:
        path_or_url = r"C:\Users\wongi\Desktop\mlrs\input"

    # 2. 새로운 추출기를 사용하여 콘텐츠 로드
    content, content_type = get_content_from_input(path_or_url)
    if content is None:
        print(f"[오류] '{path_or_url}'에서 콘텐츠를 처리할 수 없습니다. 입력을 확인해주세요.")
        return

    # 3. 공유 컨텍스트 및 오케스트레이터 초기화
    shared_context = SharedContext(initial_goal=user_request)
    shared_context.set("input_source_path_or_url", path_or_url) # 원본 소스 경로 저장

    # 콘텐츠 유형에 따라 컨텍스트 설정
    if content_type == "file_path":
        shared_context.set("content_file_path", content)
        shared_context.set("content_text", "")
    else: # article, youtube, text file 등
        shared_context.set("content_file_path", None)
        shared_context.set("content_text", content)

    # 4. 새로운 워크플로우 실행
    orchestrator = OrchestratorAgent(shared_context)
    orchestrator.run_workflow()

    # 5. 최종 보고서 저장
    final_report = shared_context.get("final_report")
    if final_report:
        report_filename = f"HR_Insight_Report_{os.path.basename(path_or_url).split('.')[0]}.txt"
        save_output(report_filename, final_report)
        print("...보고서 저장 완료.")

    print("\n" + "=" * 60)
    print("모든 분석이 완료되었습니다.")


if __name__ == "__main__":
    main()