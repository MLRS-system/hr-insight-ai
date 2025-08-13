# agents/reviewer.py

import os
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from core.shared_context import SharedContext
from langchain_core.prompts import PromptTemplate

try:
    from config import GCLOUD_PROJECT_ID
except ImportError:
    print("[CRITICAL ERROR] config.py 파일에서 GCLOUD_PROJECT_ID를 찾을 수 없습니다.")
    GCLOUD_PROJECT_ID = None

class ReviewerAgent:
    def __init__(self, shared_context: SharedContext):
        self.shared_context = shared_context
        self.project_id = GCLOUD_PROJECT_ID
        self.location = "us-central1"
        print("    (Agent: Reviewer) - Initializing with Vertex AI SDK...")

        if not self.project_id or self.project_id == "YOUR_GCLOUD_PROJECT_ID_HERE":
            raise ValueError("ReviewerAgent: Google Cloud 프로젝트 ID가 설정되지 않았습니다. config.py를 확인해주세요.")

        try:
            vertexai.init(project=self.project_id, location=self.location)
            
            # 모델 ID를 가장 최신 버전인 gemini-2.5-pro로 지정합니다.
            self.model = GenerativeModel("gemini-2.5-pro")
            
            self.generation_config = GenerationConfig(
                temperature=0.3,
                max_output_tokens=8192,
            )

            # 프롬프트 템플릿 로드
            review_prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'reviewer.prompt')
            with open(review_prompt_path, 'r', encoding='utf-8') as f:
                self.review_template = PromptTemplate.from_template(f.read())

            final_report_prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'final_report.prompt')
            with open(final_report_prompt_path, 'r', encoding='utf-8') as f:
                self.final_report_template = PromptTemplate.from_template(f.read())
                
            print("    (Agent: Reviewer) - Vertex AI SDK and prompts loaded successfully.")

        except Exception as e:
            print(f"    [CRITICAL ERROR] Vertex AI SDK 초기화 또는 프롬프트 로딩 실패: {e}")
            raise
        
        print("    (Agent: Reviewer) - Initialization complete.")

    def _call_gemini(self, prompt: str) -> str:
        """Helper function to call Gemini model via Vertex AI SDK."""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            return response.text
        except Exception as e:
            print(f"    [ERROR] Gemini SDK 호출 중 오류 발생: {e}")
            raise 

    def review(self, content_type: str):
        print(f"    (Agent: Reviewer) - Reviewing '{content_type}' with Vertex AI SDK...")
        content = self.shared_context.get(content_type)
        if not content or not content.strip():
            self.shared_context.add_feedback(f"'{content_type}'에 대한 내용이 없어 검토를 진행할 수 없습니다.")
            return

        prompt = self.review_template.format(text_to_review=content)

        try:
            feedback = self._call_gemini(prompt)
            print(f"    --- Review Feedback (from Gemini) ---\n{feedback}")
            self.shared_context.add_feedback(feedback)
            self.shared_context.set(f"{content_type}_reviewed", True)
            self.shared_context.add_history("ReviewerAgent", f"Review {content_type}", feedback)
            print(f"    (Agent: Reviewer) - '{content_type}' review complete.")
        except Exception as e:
            error_message = f"Vertex AI SDK execution failed during review: {e}"
            print(f"    [CRITICAL ERROR] {error_message}")
            self.shared_context.add_feedback(f"ReviewerAgent: {error_message}")

    def generate_final_report(self):
        print("    (Agent: Reviewer) - Generating Final Report with Vertex AI SDK...")
        content_summary = self.shared_context.get("content_summary")
        hr_ideas = self.shared_context.get("hr_ideas")
        training_program = self.shared_context.get("training_program", "아직 생성되지 않음")

        if not all([content_summary, hr_ideas]):
            error_message = "최종 보고서를 생성하기 위한 모든 구성요소(요약, HR 아이디어)가 준비되지 않았습니다."
            print(f"    [ERROR] {error_message}")
            self.shared_context.add_feedback(error_message)
            return
            
        prompt = self.final_report_template.format(
            content_summary=content_summary,
            hr_ideas=hr_ideas,
            training_program=training_program
        )

        try:
            final_report = self._call_gemini(prompt)
            print("    --- Final Report (from Gemini) ---")
            print(final_report[:500] + "\n...") 
            
            self.shared_context.set("final_report", final_report)
            self.shared_context.add_history("ReviewerAgent", "Generate Final Report", "최종 보고서 생성 완료.")
            print("    (Agent: Reviewer) - Final Report generation complete.")
        except Exception as e:
            error_message = f"Vertex AI SDK execution failed during final report generation: {e}"
            print(f"    [CRITICAL ERROR] {error_message}")
            self.shared_context.add_feedback(f"ReviewerAgent: {error_message}")