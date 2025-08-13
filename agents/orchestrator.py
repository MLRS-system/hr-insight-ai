import os
from core.shared_context import SharedContext
from agents.content_processor import ContentProcessorAgent
from agents.hr_analyzer import HRApplicabilityAgent
from agents.program_generator import ProgramGeneratorAgent
from agents.reviewer import ReviewerAgent # 리뷰어는 나중에 Gemini로 연동

class OrchestratorAgent:
    def __init__(self, shared_context: SharedContext):
        self.shared_context = shared_context
        # 에이전트들을 미리 초기화해두고 필요할 때 사용
        self.agents = {
            "content_processor": ContentProcessorAgent(self.shared_context),
            "hr_analyzer": HRApplicabilityAgent(self.shared_context),
            "program_generator": ProgramGeneratorAgent(self.shared_context),
            "reviewer": ReviewerAgent(self.shared_context) # Gemini CLI 연동 시 여기에 추가
        }
        print("OrchestratorAgent 초기화됨 (동적 플래닝 모드).")

    def _create_plan(self) -> list:
        """
        사용자의 요청사항을 분석하여 동적으로 작업 계획을 수립합니다.
        (여기서는 키워드 기반의 간단한 플래너를 구현합니다.)
        """
        user_request = self.shared_context.get("initial_goal", "").lower()
        plan = []

        # 1. 모든 요청에 기본적으로 콘텐츠 처리(요약)는 포함
        plan.append("summarize_content")

        # 2. 요청에 따라 추가 작업 결정
        if "hr" in user_request or "인사" in user_request or "적용" in user_request:
            plan.append("analyze_hr_ideas")
        
        if "교육" in user_request or "프로그램" in user_request or "제안" in user_request:
            # HR 분석이 선행되어야 프로그램 생성이 의미 있으므로, 순서 보장
            if "analyze_hr_ideas" not in plan:
                plan.append("analyze_hr_ideas")
            plan.append("generate_program")
        
        # 3. 리뷰 단계 추가 (추후 구현)
        # 3. 리뷰 단계는 최종 보고서 생성 후 실행

        print(f"    (Planner) - 생성된 작업 계획: {plan}")
        return plan


    def run_workflow(self):
        """
        생성된 작업 계획에 따라 워크플로우를 실행하고, 최종 보고서 생성을 위임합니다.
        """
        print("OrchestratorAgent: 워크플로우 시작.")
        
        plan = self._create_plan()
        
        # 1. 계획된 모든 에이전트를 순차적으로 실행
        for step in plan:
            print(f"\n--- [실행 단계: {step}] ---")
            if step == "summarize_content":
                agent = self.agents["content_processor"]
                agent.process()
            elif step == "analyze_hr_ideas":
                agent = self.agents["hr_analyzer"]
                agent.analyze()
            elif step == "generate_program":
                agent = self.agents["program_generator"]
                agent.generate()
            # 여기에 다른 단계들을 추가할 수 있습니다.
        
        print("\nOrchestratorAgent: 모든 계획된 작업 완료.")

        # --- 최종 보고서 생성 및 리뷰 단계 ---
        # 이 블록은 모든 작업이 끝난 후, for 루프 밖에서 실행됩니다.
        print("\n--- [최종 단계: generate_and_review_final_report] ---")
        
        # 2. Reviewer에게 최종 보고서 생성을 '지시'
        print("    (Orchestrator) - ReviewerAgent에게 최종 보고서 생성을 요청합니다...")
        reviewer_agent = self.agents["reviewer"]
        reviewer_agent.generate_final_report()

        # 3. (선택적) Gemini가 만든 최종 보고서를 '다시' 리뷰하여 자기검증
        if self.shared_context.get("final_report"):
            print("    (Orchestrator) - 생성된 최종 보고서에 대한 최종 검토를 요청합니다...")
            reviewer_agent.review(content_type="final_report")
        else:
            print("    [경고] 최종 보고서가 생성되지 않아 리뷰를 건너뜁니다.")