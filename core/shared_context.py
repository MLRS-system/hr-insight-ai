import json
from typing import Any, Dict, List

class SharedContext:
    """
    모든 에이전트가 작업 내용을 공유하는 중앙 데이터 저장소(블랙보드)입니다.
    작업의 현재 상태, 데이터, 이력 등 모든 정보를 포함합니다.
    """
    def __init__(self, initial_goal: str):
        self._context: Dict[str, Any] = {
            "initial_goal": initial_goal,
            "plan": [],
            "history": [],
            "results": {},
            "feedback": []
        }

    def set(self, key: str, value: Any):
        """컨텍스트에 데이터를 저장합니다."""
        self._context[key] = value
        if key == "content_text":
            print(f"[Context Updated] Key: {key}, Value: <text data of length {len(value)}>")
        else:
            value_str = str(value)
            if len(value_str) > 200:
                value_to_print = value_str[:200] + "... (truncated)"
            else:
                value_to_print = value_str
            try:
                print(f"[Context Updated] Key: {key}, Value: {value_to_print}")
            except UnicodeEncodeError:
                safe_value = value_to_print.encode('utf-8', 'ignore').decode('utf-8')
                print(f"[Context Updated] Key: {key}, Value: {safe_value}")

    def get(self, key: str, default: Any = None) -> Any:
        """컨텍스트에서 데이터를 가져옵니다."""
        return self._context.get(key, default)

    def add_history(self, agent_name: str, action: str, result_summary: str):
        """작업 이력을 추가합니다."""
        history_entry = {
            "agent": agent_name,
            "action": action,
            "summary": result_summary
        }
        self._context["history"].append(history_entry)
        print(f"[History Added] Agent: {agent_name}, Action: {action}")

    def add_feedback(self, feedback: str):
        """오케스트레이터나 사용자의 피드백을 추가합니다."""
        self._context["feedback"].append(feedback)
        print(f"[Feedback Added] {feedback}")

    def get_full_context(self) -> Dict[str, Any]:
        """현재의 모든 컨텍스트를 반환합니다."""
        return self._context

    def __str__(self) -> str:
        """컨텍스트의 현재 상태를 문자열로 반환합니다."""
        # 순환 참조를 피하기 위해 간단한 정보를 직렬화합니다.
        context_for_print = {
            "initial_goal": self.get("initial_goal"),
            "plan": self.get("plan"),
            "history_count": len(self.get("history", [])),
            "results_keys": list(self.get("results", {}).keys()),
            "feedback_count": len(self.get("feedback", []))
        }
        return json.dumps(context_for_print, indent=2, ensure_ascii=False)