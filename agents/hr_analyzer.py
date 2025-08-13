import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 새로운 하이브리드 웹 검색 함수를 임포트합니다.
from utils.web_search import perform_web_search
from core.shared_context import SharedContext

# config.py에서 설정을 가져옵니다.
try:
    from config import (
        HR_ANALYZER_MODEL_PATH,
        USE_4BIT_QUANTIZATION,
        HF_CACHE_DIR,
        MAX_SEQUENCE_LENGTH,
        BATCH_SIZE
    )
except ImportError:
    print("[Error] config.py not found or not configured correctly.")
    # 기본 폴백 설정
    HR_ANALYZER_MODEL_PATH = "yanolja/EEVE-Korean-Instruct-7B-v2.0-Preview"
    USE_4BIT_QUANTIZATION = True
    HF_CACHE_DIR = None
    MAX_SEQUENCE_LENGTH = 4096
    BATCH_SIZE = 4

class HRApplicabilityAgent:
    """Agent to analyze content and generate HR/HRD applicability ideas using a local LLM and hybrid web search."""

    def __init__(self, shared_context: SharedContext):
        """에이전트를 초기화하고, 모델, 토크나이저, 프롬프트를 로드합니다."""
        self.shared_context = shared_context
        print("    (Agent: HRApplicability) - Initializing...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"    (Agent: HRApplicability) - Using device: {self.device}")

        try:
            print(f"    (Agent: HRApplicability) - Loading tokenizer: {HR_ANALYZER_MODEL_PATH}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                HR_ANALYZER_MODEL_PATH, cache_dir=HF_CACHE_DIR, trust_remote_code=True
            )

            print(f"    (Agent: HRApplicability) - Loading model: {HR_ANALYZER_MODEL_PATH}")
            quantization_config = None
            if USE_4BIT_QUANTIZATION:
                print("    (Agent: HRApplicability) - Using 4-bit quantization.")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                HR_ANALYZER_MODEL_PATH,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                cache_dir=HF_CACHE_DIR,
                trust_remote_code=True
            ).to(self.device)

        except Exception as e:
            print(f"    [CRITICAL ERROR] Failed to load model or tokenizer for HRApplicabilityAgent: {e}")
            raise

        # 프롬프트 템플릿 로드
        prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'hr_applicability.prompt')
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.prompt_template = f.read()
        print("    (Agent: HRApplicability) - Initialization complete.")

    def _extract_summary_for_search(self, analysis_result: str) -> str:
        """웹 검색을 위해 분석 결과에서 핵심 요약 부분을 추출합니다."""
        match = re.search(r"(1\.\s*Executive Summary|\*\*1\. Executive Summary\*\*)\s*\n(.*?)(?=\n\n|2\.|\*\*2\.)", analysis_result, re.DOTALL | re.IGNORECASE)
        if match:
            summary = match.group(2).strip()
            return summary[:200]
        return analysis_result[:200]

    def analyze(self):
        """
        SharedContext에서 콘텐츠 분석 결과를 읽고, 하이브리드 웹 검색을 통해
        HR/HRD 적용 아이디어를 생성합니다.
        """
        print("    (Agent: HRApplicability) - Analyzing content from SharedContext...")
        analysis_result = self.shared_context.get("content_summary")

        if not analysis_result:
            print("    (Agent: HRApplicability) - No content analysis result found in SharedContext.")
            self.shared_context.add_feedback("HRApplicabilityAgent: No content analysis result found. Requires content_summary.")
            return

        # --- Hybrid Web Search Integration ---
        summary_for_search = self._extract_summary_for_search(analysis_result)
        search_query = f'"{summary_for_search}" 최신 HRD 트렌드 적용 사례'
        
        print(f"    (Agent: HRApplicability) - Performing hybrid web search...")
        search_results = perform_web_search(search_query) # <<< 함수 호출 변경
        
        if search_results:
            print("    (Agent: HRApplicability) - Web search results found.")
            web_context = f"\n\n[웹 검색 추가 정보]:\n{search_results}\n"
        else:
            print("    (Agent: HRApplicability) - No web search results found or search failed.")
            web_context = "\n\n[웹 검색 추가 정보]:\n검색된 정보 없음."

        prompt = self.prompt_template.format(
            analysis_result=analysis_result,
            web_context=web_context
        )

        try:
            chat = [{"role": "user", "content": prompt}]
            input_text = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        except Exception:
            input_text = prompt

        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=MAX_SEQUENCE_LENGTH // 2, truncation=True).to(self.device)

        try:
            outputs = self.model.generate(**inputs, max_new_tokens=MAX_SEQUENCE_LENGTH // 2, do_sample=True, temperature=0.7)
            response_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            hr_analysis_result = response_text.strip()

            self.shared_context.set("hr_ideas", hr_analysis_result)
            self.shared_context.add_history("HRApplicabilityAgent", "Analyze HR Applicability", "HR/HRD 적용 아이디어 생성 완료.")
            print("    (Agent: HRApplicability) - HR/HRD analysis complete. Result stored in SharedContext.")

        except Exception as e:
            print(f"    [ERROR] HRApplicabilityAgent processing failed: {e}")
            self.shared_context.add_feedback(f"HRApplicabilityAgent: Error during analysis: {e}")

    def refine(self, feedback: str):
        """피드백을 바탕으로 기존 HR 아이디어를 개선합니다."""
        print("    (Agent: HRApplicability) - Refining HR ideas based on feedback...")
        original_ideas = self.shared_context.get("hr_ideas")
        
        refine_prompt = f"기존 HR 아이디어:\n{original_ideas}\n\n피드백:\n{feedback}\n\n위 피드백을 반영하여 HR 아이디어를 개선해주세요. 개선된 아이디어만 간결하게 제시해주세요."
        
        try:
            chat = [{"role": "user", "content": refine_prompt}]
            input_text = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        except Exception:
            input_text = refine_prompt

        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=MAX_SEQUENCE_LENGTH // 2, truncation=True).to(self.device)

        try:
            outputs = self.model.generate(**inputs, max_new_tokens=MAX_SEQUENCE_LENGTH // 2, do_sample=True, temperature=0.7)
            response_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            refined_ideas = response_text.strip()

            self.shared_context.set("hr_ideas", refined_ideas)
            self.shared_context.add_history("HRApplicabilityAgent", "Refine HR Ideas", "피드백 기반 HR 아이디어 개선 완료.")
            print("    (Agent: HRApplicability) - HR ideas refinement complete.")

        except Exception as e:
            print(f"    [ERROR] HRApplicabilityAgent refinement failed: {e}")
            self.shared_context.add_feedback(f"HRApplicabilityAgent: Error during refinement: {e}")