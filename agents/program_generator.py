import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import Accelerator

# 새로운 하이브리드 웹 검색 함수를 임포트합니다.
from utils.web_search import perform_web_search
from core.shared_context import SharedContext

# Import configuration from config.py
try:
    from config import (
        PROGRAM_GENERATOR_MODEL_PATH,
        USE_4BIT_QUANTIZATION,
        HF_CACHE_DIR,
        MAX_SEQUENCE_LENGTH,
        BATCH_SIZE
    )
except ImportError:
    print("[Error] config.py not found or not configured correctly.")
    # Set default fallbacks
    PROGRAM_GENERATOR_MODEL_PATH = "kakaocorp/kanana-1.5-8b-instruct-2505"
    USE_4BIT_QUANTIZATION = True
    HF_CACHE_DIR = None
    MAX_SEQUENCE_LENGTH = 4096
    BATCH_SIZE = 1

class ProgramGeneratorAgent:
    """Agent to generate a detailed training program based on HR ideas using a local LLM and hybrid web search."""

    def __init__(self, shared_context: SharedContext):
        """Initializes the agent, loads the model, tokenizer, and prompt."""
        self.shared_context = shared_context
        print("    (Agent: ProgramGenerator) - Initializing...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"    (Agent: ProgramGenerator) - Using device: {self.device}")

        try:
            print(f"    (Agent: ProgramGenerator) - Loading tokenizer: {PROGRAM_GENERATOR_MODEL_PATH}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                PROGRAM_GENERATOR_MODEL_PATH, cache_dir=HF_CACHE_DIR, trust_remote_code=True
            )

            print(f"    (Agent: ProgramGenerator) - Loading model: {PROGRAM_GENERATOR_MODEL_PATH}")
            quantization_config = None
            if USE_4BIT_QUANTIZATION:
                print("    (Agent: ProgramGenerator) - Using 4-bit quantization.")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                PROGRAM_GENERATOR_MODEL_PATH,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                cache_dir=HF_CACHE_DIR,
                trust_remote_code=True
            ).to(self.device)
            
        except Exception as e:
            print(f"    [CRITICAL ERROR] Failed to load model or tokenizer for ProgramGeneratorAgent: {e}")
            raise

        # Load the prompt template
        prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'program_generation.prompt')
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.prompt_template = f.read()
        print("    (Agent: ProgramGenerator) - Initialization complete.")

    def _extract_initiatives_for_search(self, hr_ideas: str) -> str:
        """Extracts the Priority Initiatives for a focused web search."""
        match = re.search(r"(2\.\s*Priority Initiatives|\*\*2\. Priority Initiatives\*\*)\s*\n(.*?)(?=\n\n|3\.|\*\*3\.)", hr_ideas, re.DOTALL | re.IGNORECASE)
        if match:
            initiatives = match.group(2).strip()
            keywords = re.findall(r"\*\*(.*?)\*\*", initiatives)
            if keywords:
                return ", ".join(keywords)
        return hr_ideas[:150]

    def generate(self):
        """
        SharedContext에서 HR 아이디어를 읽고, 하이브리드 웹 검색을 통해 훈련 프로그램을 생성합니다.
        """
        print("    (Agent: ProgramGenerator) - Generating program from SharedContext...")

        hr_ideas = self.shared_context.get("hr_ideas")

        if not hr_ideas:
            print("    (Agent: ProgramGenerator) - No HR ideas found in SharedContext.")
            self.shared_context.add_feedback("ProgramGenerator: No HR ideas found. Requires hr_ideas.")
            return

        # --- Hybrid Web Search Integration ---
        initiatives_for_search = self._extract_initiatives_for_search(hr_ideas)
        search_query = f'"{initiatives_for_search}" 관련 기업 교육 프로그램 사례'
        
        print(f"    (Agent: ProgramGenerator) - Performing hybrid web search for: {search_query}")
        search_results = perform_web_search(search_query) # <<< 함수 호출 변경

        if search_results:
            print(f"    (Agent: ProgramGenerator) - Web search results found.")
            web_context = f"\n\n[웹 검색 추가 정보]\n{search_results}\n"
        else:
            print("    (Agent: ProgramGenerator) - No web search results found or search disabled.")
            web_context = "\n\n[웹 검색 추가 정보]\n검색된 정보 없음."

        prompt = self.prompt_template.format(
            hr_ideas=hr_ideas,
            web_context=web_context
        )

        try:
            chat = [{"role": "user", "content": prompt}]
            input_text = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        except Exception:
            input_text = prompt

        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=MAX_SEQUENCE_LENGTH // 2, truncation=True).to(self.device)

        try:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_SEQUENCE_LENGTH // 2,
                do_sample=True,
                temperature=0.7
            )
            response_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            generated_program = response_text.strip()

            self.shared_context.set("training_program", generated_program)
            self.shared_context.add_history("ProgramGeneratorAgent", "Generate Program", "훈련 프로그램 생성 완료.")
            print("    (Agent: ProgramGenerator) - Program generation complete. Result stored in SharedContext.")

        except Exception as e:
            print(f"    [ERROR] ProgramGeneratorAgent processing failed: {e}")
            self.shared_context.add_feedback(f"ProgramGenerator: Error during generation: {e}")

    def refine(self, feedback: str):
        """
        피드백을 바탕으로 기존 교육 프로그램을 개선합니다.
        """
        print("    (Agent: ProgramGenerator) - Refining training program based on feedback...")
        original_program = self.shared_context.get("training_program")
        
        refine_prompt = f"기존 교육 프로그램:\n{original_program}\n\n피드백:\n{feedback}\n\n위 피드백을 반영하여 교육 프로그램을 개선해주세요. 개선된 프로그램만 간결하게 제시해주세요."
        
        try:
            chat = [{"role": "user", "content": refine_prompt}]
            input_text = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        except Exception:
            input_text = refine_prompt

        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=MAX_SEQUENCE_LENGTH // 2, truncation=True).to(self.device)

        try:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_SEQUENCE_LENGTH // 2,
                do_sample=True,
                temperature=0.7
            )
            response_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            refined_program = response_text.strip()

            self.shared_context.set("training_program", refined_program)
            self.shared_context.add_history("ProgramGeneratorAgent", "Refine Program", "피드백 기반 교육 프로그램 개선 완료.")
            print("    (Agent: ProgramGenerator) - Training program refinement complete.")

        except Exception as e:
            print(f"    [ERROR] ProgramGeneratorAgent refinement failed: {e}")
            self.shared_context.add_feedback(f"ProgramGenerator: Error during refinement: {e}")