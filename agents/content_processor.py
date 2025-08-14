# agents/content_processor.py

import os
import torch
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from core.shared_context import SharedContext
# easyocr 대신 새로운 하이브리드 OCR 핸들러를 임포트합니다.
from utils.ocr_handler import perform_hybrid_ocr 
from config import USE_4BIT_QUANTIZATION, HF_CACHE_DIR, MAX_SEQUENCE_LENGTH, CAPTIONING_MODEL_PATH, SYNTHESIS_LLM_PATH

class ContentProcessorAgent:
    """
    하이브리드 OCR 시스템(Google Vision + Tesseract)이 적용된 최종 콘텐츠 처리 에이전트.
    """
    def __init__(self, shared_context: SharedContext):
        """에이전트를 초기화하지만, easyocr를 포함한 무거운 모델들은 로드하지 않습니다."""
        self.shared_context = shared_context
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # easyocr 리더를 제거하고 나머지 변수들을 초기화합니다.
        self.caption_processor = None
        self.caption_model = None
        self.llm_tokenizer = None
        self.llm_model = None
        self.models_loaded = False
        self.prompt_template = ""
        
        try:
            prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'content_summary.prompt')
            with open(prompt_path, 'r', encoding='utf-8') as f:
                self.prompt_template = f.read()
        except Exception as e:
            print(f"    [CRITICAL ERROR] ContentProcessorAgent: 프롬프트 파일 로딩 실패: {e}")

        print("    (Agent: ContentProcessor) - Initialized (Lazy Loading Mode).")
    
    def _load_models_if_needed(self):
        """실제로 필요할 때 캡셔닝 및 요약 모델을 로드합니다."""
        if self.models_loaded:
            return
            
        print("    (Agent: ContentProcessor) - Loading models for the first time...")
        self.shared_context.add_history("ContentProcessorAgent", "Model Loading", "콘텐츠 분석 모델 로드 중...")
        try:
            # easyocr 로딩 부분을 완전히 제거합니다.
            print(f"    (Sub-task) Loading Image Captioning model ({CAPTIONING_MODEL_PATH})...")
            self.caption_processor = AutoProcessor.from_pretrained(CAPTIONING_MODEL_PATH, cache_dir=HF_CACHE_DIR)
            self.caption_model = BlipForConditionalGeneration.from_pretrained(CAPTIONING_MODEL_PATH, cache_dir=HF_CACHE_DIR).to(self.device)

            print(f"    (Sub-task) Loading Synthesis LLM ({SYNTHESIS_LLM_PATH})...")
            self.llm_tokenizer = AutoTokenizer.from_pretrained(SYNTHESIS_LLM_PATH, cache_dir=HF_CACHE_DIR)
            
            quantization_config = None
            if USE_4BIT_QUANTIZATION and self.device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
                )
            
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                SYNTHESIS_LLM_PATH,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                cache_dir=HF_CACHE_DIR,
                trust_remote_code=True
            ).to(self.device)
            
            self.models_loaded = True
            print("    (Agent: ContentProcessor) - All models loaded successfully.")
            self.shared_context.add_history("ContentProcessorAgent", "Model Loading", "✓ 콘텐츠 분석 모델 로드 완료")
        except Exception as e:
            print(f"    [CRITICAL ERROR] Failed to load models for ContentProcessorAgent: {e}")
            raise

    def process(self):
        """입력 유형에 따라 올바른 파이프라인을 실행하며, OCR은 ocr_handler에게 위임합니다."""
        self._load_models_if_needed()
        
        file_path_input = self.shared_context.get("content_file_path")
        text_input = self.shared_context.get("content_text")

        # 1. 파일 경로가 있는 경우 (이미지 처리)
        if file_path_input and os.path.exists(file_path_input):
            try:
                Image.open(file_path_input).verify()
                self.shared_context.add_history("ContentProcessorAgent", "Processing", "이미지 분석 파이프라인 시작...")
                image = Image.open(file_path_input).convert("RGB")
                
                # --- OCR 로직을 새로운 하이브리드 핸들러 호출로 변경 ---
                self.shared_context.add_history("ContentProcessorAgent", "Processing", "이미지에서 텍스트 추출 중 (Hybrid OCR)...")
                extracted_text = perform_hybrid_ocr(file_path_input)
                self.shared_context.add_history("ContentProcessorAgent", "Processing", "✓ 텍스트 추출 완료")

                # --- Captioning ---
                self.shared_context.add_history("ContentProcessorAgent", "Processing", "시각적 특징 분석 중 (Captioning)...")
                caption_inputs = self.caption_processor(images=image, return_tensors="pt").to(self.device)
                caption_outputs = self.caption_model.generate(**caption_inputs, max_new_tokens=128)
                visual_caption = self.caption_processor.decode(caption_outputs[0], skip_special_tokens=True) if caption_outputs else "이미지의 시각적 특징을 분석할 수 없습니다."
                self.shared_context.add_history("ContentProcessorAgent", "Processing", "✓ 시각적 분석 완료")
                
                combined_content = f"[이미지에서 추출된 텍스트]:\n{extracted_text}\n\n[이미지의 시각적 요약]:\n{visual_caption}"
                self._summarize_text(combined_content)
                return

            except Exception as e:
                error_msg = f"이미지 처리 중 오류 발생: {e}"
                print(f"    (ContentProcessor) - {error_msg}")
                self.shared_context.set("content_summary", f"오류: {error_msg}")
                return

        # 2. 텍스트 입력만 있는 경우
        elif text_input:
            self.shared_context.add_history("ContentProcessorAgent", "Processing", "텍스트 입력 처리 중...")
            self._summarize_text(text_input)
            return

        # 3. 유효한 입력이 없는 경우
        self.shared_context.set("content_summary", "오류: 분석할 콘텐츠가 없습니다.")

    def _summarize_text(self, content: str):
        """LLM이 실패하더라도 절대 멈추지 않는 최종 요약 함수"""
        self.shared_context.add_history("ContentProcessorAgent", "Processing", "정보 종합 및 요약 중...")
        final_summary = "오류: 콘텐츠 요약에 실패했습니다." # 기본 오류 메시지
        try:
            final_prompt = self.prompt_template.format(content=content)
            inputs = self.llm_tokenizer(final_prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQUENCE_LENGTH // 2).to(self.device)
            
            outputs = self.llm_model.generate(**inputs, max_new_tokens=MAX_SEQUENCE_LENGTH // 2, temperature=0.7)
            
            # 모델 출력이 비어있지 않은지 꼼꼼하게 확인
            if outputs is not None and len(outputs) > 0:
                decoded_summary = self.llm_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
                if decoded_summary: # 디코딩된 결과도 비어있지 않은지 확인
                    final_summary = decoded_summary

            self.shared_context.add_history("ContentProcessorAgent", "Processing", "✓ 콘텐츠 요약 완료")
        except Exception as e:
            error_message = f"요약 중 오류 발생: {e}"
            print(f"    [ERROR] Text summarization failed: {e}")
            final_summary = f"오류: 콘텐츠를 요약하는 중 문제가 발생했습니다.\n세부 정보: {e}"
            self.shared_context.add_history("ContentProcessorAgent", "Processing", error_message)
        finally:
            # 어떤 경우에도 content_summary를 설정하여 NoneType 오류를 원천 차단
            self.shared_context.set("content_summary", final_summary)

    def refine(self, feedback: str):
        """피드백을 바탕으로 요약문을 개선합니다."""
        self._load_models_if_needed()
        
        original_summary = self.shared_context.get("content_summary")
        refine_prompt = f"기존 요약문:\n{original_summary}\n\n피드백:\n{feedback}\n\n위 피드백을 반영하여 요약문을 개선해주세요. 개선된 요약문만 간결하게 제시해주세요."
        
        self.shared_context.add_history("ContentProcessorAgent", "Refining", "피드백을 반영하여 요약문을 개선하는 중...")
        try:
            inputs = self.llm_tokenizer(refine_prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQUENCE_LENGTH // 2).to(self.device)
            outputs = self.llm_model.generate(**inputs, max_new_tokens=MAX_SEQUENCE_LENGTH // 2, do_sample=True, temperature=0.7)
            refined_summary = self.llm_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

            self.shared_context.set("content_summary", refined_summary)
            self.shared_context.add_history("ContentProcessorAgent", "Refining", "✓ 피드백 기반 요약문 개선 완료")
        except Exception as e:
            print(f"    [ERROR] Summary refinement failed: {e}")
            self.shared_context.add_history("ContentProcessorAgent", "Refining", f"개선 중 오류 발생: {e}")