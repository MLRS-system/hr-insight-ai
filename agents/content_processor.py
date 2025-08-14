import os
import torch
from PIL import Image
# 필요한 라이브러리들을 미리 임포트합니다.
import easyocr
from transformers import AutoProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from core.shared_context import SharedContext
from config import USE_4BIT_QUANTIZATION, HF_CACHE_DIR, MAX_SEQUENCE_LENGTH, CAPTIONING_MODEL_PATH, SYNTHESIS_LLM_PATH

class ContentProcessorAgent:
    """
    지연 로딩, 조건부 양자화, 상세 로깅이 적용된 콘텐츠 처리 에이전트.
    """
    def __init__(self, shared_context: SharedContext):
        """에이전트를 초기화하지만, 무거운 모델들은 로드하지 않습니다."""
        self.shared_context = shared_context
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.ocr_reader = None
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
        """실제로 필요할 때 모델들을 로드하는 내부 함수"""
        if self.models_loaded:
            return
            
        print("    (Agent: ContentProcessor) - Loading models for the first time...")
        self.shared_context.add_history("ContentProcessorAgent", "Model Loading", "AI 모델들을 메모리에 로드하는 중...")
        try:
            print("    (Sub-task) Loading OCR model (easyocr)...")
            self.ocr_reader = easyocr.Reader(['ko', 'en'], gpu=torch.cuda.is_available())

            print(f"    (Sub-task) Loading Image Captioning model ({CAPTIONING_MODEL_PATH})...")
            self.caption_processor = AutoProcessor.from_pretrained(CAPTIONING_MODEL_PATH, cache_dir=HF_CACHE_DIR, use_fast=True)
            self.caption_model = BlipForConditionalGeneration.from_pretrained(CAPTIONING_MODEL_PATH, cache_dir=HF_CACHE_DIR).to(self.device)

            print(f"    (Sub-task) Loading Synthesis LLM ({SYNTHESIS_LLM_PATH})...")
            self.llm_tokenizer = AutoTokenizer.from_pretrained(SYNTHESIS_LLM_PATH, cache_dir=HF_CACHE_DIR)
            
            quantization_config = None
            if USE_4BIT_QUANTIZATION and self.device == "cuda":
                print("    (Sub-task) - Applying 4-bit quantization.")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
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
            self.shared_context.add_history("ContentProcessorAgent", "Model Loading", "✓ AI 모델 로드 완료")
        except Exception as e:
            if "bitsandbytes" in str(e):
                 print(f"    [CRITICAL ERROR] bitsandbytes 라이브러리 로딩 실패. Cloud CPU 환경에서는 지원되지 않을 수 있습니다: {e}")
            else:
                print(f"    [CRITICAL ERROR] Failed to load models for ContentProcessorAgent: {e}")
            raise

    def process(self):
        """
        입력 유형(파일/텍스트)에 따라 올바른 파이프라인을 실행하도록 수정된 process 함수
        """
        self._load_models_if_needed()
        
        file_path_input = self.shared_context.get("content_file_path")
        text_input = self.shared_context.get("content_text")

        # 1. 파일 입력(이미지, 문서 등)이 최우선 처리 대상
        if file_path_input and os.path.exists(file_path_input):
            self.shared_context.add_history("ContentProcessorAgent", "Processing", "이미지 분석 파이프라인 시작...")
            try:
                # OCR과 캡셔닝을 모두 수행
                self.shared_context.add_history("ContentProcessorAgent", "Processing", "이미지에서 텍스트 추출 중 (OCR)...")
                image = Image.open(file_path_input).convert("RGB")
                ocr_results = self.ocr_reader.readtext(file_path_input, paragraph=True)
                extracted_text = "\n".join([res[1] for res in ocr_results])
                self.shared_context.add_history("ContentProcessorAgent", "Processing", "✓ 텍스트 추출 완료")

                self.shared_context.add_history("ContentProcessorAgent", "Processing", "시각적 특징 분석 중 (Captioning)...")
                caption_inputs = self.caption_processor(images=image, return_tensors="pt").to(self.device)
                caption_outputs = self.caption_model.generate(**caption_inputs, max_new_tokens=128)
                visual_caption = self.caption_processor.decode(caption_outputs[0], skip_special_tokens=True)
                self.shared_context.add_history("ContentProcessorAgent", "Processing", "✓ 시각적 분석 완료")
                
                combined_content = f"[이미지에서 추출된 텍스트]:\n{extracted_text}\n\n[이미지의 시각적 요약]:\n{visual_caption}"
                self._summarize_text(combined_content)
                return

            except Exception as e:
                print(f"    [ERROR] ContentProcessorAgent 이미지 처리 실패: {e}")
                self.shared_context.add_history("ContentProcessorAgent", "Processing", f"이미지 처리 중 오류 발생: {e}")
                return

        # 2. 파일 입력이 없고, 텍스트 입력만 있을 경우
        elif text_input:
            self.shared_context.add_history("ContentProcessorAgent", "Processing", "텍스트 입력 처리 중...")
            self._summarize_text(text_input)
            return

        # 3. 유효한 입력이 아무것도 없는 경우
        print("    (Agent: ContentProcessor) - 유효한 입력 없음.")
        self.shared_context.add_feedback("ContentProcessor: 유효한 파일이나 텍스트 입력이 없습니다.")

    def _summarize_text(self, content: str):
        """로드된 LLM을 사용하여 텍스트를 요약하는 헬퍼 함수"""
        try:
            final_prompt = self.prompt_template.format(content=content)
            inputs = self.llm_tokenizer(final_prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQUENCE_LENGTH // 2).to(self.device)
            outputs = self.llm_model.generate(**inputs, max_new_tokens=MAX_SEQUENCE_LENGTH // 2, do_sample=True, temperature=0.7)
            final_summary = self.llm_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

            self.shared_context.set("content_summary", final_summary)
            self.shared_context.add_history("ContentProcessorAgent", "Processing", "✓ 콘텐츠 요약 및 합성 완료")
        except Exception as e:
            print(f"    [ERROR] Text summarization failed: {e}")
            self.shared_context.add_feedback(f"ContentProcessor: Error during text summarization: {e}")
            self.shared_context.set("content_summary", f"Error: {e}")
            self.shared_context.add_history("ContentProcessorAgent", "Processing", f"요약 중 오류 발생: {e}")

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