# agents/content_processor.py

import os
import torch
from PIL import Image
import easyocr
from transformers import AutoProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import Accelerator

from core.shared_context import SharedContext
# 수정된 config 파일에서 새로운 모델 경로들을 가져옵니다.
from config import (
    USE_4BIT_QUANTIZATION,
    HF_CACHE_DIR,
    MAX_SEQUENCE_LENGTH,
    CAPTIONING_MODEL_PATH,
    SYNTHESIS_LLM_PATH  # MULTIMODAL_MODEL_PATH 대신 사용
)

class ContentProcessorAgent:
    """
    A redesigned agent that uses a local pipeline (OCR + Captioning + LLM)
    to robustly process content without relying on a single, complex multimodal model.
    """

    def __init__(self, shared_context: SharedContext):
        self.shared_context = shared_context
        print("    (Agent: ContentProcessor) - Initializing Local Pipeline...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            # 1. Initialize Local OCR Reader
            print("    (Agent: ContentProcessor) - Loading OCR model (easyocr)...")
            self.ocr_reader = easyocr.Reader(['ko', 'en'], gpu=torch.cuda.is_available())

            # 2. Initialize Local Image Captioning Model (BLIP)
            print(f"    (Agent: ContentProcessor) - Loading Image Captioning model ({CAPTIONING_MODEL_PATH})...")
            self.caption_processor = AutoProcessor.from_pretrained(CAPTIONING_MODEL_PATH, cache_dir=HF_CACHE_DIR, use_fast=True)
            self.caption_model = BlipForConditionalGeneration.from_pretrained(CAPTIONING_MODEL_PATH, cache_dir=HF_CACHE_DIR).to(self.device)

            # 3. Initialize Local Text LLM for Synthesis (e.g., EXAONE or Gemma)
            print(f"    (Agent: ContentProcessor) - Loading Synthesis LLM ({SYNTHESIS_LLM_PATH})...")
            self.llm_tokenizer = AutoTokenizer.from_pretrained(SYNTHESIS_LLM_PATH, cache_dir=HF_CACHE_DIR)
            
            quantization_config = None
            if USE_4BIT_QUANTIZATION:
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
            
            # Load the prompt template
            prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'content_summary.prompt')
            with open(prompt_path, 'r', encoding='utf-8') as f:
                self.prompt_template = f.read()
            print("    (Agent: ContentProcessor) - Initialization complete.")

        except Exception as e:
            print(f"    [CRITICAL ERROR] Failed to load models for ContentProcessorAgent: {e}")
            raise

    def process(self):
        """Executes the local OCR -> Captioning -> Synthesis pipeline."""
        print("    (Agent: ContentProcessor) - Executing local pipeline...")
        file_path_input = self.shared_context.get("content_file_path")
        text_input = self.shared_context.get("content_text")

        # 텍스트만 입력된 경우, 텍스트 LLM으로 바로 요약
        if text_input and not file_path_input:
            print("    (Agent: ContentProcessor) - Processing text-only input...")
            self._summarize_text(text_input)
            return

        if not file_path_input or not os.path.exists(file_path_input):
            print("    (Agent: ContentProcessor) - No valid input found.")
            self.shared_context.add_feedback("ContentProcessor: No valid input file found.")
            return

        try:
            image = Image.open(file_path_input).convert("RGB")

            # --- STEP 1: Local OCR ---
            print("    (Pipeline 1/3) - Extracting text with easyocr...")
            ocr_results = self.ocr_reader.readtext(file_path_input, paragraph=True)
            extracted_text = "\n".join([res[1] for res in ocr_results])
            print(f"    (Pipeline 1/3) - OCR found text: {extracted_text[:100]}...")

            # --- STEP 2: Local Image Captioning ---
            print("    (Pipeline 2/3) - Generating visual caption with BLIP...")
            caption_inputs = self.caption_processor(images=image, return_tensors="pt").to(self.device)
            caption_outputs = self.caption_model.generate(**caption_inputs, max_new_tokens=128)
            visual_caption = self.caption_processor.decode(caption_outputs[0], skip_special_tokens=True)
            print(f"    (Pipeline 2/3) - Visual caption: {visual_caption}")
            
            # --- STEP 3: Summarization with Local Text LLM ---
            print("    (Pipeline 3/3) - Synthesizing summary with local LLM...")
            combined_content = f"""[이미지에서 추출된 텍스트 정보]:
{extracted_text}

[이미지에 대한 핵심 시각 정보]:
{visual_caption}
"""
            self._summarize_text(combined_content)

        except Exception as e:
            print(f"    [ERROR] ContentProcessorAgent pipeline failed: {e}")
            self.shared_context.add_feedback(f"ContentProcessor: Error during local pipeline: {e}")

    def _summarize_text(self, content: str):
        """Helper function to summarize text using the loaded text LLM."""
        try:
            # content_summary.prompt 템플릿에 맞게 포맷팅
            final_prompt = self.prompt_template.format(content=content)
            
            inputs = self.llm_tokenizer(final_prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQUENCE_LENGTH // 2).to(self.device)
            outputs = self.llm_model.generate(**inputs, max_new_tokens=MAX_SEQUENCE_LENGTH // 2, do_sample=True, temperature=0.7)
            final_summary = self.llm_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

            self.shared_context.set("content_summary", final_summary)
            self.shared_context.add_history("ContentProcessorAgent", "Local Pipeline Process", "콘텐츠 요약/합성 완료.")
            print("    (Agent: ContentProcessor) - Summary generation complete.")
        except Exception as e:
            print(f"    [ERROR] Text summarization failed: {e}")
            self.shared_context.add_feedback(f"ContentProcessor: Error during text summarization: {e}")
            self.shared_context.set("content_summary", f"Error: {e}")


    def refine(self, feedback: str):
        """Refines the summary using the local text LLM."""
        print("    (Agent: ContentProcessor) - Refining summary with local LLM...")
        original_summary = self.shared_context.get("content_summary")
        # refine 프롬프트는 헬퍼 함수를 재사용하기 위해 content 인자로 전달합니다.
        refine_prompt = f"기존 요약문:\n{original_summary}\n\n피드백:\n{feedback}\n\n위 피드백을 반영하여 요약문을 개선해주세요. 개선된 요약문만 간결하게 제시해주세요."
        
        # 헬퍼 함수를 재사용하되, 프롬프트 템플릿을 적용하지 않고 직접 사용
        try:
            inputs = self.llm_tokenizer(refine_prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQUENCE_LENGTH // 2).to(self.device)
            outputs = self.llm_model.generate(**inputs, max_new_tokens=MAX_SEQUENCE_LENGTH // 2, do_sample=True, temperature=0.7)
            refined_summary = self.llm_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

            self.shared_context.set("content_summary", refined_summary)
            self.shared_context.add_history("ContentProcessorAgent", "Refine Content", "피드백 기반 요약문 개선 완료.")
            print("    (Agent: ContentProcessor) - Summary refinement complete.")
        except Exception as e:
            print(f"    [ERROR] Summary refinement failed: {e}")