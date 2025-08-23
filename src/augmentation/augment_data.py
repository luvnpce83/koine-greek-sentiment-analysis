# -*- coding: utf-8 -*-
"""
GCP 환경에서 실행되는 데이터 증강 스크립트 (v4 - Aggressive Diversity).
Paraphrasing, 높은 Temperature, 유사성 체크를 통해 데이터 다양성을 극대화합니다.
"""

import os
import time
import pandas as pd
import google.generativeai as genai
from tqdm.asyncio import tqdm
import asyncio

# --- 1. 설정 (Configuration) ---
SERVICE_ACCOUNT_KEY_FILE = "sentiment-analysis-469221-64e5ee43271c.json" 
MODEL_NAME = "gemini-2.5-flash"
INPUT_CSV = "data/primary_emotion_data.csv"
FINAL_OUTPUT_CSV = "data/augmented_emotion_data.csv"
LABEL_FOR_GENERATIVE = 'Surprise'
LABEL_FOR_ENGLISH_ONLY_BACKTRANSLATION = 'Disgust'
CONCURRENT_REQUEST_LIMIT = 20

# --- 2. GCP 인증 및 모델 설정 ---
try:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_KEY_FILE
    model = genai.GenerativeModel(MODEL_NAME)
    print("✅ GCP 인증 및 모델 설정 완료.")
except Exception as e:
    print(f"❌ GCP 인증 중 오류 발생: {e}")
    exit()

# --- 3. 향상된 프롬프트 템플릿 정의 (v5) ---
BACK_TRANSLATE_PROMPT_1 = """Paraphrase the following Koine Greek sentence in modern English. 
CRITICAL: Do not perform a literal, word-for-word translation. Instead, capture the core meaning and sentiment and express it using a completely different sentence structure.

Koine Greek: "{text}"
English Paraphrase:"""

# Revised with a strict output format instruction
BACK_TRANSLATE_PROMPT_2 = """You are an expert scholar of Koine Greek. Take the following {intermediate_language} paraphrase and express its meaning in Koine Greek, as if you were a different author from the same period.
CRITICAL: Avoid simple, direct translation. Use different vocabulary and grammatical structures to create a distinct but semantically equivalent sentence.
The final style must be consistent with 1st-century New Testament Greek.
CRITICAL OUTPUT FORMAT: Your response must contain ONLY the single Koine Greek sentence and nothing else. Do not add explanations, apologies, or any other text.

{intermediate_language} Paraphrase: "{translated_text}"
Koine Greek Expression:"""

# Revised with a strict output format instruction
GENERATIVE_PROMPT = """You are a creative writer and expert in Koine Greek, tasked with creating diverse training data. The following sentence expresses 'Surprise': "{text}".
Your task is to write 6 new, completely different sentences that also convey 'Surprise', but from varied perspectives.
CRITICAL: Do not just rephrase the original. Imagine how different people (a disciple, a Roman soldier, a narrator) would express surprise at the same event. Use different vocabulary and sentence structures for each.
CRITICAL OUTPUT FORMAT: Your response must contain ONLY the 6 new Koine Greek sentences, each on a new line. Do not add titles, explanations, or any other surrounding text."""

# --- 4. 비동기(Asynchronous) 핵심 기능 함수 ---
semaphore = asyncio.Semaphore(CONCURRENT_REQUEST_LIMIT)

async def call_gemini_api_async(prompt, temperature):
    async with semaphore:
        try:
            generation_config = genai.types.GenerationConfig(temperature=temperature)
            response = await model.generate_content_async(prompt, generation_config=generation_config)
            await asyncio.sleep(1)
            return response.text.strip()
        except Exception as e:
            print(f"API 호출 중 오류 발생: {e}")
            return None

async def process_row(row):
    original_text, label = row['text'], row['label']
    augmented_results = []

    # NEW: Higher temperatures to force creativity
    async def back_translate_async(text, lang):
        prompt1 = BACK_TRANSLATE_PROMPT_1.format(text=text)
        paraphrased = await call_gemini_api_async(prompt1, temperature=0.9) # High temp for paraphrasing
        if not paraphrased: return
        
        prompt2 = BACK_TRANSLATE_PROMPT_2.format(intermediate_language=lang, translated_text=paraphrased)
        back_translated = await call_gemini_api_async(prompt2, temperature=0.8) # High-ish temp for creative translation
        
        # NEW: Similarity Check - only add if it's different
        if back_translated and back_translated.strip() != text.strip():
            augmented_results.append({'text': back_translated, 'label': label})

    async def generate_async(text):
        prompt = GENERATIVE_PROMPT.format(text=text)
        generated_text = await call_gemini_api_async(prompt, temperature=1.0) # Max temp for generation
        if generated_text:
            for sentence in generated_text.split('\n'):
                clean_sentence = sentence.strip()
                # NEW: Similarity Check
                if clean_sentence and clean_sentence != text.strip():
                    augmented_results.append({'text': clean_sentence, 'label': label})

    tasks = []
    if label == LABEL_FOR_GENERATIVE:
        tasks.append(generate_async(original_text))
    elif label == LABEL_FOR_ENGLISH_ONLY_BACKTRANSLATION:
        tasks.append(back_translate_async(original_text, "English"))
    else:
        # NOTE: For faster processing and cost saving, you might choose only ONE path here
        # For now, keeping both to maximize diversity.
        tasks.append(back_translate_async(original_text, "English"))
        tasks.append(back_translate_async(original_text, "Modern Greek"))
        
    await asyncio.gather(*tasks)
    return augmented_results

# --- 5. 메인 실행 로직 ---
async def main():
    print(f"'{INPUT_CSV}' 파일을 불러옵니다.")
    try:
        original_df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"❌ 오류: '{INPUT_CSV}' 파일을 찾을 수 없습니다.")
        return

    tasks = [process_row(row) for _, row in original_df.iterrows()]
    print(f"총 {len(original_df)}개 항목에 대한 증강을 병렬로 시작합니다 (v4: Aggressive Diversity)...")
    
    results = await tqdm.gather(*tasks, desc="Processing All Rows Concurrently")
    
    all_augmented_data = []
    for result in results:
        all_augmented_data.extend(result)
        
    print(f"\n🎉 모든 작업 완료! {len(all_augmented_data)}개의 새로운 데이터가 생성되었습니다.")
    
    augmented_df = pd.DataFrame(all_augmented_data)
    final_df = pd.concat([original_df, augmented_df], ignore_index=True)
    final_df.to_csv(FINAL_OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"최종 결과가 '{FINAL_OUTPUT_CSV}' 파일에 저장되었습니다. 총 데이터 수: {len(final_df)}")

if __name__ == "__main__":
    asyncio.run(main())