# -*- coding: utf-8 -*-
"""
데이터 증강 스크립트 for Valence Regression.
Valence score를 기반으로 문장을 생성하고 번역합니다. (v2)
"""

import os
import pandas as pd
import google.generativeai as genai
from tqdm.asyncio import tqdm
import asyncio

# --- 1. 설정 (Configuration) ---
SERVICE_ACCOUNT_KEY_FILE = "[Your_Key_File]" 
MODEL_NAME = "gemini-2.5-flash"
INPUT_CSV = "nt_train_seeds.csv"
FINAL_OUTPUT_CSV = "augmented_nt_train.csv"
CONCURRENT_REQUEST_LIMIT = 20

# --- 2. GCP 인증 및 모델 설정 (Original, Stable Method) ---
try:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_KEY_FILE
    model = genai.GenerativeModel(MODEL_NAME)
    print("✅ GCP 인증 및 모델 설정 완료.")
except Exception as e:
    print(f"❌ GCP 인증 중 오류 발생: {e}")
    exit()

# --- 3. Valence Regression을 위한 프롬프트 템플릿 ---
BACK_TRANSLATE_PROMPT_1 = """Paraphrase the following Koine Greek sentence in modern English. 
CRITICAL: Do not perform a literal, word-for-word translation. Instead, capture the core meaning and sentiment (positive/negative intensity) and express it using a completely different sentence structure.

Koine Greek: "{text}"
English Paraphrase:"""

BACK_TRANSLATE_PROMPT_2 = """You are an expert scholar of Koine Greek. Take the following {intermediate_language} paraphrase and express its meaning in Koine Greek, as if you were a different author from the same period.
CRITICAL: Avoid simple, direct translation. Use different vocabulary and grammatical structures to create a distinct but semantically equivalent sentence that preserves the original sentiment.
The final style must be consistent with 1st-century New Testament Greek.
CRITICAL OUTPUT FORMAT: Your response must contain ONLY the single Koine Greek sentence and nothing else.

{intermediate_language} Paraphrase: "{translated_text}"
Koine Greek Expression:"""

GENERATIVE_PROMPT = """You are a creative writer and expert in Koine Greek, tasked with creating diverse training data for a sentiment analysis model.
The following sentence has a calculated valence score of **{valence_score:.2f}** (where -1.0 is very negative and +1.0 is very positive).
Your task is to write **6** new, completely different Koine Greek sentences that would likely receive a similar valence score. Capture the same intensity of positivity or negativity.
CRITICAL: Do not just rephrase the original. Imagine different contexts or speakers expressing a similar sentiment. Use varied vocabulary and sentence structures.

Original Koine Greek Sentence: "{text}"
CRITICAL OUTPUT FORMAT: Your response must contain ONLY the 6 new Koine Greek sentences, each on a new line. Do not add titles, explanations, or any other text.
"""

# --- 4. 비동기(Asynchronous) 핵심 기능 함수 ---
semaphore = asyncio.Semaphore(CONCURRENT_REQUEST_LIMIT)

async def call_gemini_api_async(prompt, temperature):
    async with semaphore:
        try:
            generation_config = genai.types.GenerationConfig(temperature=temperature)
            response = await model.generate_content_async(prompt, generation_config=generation_config)
            await asyncio.sleep(1.5) # Respect rate limits
            return response.text.strip()
        except Exception as e:
            print(f"API 호출 중 오류 발생: {e}. Prompt: {prompt[:100]}...")
            return None

async def process_row(row):
    original_text, valence_score = row['text'], row['valence_score']
    augmented_results = []

    async def back_translate_async(text, lang):
        prompt1 = BACK_TRANSLATE_PROMPT_1.format(text=text)
        paraphrased = await call_gemini_api_async(prompt1, temperature=0.9)
        if not paraphrased: return
        
        prompt2 = BACK_TRANSLATE_PROMPT_2.format(intermediate_language=lang, translated_text=paraphrased)
        back_translated = await call_gemini_api_async(prompt2, temperature=0.7)
        
        if back_translated and back_translated.strip() and back_translated.strip() != text.strip():
            augmented_results.append({'text': back_translated, 'valence_score': valence_score})

    async def generate_async(text, score):
        prompt = GENERATIVE_PROMPT.format(text=text, valence_score=score)
        generated_text = await call_gemini_api_async(prompt, temperature=1.0)
        if generated_text:
            for sentence in generated_text.split('\n'):
                clean_sentence = sentence.strip()
                if clean_sentence and clean_sentence != text.strip():
                    augmented_results.append({'text': clean_sentence, 'valence_score': score})

    tasks = [
        back_translate_async(original_text, "English"),
        back_translate_async(original_text, "Modern Greek"),
        generate_async(original_text, valence_score)
    ]

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
    print(f"총 {len(original_df)}개 원본(seed) 항목에 대한 증강을 시작합니다...")
    
    results = await tqdm.gather(*tasks, desc="Augmenting NT Training Seeds")
    
    all_augmented_data = []
    for result in results:
        all_augmented_data.extend(result)
        
    print(f"\n🎉 증강 완료! {len(all_augmented_data)}개의 새로운 데이터가 생성되었습니다.")
    
    augmented_df = pd.DataFrame(all_augmented_data)
    final_df = pd.concat([original_df, augmented_df], ignore_index=True)
    final_df.drop_duplicates(subset=['text'], inplace=True)

    final_df.to_csv(FINAL_OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"최종 결과가 '{FINAL_OUTPUT_CSV}' 파일에 저장되었습니다. 총 데이터 수: {len(final_df)}")

if __name__ == "__main__":
    asyncio.run(main())