#!/usr/bin/env python
"""
일본어 감정 샘플 생성기
제공된 감정 모델을 활용하여 고품질 일본어 음성 생성
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
import time

# Bark 모듈 import 전에 환경변수 설정 (최대 품질)
os.environ["SUNO_USE_SMALL_MODELS"] = "False"
os.environ.setdefault("SUNO_ENABLE_MPS", "True")
os.environ["SUNO_OFFLOAD_CPU"] = "False"
os.environ["BARK_FORCE_CPU"] = "False"

from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav

# PyTorch 호환성 패치
_original_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _torch_load_compat

class JapaneseEmotionGenerator:
    def __init__(self):
        self.device = self._get_optimal_device()
        self.models_loaded = False
        
        # 일본어 감정별 문장
        self.japanese_sentences = {
            "우울": [
                "今日も辛い一日が過ぎていきますね。",
                "すべてが無意味に感じられます。",
                "一人でいる時間が長すぎます。",
                "何もしたくありません。",
                "心が重くて耐えられません。",
                "世界が灰色にしか見えません。",
                "笑うことさえ難しくなりました。",
                "この気持ちはいつまで続くのでしょうか。"
            ],
            "슬픔": [
                "別れは本当に辛いことです。",
                "涙が止まりません。",
                "大切なものを失いました。",
                "もう戻らない時間です。",
                "胸が詰まって痛いです。",
                "思い出だけが残りました。",
                "なぜこんなに悲しいのでしょう。",
                "心の傷が癒えません。"
            ],
            "기쁨": [
                "今日は本当に気分がいいです！",
                "ついに夢が叶いました！",
                "嬉しくて踊りたいです！",
                "世界がこんなに美しいとは知りませんでした！",
                "笑いが自然に出てきます！",
                "すべてが完璧に見えます！",
                "この瞬間が永遠に続けばいいのに！",
                "心が飛んでいきそうです！"
            ]
        }
        
        # 일본어 화자 설정
        self.japanese_speakers = [
            "v2/ja_speaker_0", "v2/ja_speaker_1", "v2/ja_speaker_2",
            "v2/ja_speaker_3", "v2/ja_speaker_4", "v2/ja_speaker_5",
            "v2/ja_speaker_6", "v2/ja_speaker_7", "v2/ja_speaker_8"
        ]
        
        print(f"🎌 일본어 감정 생성기 초기화 완료")
        print(f"📱 디바이스: {self.device}")

    def _get_optimal_device(self) -> str:
        """최적 디바이스 선택"""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def load_emotion_configs(self):
        """감정 설정 로드"""
        try:
            with open("emotion_models_analysis.json", 'r', encoding='utf-8') as f:
                analysis = json.load(f)
            
            self.emotion_configs = {}
            for emotion in analysis["emotions_analyzed"]:
                features = analysis["extracted_features"][emotion]
                self.emotion_configs[emotion] = {
                    "text_temp": features["suggested_temp"]["text"],
                    "waveform_temp": features["suggested_temp"]["waveform"],
                    "emotion_tags": analysis["bark_config"]["emotion_speakers"][emotion]["emotion_tags"]
                }
            
            print(f"   ✅ {len(self.emotion_configs)}개 감정 설정 로드 완료")
            return True
            
        except Exception as e:
            print(f"   ❌ 감정 설정 로드 실패: {e}")
            return False

    def initialize_models(self):
        """Bark 모델 초기화"""
        print("🚀 Bark 모델 초기화 중...")
        
        try:
            preload_models(
                text_use_gpu=True,
                text_use_small=False,
                coarse_use_gpu=True,
                coarse_use_small=False,
                fine_use_gpu=True,
                fine_use_small=False,
                codec_use_gpu=True,
                force_reload=False
            )
            
            self.models_loaded = True
            print("   ✅ 모든 모델 로드 완료")
            return True
            
        except Exception as e:
            print(f"   ❌ 모델 초기화 실패: {e}")
            return False

    def generate_japanese_emotion_samples(self):
        """일본어 감정 샘플 생성"""
        print("🎌 일본어 감정 샘플 생성 시작")
        print("=" * 50)
        
        # 출력 디렉토리
        output_dir = Path("japanese_emotion_samples")
        output_dir.mkdir(exist_ok=True)
        
        all_generated = {}
        
        for emotion in ["우울", "슬픔", "기쁨"]:
            print(f"\n🎭 {emotion} 감정 일본어 샘플 생성 중...")
            
            emotion_dir = output_dir / emotion
            emotion_dir.mkdir(exist_ok=True)
            
            config = self.emotion_configs[emotion]
            sentences = self.japanese_sentences[emotion]
            generated_files = []
            
            # 각 감정별로 3개 샘플 생성
            for i in range(3):
                try:
                    sentence = sentences[i % len(sentences)]
                    emotion_tag = config["emotion_tags"][0]  # 주요 감정 태그
                    
                    # 일본어 화자 선택 (감정별로 다른 화자)
                    speaker_idx = {"우울": 1, "슬픔": 3, "기쁨": 0}[emotion]
                    
                    # 성별 태그 (일본어)
                    gender_tags = ["[WOMAN]", "[MAN]"]
                    gender_tag = gender_tags[i % 2]
                    
                    # 최종 텍스트 구성
                    final_text = f"{gender_tag} {emotion_tag} {sentence}"
                    
                    print(f"   🎤 샘플 {i+1}: {sentence}")
                    print(f"      📝 생성 텍스트: {final_text}")
                    
                    # 고품질 생성
                    start_time = time.time()
                    audio_array = generate_audio(
                        final_text,
                        text_temp=config["text_temp"],
                        waveform_temp=config["waveform_temp"],
                        silent=True
                    )
                    generation_time = time.time() - start_time
                    
                    # 저장
                    output_file = emotion_dir / f"japanese_{emotion}_sample_{i+1:02d}.wav"
                    write_wav(output_file, SAMPLE_RATE, audio_array)
                    
                    duration = len(audio_array) / SAMPLE_RATE
                    print(f"      ✅ 생성 완료: {duration:.1f}초 ({generation_time:.1f}초 소요)")
                    
                    generated_files.append(str(output_file))
                    
                except Exception as e:
                    print(f"      ❌ 샘플 {i+1} 생성 실패: {e}")
                    continue
            
            all_generated[emotion] = generated_files
            print(f"   🎉 {emotion} 일본어 샘플 {len(generated_files)}개 생성 완료")
        
        return all_generated

    def generate_japanese_conversation(self):
        """일본어 대화 스크립트 생성"""
        print("🎬 일본어 대화 스크립트 생성 중...")
        
        # 일본어 회전초밥 대화
        japanese_script = [
            ("ヒョンジョン", "기쁨", "[joyful]", "回転寿司を食べに行こう！"),
            ("キム・ファンソク", "중립", "[neutral]", "いいね。"),
            ("ヒョンジョン", "기쁨", "[excited]", "ソン・チホも一緒に行こう。"),
            ("ソン・チホ", "중립", "[polite]", "はい。"),
            ("キム・ファンソク", "우울", "[confused]", "でも、なぜクルーズに乗らなければならないの？"),
            ("ヒョンジョン", "기쁨", "[happy]", "五大洋を回りながら回転寿司を食べるのよ。"),
            ("キム・ファンソク", "슬픔", "[resigned]", "あ、船が回るという意味なんだね。"),
            ("ヒョンジョン", "기쁨", "[commanding]", "エビ寿司20個を予約しておいて。"),
        ]
        
        audio_segments = []
        silence = np.zeros(int(SAMPLE_RATE * 0.3))  # 0.3초 무음
        
        print("🎤 대사별 일본어 생성:")
        
        for i, (speaker, emotion_category, emotion_tag, text) in enumerate(japanese_script):
            print(f"\n🎤 {i+1}/8 - {speaker} ({emotion_category}): {text}")
            
            try:
                # 화자별 성별 태그
                gender_tag = "[WOMAN]" if "ヒョンジョン" in speaker else "[MAN]"
                
                # 감정 설정
                if emotion_category in self.emotion_configs:
                    config = self.emotion_configs[emotion_category]
                    text_temp = config["text_temp"]
                    waveform_temp = config["waveform_temp"]
                else:
                    text_temp = 0.5
                    waveform_temp = 0.6
                
                # 최종 텍스트
                final_text = f"{gender_tag} {emotion_tag} {text}"
                
                print(f"   📝 생성 텍스트: {final_text}")
                print(f"   ⚙️ 온도: text={text_temp}, wave={waveform_temp}")
                
                # 음성 생성
                start_time = time.time()
                audio_array = generate_audio(
                    final_text,
                    text_temp=text_temp,
                    waveform_temp=waveform_temp,
                    silent=True
                )
                generation_time = time.time() - start_time
                
                duration = len(audio_array) / SAMPLE_RATE
                print(f"   ✅ 생성 완료: {duration:.1f}초 ({generation_time:.1f}초 소요)")
                
                # 세그먼트 추가
                audio_segments.append(audio_array)
                if i < len(japanese_script) - 1:
                    audio_segments.append(silence)
                
            except Exception as e:
                print(f"   ❌ 생성 실패: {e}")
                continue
        
        # 최종 결합 및 저장
        if audio_segments:
            print(f"\n🔗 {len(audio_segments)//2 + 1}개 음성 세그먼트 결합 중...")
            final_audio = np.concatenate(audio_segments)
            
            # 정규화
            final_audio = final_audio / np.max(np.abs(final_audio)) * 0.95
            
            # 저장
            output_file = "japanese_emotion_conversation.wav"
            write_wav(output_file, SAMPLE_RATE, final_audio)
            
            total_duration = len(final_audio) / SAMPLE_RATE
            print(f"🎉 일본어 대화 스크립트 완성!")
            print(f"📁 파일: {output_file}")
            print(f"⏱️ 총 길이: {total_duration:.1f}초")
            
            return output_file
        
        return None

    def run_japanese_generation(self):
        """일본어 생성 프로세스 실행"""
        print("🎌 일본어 감정 샘플 생성 시작")
        print("=" * 60)
        
        # 1. 감정 설정 로드
        if not self.load_emotion_configs():
            print("❌ 감정 설정 로드 실패")
            return False
        
        # 2. 모델 초기화
        if not self.initialize_models():
            print("❌ 모델 초기화 실패")
            return False
        
        # 3. 일본어 감정 샘플 생성
        samples = self.generate_japanese_emotion_samples()
        
        # 4. 일본어 대화 스크립트 생성
        conversation = self.generate_japanese_conversation()
        
        # 결과 요약
        print(f"\n🎉 일본어 생성 완료!")
        print(f"📊 결과 요약:")
        
        total_samples = sum(len(files) for files in samples.values())
        print(f"  🎵 일본어 감정 샘플: {total_samples}개")
        
        if conversation:
            print(f"  🎬 일본어 대화 스크립트: {conversation}")
        
        return True

def main():
    generator = JapaneseEmotionGenerator()
    success = generator.run_japanese_generation()
    
    if success:
        print("\n🎊 일본어 감정 샘플 생성이 완료되었습니다!")
    else:
        print("\n❌ 일본어 생성 중 오류가 발생했습니다.")

if __name__ == "__main__":
    main() 