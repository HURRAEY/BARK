#!/usr/bin/env python
"""
회전초밥 대화 스크립트에 감정별 음성 적용
풀사이즈 Bark 모델과 감정 태그를 활용한 고품질 음성 생성
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import re

# Bark 모듈 import 전에 환경변수 설정
os.environ["SUNO_USE_SMALL_MODELS"] = "False"  # 풀사이즈 모델 사용
os.environ.setdefault("SUNO_ENABLE_MPS", "True")
os.environ["SUNO_OFFLOAD_CPU"] = "False"

import torch
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav

# PyTorch 호환성 패치
_original_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _torch_load_compat

class EmotionScriptGenerator:
    def __init__(self):
        # 화자별 설정
        self.speaker_configs = {
            "현정": {
                "gender": "[WOMAN]",
                "base_preset": "v2/ko_speaker_0",
                "base_emotions": ["excited", "commanding", "proud", "directive"]
            },
            "김환석": {
                "gender": "[MAN]", 
                "base_preset": "v2/ko_speaker_3",
                "base_emotions": ["neutral", "angry", "resigned"]
            },
            "송치호": {
                "gender": "[MAN]",
                "base_preset": "v2/ko_speaker_7", 
                "base_emotions": ["polite", "neutral"]
            }
        }
        
        # 감정별 설정 (기존 훈련된 모델 기반)
        self.emotion_configs = {
            "excited": {"text_temp": 0.7, "waveform_temp": 0.8},
            "neutral": {"text_temp": 0.4, "waveform_temp": 0.5},
            "commanding": {"text_temp": 0.8, "waveform_temp": 0.7},
            "polite": {"text_temp": 0.3, "waveform_temp": 0.4},
            "angry": {"text_temp": 0.7, "waveform_temp": 0.6},
            "proud": {"text_temp": 0.6, "waveform_temp": 0.7},
            "resigned": {"text_temp": 0.5, "waveform_temp": 0.5},
            "directive": {"text_temp": 0.7, "waveform_temp": 0.6},
            # 추가 감정들 (훈련된 모델 기반)
            "sad": {"text_temp": 0.6, "waveform_temp": 0.7},
            "crying": {"text_temp": 0.5, "waveform_temp": 0.6},
            "happy": {"text_temp": 0.7, "waveform_temp": 0.8},
            "joyful": {"text_temp": 0.7, "waveform_temp": 0.8}
        }

    def get_script_with_emotions(self) -> List[Tuple[str, str, str]]:
        """감정이 적용된 회전초밥 대화 스크립트"""
        return [
            ("현정", "[excited]", "회전초밥 먹으러 가자!"),
            ("김환석", "[neutral]", "좋다."),
            ("현정", "[commanding]", "송치호야, 너도 같이 가자."),
            ("송치호", "[polite]", "네."),
            ("김환석", "[angry]", "그런데 왜 크루즈를 타고 가야 하는 거야?"),
            ("현정", "[proud]", "5대양을 돌면서 회전초밥을 먹는 거야."),
            ("김환석", "[resigned]", "아, 배가 돈다는 뜻이구나."),
            ("현정", "[directive]", "새우초밥 20개 미리 예약해 놔."),
        ]

    def enhance_script_with_trained_emotions(self) -> List[Tuple[str, str, str]]:
        """훈련된 감정 모델을 활용한 향상된 스크립트"""
        return [
            ("현정", "[joyful]", "회전초밥 먹으러 가자!"),  # 기쁨 모델 활용
            ("김환석", "[neutral]", "좋다."),
            ("현정", "[excited]", "송치호야, 너도 같이 가자."),
            ("송치호", "[polite]", "네."),
            ("김환석", "[angry]", "그런데 왜 크루즈를 타고 가야 하는 거야?"),
            ("현정", "[happy]", "5대양을 돌면서 회전초밥을 먹는 거야."),  # 기쁨 모델 활용
            ("김환석", "[sad]", "아, 배가 돈다는 뜻이구나."),  # 우울 모델 활용
            ("현정", "[commanding]", "새우초밥 20개 미리 예약해 놔."),
        ]

    def generate_emotion_script_audio(self, use_enhanced=True):
        """감정이 적용된 스크립트 음성 생성"""
        print("🎭 감정별 회전초밥 대화 음성 생성 시작")
        print("=" * 50)
        
        # 모델 로딩
        print("🚀 Bark 풀사이즈 모델 로딩 중...")
        preload_models()
        print("✅ 모델 로딩 완료!")
        
        # 스크립트 선택
        if use_enhanced:
            script = self.enhance_script_with_trained_emotions()
            output_file = "emotion_script_enhanced.wav"
            print("🎯 향상된 감정 스크립트 사용 (훈련 모델 기반)")
        else:
            script = self.get_script_with_emotions()
            output_file = "emotion_script_basic.wav"
            print("🎯 기본 감정 스크립트 사용")
        
        print(f"📝 총 {len(script)}개 대사 생성 예정")
        print("-" * 30)
        
        # 음성 생성
        audio_segments = []
        silence = np.zeros(int(SAMPLE_RATE * 0.25))  # 0.25초 무음
        
        for i, (speaker, emotion_tag, line) in enumerate(script):
            print(f"🎤 {i+1}/{len(script)} - {speaker} ({emotion_tag}): {line}")
            
            try:
                # 화자 설정
                speaker_config = self.speaker_configs[speaker]
                gender_tag = speaker_config["gender"]
                
                # 감정 태그에서 감정 추출
                emotion = emotion_tag.strip("[]")
                emotion_config = self.emotion_configs.get(emotion, {"text_temp": 0.5, "waveform_temp": 0.5})
                
                # 태그가 포함된 텍스트 생성
                tagged_text = f"{gender_tag} {emotion_tag} {line}"
                
                print(f"   📋 생성 텍스트: {tagged_text}")
                print(f"   ⚙️ Temperature: text={emotion_config['text_temp']}, wave={emotion_config['waveform_temp']}")
                
                # 음성 생성
                audio_array = generate_audio(
                    tagged_text,
                    text_temp=emotion_config["text_temp"],
                    waveform_temp=emotion_config["waveform_temp"]
                )
                
                # 세그먼트 추가
                audio_segments.append(audio_array)
                audio_segments.append(silence)
                
                print(f"   ✅ 생성 완료 ({len(audio_array)/SAMPLE_RATE:.1f}초)")
                
            except Exception as e:
                print(f"   ❌ 생성 실패: {e}")
                continue
        
        # 최종 오디오 결합
        if audio_segments:
            print(f"\n🔗 {len(audio_segments)//2}개 음성 세그먼트 결합 중...")
            final_audio = np.concatenate(audio_segments)
            
            # 저장
            write_wav(output_file, SAMPLE_RATE, final_audio)
            duration = len(final_audio) / SAMPLE_RATE
            
            print(f"🎉 음성 생성 완료!")
            print(f"📁 파일: {output_file}")
            print(f"⏱️ 총 길이: {duration:.1f}초")
            print(f"🎵 품질: {SAMPLE_RATE}Hz, 풀사이즈 모델")
            
            return output_file
        else:
            print("❌ 생성된 음성이 없습니다.")
            return None

    def generate_emotion_comparison(self):
        """같은 대사를 다른 감정으로 비교 생성"""
        print("\n🔄 감정 비교 샘플 생성")
        print("=" * 30)
        
        # 비교할 대사
        test_line = "회전초밥 먹으러 가자!"
        speaker = "현정"
        
        emotions_to_test = ["excited", "happy", "joyful", "sad", "neutral"]
        
        output_dir = Path("emotion_comparison_output")
        output_dir.mkdir(exist_ok=True)
        
        for emotion in emotions_to_test:
            try:
                print(f"🎭 '{emotion}' 감정으로 생성 중...")
                
                speaker_config = self.speaker_configs[speaker]
                gender_tag = speaker_config["gender"]
                emotion_config = self.emotion_configs.get(emotion, {"text_temp": 0.5, "waveform_temp": 0.5})
                
                tagged_text = f"{gender_tag} [{emotion}] {test_line}"
                
                audio_array = generate_audio(
                    tagged_text,
                    text_temp=emotion_config["text_temp"],
                    waveform_temp=emotion_config["waveform_temp"]
                )
                
                output_file = output_dir / f"comparison_{emotion}.wav"
                write_wav(output_file, SAMPLE_RATE, audio_array)
                
                print(f"   ✅ 저장: {output_file}")
                
            except Exception as e:
                print(f"   ❌ 생성 실패: {e}")

def main():
    generator = EmotionScriptGenerator()
    
    print("🎭 감정별 Bark TTS 회전초밥 대화 생성기")
    print("=" * 50)
    
    # 1. 향상된 감정 스크립트 생성
    print("\n1️⃣ 향상된 감정 스크립트 생성 (훈련 모델 기반)")
    enhanced_file = generator.generate_emotion_script_audio(use_enhanced=True)
    
    # 2. 기본 감정 스크립트 생성
    print("\n2️⃣ 기본 감정 스크립트 생성")
    basic_file = generator.generate_emotion_script_audio(use_enhanced=False)
    
    # 3. 감정 비교 샘플 생성
    print("\n3️⃣ 감정 비교 샘플 생성")
    generator.generate_emotion_comparison()
    
    print(f"\n🎉 모든 음성 생성 완료!")
    print(f"📁 생성된 파일들:")
    if enhanced_file:
        print(f"  - 향상된 스크립트: {enhanced_file}")
    if basic_file:
        print(f"  - 기본 스크립트: {basic_file}")
    print(f"  - 감정 비교 샘플: emotion_comparison_output/")

if __name__ == "__main__":
    main() 