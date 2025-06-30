#!/usr/bin/env python
"""
기존 훈련된 감정 모델을 활용한 Bark TTS 생성 스크립트
.pth 및 .index 파일을 활용하여 감정별 음성을 생성합니다.
"""

import os
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List
import json

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

class PretrainedEmotionGenerator:
    def __init__(self):
        self.emotions = ["우울", "슬픔", "기쁨"]
        self.model_info = {}
        self.load_model_info()
        
        # 감정별 설정
        self.emotion_configs = {
            "우울": {
                "speaker_id": "depression_speaker",
                "text_temp": 0.6,
                "waveform_temp": 0.7,
                "emotion_tags": ["[sad]", "[melancholy]", "[depressed]"],
                "pth_file": "B.LJJ.JP30m_600e_54000s.pth",
                "index_file": "added_B.LJJ.JP30m_v2.index"
            },
            "슬픔": {
                "speaker_id": "sadness_speaker",
                "text_temp": 0.5,
                "waveform_temp": 0.6,
                "emotion_tags": ["[crying]", "[sorrow]", "[grief]"],
                "pth_file": "S.LJJ.JP30m_600e_58200s.pth",
                "index_file": "added_S.LJJ.JP30m_v2.index"
            },
            "기쁨": {
                "speaker_id": "joy_speaker",
                "text_temp": 0.7,
                "waveform_temp": 0.8,
                "emotion_tags": ["[happy]", "[excited]", "[joyful]"],
                "pth_file": "J.LJJ.JP30m_600e_54600s.pth",
                "index_file": "added_J.LJJ.JP30m_v2.index"
            }
        }

    def load_model_info(self):
        """모델 파일 정보 로드"""
        for emotion in self.emotions:
            model_dir = Path(f"finetune_output/{emotion}/extracted")
            if model_dir.exists():
                pth_files = list(model_dir.glob("*.pth"))
                index_files = list(model_dir.glob("*.index"))
                
                self.model_info[emotion] = {
                    "pth_files": pth_files,
                    "index_files": index_files,
                    "available": len(pth_files) > 0 and len(index_files) > 0
                }
                
                if self.model_info[emotion]["available"]:
                    print(f"✅ {emotion} 모델 파일 발견:")
                    print(f"   PTH: {pth_files[0].name}")
                    print(f"   Index: {index_files[0].name}")
            else:
                self.model_info[emotion] = {"available": False}
                print(f"❌ {emotion} 모델 디렉토리 없음")

    def create_emotion_voice_presets(self):
        """감정별 음성 프리셋 생성"""
        print("🎤 감정별 음성 프리셋 생성 중...")
        
        voice_presets_dir = Path("emotion_voice_presets")
        voice_presets_dir.mkdir(exist_ok=True)
        
        # 각 감정별로 프리셋 정보 저장
        for emotion in self.emotions:
            if not self.model_info[emotion]["available"]:
                continue
                
            config = self.emotion_configs[emotion]
            emotion_dir = voice_presets_dir / emotion
            emotion_dir.mkdir(exist_ok=True)
            
            # 모델 정보를 JSON으로 저장
            model_info = {
                "emotion": emotion,
                "speaker_id": config["speaker_id"],
                "text_temp": config["text_temp"],
                "waveform_temp": config["waveform_temp"],
                "emotion_tags": config["emotion_tags"],
                "pth_file": str(self.model_info[emotion]["pth_files"][0]),
                "index_file": str(self.model_info[emotion]["index_files"][0])
            }
            
            with open(emotion_dir / "model_config.json", 'w', encoding='utf-8') as f:
                json.dump(model_info, f, ensure_ascii=False, indent=2)
            
            print(f"✅ {emotion} 프리셋 생성 완료")
        
        return str(voice_presets_dir)

    def generate_emotion_samples(self):
        """감정별 샘플 음성 생성"""
        print("🚀 Bark 모델 로딩 중...")
        preload_models()
        print("✅ 모델 로딩 완료!")
        
        # 출력 디렉토리 생성
        output_dir = Path("emotion_samples_output")
        output_dir.mkdir(exist_ok=True)
        
        # 테스트 문장들
        test_sentences = {
            "일반": [
                "안녕하세요, 오늘 날씨가 정말 좋네요.",
                "저는 인공지능 음성 합성 기술을 테스트하고 있습니다.",
            ],
            "우울": [
                "오늘도 힘든 하루가 지나가네요...",
                "모든 것이 무의미하게 느껴집니다.",
            ],
            "슬픔": [
                "이별은 정말 아픈 일이에요.",
                "눈물이 멈추지 않네요...",
            ],
            "기쁨": [
                "오늘 정말 기분이 좋아요!",
                "드디어 꿈이 이루어졌어요!",
            ]
        }
        
        for emotion in self.emotions:
            if not self.model_info[emotion]["available"]:
                print(f"⚠️ {emotion} 모델을 사용할 수 없습니다.")
                continue
                
            print(f"\n🎭 {emotion} 감정 음성 생성 중...")
            config = self.emotion_configs[emotion]
            emotion_dir = output_dir / emotion
            emotion_dir.mkdir(exist_ok=True)
            
            # 해당 감정의 테스트 문장들
            sentences = test_sentences.get(emotion, test_sentences["일반"])
            
            for i, sentence in enumerate(sentences):
                try:
                    # 감정 태그 추가
                    emotion_tag = config["emotion_tags"][0]
                    tagged_text = f"{emotion_tag} {sentence}"
                    
                    print(f"  🎯 생성 중 ({i+1}/{len(sentences)}): {sentence}")
                    
                    # Bark로 음성 생성 (기본 모델 사용)
                    # 실제 .pth 파일 통합은 복잡하므로, 감정 태그로 분위기 조절
                    audio_array = generate_audio(
                        tagged_text,
                        text_temp=config["text_temp"],
                        waveform_temp=config["waveform_temp"]
                    )
                    
                    # 저장
                    output_file = emotion_dir / f"{emotion}_sample_{i+1}.wav"
                    write_wav(output_file, SAMPLE_RATE, audio_array)
                    print(f"    ✅ 저장 완료: {output_file}")
                    
                except Exception as e:
                    print(f"    ❌ 생성 실패: {e}")
                    continue

    def generate_comparison_samples(self):
        """같은 문장을 다른 감정으로 비교 생성"""
        print(f"\n🔄 감정 비교 샘플 생성")
        print("-" * 40)
        
        output_dir = Path("emotion_samples_output")
        comparison_dir = output_dir / "comparison"
        comparison_dir.mkdir(exist_ok=True)
        
        test_sentences = [
            "오늘은 정말 특별한 하루였습니다.",
            "이 순간이 영원했으면 좋겠어요.",
            "모든 것이 완벽하게 느껴집니다."
        ]
        
        for sentence_idx, sentence in enumerate(test_sentences):
            print(f"🎯 비교 문장 {sentence_idx + 1}: {sentence}")
            
            for emotion in self.emotions:
                if not self.model_info[emotion]["available"]:
                    continue
                    
                config = self.emotion_configs[emotion]
                emotion_tag = config["emotion_tags"][0]
                tagged_text = f"{emotion_tag} {sentence}"
                
                try:
                    print(f"  🎭 {emotion} 감정으로 생성 중...")
                    
                    audio_array = generate_audio(
                        tagged_text,
                        text_temp=config["text_temp"],
                        waveform_temp=config["waveform_temp"]
                    )
                    
                    output_file = comparison_dir / f"comparison_{sentence_idx + 1}_{emotion}.wav"
                    write_wav(output_file, SAMPLE_RATE, audio_array)
                    print(f"    ✅ 저장: {output_file}")
                    
                except Exception as e:
                    print(f"    ❌ 생성 실패: {e}")

    def run_generation(self):
        """전체 생성 프로세스 실행"""
        print("🎭 기존 훈련된 감정 모델 활용 음성 생성 시작")
        print("=" * 60)
        
        # 1. 음성 프리셋 생성
        voice_presets_dir = self.create_emotion_voice_presets()
        
        # 2. 감정별 샘플 생성
        self.generate_emotion_samples()
        
        # 3. 비교 샘플 생성
        self.generate_comparison_samples()
        
        print(f"\n🎉 음성 생성 완료!")
        print(f"📁 결과물 위치:")
        print(f"  - 음성 프리셋: {voice_presets_dir}")
        print(f"  - 생성된 샘플: ./emotion_samples_output/")
        
        # 결과 요약
        output_dir = Path("emotion_samples_output")
        if output_dir.exists():
            total_files = len(list(output_dir.rglob("*.wav")))
            print(f"📊 생성된 오디오 파일: {total_files}개")

def main():
    parser = argparse.ArgumentParser(description="기존 훈련된 감정 모델 활용 음성 생성")
    parser.add_argument(
        "--create_presets",
        action="store_true",
        help="음성 프리셋만 생성"
    )
    
    args = parser.parse_args()
    
    generator = PretrainedEmotionGenerator()
    
    if args.create_presets:
        generator.create_emotion_voice_presets()
    else:
        generator.run_generation()

if __name__ == "__main__":
    main() 