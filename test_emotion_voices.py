#!/usr/bin/env python
"""
감정별 Bark TTS 테스트 스크립트
파인튜닝된 감정 모델들을 테스트합니다.
"""

import os
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List

# Bark 모듈 import 전에 환경변수 설정
os.environ["SUNO_USE_SMALL_MODELS"] = "False"
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

class EmotionVoiceTester:
    def __init__(self):
        self.emotions = ["우울", "슬픔", "기쁨"]
        self.emotion_configs = {
            "우울": {
                "speaker_id": "depression_speaker",
                "text_temp": 0.6,
                "waveform_temp": 0.7,
                "emotion_tags": ["[sad]", "[melancholy]", "[depressed]"],
                "voice_dir": "./finetune_output/우울/voice_presets"
            },
            "슬픔": {
                "speaker_id": "sadness_speaker", 
                "text_temp": 0.5,
                "waveform_temp": 0.6,
                "emotion_tags": ["[crying]", "[sorrow]", "[grief]"],
                "voice_dir": "./finetune_output/슬픔/voice_presets"
            },
            "기쁨": {
                "speaker_id": "joy_speaker",
                "text_temp": 0.7, 
                "waveform_temp": 0.8,
                "emotion_tags": ["[happy]", "[excited]", "[joyful]"],
                "voice_dir": "./finetune_output/기쁨/voice_presets"
            }
        }
        
        # 테스트 문장들
        self.test_sentences = {
            "일반": [
                "안녕하세요, 오늘 날씨가 정말 좋네요.",
                "저는 인공지능 음성 합성 기술을 테스트하고 있습니다.",
                "이 문장을 통해 감정 표현이 잘 되는지 확인해보겠습니다."
            ],
            "우울": [
                "오늘도 힘든 하루가 지나가네요...",
                "모든 것이 무의미하게 느껴집니다.",
                "언제쯤 이런 기분에서 벗어날 수 있을까요."
            ],
            "슬픔": [
                "이별은 정말 아픈 일이에요.",
                "눈물이 멈추지 않네요...",
                "소중한 것을 잃었을 때의 마음은 이런 거군요."
            ],
            "기쁨": [
                "오늘 정말 기분이 좋아요!",
                "드디어 꿈이 이루어졌어요!",
                "이렇게 행복한 순간이 있다니 믿기지 않아요!"
            ]
        }

    def load_models(self):
        """Bark 모델 로드"""
        print("📥 Bark 모델 로딩 중...")
        preload_models()
        print("✅ 모델 로딩 완료!")

    def generate_emotion_speech(self, text: str, emotion: str, use_emotion_tag: bool = True) -> np.ndarray:
        """감정별 음성 생성"""
        config = self.emotion_configs[emotion]
        
        # 감정 태그 추가
        if use_emotion_tag:
            emotion_tag = config["emotion_tags"][0]
            tagged_text = f"{emotion_tag} {text}"
        else:
            tagged_text = text
            
        # 음성 프리셋 경로 확인
        voice_dir = Path(config["voice_dir"])
        speaker_path = voice_dir / config["speaker_id"] / "speaker.wav"
        
        if not speaker_path.exists():
            print(f"⚠️ 음성 프리셋을 찾을 수 없습니다: {speaker_path}")
            print("기본 설정으로 생성합니다.")
            history_prompt = None
        else:
            history_prompt = str(speaker_path)
        
        try:
            audio_array = generate_audio(
                tagged_text,
                history_prompt=history_prompt,
                text_temp=config["text_temp"],
                waveform_temp=config["waveform_temp"]
            )
            return audio_array
        except Exception as e:
            print(f"❌ 음성 생성 실패: {e}")
            return None

    def test_single_emotion(self, emotion: str, output_dir: Path):
        """특정 감정 테스트"""
        print(f"\n🎭 {emotion} 감정 음성 테스트")
        print("-" * 40)
        
        emotion_dir = output_dir / emotion
        emotion_dir.mkdir(exist_ok=True)
        
        # 해당 감정의 테스트 문장들
        sentences = self.test_sentences.get(emotion, self.test_sentences["일반"])
        
        for i, sentence in enumerate(sentences):
            print(f"🎯 생성 중 ({i+1}/{len(sentences)}): {sentence}")
            
            # 감정 태그 있는 버전
            audio_with_tag = self.generate_emotion_speech(sentence, emotion, use_emotion_tag=True)
            if audio_with_tag is not None:
                output_file = emotion_dir / f"{emotion}_tagged_{i+1}.wav"
                write_wav(output_file, SAMPLE_RATE, audio_with_tag)
                print(f"  ✅ 태그 버전 저장: {output_file}")
            
            # 감정 태그 없는 버전 (음성 프리셋만 사용)
            audio_without_tag = self.generate_emotion_speech(sentence, emotion, use_emotion_tag=False)
            if audio_without_tag is not None:
                output_file = emotion_dir / f"{emotion}_preset_{i+1}.wav"
                write_wav(output_file, SAMPLE_RATE, audio_without_tag)
                print(f"  ✅ 프리셋 버전 저장: {output_file}")

    def test_comparison(self, output_dir: Path):
        """같은 문장을 다른 감정으로 비교 생성"""
        print(f"\n🔄 감정 비교 테스트")
        print("-" * 40)
        
        comparison_dir = output_dir / "comparison"
        comparison_dir.mkdir(exist_ok=True)
        
        test_sentence = "오늘은 정말 특별한 하루였습니다."
        print(f"🎯 비교 문장: {test_sentence}")
        
        for emotion in self.emotions:
            print(f"  🎭 {emotion} 감정으로 생성 중...")
            
            audio_array = self.generate_emotion_speech(test_sentence, emotion, use_emotion_tag=True)
            if audio_array is not None:
                output_file = comparison_dir / f"comparison_{emotion}.wav"
                write_wav(output_file, SAMPLE_RATE, audio_array)
                print(f"    ✅ 저장: {output_file}")

    def run_test(self, emotions: List[str] = None):
        """전체 테스트 실행"""
        if emotions is None:
            emotions = self.emotions
            
        print("🎤 감정별 Bark TTS 테스트 시작")
        print("=" * 50)
        
        # 출력 디렉토리 생성
        output_dir = Path("./emotion_test_results")
        output_dir.mkdir(exist_ok=True)
        
        # 모델 로드
        self.load_models()
        
        # 각 감정별 테스트
        for emotion in emotions:
            if emotion in self.emotion_configs:
                self.test_single_emotion(emotion, output_dir)
            else:
                print(f"⚠️ 지원하지 않는 감정: {emotion}")
        
        # 비교 테스트
        if len(emotions) > 1:
            self.test_comparison(output_dir)
        
        print(f"\n🎉 테스트 완료!")
        print(f"📁 결과물 위치: {output_dir}")
        
        # 결과 요약
        total_files = len(list(output_dir.rglob("*.wav")))
        print(f"📊 생성된 오디오 파일: {total_files}개")

def main():
    parser = argparse.ArgumentParser(description="감정별 Bark TTS 테스트")
    parser.add_argument(
        "--emotions",
        nargs="+",
        choices=["우울", "슬픔", "기쁨"],
        default=["우울", "슬픔", "기쁨"],
        help="테스트할 감정들 선택"
    )
    
    args = parser.parse_args()
    
    tester = EmotionVoiceTester()
    tester.run_test(args.emotions)

if __name__ == "__main__":
    main() 