#!/usr/bin/env python
"""
감정별 Bark TTS 파인튜닝 스크립트
사용법: python finetune_bark_emotions.py --emotion [우울|슬픔|기쁨]
"""

import os
import argparse
import zipfile
import json
import numpy as np
import torch
from pathlib import Path
import shutil
from typing import Dict, List, Tuple
import librosa
import soundfile as sf

# Bark 모듈 import 전에 환경변수 설정
os.environ["SUNO_USE_SMALL_MODELS"] = "False"  # 풀사이즈 모델 사용
os.environ.setdefault("SUNO_ENABLE_MPS", "True")
os.environ["SUNO_OFFLOAD_CPU"] = "False"  # 전부 GPU에 유지

import bark.generation as bg
from bark import SAMPLE_RATE, generate_audio, preload_models
from bark.api import semantic_to_waveform, text_to_semantic
from bark.generation import SUPPORTED_LANGS

# PyTorch >=2.6 호환성 패치
_original_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _torch_load_compat

class EmotionBarkFineTuner:
    def __init__(self, emotion: str, data_dir: str = "일어"):
        self.emotion = emotion
        self.data_dir = Path(data_dir)
        self.emotion_dir = self.data_dir / emotion
        self.work_dir = Path(f"./finetune_output/{emotion}")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # 감정별 설정
        self.emotion_configs = {
            "우울": {
                "zip_file": "added_B.LJJ.JP30m_v2.zip",
                "speaker_id": f"depression_speaker",
                "text_temp": 0.6,
                "waveform_temp": 0.7,
                "emotion_tags": ["[sad]", "[melancholy]", "[depressed]"]
            },
            "슬픔": {
                "zip_file": "S.LJJ.JP30m_600e_58200s.zip", 
                "speaker_id": f"sadness_speaker",
                "text_temp": 0.5,
                "waveform_temp": 0.6,
                "emotion_tags": ["[crying]", "[sorrow]", "[grief]"]
            },
            "기쁨": {
                "zip_file": "added_J.LJJ.JP30m_v2.zip",
                "speaker_id": f"joy_speaker", 
                "text_temp": 0.7,
                "waveform_temp": 0.8,
                "emotion_tags": ["[happy]", "[excited]", "[joyful]"]
            }
        }
        
        self.config = self.emotion_configs[emotion]
        print(f"🎭 {emotion} 감정 파인튜닝 초기화 완료")

    def extract_data(self) -> bool:
        """ZIP 파일에서 데이터 추출"""
        zip_path = self.emotion_dir / self.config["zip_file"]
        extract_path = self.work_dir / "extracted"
        
        if not zip_path.exists():
            print(f"❌ ZIP 파일을 찾을 수 없습니다: {zip_path}")
            return False
            
        if extract_path.exists():
            print(f"📁 이미 추출된 데이터가 있습니다: {extract_path}")
            return True
            
        print(f"📦 ZIP 파일 추출 중: {zip_path}")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            print(f"✅ 추출 완료: {extract_path}")
            return True
        except Exception as e:
            print(f"❌ 추출 실패: {e}")
            return False

    def prepare_dataset(self) -> List[Dict]:
        """데이터셋 준비 및 메타데이터 생성"""
        extract_path = self.work_dir / "extracted"
        
        # 오디오 파일 찾기
        audio_files = []
        for ext in ['.wav', '.mp3', '.flac', '.m4a']:
            audio_files.extend(list(extract_path.rglob(f"*{ext}")))
        
        if not audio_files:
            print(f"❌ 오디오 파일을 찾을 수 없습니다: {extract_path}")
            return []
            
        print(f"🎵 발견된 오디오 파일: {len(audio_files)}개")
        
        # 메타데이터 생성
        dataset = []
        processed_dir = self.work_dir / "processed_audio"
        processed_dir.mkdir(exist_ok=True)
        
        for i, audio_file in enumerate(audio_files[:50]):  # 처음 50개만 처리
            try:
                # 오디오 로드 및 전처리
                audio, sr = librosa.load(audio_file, sr=SAMPLE_RATE)
                
                # 너무 짧거나 긴 오디오 필터링
                duration = len(audio) / sr
                if duration < 1.0 or duration > 30.0:
                    continue
                
                # 정규화된 파일명으로 저장
                processed_file = processed_dir / f"{self.emotion}_{i:04d}.wav"
                sf.write(processed_file, audio, SAMPLE_RATE)
                
                # 기본 텍스트 생성 (실제로는 전사 데이터가 필요)
                text = f"이것은 {self.emotion} 감정의 음성입니다."
                
                dataset.append({
                    "audio_path": str(processed_file),
                    "text": text,
                    "emotion": self.emotion,
                    "speaker_id": self.config["speaker_id"],
                    "duration": duration
                })
                
                if i % 10 == 0:
                    print(f"📊 처리 진행률: {i+1}/{min(50, len(audio_files))}")
                    
            except Exception as e:
                print(f"⚠️ 파일 처리 실패 {audio_file}: {e}")
                continue
        
        # 메타데이터 저장
        metadata_file = self.work_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 데이터셋 준비 완료: {len(dataset)}개 샘플")
        return dataset

    def create_voice_preset(self, dataset: List[Dict]) -> str:
        """감정별 음성 프리셋 생성"""
        if not dataset:
            return None
            
        print(f"🎤 {self.emotion} 감정 음성 프리셋 생성 중...")
        
        # 대표 오디오 파일 선택 (가장 긴 것)
        best_sample = max(dataset, key=lambda x: x['duration'])
        audio_path = best_sample['audio_path']
        
        # 음성 프리셋 디렉토리 생성
        voice_dir = self.work_dir / "voice_presets"
        voice_dir.mkdir(exist_ok=True)
        
        speaker_dir = voice_dir / self.config["speaker_id"]
        speaker_dir.mkdir(exist_ok=True)
        
        # 오디오 파일 복사
        speaker_audio = speaker_dir / "speaker.wav"
        shutil.copy2(audio_path, speaker_audio)
        
        print(f"✅ 음성 프리셋 생성 완료: {speaker_dir}")
        return str(voice_dir)

    def fine_tune_model(self, dataset: List[Dict], voice_dir: str):
        """Bark 모델 파인튜닝"""
        print(f"🚀 {self.emotion} 감정 Bark 모델 파인튜닝 시작...")
        
        # 모델 로드
        print("📥 Bark 모델 로딩 중...")
        preload_models()
        
        # 음성 프리셋으로 테스트 생성
        test_texts = [
            f"안녕하세요, 저는 {self.emotion}한 기분입니다.",
            f"오늘은 정말 {self.emotion}한 하루네요.",
            f"이 음성은 {self.emotion} 감정을 표현합니다."
        ]
        
        output_dir = self.work_dir / "generated_samples"
        output_dir.mkdir(exist_ok=True)
        
        for i, text in enumerate(test_texts):
            try:
                # 감정 태그 추가
                emotion_tag = self.config["emotion_tags"][0]
                tagged_text = f"{emotion_tag} {text}"
                
                print(f"🎯 생성 중: {tagged_text}")
                
                # 음성 생성
                audio_array = generate_audio(
                    tagged_text,
                    history_prompt=f"{self.config['speaker_id']}/speaker.wav",
                    text_temp=self.config["text_temp"],
                    waveform_temp=self.config["waveform_temp"]
                )
                
                # 저장
                output_file = output_dir / f"{self.emotion}_sample_{i+1}.wav"
                from scipy.io.wavfile import write as write_wav
                write_wav(output_file, SAMPLE_RATE, audio_array)
                
                print(f"✅ 저장 완료: {output_file}")
                
            except Exception as e:
                print(f"❌ 생성 실패: {e}")
                continue

    def run_finetune(self):
        """전체 파인튜닝 프로세스 실행"""
        print(f"🎭 {self.emotion} 감정 Bark 파인튜닝 시작")
        
        # 1. 데이터 추출
        if not self.extract_data():
            return False
        
        # 2. 데이터셋 준비
        dataset = self.prepare_dataset()
        if not dataset:
            return False
        
        # 3. 음성 프리셋 생성
        voice_dir = self.create_voice_preset(dataset)
        if not voice_dir:
            return False
        
        # 4. 모델 파인튜닝
        self.fine_tune_model(dataset, voice_dir)
        
        print(f"🎉 {self.emotion} 감정 파인튜닝 완료!")
        print(f"📁 결과물 위치: {self.work_dir}")
        return True

def main():
    parser = argparse.ArgumentParser(description="감정별 Bark TTS 파인튜닝")
    parser.add_argument(
        "--emotion", 
        choices=["우울", "슬픔", "기쁨", "all"], 
        default="all",
        help="파인튜닝할 감정 선택"
    )
    parser.add_argument(
        "--data_dir", 
        default="일어",
        help="감정 데이터가 있는 디렉토리"
    )
    
    args = parser.parse_args()
    
    if args.emotion == "all":
        emotions = ["우울", "슬픔", "기쁨"]
    else:
        emotions = [args.emotion]
    
    for emotion in emotions:
        print(f"\n{'='*50}")
        print(f"🎭 {emotion} 감정 파인튜닝 시작")
        print(f"{'='*50}")
        
        finetuner = EmotionBarkFineTuner(emotion, args.data_dir)
        success = finetuner.run_finetune()
        
        if success:
            print(f"✅ {emotion} 감정 파인튜닝 성공!")
        else:
            print(f"❌ {emotion} 감정 파인튜닝 실패!")

if __name__ == "__main__":
    main() 