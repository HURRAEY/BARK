#!/usr/bin/env python
"""
오디오 노이즈 제거 및 품질 향상 스크립트
Bark TTS로 생성된 음성에서 잡음, 노래소리, 배경음을 제거합니다.
"""

import os
import numpy as np
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
from pathlib import Path
import argparse
from typing import List, Tuple
import librosa
from scipy import signal
import matplotlib.pyplot as plt

class AudioCleaner:
    def __init__(self):
        self.sample_rate = 24000  # Bark의 기본 샘플레이트
        
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """오디오 파일 로드"""
        try:
            audio, sr = sf.read(file_path)
            
            # 스테레오를 모노로 변환
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # 샘플레이트 맞추기
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                sr = self.sample_rate
            
            return audio, sr
        except Exception as e:
            print(f"❌ 오디오 로드 실패: {e}")
            return None, None

    def remove_background_noise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """배경 잡음 제거"""
        print("🔧 배경 잡음 제거 중...")
        
        # noisereduce를 사용한 스펙트럼 기반 노이즈 제거
        # 첫 0.5초를 노이즈 샘플로 사용
        noise_sample_length = min(int(sr * 0.5), len(audio) // 4)
        
        # 여러 방법으로 노이즈 제거 시도
        try:
            # 방법 1: 통계적 노이즈 제거 (기본)
            reduced_noise = nr.reduce_noise(
                y=audio, 
                sr=sr,
                stationary=False,  # 비정상 노이즈도 제거
                prop_decrease=0.8   # 노이즈 감소 비율
            )
            print("   ✅ 통계적 노이즈 제거 완료")
            
            # 방법 2: 스펙트럼 게이팅 추가
            reduced_noise = nr.reduce_noise(
                y=reduced_noise,
                sr=sr,
                use_tensorflow=False
            )
            print("   ✅ 스펙트럼 게이팅 완료")
            
            return reduced_noise
            
        except Exception as e:
            print(f"   ⚠️ 고급 노이즈 제거 실패, 기본 방법 사용: {e}")
            return nr.reduce_noise(y=audio, sr=sr)

    def remove_music_artifacts(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """음악/노래 요소 제거"""
        print("🎵 음악 요소 제거 중...")
        
        try:
            # 주파수 분석을 통한 음악 성분 제거
            # 1. 하모닉 성분 분리
            harmonic, percussive = librosa.effects.hpss(audio)
            
            # 2. 음성 주파수 대역 강조 (80Hz - 8kHz)
            # 고주파 및 저주파 음악 성분 제거
            nyquist = sr // 2
            low_freq = 80 / nyquist
            high_freq = 8000 / nyquist
            
            # 밴드패스 필터 적용
            b, a = signal.butter(4, [low_freq, high_freq], btype='band')
            filtered_audio = signal.filtfilt(b, a, audio)
            
            # 3. 음성과 음악 성분 분리
            # 하모닉 성분을 줄이고 음성 성분 강조
            cleaned_audio = filtered_audio * 0.7 + harmonic * 0.3
            
            print("   ✅ 음악 요소 제거 완료")
            return cleaned_audio
            
        except Exception as e:
            print(f"   ⚠️ 음악 제거 실패, 원본 사용: {e}")
            return audio

    def enhance_speech(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """음성 품질 향상"""
        print("🎤 음성 품질 향상 중...")
        
        try:
            # 1. 다이나믹 레인지 압축
            # PyDub으로 변환
            audio_int16 = (audio * 32767).astype(np.int16)
            audio_segment = AudioSegment(
                audio_int16.tobytes(),
                frame_rate=sr,
                sample_width=2,
                channels=1
            )
            
            # 2. 정규화 및 압축
            normalized = normalize(audio_segment)
            compressed = compress_dynamic_range(normalized, threshold=-20.0, ratio=4.0)
            
            # 3. 다시 numpy로 변환
            enhanced_audio = np.array(compressed.get_array_of_samples(), dtype=np.float32) / 32767.0
            
            print("   ✅ 음성 품질 향상 완료")
            return enhanced_audio
            
        except Exception as e:
            print(f"   ⚠️ 음성 향상 실패, 원본 사용: {e}")
            return audio

    def remove_silence_gaps(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """과도한 무음 구간 제거"""
        print("🔇 무음 구간 최적화 중...")
        
        try:
            # 무음 구간 탐지 및 조정
            intervals = librosa.effects.split(
                audio, 
                top_db=30,  # 무음 임계값
                frame_length=2048,
                hop_length=512
            )
            
            # 무음 구간을 0.25초로 제한
            max_silence = int(sr * 0.25)
            
            cleaned_segments = []
            for i, (start, end) in enumerate(intervals):
                # 음성 구간 추가
                cleaned_segments.append(audio[start:end])
                
                # 마지막 구간이 아니면 짧은 무음 추가
                if i < len(intervals) - 1:
                    silence = np.zeros(max_silence)
                    cleaned_segments.append(silence)
            
            if cleaned_segments:
                result = np.concatenate(cleaned_segments)
                print(f"   ✅ 무음 최적화 완료 ({len(audio)/sr:.1f}초 → {len(result)/sr:.1f}초)")
                return result
            else:
                return audio
                
        except Exception as e:
            print(f"   ⚠️ 무음 최적화 실패: {e}")
            return audio

    def clean_audio_file(self, input_file: str, output_file: str = None) -> str:
        """오디오 파일 전체 정리"""
        if output_file is None:
            path = Path(input_file)
            output_file = str(path.parent / f"{path.stem}_cleaned{path.suffix}")
        
        print(f"🧹 오디오 정리 시작: {input_file}")
        print("=" * 50)
        
        # 1. 오디오 로드
        audio, sr = self.load_audio(input_file)
        if audio is None:
            return None
        
        original_duration = len(audio) / sr
        print(f"📊 원본 길이: {original_duration:.1f}초")
        
        # 2. 배경 잡음 제거
        audio = self.remove_background_noise(audio, sr)
        
        # 3. 음악/노래 요소 제거
        audio = self.remove_music_artifacts(audio, sr)
        
        # 4. 음성 품질 향상
        audio = self.enhance_speech(audio, sr)
        
        # 5. 무음 구간 최적화
        audio = self.remove_silence_gaps(audio, sr)
        
        # 6. 최종 정규화
        audio = audio / np.max(np.abs(audio)) * 0.95
        
        # 7. 저장
        try:
            sf.write(output_file, audio, sr)
            final_duration = len(audio) / sr
            
            print(f"\n🎉 정리 완료!")
            print(f"📁 출력 파일: {output_file}")
            print(f"⏱️ 최종 길이: {final_duration:.1f}초")
            print(f"📉 길이 변화: {original_duration:.1f}초 → {final_duration:.1f}초")
            
            return output_file
            
        except Exception as e:
            print(f"❌ 저장 실패: {e}")
            return None

    def batch_clean(self, file_pattern: str = "*.wav"):
        """여러 파일 일괄 정리"""
        print("🔄 일괄 오디오 정리 시작")
        print("=" * 50)
        
        files = list(Path(".").glob(file_pattern))
        if not files:
            print(f"❌ {file_pattern} 패턴에 맞는 파일이 없습니다.")
            return
        
        print(f"📁 발견된 파일: {len(files)}개")
        
        cleaned_files = []
        for i, file_path in enumerate(files):
            print(f"\n📂 {i+1}/{len(files)}: {file_path.name}")
            print("-" * 30)
            
            output_file = self.clean_audio_file(str(file_path))
            if output_file:
                cleaned_files.append(output_file)
        
        print(f"\n🎉 일괄 정리 완료!")
        print(f"✅ 성공: {len(cleaned_files)}개")
        print(f"❌ 실패: {len(files) - len(cleaned_files)}개")
        
        return cleaned_files

def main():
    parser = argparse.ArgumentParser(description="오디오 노이즈 제거 및 품질 향상")
    parser.add_argument("--file", "-f", help="정리할 오디오 파일")
    parser.add_argument("--output", "-o", help="출력 파일명")
    parser.add_argument("--batch", "-b", action="store_true", help="모든 WAV 파일 일괄 처리")
    parser.add_argument("--pattern", "-p", default="*.wav", help="일괄 처리 파일 패턴")
    
    args = parser.parse_args()
    
    cleaner = AudioCleaner()
    
    if args.batch:
        cleaner.batch_clean(args.pattern)
    elif args.file:
        cleaner.clean_audio_file(args.file, args.output)
    else:
        print("사용법:")
        print("  단일 파일: python audio_noise_cleaner.py -f input.wav")
        print("  일괄 처리: python audio_noise_cleaner.py -b")
        print("  특정 패턴: python audio_noise_cleaner.py -b -p 'emotion_*.wav'")

if __name__ == "__main__":
    main() 