#!/usr/bin/env python
"""
부드러운 오디오 클리너 - 음성 보존하며 가볍게 정리
음성 품질을 유지하면서 약간의 잡음만 제거하는 시스템
"""

import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr
from scipy import signal
import argparse
import os
from pathlib import Path

class GentleAudioCleaner:
    def __init__(self):
        self.sample_rate = 24000
        
    def gentle_noise_reduction(self, audio, sr):
        """부드러운 노이즈 제거"""
        print("🔇 부드러운 잡음 제거 중...")
        
        try:
            # 매우 부드러운 노이즈 제거
            audio_cleaned = nr.reduce_noise(
                y=audio, 
                sr=sr,
                stationary=False,
                prop_decrease=0.3,  # 30%만 제거 (원래 80%)
                n_grad_freq=2,
                n_grad_time=2,
                n_fft=2048,
                win_length=2048,
                hop_length=512
            )
        except:
            # 기본 방법
            audio_cleaned = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.3)
        
        print("   ✅ 부드러운 잡음 제거 완료")
        return audio_cleaned
    
    def gentle_volume_normalize(self, audio):
        """부드러운 볼륨 정규화"""
        print("🎵 볼륨 정규화 중...")
        
        # 피크 정규화 (너무 크거나 작은 소리 조정)
        peak = np.max(np.abs(audio))
        if peak > 0:
            # 0.8로 정규화 (클리핑 방지)
            audio_normalized = audio * (0.8 / peak)
        else:
            audio_normalized = audio
            
        print("   ✅ 볼륨 정규화 완료")
        return audio_normalized
    
    def gentle_silence_trim(self, audio, sr):
        """부드러운 무음 제거 (앞뒤만)"""
        print("✂️ 앞뒤 무음 제거 중...")
        
        # 앞뒤 무음만 제거 (중간 무음은 보존)
        audio_trimmed, _ = librosa.effects.trim(
            audio, 
            top_db=30,  # 부드러운 임계값 (원래 25)
            frame_length=2048,
            hop_length=512
        )
        
        print("   ✅ 앞뒤 무음 제거 완료")
        return audio_trimmed
    
    def clean_audio_gently(self, input_file, output_file=None):
        """부드러운 오디오 정리"""
        input_path = Path(input_file)
        
        if output_file is None:
            output_file = input_path.parent / f"{input_path.stem}_gently_cleaned{input_path.suffix}"
        
        print(f"🧹 부드러운 오디오 정리 시작: {input_file}")
        print("=" * 60)
        
        # 오디오 로드
        try:
            audio, sr = librosa.load(input_file, sr=self.sample_rate)
            original_duration = len(audio) / sr
            print(f"📊 원본 길이: {original_duration:.1f}초")
        except Exception as e:
            print(f"❌ 파일 로드 실패: {e}")
            return False
        
        # 1. 부드러운 노이즈 제거
        audio = self.gentle_noise_reduction(audio, sr)
        
        # 2. 볼륨 정규화
        audio = self.gentle_volume_normalize(audio)
        
        # 3. 앞뒤 무음 제거
        audio = self.gentle_silence_trim(audio, sr)
        
        # 최종 길이 계산
        final_duration = len(audio) / sr
        
        # 저장
        try:
            sf.write(output_file, audio, sr, format='WAV', subtype='PCM_16')
            print("🎉 부드러운 정리 완료!")
            print(f"📁 출력 파일: {output_file}")
            print(f"⏱️ 최종 길이: {final_duration:.1f}초")
            print(f"📉 길이 변화: {original_duration:.1f}초 → {final_duration:.1f}초")
            print("🎯 음성 품질 보존하며 가볍게 정리")
            return True
        except Exception as e:
            print(f"❌ 파일 저장 실패: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="부드러운 오디오 클리너")
    parser.add_argument("-f", "--file", required=True, help="입력 오디오 파일")
    parser.add_argument("-o", "--output", help="출력 파일 (선택사항)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"❌ 파일을 찾을 수 없습니다: {args.file}")
        return
    
    cleaner = GentleAudioCleaner()
    success = cleaner.clean_audio_gently(args.file, args.output)
    
    if success:
        print("\n✅ 부드러운 오디오 정리 완료!")
    else:
        print("\n❌ 오디오 정리 실패!")

if __name__ == "__main__":
    main() 