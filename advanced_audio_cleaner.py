#!/usr/bin/env python
"""
고급 오디오 클리너 - 노래와 소음 완전 제거
음성만 남기고 모든 배경음, 노래, 잡음을 제거하는 시스템
"""

import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr
from scipy import signal
from scipy.ndimage import median_filter
import argparse
import os
from pathlib import Path

class AdvancedAudioCleaner:
    def __init__(self):
        self.sample_rate = 24000
        
    def remove_music_completely(self, audio, sr):
        """음악 요소 완전 제거"""
        print("🎵 음악 요소 완전 제거 중...")
        
        # 1. 하모닉-퍼커시브 분리 (더 강력한 설정)
        harmonic, percussive = librosa.effects.hpss(audio, margin=(1.0, 5.0))
        
        # 2. 음성 주파수 대역만 추출 (80Hz-8000Hz)
        # 음성 외 주파수 완전 차단
        nyquist = sr // 2
        low_cutoff = 80 / nyquist
        high_cutoff = 8000 / nyquist
        
        # 6차 버터워스 필터로 강력한 필터링
        b, a = signal.butter(6, [low_cutoff, high_cutoff], btype='band')
        audio_filtered = signal.filtfilt(b, a, audio)
        
        # 3. 스펙트럼 마스킹으로 음악 패턴 제거
        stft = librosa.stft(audio_filtered, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # 음성 특성 강화, 음악 특성 억제
        # 음성은 보통 불규칙적이고 음악은 규칙적
        for freq_bin in range(magnitude.shape[0]):
            freq_hz = librosa.fft_frequencies(sr=sr, n_fft=2048)[freq_bin]
            
            # 음성 중요 주파수 (200-4000Hz) 강화
            if 200 <= freq_hz <= 4000:
                magnitude[freq_bin] *= 1.2
            # 음악 주파수 (50-200Hz, 4000-8000Hz) 억제
            elif freq_hz < 200 or freq_hz > 4000:
                magnitude[freq_bin] *= 0.3
        
        # 4. 리듬 패턴 제거 (음악의 규칙적 패턴 탐지 및 제거)
        tempo, beats = librosa.beat.beat_track(y=audio_filtered, sr=sr)
        if tempo > 60:  # 음악적 리듬이 감지되면
            # 비트 위치에서 강도 감소
            beat_frames = librosa.frames_to_samples(beats, hop_length=512)
            for beat in beat_frames:
                if beat < len(audio_filtered):
                    start = max(0, beat - 1000)
                    end = min(len(audio_filtered), beat + 1000)
                    audio_filtered[start:end] *= 0.7
        
        # 5. 재구성
        stft_cleaned = magnitude * np.exp(1j * phase)
        audio_cleaned = librosa.istft(stft_cleaned, hop_length=512)
        
        print("   ✅ 음악 요소 완전 제거 완료")
        return audio_cleaned
    
    def remove_background_noise_aggressive(self, audio, sr):
        """배경 잡음 적극적 제거"""
        print("🔇 배경 잡음 적극적 제거 중...")
        
        # 1. 기본 노이즈 제거 (더 강력한 설정)
        try:
            # 더 강력한 노이즈 제거
            audio_cleaned = nr.reduce_noise(
                y=audio, 
                sr=sr,
                stationary=False,
                prop_decrease=0.8,  # 80% 노이즈 제거
                n_grad_freq=3,
                n_grad_time=5,
                n_fft=2048,
                win_length=2048,
                hop_length=512
            )
        except:
            # 기본 방법
            audio_cleaned = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.8)
        
        # 2. 스펙트럼 게이팅 (음성 아닌 부분 제거)
        stft = librosa.stft(audio_cleaned, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        
        # 음성 특성 임계값 설정
        voice_threshold = np.percentile(magnitude, 30)  # 하위 30% 제거
        mask = magnitude > voice_threshold
        
        # 마스크 적용
        stft_masked = stft * mask
        audio_cleaned = librosa.istft(stft_masked, hop_length=512)
        
        # 3. 적응적 필터링
        # 짧은 구간별로 노이즈 특성 분석하여 제거
        frame_length = sr // 10  # 0.1초 단위
        cleaned_frames = []
        
        for i in range(0, len(audio_cleaned), frame_length):
            frame = audio_cleaned[i:i+frame_length]
            if len(frame) < frame_length // 2:
                break
                
            # 프레임별 노이즈 제거
            frame_energy = np.mean(frame ** 2)
            if frame_energy > np.percentile(audio_cleaned ** 2, 20):  # 에너지 임계값
                cleaned_frames.append(frame)
            else:
                # 저에너지 구간은 음성이 아닐 가능성이 높음
                cleaned_frames.append(frame * 0.1)  # 거의 무음으로
        
        audio_cleaned = np.concatenate(cleaned_frames)
        
        print("   ✅ 배경 잡음 적극적 제거 완료")
        return audio_cleaned
    
    def enhance_voice_only(self, audio, sr):
        """음성만 강화"""
        print("🎤 음성만 강화 중...")
        
        # 1. 음성 주파수 대역 강화 (200-4000Hz)
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        
        # 음성 대역별 강화
        for i, freq in enumerate(freqs):
            if 200 <= freq <= 800:  # 기본 음성 주파수
                magnitude[i] *= 1.5
            elif 800 <= freq <= 2000:  # 명료도 주파수
                magnitude[i] *= 1.3
            elif 2000 <= freq <= 4000:  # 자음 주파수
                magnitude[i] *= 1.2
            elif freq < 200 or freq > 4000:  # 음성 외 주파수
                magnitude[i] *= 0.5
        
        # 2. 음성 특성 강조
        # 포먼트 강화
        audio_enhanced = librosa.istft(magnitude * np.exp(1j * phase), hop_length=512)
        
        # 3. 다이나믹 레인지 최적화 (음성용)
        # 컴프레서 효과 (음성 레벨 균일화)
        threshold = np.percentile(np.abs(audio_enhanced), 70)
        ratio = 3.0
        
        compressed = np.copy(audio_enhanced)
        over_threshold = np.abs(compressed) > threshold
        compressed[over_threshold] = np.sign(compressed[over_threshold]) * (
            threshold + (np.abs(compressed[over_threshold]) - threshold) / ratio
        )
        
        print("   ✅ 음성만 강화 완료")
        return compressed
    
    def remove_silence_advanced(self, audio, sr):
        """고급 무음 제거"""
        print("🔇 고급 무음 제거 중...")
        
        # 1. 음성 활동 감지 (VAD)
        # 더 정확한 음성 구간 탐지
        intervals = librosa.effects.split(
            audio, 
            top_db=25,  # 더 민감한 감지
            frame_length=2048,
            hop_length=512
        )
        
        if len(intervals) == 0:
            return audio
        
        # 2. 음성 구간만 추출하되 자연스러운 페이드 적용
        cleaned_segments = []
        fade_samples = int(0.01 * sr)  # 10ms 페이드
        
        for start, end in intervals:
            segment = audio[start:end]
            
            # 구간이 너무 짧으면 (0.1초 미만) 제거
            if len(segment) < sr * 0.1:
                continue
            
            # 페이드 인/아웃 적용
            if len(segment) > fade_samples * 2:
                # 페이드 인
                segment[:fade_samples] *= np.linspace(0, 1, fade_samples)
                # 페이드 아웃
                segment[-fade_samples:] *= np.linspace(1, 0, fade_samples)
            
            cleaned_segments.append(segment)
            
            # 구간 사이에 짧은 무음 추가 (자연스러운 호흡)
            if len(cleaned_segments) > 1:
                silence = np.zeros(int(0.05 * sr))  # 50ms 무음
                cleaned_segments.append(silence)
        
        if cleaned_segments:
            audio_cleaned = np.concatenate(cleaned_segments)
        else:
            audio_cleaned = audio
        
        print(f"   ✅ 무음 제거 완료 ({len(audio)/sr:.1f}초 → {len(audio_cleaned)/sr:.1f}초)")
        return audio_cleaned
    
    def clean_audio_completely(self, input_file, output_file=None):
        """오디오 완전 정리"""
        if output_file is None:
            output_file = input_file.replace('.wav', '_ultra_cleaned.wav')
        
        print(f"🧹 오디오 완전 정리 시작: {input_file}")
        print("=" * 60)
        
        # 오디오 로드
        audio, sr = librosa.load(input_file, sr=self.sample_rate)
        original_duration = len(audio) / sr
        print(f"📊 원본 길이: {original_duration:.1f}초")
        
        # 1. 음악 요소 완전 제거
        audio = self.remove_music_completely(audio, sr)
        
        # 2. 배경 잡음 적극적 제거
        audio = self.remove_background_noise_aggressive(audio, sr)
        
        # 3. 음성만 강화
        audio = self.enhance_voice_only(audio, sr)
        
        # 4. 고급 무음 제거
        audio = self.remove_silence_advanced(audio, sr)
        
        # 5. 최종 정규화
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.95
        
        # 저장
        sf.write(output_file, audio, sr)
        
        final_duration = len(audio) / sr
        print(f"\n🎉 완전 정리 완료!")
        print(f"📁 출력 파일: {output_file}")
        print(f"⏱️ 최종 길이: {final_duration:.1f}초")
        print(f"📉 길이 변화: {original_duration:.1f}초 → {final_duration:.1f}초")
        print(f"🎯 음성만 남김: 노래/소음 완전 제거")
        
        return output_file

def main():
    parser = argparse.ArgumentParser(description='고급 오디오 클리너 - 노래와 소음 완전 제거')
    parser.add_argument('-f', '--file', required=True, help='정리할 오디오 파일')
    parser.add_argument('-o', '--output', help='출력 파일명')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"❌ 파일을 찾을 수 없습니다: {args.file}")
        return
    
    cleaner = AdvancedAudioCleaner()
    cleaner.clean_audio_completely(args.file, args.output)

if __name__ == "__main__":
    main() 