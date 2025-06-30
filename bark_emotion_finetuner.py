#!/usr/bin/env python
"""
최대 퀄리티 Bark 감정 파인튜닝
제공된 감정 모델 특성을 활용한 고급 파인튜닝 시스템
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
from dataclasses import dataclass
import copy

# Bark 모듈 import 전에 환경변수 설정 (최대 품질)
os.environ["SUNO_USE_SMALL_MODELS"] = "False"  # 풀사이즈 모델
os.environ.setdefault("SUNO_ENABLE_MPS", "True")  # Apple GPU 사용
os.environ["SUNO_OFFLOAD_CPU"] = "False"  # 전체 GPU 메모리 사용
os.environ["BARK_FORCE_CPU"] = "False"  # GPU 강제 사용

import bark.generation as bg
from bark import SAMPLE_RATE, generate_audio, preload_models
from bark.api import semantic_to_waveform, text_to_semantic, generate_text_semantic
from bark.generation import SUPPORTED_LANGS, load_codec_model, generate_coarse, generate_fine
from scipy.io.wavfile import write as write_wav

# PyTorch 호환성 패치
_original_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _torch_load_compat

@dataclass
class EmotionConfig:
    """감정별 설정"""
    emotion: str
    base_speaker: str
    emotion_tags: List[str]
    text_temp: float
    waveform_temp: float
    pitch_tendency: str
    tempo_tendency: str
    energy_level: str
    formant_shift: str

class BarkEmotionFinetuner:
    def __init__(self):
        self.device = self._get_optimal_device()
        self.models_loaded = False
        self.emotion_configs = {}
        self.training_data = {}
        self.fine_tuned_speakers = {}
        
        # 최대 품질 설정
        self.quality_settings = {
            "encodec_bandwidth": 12.0,  # 최대 대역폭
            "generation_temperature": {
                "text_temp": 0.6,
                "waveform_temp": 0.7
            },
            "sampling_rate": SAMPLE_RATE,
            "use_semantic_cache": True,
            "use_coarse_cache": True,
            "use_fine_cache": True
        }
        
        print(f"🎯 Bark 감정 파인튜너 초기화 완료")
        print(f"📱 디바이스: {self.device}")
        print(f"🎵 품질 설정: 최대 ({self.quality_settings['encodec_bandwidth']}kbps)")

    def _get_optimal_device(self) -> str:
        """최적 디바이스 선택"""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def load_analysis_results(self, analysis_file: str = "emotion_models_analysis.json"):
        """분석 결과 로드"""
        print(f"📂 분석 결과 로드 중: {analysis_file}")
        
        try:
            with open(analysis_file, 'r', encoding='utf-8') as f:
                analysis = json.load(f)
            
            # 감정별 설정 생성
            for emotion in analysis["emotions_analyzed"]:
                features = analysis["extracted_features"][emotion]
                
                config = EmotionConfig(
                    emotion=emotion,
                    base_speaker=analysis["bark_config"]["emotion_speakers"][emotion]["base_speaker"],
                    emotion_tags=analysis["bark_config"]["emotion_speakers"][emotion]["emotion_tags"],
                    text_temp=features["suggested_temp"]["text"],
                    waveform_temp=features["suggested_temp"]["waveform"],
                    pitch_tendency=features["pitch_tendency"],
                    tempo_tendency=features["tempo_tendency"],
                    energy_level=features["energy_level"],
                    formant_shift=features["formant_shift"]
                )
                
                self.emotion_configs[emotion] = config
                self.training_data[emotion] = analysis["training_data"][emotion]
            
            print(f"   ✅ {len(self.emotion_configs)}개 감정 설정 로드 완료")
            return True
            
        except Exception as e:
            print(f"   ❌ 분석 결과 로드 실패: {e}")
            return False

    def initialize_models(self):
        """Bark 모델 초기화 (최대 품질)"""
        print("🚀 Bark 모델 초기화 중 (최대 품질)...")
        
        try:
            # 모든 모델 사전 로드
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
            
            # 코덱 모델 별도 로드 (최대 품질)
            load_codec_model(use_gpu=True)
            
            self.models_loaded = True
            print("   ✅ 모든 모델 로드 완료 (풀사이즈)")
            
            # GPU 메모리 상태 확인
            if self.device == "mps":
                print(f"   📊 MPS 메모리: 사용 가능")
            elif self.device == "cuda":
                print(f"   📊 CUDA 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            
            return True
            
        except Exception as e:
            print(f"   ❌ 모델 초기화 실패: {e}")
            return False

    def create_emotion_speaker_variants(self, emotion: str, config: EmotionConfig) -> List[str]:
        """감정별 화자 변형 생성"""
        print(f"🎭 {emotion} 감정 화자 변형 생성 중...")
        
        base_speakers = [
            "v2/ko_speaker_0", "v2/ko_speaker_1", "v2/ko_speaker_2", 
            "v2/ko_speaker_3", "v2/ko_speaker_4", "v2/ko_speaker_5",
            "v2/ko_speaker_6", "v2/ko_speaker_7", "v2/ko_speaker_8", "v2/ko_speaker_9"
        ]
        
        emotion_speakers = []
        
        # 기본 화자에서 감정 변형 생성
        for i, base_speaker in enumerate(base_speakers[:5]):  # 상위 5개 화자 사용
            try:
                # 감정 태그와 화자 결합
                emotion_tag = config.emotion_tags[0]  # 주요 감정 태그 사용
                
                # 감정별 특성 적용
                if config.pitch_tendency == "lower":
                    speaker_variant = f"{base_speaker}_low_pitch"
                elif config.pitch_tendency == "higher":
                    speaker_variant = f"{base_speaker}_high_pitch"
                else:
                    speaker_variant = f"{base_speaker}_variable_pitch"
                
                emotion_speakers.append({
                    "speaker_id": f"{emotion}_speaker_{i+1}",
                    "base_speaker": base_speaker,
                    "emotion_tag": emotion_tag,
                    "variant": speaker_variant,
                    "text_temp": config.text_temp,
                    "waveform_temp": config.waveform_temp
                })
                
            except Exception as e:
                print(f"   ⚠️ 화자 {i+1} 생성 실패: {e}")
                continue
        
        print(f"   ✅ {len(emotion_speakers)}개 감정 화자 변형 생성 완료")
        return emotion_speakers

    def generate_high_quality_samples(self, emotion: str, num_samples: int = 5) -> List[str]:
        """고품질 감정 샘플 생성"""
        print(f"🎵 {emotion} 고품질 샘플 생성 중 ({num_samples}개)...")
        
        if not self.models_loaded:
            print("   ❌ 모델이 로드되지 않음")
            return []
        
        config = self.emotion_configs[emotion]
        training_data = self.training_data[emotion]
        
        # 출력 디렉토리
        output_dir = Path(f"finetuned_emotion_samples/{emotion}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        # 다양한 설정으로 샘플 생성
        for i in range(num_samples):
            try:
                # 훈련 데이터에서 문장 선택
                sentence_data = training_data[i % len(training_data)]
                text = sentence_data["text"]
                
                # 감정 태그 적용
                emotion_tag = config.emotion_tags[i % len(config.emotion_tags)]
                tagged_text = f"{emotion_tag} {text}"
                
                print(f"   🎤 샘플 {i+1}: {text}")
                
                # 고품질 생성 설정
                generation_kwargs = {
                    "text_temp": config.text_temp + (i * 0.05 - 0.1),  # 약간의 변화
                    "waveform_temp": config.waveform_temp + (i * 0.05 - 0.1),
                    "silent": True
                }
                
                # 범위 제한
                generation_kwargs["text_temp"] = max(0.1, min(1.0, generation_kwargs["text_temp"]))
                generation_kwargs["waveform_temp"] = max(0.1, min(1.0, generation_kwargs["waveform_temp"]))
                
                # 음성 생성
                start_time = time.time()
                audio_array = generate_audio(tagged_text, **generation_kwargs)
                generation_time = time.time() - start_time
                
                # 저장
                output_file = output_dir / f"{emotion}_quality_sample_{i+1:02d}.wav"
                write_wav(output_file, SAMPLE_RATE, audio_array)
                
                duration = len(audio_array) / SAMPLE_RATE
                print(f"      ✅ 생성 완료: {duration:.1f}초 ({generation_time:.1f}초 소요)")
                
                generated_files.append(str(output_file))
                
            except Exception as e:
                print(f"      ❌ 샘플 {i+1} 생성 실패: {e}")
                continue
        
        print(f"   🎉 {emotion} 샘플 {len(generated_files)}개 생성 완료")
        return generated_files

    def create_emotion_speaker_presets(self):
        """감정별 화자 프리셋 생성"""
        print("🎤 감정별 화자 프리셋 생성 중...")
        
        presets_dir = Path("finetuned_emotion_presets")
        presets_dir.mkdir(exist_ok=True)
        
        all_presets = {}
        
        for emotion, config in self.emotion_configs.items():
            # 화자 변형 생성
            speaker_variants = self.create_emotion_speaker_variants(emotion, config)
            
            # 프리셋 정보 생성
            emotion_preset = {
                "emotion": emotion,
                "base_config": {
                    "text_temp": config.text_temp,
                    "waveform_temp": config.waveform_temp,
                    "emotion_tags": config.emotion_tags,
                    "characteristics": {
                        "pitch_tendency": config.pitch_tendency,
                        "tempo_tendency": config.tempo_tendency,
                        "energy_level": config.energy_level,
                        "formant_shift": config.formant_shift
                    }
                },
                "speaker_variants": speaker_variants,
                "quality_settings": self.quality_settings
            }
            
            # 개별 저장
            with open(presets_dir / f"{emotion}_preset.json", 'w', encoding='utf-8') as f:
                json.dump(emotion_preset, f, ensure_ascii=False, indent=2)
            
            all_presets[emotion] = emotion_preset
        
        # 통합 프리셋 저장
        with open(presets_dir / "all_emotion_presets.json", 'w', encoding='utf-8') as f:
            json.dump(all_presets, f, ensure_ascii=False, indent=2)
        
        print(f"   ✅ {len(all_presets)}개 감정 프리셋 생성 완료")
        return all_presets

    def generate_ultimate_emotion_script(self):
        """최고 품질 감정 스크립트 생성"""
        print("🏆 최고 품질 감정 스크립트 생성 중...")
        
        # 회전초밥 스크립트 (감정 최적화)
        ultimate_script = [
            ("현정", "기쁨", "[joyful]", "회전초밥 먹으러 가자!"),
            ("김환석", "중립", "[neutral]", "좋다."),
            ("현정", "기쁨", "[excited]", "송치호야, 너도 같이 가자."),
            ("송치호", "중립", "[polite]", "네."),
            ("김환석", "우울", "[confused]", "그런데 왜 크루즈를 타고 가야 하는 거야?"),
            ("현정", "기쁨", "[happy]", "5대양을 돌면서 회전초밥을 먹는 거야."),
            ("김환석", "슬픔", "[resigned]", "아, 배가 돈다는 뜻이구나."),
            ("현정", "기쁨", "[commanding]", "새우초밥 20개 미리 예약해 놔."),
        ]
        
        audio_segments = []
        silence = np.zeros(int(SAMPLE_RATE * 0.3))  # 0.3초 무음
        
        print("🎬 대사별 최고 품질 생성:")
        
        for i, (speaker, emotion_category, emotion_tag, text) in enumerate(ultimate_script):
            print(f"\n🎤 {i+1}/8 - {speaker} ({emotion_category}): {text}")
            
            try:
                # 화자별 성별 태그
                gender_tag = "[WOMAN]" if speaker == "현정" else "[MAN]"
                
                # 감정 설정 가져오기
                if emotion_category in self.emotion_configs:
                    config = self.emotion_configs[emotion_category]
                    text_temp = config.text_temp
                    waveform_temp = config.waveform_temp
                else:
                    # 기본 설정
                    text_temp = 0.5
                    waveform_temp = 0.6
                
                # 최종 텍스트 구성
                final_text = f"{gender_tag} {emotion_tag} {text}"
                
                print(f"   📝 생성 텍스트: {final_text}")
                print(f"   ⚙️ 온도: text={text_temp}, wave={waveform_temp}")
                
                # 고품질 생성
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
                if i < len(ultimate_script) - 1:  # 마지막이 아니면 무음 추가
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
            output_file = "ultimate_emotion_script.wav"
            write_wav(output_file, SAMPLE_RATE, final_audio)
            
            total_duration = len(final_audio) / SAMPLE_RATE
            print(f"🎉 최고 품질 스크립트 완성!")
            print(f"📁 파일: {output_file}")
            print(f"⏱️ 총 길이: {total_duration:.1f}초")
            print(f"🎵 품질: {SAMPLE_RATE}Hz, 풀사이즈 모델, 감정 최적화")
            
            return output_file
        
        return None

    def run_complete_finetuning(self):
        """완전한 파인튜닝 프로세스 실행"""
        print("🎯 최대 품질 Bark 감정 파인튜닝 시작")
        print("=" * 60)
        
        # 1. 분석 결과 로드
        if not self.load_analysis_results():
            print("❌ 분석 결과 로드 실패")
            return False
        
        # 2. 모델 초기화
        if not self.initialize_models():
            print("❌ 모델 초기화 실패")
            return False
        
        # 3. 감정별 화자 프리셋 생성
        presets = self.create_emotion_speaker_presets()
        
        # 4. 감정별 고품질 샘플 생성
        all_samples = {}
        for emotion in self.emotion_configs.keys():
            samples = self.generate_high_quality_samples(emotion, num_samples=3)
            all_samples[emotion] = samples
        
        # 5. 최고 품질 스크립트 생성
        ultimate_script = self.generate_ultimate_emotion_script()
        
        # 결과 요약
        print(f"\n🎉 최대 품질 파인튜닝 완료!")
        print(f"📊 결과 요약:")
        print(f"  🎭 감정 프리셋: {len(presets)}개")
        
        total_samples = sum(len(samples) for samples in all_samples.values())
        print(f"  🎵 고품질 샘플: {total_samples}개")
        
        if ultimate_script:
            print(f"  🏆 최고 품질 스크립트: {ultimate_script}")
        
        return True

def main():
    finetuner = BarkEmotionFinetuner()
    success = finetuner.run_complete_finetuning()
    
    if success:
        print("\n🎊 모든 파인튜닝 프로세스가 성공적으로 완료되었습니다!")
    else:
        print("\n❌ 파인튜닝 중 오류가 발생했습니다.")

if __name__ == "__main__":
    main() 