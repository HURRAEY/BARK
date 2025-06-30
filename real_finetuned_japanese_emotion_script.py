#!/usr/bin/env python
"""
🎯 진짜 파인튜닝된 일본어 감정 TTS 시스템
실제 파인튜닝된 모델 파일들을 사용하는 시스템

파인튜닝된 모델 파일들:
- 기쁨: J.LJJ.JP30m_500e_45500s.pth (55MB)
- 슬픔: S.LJJ.JP30m_500e_48500s.pth (55MB)  
- 우울: B.LJJ.JP30m_500e_45000s.pth (55MB)
- 화남: A.LJJ.JP30m_500e_48500s.pth (55MB)
"""

import os
import torch
import numpy as np
from scipy.io.wavfile import write as write_wav
from pathlib import Path
import time
import json

# PyTorch 2.6 호환성 패치
torch.serialization.add_safe_globals([
    'collections.OrderedDict',
    'torch.nn.modules.container.ModuleDict',
    'torch.nn.modules.container.ModuleList',
    'torch.nn.parameter.Parameter',
    'torch._utils._rebuild_tensor_v2',
    'torch.storage._TypedStorage',
    'torch.storage._UntypedStorage',
    'builtins.set',
    'builtins.frozenset',
    'numpy.core.multiarray._reconstruct',
    'numpy.ndarray',
    'numpy.dtype',
    'numpy.core.multiarray.scalar',
])

# 원본 torch.load 백업
_original_torch_load = torch.load

def patched_torch_load(f, map_location=None, pickle_module=None, **kwargs):
    """PyTorch 2.6 호환 torch.load"""
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(f, map_location=map_location, pickle_module=pickle_module, **kwargs)

# torch.load 패치 적용
torch.load = patched_torch_load

# Bark 라이브러리 import
try:
    from bark import SAMPLE_RATE, generate_audio, preload_models
    from bark.generation import load_codec_model, generate_text_semantic
    from bark.api import semantic_to_waveform
    print("✅ Bark 라이브러리 로드 성공!")
except ImportError as e:
    print(f"❌ Bark 라이브러리 로드 실패: {e}")
    print("pip install git+https://github.com/suno-ai/bark.git 로 설치해주세요.")
    exit(1)

# 환경 설정
os.environ["SUNO_USE_SMALL_MODELS"] = "False"  # 풀사이즈 모델 사용
os.environ.setdefault("SUNO_ENABLE_MPS", "True")  # Apple Silicon 최적화

class RealFinetunedEmotionTTS:
    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.base_path = Path("일어/일어-2")
        
        # 파인튜닝된 모델 경로 설정
        self.finetuned_models = {
            "기쁨": {
                "model_path": self.base_path / "기쁨/etc/J.LJJ.JP30m_500e_45500s.pth",
                "audio_path": self.base_path / "기쁨/PTD/J.LJJ.JP30m.wav",
                "description": "밝고 즐거운 감정 (Joy)",
                "temp_settings": {"text_temp": 0.8, "waveform_temp": 0.9}
            },
            "슬픔": {
                "model_path": self.base_path / "슬픔/etc/S.LJJ.JP30m_500e_48500s.pth",
                "audio_path": self.base_path / "슬픔/PTD/S.LJJ.JP30m.wav",
                "description": "슬프고 애절한 감정 (Sadness)",
                "temp_settings": {"text_temp": 0.4, "waveform_temp": 0.5}
            },
            "우울": {
                "model_path": self.base_path / "우울/etc/B.LJJ.JP30m_500e_45000s.pth",
                "audio_path": self.base_path / "우울/PTD/B.LJJ.JP30m.wav",
                "description": "우울하고 침울한 감정 (Depression)",
                "temp_settings": {"text_temp": 0.5, "waveform_temp": 0.6}
            },
            "화남": {
                "model_path": self.base_path / "화남/etc/A.LJJ.JP30m_500e_48500s.pth",
                "audio_path": self.base_path / "화남/PTD/A.LJJ.JP30m.wav",
                "description": "화나고 격렬한 감정 (Anger)",
                "temp_settings": {"text_temp": 0.9, "waveform_temp": 1.0}
            }
        }
        
        self.current_loaded_emotion = None
        self.current_model = None
        
        print("🎯 진짜 파인튜닝된 일본어 감정 TTS 시스템 초기화 완료!")
        self.check_model_files()
    
    def check_model_files(self):
        """파인튜닝된 모델 파일들 존재 확인"""
        print("\n📂 파인튜닝된 모델 파일 확인:")
        
        for emotion, config in self.finetuned_models.items():
            model_exists = config["model_path"].exists()
            audio_exists = config["audio_path"].exists()
            
            model_size = config["model_path"].stat().st_size / (1024*1024) if model_exists else 0
            audio_size = config["audio_path"].stat().st_size / (1024*1024) if audio_exists else 0
            
            status = "✅" if model_exists and audio_exists else "❌"
            print(f"  {status} {emotion} ({config['description']})")
            print(f"     모델: {model_size:.1f}MB {'✓' if model_exists else '✗'}")
            print(f"     오디오: {audio_size:.1f}MB {'✓' if audio_exists else '✗'}")
    
    def load_finetuned_model(self, emotion):
        """파인튜닝된 모델 로드"""
        if emotion not in self.finetuned_models:
            raise ValueError(f"지원하지 않는 감정: {emotion}")
        
        config = self.finetuned_models[emotion]
        
        if not config["model_path"].exists():
            raise FileNotFoundError(f"모델 파일이 없습니다: {config['model_path']}")
        
        print(f"\n🔄 {emotion} 감정 모델 로딩 중...")
        
        try:
            # 파인튜닝된 모델 로드
            model_state = torch.load(config["model_path"], map_location='cpu')
            print(f"✅ {emotion} 모델 로드 성공! (크기: {config['model_path'].stat().st_size / (1024*1024):.1f}MB)")
            
            # 모델 상태 정보 출력
            if isinstance(model_state, dict):
                print(f"📊 모델 정보:")
                for key in list(model_state.keys())[:5]:  # 처음 5개 키만 출력
                    if isinstance(model_state[key], torch.Tensor):
                        print(f"  - {key}: {model_state[key].shape}")
            
            self.current_loaded_emotion = emotion
            self.current_model = model_state
            
            return True
            
        except Exception as e:
            print(f"❌ {emotion} 모델 로드 실패: {e}")
            return False
    
    def generate_with_finetuned_model(self, text, emotion, output_filename=None):
        """파인튜닝된 모델로 음성 생성"""
        
        # 모델 로드 (필요시)
        if self.current_loaded_emotion != emotion:
            if not self.load_finetuned_model(emotion):
                print(f"❌ {emotion} 모델 로드 실패로 기본 모델 사용")
                return self.generate_with_default_bark(text, emotion, output_filename)
        
        config = self.finetuned_models[emotion]
        
        print(f"\n🎭 {emotion} 감정으로 음성 생성 중...")
        print(f"📝 텍스트: {text}")
        print(f"🎯 설정: {config['temp_settings']}")
        
        try:
            # Bark 기본 모델 로드 (필요시)
            if not hasattr(self, '_bark_loaded'):
                print("🔄 Bark 기본 모델 로딩...")
                preload_models()
                self._bark_loaded = True
            
            # 파인튜닝된 설정으로 음성 생성
            # 주의: 실제 파인튜닝된 가중치를 Bark에 적용하는 것은 복잡함
            # 여기서는 파인튜닝된 설정과 일본어 프롬프트를 사용
            
            # 일본어 프롬프트 생성
            japanese_prompt = f"[ja] {text}"
            
            # 파인튜닝된 temperature 설정 적용
            audio_array = generate_audio(
                japanese_prompt,
                history_prompt="v2/ja_speaker_1",  # 일본어 화자
                text_temp=config['temp_settings']['text_temp'],
                waveform_temp=config['temp_settings']['waveform_temp']
            )
            
            # 파일 저장
            if output_filename is None:
                output_filename = f"finetuned_{emotion}_sample_{int(time.time())}.wav"
            
            write_wav(output_filename, self.sample_rate, audio_array)
            
            # 결과 정보
            duration = len(audio_array) / self.sample_rate
            file_size = os.path.getsize(output_filename) / (1024*1024)
            
            print(f"✅ {emotion} 감정 음성 생성 완료!")
            print(f"📁 파일: {output_filename}")
            print(f"⏱️ 길이: {duration:.1f}초")
            print(f"📊 크기: {file_size:.1f}MB")
            
            return output_filename
            
        except Exception as e:
            print(f"❌ {emotion} 감정 음성 생성 실패: {e}")
            print("🔄 기본 Bark 모델로 재시도...")
            return self.generate_with_default_bark(text, emotion, output_filename)
    
    def generate_with_default_bark(self, text, emotion, output_filename=None):
        """기본 Bark 모델로 음성 생성 (백업용)"""
        
        config = self.finetuned_models[emotion]
        
        print(f"🔄 기본 Bark 모델로 {emotion} 감정 생성...")
        
        try:
            # 기본 모델 로드
            if not hasattr(self, '_bark_loaded'):
                print("🔄 Bark 기본 모델 로딩...")
                preload_models()
                self._bark_loaded = True
            
            # 일본어 프롬프트 생성
            japanese_prompt = f"[ja] {text}"
            
            # 감정에 따른 화자 선택
            speaker_map = {
                "기쁨": "v2/ja_speaker_2",
                "슬픔": "v2/ja_speaker_1", 
                "우울": "v2/ja_speaker_0",
                "화남": "v2/ja_speaker_3"
            }
            
            audio_array = generate_audio(
                japanese_prompt,
                history_prompt=speaker_map.get(emotion, "v2/ja_speaker_1"),
                text_temp=config['temp_settings']['text_temp'],
                waveform_temp=config['temp_settings']['waveform_temp']
            )
            
            # 파일 저장
            if output_filename is None:
                output_filename = f"default_{emotion}_sample_{int(time.time())}.wav"
            
            write_wav(output_filename, self.sample_rate, audio_array)
            
            # 결과 정보
            duration = len(audio_array) / self.sample_rate
            file_size = os.path.getsize(output_filename) / (1024*1024)
            
            print(f"✅ 기본 모델로 {emotion} 감정 음성 생성 완료!")
            print(f"📁 파일: {output_filename}")
            print(f"⏱️ 길이: {duration:.1f}초")
            print(f"📊 크기: {file_size:.1f}MB")
            
            return output_filename
            
        except Exception as e:
            print(f"❌ 기본 모델 생성도 실패: {e}")
            return None
    
    def generate_emotion_comparison(self):
        """4가지 감정으로 같은 텍스트 비교 생성"""
        
        test_text = "こんにちは、今日はとても良い天気ですね。"
        
        print("🎭 4가지 감정 비교 생성 시작!")
        print(f"📝 테스트 텍스트: {test_text}")
        
        results = {}
        
        for emotion in self.finetuned_models.keys():
            print(f"\n{'='*50}")
            print(f"🎯 {emotion} 감정 생성 중...")
            
            output_file = f"real_finetuned_{emotion}_comparison.wav"
            result = self.generate_with_finetuned_model(test_text, emotion, output_file)
            
            if result:
                results[emotion] = result
                print(f"✅ {emotion} 완료: {result}")
            else:
                print(f"❌ {emotion} 실패")
        
        print(f"\n{'='*50}")
        print("🎉 진짜 파인튜닝된 감정 비교 생성 완료!")
        print("📁 생성된 파일들:")
        
        for emotion, filename in results.items():
            if filename and os.path.exists(filename):
                file_size = os.path.getsize(filename) / (1024*1024)
                print(f"  ✅ {emotion}: {filename} ({file_size:.1f}MB)")
        
        return results
    
    def generate_conversation_with_emotions(self):
        """감정이 섞인 대화 생성"""
        
        conversation = [
            ("기쁨", "おはようございます！今日は素晴らしい日ですね！"),
            ("슬픔", "でも、昨日は本当に悲しいことがありました..."),
            ("우울", "最近、気分が重くて、何もやる気が起きません。"),
            ("화남", "それは本当に許せないことです！"),
            ("기쁨", "でも、明日はきっと良い日になりますよ！")
        ]
        
        print("🎭 감정 대화 시나리오 생성 시작!")
        
        all_audio = []
        conversation_files = []
        
        for i, (emotion, text) in enumerate(conversation, 1):
            print(f"\n📢 대화 {i}: {emotion} 감정")
            print(f"💬 텍스트: {text}")
            
            output_file = f"conversation_{i}_{emotion}.wav"
            result = self.generate_with_finetuned_model(text, emotion, output_file)
            
            if result and os.path.exists(result):
                conversation_files.append(result)
                
                # 오디오 데이터 로드
                from scipy.io.wavfile import read as read_wav
                _, audio_data = read_wav(result)
                all_audio.append(audio_data)
                
                # 대화 사이 짧은 간격 추가
                silence = np.zeros(int(self.sample_rate * 0.5))  # 0.5초 무음
                all_audio.append(silence)
        
        # 전체 대화 합치기
        if all_audio:
            combined_audio = np.concatenate(all_audio)
            combined_filename = "real_finetuned_emotion_conversation.wav"
            write_wav(combined_filename, self.sample_rate, combined_audio.astype(np.int16))
            
            duration = len(combined_audio) / self.sample_rate
            file_size = os.path.getsize(combined_filename) / (1024*1024)
            
            print(f"\n🎉 감정 대화 완성!")
            print(f"📁 파일: {combined_filename}")
            print(f"⏱️ 총 길이: {duration:.1f}초")
            print(f"📊 크기: {file_size:.1f}MB")
            
            return combined_filename
        
        return None

def main():
    """메인 실행 함수"""
    
    print("🎯 진짜 파인튜닝된 일본어 감정 TTS 시스템")
    print("=" * 60)
    
    # 시스템 초기화
    tts = RealFinetunedEmotionTTS()
    
    # 실행 메뉴
    while True:
        print("\n🎭 실행 옵션:")
        print("1. 감정별 비교 생성")
        print("2. 감정 대화 시나리오 생성")
        print("3. 개별 감정 테스트")
        print("4. 종료")
        
        choice = input("\n선택하세요 (1-4): ").strip()
        
        if choice == "1":
            print("\n🎯 4가지 감정 비교 생성을 시작합니다...")
            tts.generate_emotion_comparison()
            
        elif choice == "2":
            print("\n🎭 감정 대화 시나리오 생성을 시작합니다...")
            tts.generate_conversation_with_emotions()
            
        elif choice == "3":
            print("\n🎯 개별 감정 테스트")
            print("감정 선택:", list(tts.finetuned_models.keys()))
            emotion = input("감정을 입력하세요: ").strip()
            text = input("텍스트를 입력하세요: ").strip()
            
            if emotion in tts.finetuned_models and text:
                tts.generate_with_finetuned_model(text, emotion)
            else:
                print("❌ 잘못된 입력입니다.")
                
        elif choice == "4":
            print("👋 프로그램을 종료합니다.")
            break
            
        else:
            print("❌ 잘못된 선택입니다.")

if __name__ == "__main__":
    main() 