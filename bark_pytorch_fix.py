#!/usr/bin/env python3
"""
🔧 Bark TTS PyTorch 2.6 호환성 패치
웹 검색 결과를 바탕으로 weights_only 문제 해결
"""

import torch
import os
import sys
from pathlib import Path

def patch_torch_load():
    """PyTorch 2.6의 weights_only=True 기본값 문제 해결"""
    print("🔧 PyTorch 2.6 호환성 패치 적용 중...")
    
    # torch.serialization에 안전한 globals 추가
    try:
        import numpy as np
        torch.serialization.add_safe_globals([
            np.core.multiarray.scalar,
            np.ndarray,
            np.dtype,
            np.core.multiarray._reconstruct,
            np.core.multiarray.scalar,
        ])
        print("✅ NumPy globals 추가 완료")
    except Exception as e:
        print(f"⚠️ NumPy globals 추가 실패: {e}")
    
    # Bark 관련 안전한 globals 추가
    try:
        # 일반적인 Python 타입들
        safe_globals = [
            dict, list, tuple, set, frozenset,
            int, float, str, bool, bytes,
            type(None), slice, range,
            complex, memoryview
        ]
        torch.serialization.add_safe_globals(safe_globals)
        print("✅ 기본 Python 타입 globals 추가 완료")
    except Exception as e:
        print(f"⚠️ 기본 타입 globals 추가 실패: {e}")

def monkey_patch_bark_load():
    """Bark 라이브러리의 torch.load 호출을 패치"""
    print("🐒 Bark torch.load 몽키 패치 적용 중...")
    
    try:
        # bark.generation 모듈 패치
        import bark.generation as bark_gen
        
        # 원본 _load_model 함수 백업
        original_load_model = bark_gen._load_model
        
        def patched_load_model(ckpt_path, device):
            """패치된 모델 로드 함수"""
            try:
                # 먼저 weights_only=False로 시도
                checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
                return original_load_model.__wrapped__(checkpoint, device) if hasattr(original_load_model, '__wrapped__') else checkpoint
            except Exception as e:
                print(f"   ⚠️ weights_only=False 로드 실패, 안전 모드 시도: {e}")
                # 안전 모드로 재시도
                with torch.serialization.safe_globals([
                    dict, list, tuple, int, float, str, bool, 
                    type(None), torch.Tensor, torch.nn.Parameter
                ]):
                    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
                    return checkpoint
        
        # 몽키 패치 적용
        bark_gen._load_model = patched_load_model
        print("✅ bark.generation._load_model 패치 완료")
        
    except ImportError:
        print("⚠️ bark.generation 모듈을 찾을 수 없음")
    except Exception as e:
        print(f"❌ 몽키 패치 실패: {e}")

def create_fixed_bark_script():
    """수정된 Bark 스크립트 생성"""
    print("📝 수정된 Bark 스크립트 생성 중...")
    
    fixed_script = '''#!/usr/bin/env python3
"""
🐶 Bark TTS PyTorch 2.6 호환 버전
웹 검색 결과를 바탕으로 수정됨
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
import librosa
from scipy.io.wavfile import write as write_wav
from datetime import datetime

# PyTorch 2.6 호환성 패치 적용
def apply_pytorch_26_fix():
    """PyTorch 2.6 weights_only 문제 해결"""
    # 안전한 globals 추가
    safe_globals = [
        dict, list, tuple, set, frozenset,
        int, float, str, bool, bytes, type(None),
        torch.Tensor, torch.nn.Parameter,
        np.ndarray, np.dtype, np.core.multiarray.scalar
    ]
    
    try:
        torch.serialization.add_safe_globals(safe_globals)
        print("✅ 안전한 globals 추가 완료")
    except:
        pass
    
    # torch.load 몽키 패치
    original_torch_load = torch.load
    def patched_torch_load(f, map_location=None, pickle_module=None, **kwargs):
        # weights_only가 명시적으로 설정되지 않은 경우 False로 설정
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return original_torch_load(f, map_location=map_location, pickle_module=pickle_module, **kwargs)
    
    torch.load = patched_torch_load
    print("✅ torch.load 패치 적용 완료")

# 패치 적용
apply_pytorch_26_fix()

# Bark 임포트 (패치 적용 후)
try:
    from bark import SAMPLE_RATE, generate_audio, preload_models
    from bark.generation import SUPPORTED_LANGS
    BARK_AVAILABLE = True
    print("✅ Bark 라이브러리 로드 성공")
except ImportError as e:
    BARK_AVAILABLE = False
    print(f"❌ Bark 라이브러리 로드 실패: {e}")

class FixedBarkTTS:
    """PyTorch 2.6 호환 Bark TTS 클래스"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"🎯 사용 디바이스: {self.device}")
        self.models_loaded = False
    
    def load_models(self):
        """모델 로드 (패치 적용됨)"""
        if not BARK_AVAILABLE:
            print("❌ Bark 라이브러리가 없습니다")
            return False
        
        try:
            print("📥 Bark 모델 로드 중...")
            preload_models()
            self.models_loaded = True
            print("✅ 모델 로드 완료")
            return True
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            return False
    
    def generate_speech(self, text, voice_preset="v2/ko_speaker_0", text_temp=0.7, waveform_temp=0.7):
        """음성 생성"""
        if not self.models_loaded:
            if not self.load_models():
                return None
        
        try:
            print(f"🎵 음성 생성 중: {text[:50]}...")
            audio_array = generate_audio(
                text,
                history_prompt=voice_preset,
                text_temp=text_temp,
                waveform_temp=waveform_temp
            )
            print("✅ 음성 생성 완료")
            return audio_array
        except Exception as e:
            print(f"❌ 음성 생성 실패: {e}")
            return None
    
    def create_emotion_conversation(self, output_file="fixed_bark_conversation.wav"):
        """감정별 대화 생성"""
        print("🎭 감정별 대화 생성 시작...")
        
        # 감정별 설정
        emotions = {
            "우울": {"preset": "v2/ko_speaker_1", "text_temp": 0.6, "waveform_temp": 0.7},
            "슬픔": {"preset": "v2/ko_speaker_2", "text_temp": 0.5, "waveform_temp": 0.6},
            "기쁨": {"preset": "v2/ko_speaker_3", "text_temp": 0.7, "waveform_temp": 0.8}
        }
        
        # 대화 스크립트
        conversation = [
            ("우울", "오늘 회전초밥 먹으러 왔는데... 별로 맛이 없네요."),
            ("기쁨", "어? 저는 여기 연어가 정말 맛있던데요! 한 번 드셔보세요!"),
            ("슬픔", "연어... 예전에 좋아하던 사람이 연어를 참 좋아했는데..."),
            ("기쁨", "아, 그러셨구나... 그럼 다른 거 드셔보시는 건 어때요?"),
            ("우울", "뭘 먹어도 다 똑같아요... 요즘 아무것도 맛있지 않아요."),
            ("기쁨", "그럴 때일수록 맛있는 걸 먹어야 해요! 디저트는 어때요?")
        ]
        
        # 각 대사별 음성 생성
        audio_segments = []
        for i, (emotion, text) in enumerate(conversation, 1):
            print(f"   📢 대사 {i}: {emotion} - {text[:30]}...")
            
            config = emotions[emotion]
            audio = self.generate_speech(
                text, 
                voice_preset=config["preset"],
                text_temp=config["text_temp"],
                waveform_temp=config["waveform_temp"]
            )
            
            if audio is not None:
                audio_segments.append(audio)
                # 대사 간 무음 추가
                silence = np.zeros(int(0.5 * SAMPLE_RATE))
                audio_segments.append(silence)
                print(f"      ✅ 생성 완료")
            else:
                print(f"      ❌ 생성 실패")
        
        # 음성 결합 및 저장
        if audio_segments:
            combined_audio = np.concatenate(audio_segments)
            # 정규화
            combined_audio = combined_audio / np.max(np.abs(combined_audio))
            
            # 저장
            write_wav(output_file, SAMPLE_RATE, (combined_audio * 32767).astype(np.int16))
            
            duration = len(combined_audio) / SAMPLE_RATE
            file_size = Path(output_file).stat().st_size / (1024 * 1024)
            
            print(f"✅ 대화 생성 완료: {output_file}")
            print(f"   📊 길이: {duration:.1f}초")
            print(f"   📦 크기: {file_size:.1f}MB")
            return True
        
        return False

def main():
    print("🐶 Bark TTS PyTorch 2.6 호환 버전 시작")
    print("=" * 60)
    
    # TTS 시스템 초기화
    tts = FixedBarkTTS()
    
    # 감정별 대화 생성
    success = tts.create_emotion_conversation("pytorch26_fixed_conversation.wav")
    
    if success:
        print("\n🎉 성공적으로 완료되었습니다!")
    else:
        print("\n❌ 일부 문제가 발생했습니다.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
'''
    
    with open("bark_pytorch26_fixed.py", 'w', encoding='utf-8') as f:
        f.write(fixed_script)
    
    print("✅ 수정된 스크립트 생성: bark_pytorch26_fixed.py")

def main():
    print("🔧 Bark TTS PyTorch 2.6 호환성 패치 시스템")
    print("=" * 60)
    
    # 패치 적용
    patch_torch_load()
    monkey_patch_bark_load()
    create_fixed_bark_script()
    
    print("\n✅ 모든 패치 적용 완료!")
    print("📁 실행할 파일: bark_pytorch26_fixed.py")
    print("=" * 60)

if __name__ == "__main__":
    main() 