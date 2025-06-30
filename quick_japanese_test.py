#!/usr/bin/env python3
"""
🇯🇵 빠른 일본어 감정 TTS 테스트
"""

import torch
import numpy as np
from scipy.io.wavfile import write as write_wav

# PyTorch 2.6 호환성 패치
def apply_pytorch_26_fix():
    safe_globals = [
        dict, list, tuple, set, frozenset,
        int, float, str, bool, bytes, type(None),
        torch.Tensor, torch.nn.Parameter,
        np.ndarray, np.dtype, np.core.multiarray.scalar
    ]
    
    try:
        torch.serialization.add_safe_globals(safe_globals)
        print("✅ 패치 적용 완료")
    except:
        pass
    
    original_torch_load = torch.load
    def patched_torch_load(f, map_location=None, pickle_module=None, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return original_torch_load(f, map_location=map_location, pickle_module=pickle_module, **kwargs)
    
    torch.load = patched_torch_load

apply_pytorch_26_fix()

try:
    from bark import SAMPLE_RATE, generate_audio, preload_models
    print("✅ Bark 로드 성공")
    
    print("📥 모델 로드 중... (시간이 걸릴 수 있습니다)")
    preload_models()
    print("✅ 모델 로드 완료")
    
    # 간단한 일본어 테스트
    test_text = "[ja] こんにちは！今日はとても良い天気ですね！"
    print(f"🎵 테스트 음성 생성: {test_text}")
    
    audio = generate_audio(
        test_text,
        history_prompt="v2/ja_speaker_1",
        text_temp=0.7,
        waveform_temp=0.8
    )
    
    # 저장
    output_file = "quick_japanese_test.wav"
    write_wav(output_file, SAMPLE_RATE, (audio * 32767).astype(np.int16))
    
    duration = len(audio) / SAMPLE_RATE
    print(f"✅ 테스트 완료!")
    print(f"📁 파일: {output_file}")
    print(f"⏱️ 길이: {duration:.1f}초")
    
except Exception as e:
    print(f"❌ 오류: {e}") 