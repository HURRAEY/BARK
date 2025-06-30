#!/usr/bin/env python
"""Bark TTS 생성 스크립트
사용 방법: `python generate_bark.py`
`script_output.wav` 파일이 생성됩니다.
"""

import os
import numpy as np

# Bark 모듈 import 전에 환경변수를 먼저 지정해야 적용됩니다.
os.environ["SUNO_USE_SMALL_MODELS"] = "False"  # 풀사이즈 모델 사용
os.environ.setdefault("SUNO_ENABLE_MPS", "True")
os.environ["SUNO_OFFLOAD_CPU"] = "False"  # 전부 GPU에 유지해 품질/속도 향상

import torch
from encodec import EncodecModel
import bark.generation as bg
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav

TEXT_PROMPT = """
현정
 回転寿司食べに行きます？

김환석
 良いですね。

현정
 ソン室長、クルーズの準備しなさい

송치호
 はい。

김환석
 回転寿司食べるのになんでクルーズなんですか？

현정
 ５大洋を一周しながら食べるお寿司が回転寿司じゃないの。

김환석
 皿が回るんじゃなくて船が回るんだ。

현정
 ソン室長、始めのコースはブラックタイガーのえび寿司で予約しなさい。


"""

SILENCE_SECS = 0.25  # sentence 사이 0.25초 무음

SPEAKER_PRESETS = {
    "현정": "v2/ko_speaker_0",   # 밝고 생동감
    "김환석": "v2/ko_speaker_2", # 중립적 남성
    "송치호": "v2/ko_speaker_6", # 낮고 침착한 톤
}

# 감정 태그 규칙
def emotion_tag(speaker: str, line: str) -> str:
    """간단한 규칙 기반 감정 태그 반환"""
    # 물음표가 있고 김환석이면 당황, 현정이면 호기심
    if "?" in line:
        return "[angry]" if speaker == "김환석" else "[excited]"
    # 느낌표가 있으면 긍정적 고양
    if "!" in line:
        return "[excited]"
    # 기본: 현정은 밝음, 송치호는 차분, 김환석은 중립
    base = {
        "현정": "[happy]",
        "김환석": "",
        "송치호": "[calm]",
    }
    return base.get(speaker, "")

# TEXT_PROMPT 을 라인별 분석하여 (speaker, sentence) 목록 생성
def parse_dialogue(text: str):
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    dialogues = []
    current_speaker = None
    for ln in lines:
        # 스피커 라인인지 확인
        if ln in SPEAKER_PRESETS:
            current_speaker = ln
            continue
        # 스피커가 지정된 상태면 해당 문장을 추가
        if current_speaker is not None:
            dialogues.append((current_speaker, ln))
    return dialogues

DIALOGUES = parse_dialogue(TEXT_PROMPT)

# PyTorch >=2.6 기본값 변경 대응: Bark 체크포인트 로드 오류 해결
_original_torch_load = torch.load

def _torch_load_compat(*args, **kwargs):
    # weights_only 기본값 False 로 강제
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)

torch.load = _torch_load_compat

# 모델 및 코덱 모두 미리 로드 (최초 1회 다운로드 후 캐시)
print("preload_models() 시작 – 최초 실행 시 모델 다운로드")
preload_models()

# ---------- 고대역폭(12kbps) 코덱으로 교체 ----------
def _load_codec_hi(device):
    codec = EncodecModel.encodec_model_24khz()
    codec.set_target_bandwidth(12.0)  # 기본 6 → 12 kbps
    codec.eval().to(device)
    return codec

def _load_codec_model_hi(use_gpu=True, force_reload=False):
    device = "cuda" if torch.cuda.is_available() and use_gpu else (
        "mps" if torch.backends.mps.is_available() and use_gpu else "cpu"
    )
    return _load_codec_hi(device)

# Bark 내부 load_codec_model 함수를 덮어씌워 고품질 코덱 사용
bg.load_codec_model = _load_codec_model_hi

def main() -> None:
    print(f"총 {len(DIALOGUES)} 문장 생성")
    pieces: list[np.ndarray] = []
    silence = np.zeros(int(SILENCE_SECS * SAMPLE_RATE), dtype=np.float32)

    for idx, (speaker, sentence) in enumerate(DIALOGUES, 1):
        tag = emotion_tag(speaker, sentence)
        text_in = f"{tag} {sentence}".strip()
        print(f"[{idx}/{len(DIALOGUES)}] {speaker}: {text_in}")
        preset = SPEAKER_PRESETS.get(speaker)
        audio_arr = generate_audio(
            text_in,
            history_prompt=preset,
            text_temp=0.45,
            waveform_temp=0.6,
        )
        pieces.append(audio_arr)
        pieces.append(silence.copy())

    combined = np.concatenate(pieces)
    output_path = "script_output_Rule.wav"
    write_wav(output_path, SAMPLE_RATE, combined)
    print(f"완료! → {output_path}  (총 길이: {combined.shape[0] / SAMPLE_RATE:.1f}초)")


if __name__ == "__main__":
    main() 