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

[WOMAN] [excited] 回転寿司食べに行きます？


[MAN1] [neutral] 良いですね。


[WOMAN] [commanding] ソン室長、クルーズの準備しなさい。


[MAN2] [polite] はい。


[MAN1] [angry] 回転寿司食べるのになんでクルーズなんですか？


[WOMAN] [proud] ５大洋を一周しながら食べるお寿司が回転寿司じゃないの。

[MAN1] [resigned] 皿が回るんじゃなくて船が回るんだ。


[WOMAN] [directive] ソン室長、始めのコースはブラックタイガーのえび寿司で予約しなさい。
"""

SILENCE_SECS = 0.25  # sentence 사이 0.25초 무음

SPEAKER_PRESETS = {
    # 표 기준: ko_speaker_0 만 여성
    "[WOMAN]": "v2/ko_speaker_0",   # 여성, 밝음
    "[MAN1]": "v2/ko_speaker_2",    # 남성, 중립 (호환용)
    "[MAN2]": "v2/ko_speaker_6",    # 남성, 낮고 침착
}

# TEXT_PROMPT 을 라인별 분석하여 (speaker, sentence) 목록 생성
def parse_dialogue(text: str):
    """대본 문자열을 (speaker_tag, sentence) 튜플 리스트로 변환합니다.

    - 대괄호로 시작하는 행(예: "[WOMAN] [excited] ...")을 문장으로 간주합니다.
      첫 번째 토큰(스피커 태그)로 화자를 식별합니다.
    """
    dialogues = []
    for ln in text.strip().splitlines():
        ln = ln.strip()
        if not ln:
            # 빈 줄 건너뜀
            continue
        if not ln.startswith("["):
            # 화자 이름 행 → 건너뜀
            continue

        # 스피커 태그는 첫 번째 토큰("[WOMAN]" 등)
        speaker_tag = ln.split()[0]
        dialogues.append((speaker_tag, ln))
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
        text_in = sentence  # 태그 포함 문장 그대로 사용
        print(f"[{idx}/{len(DIALOGUES)}] {speaker}: {text_in}")
        preset = SPEAKER_PRESETS.get(speaker)
        audio_arr = generate_audio(
            text_in,
            history_prompt=preset,
            text_temp=0.45,
            waveform_temp=0.45,
        )
        pieces.append(audio_arr)
        pieces.append(silence.copy())

    combined = np.concatenate(pieces)
    output_path = "script_output_Tag.wav"
    write_wav(output_path, SAMPLE_RATE, combined)
    print(f"완료! → {output_path}  (총 길이: {combined.shape[0] / SAMPLE_RATE:.1f}초)")


if __name__ == "__main__":
    main() 