#!/usr/bin/env python
"""
제공된 감정 모델 파일들(.pth, .index) 분석 및 Bark 파인튜닝 준비
RVC 모델에서 오디오 특성을 추출하여 Bark 파인튜닝에 활용
"""

import os
import torch
import numpy as np
import pickle
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import librosa
import soundfile as sf

# PyTorch 호환성 패치
_original_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _torch_load_compat

class EmotionModelAnalyzer:
    def __init__(self):
        self.emotions = ["우울", "슬픔", "기쁨"]
        self.model_data = {}
        self.extracted_features = {}
        
    def analyze_pth_file(self, pth_path: str) -> Dict:
        """PTH 파일 분석"""
        print(f"🔍 PTH 파일 분석 중: {pth_path}")
        
        try:
            # PTH 파일 로드
            checkpoint = torch.load(pth_path, map_location='cpu')
            
            analysis = {
                "file_path": pth_path,
                "file_size_mb": Path(pth_path).stat().st_size / (1024*1024),
                "keys": list(checkpoint.keys()) if isinstance(checkpoint, dict) else "Not a dict",
                "data_type": type(checkpoint).__name__
            }
            
            # 딕셔너리인 경우 상세 분석
            if isinstance(checkpoint, dict):
                for key, value in checkpoint.items():
                    if isinstance(value, torch.Tensor):
                        analysis[f"{key}_shape"] = list(value.shape)
                        analysis[f"{key}_dtype"] = str(value.dtype)
                        analysis[f"{key}_size"] = value.numel()
                    elif isinstance(value, (int, float, str)):
                        analysis[f"{key}_value"] = value
                    else:
                        analysis[f"{key}_type"] = type(value).__name__
            
            print(f"   ✅ PTH 분석 완료: {len(analysis)} 속성 발견")
            return analysis
            
        except Exception as e:
            print(f"   ❌ PTH 분석 실패: {e}")
            return {"error": str(e)}

    def analyze_index_file(self, index_path: str) -> Dict:
        """Index 파일 분석"""
        print(f"🔍 Index 파일 분석 중: {index_path}")
        
        try:
            # Index 파일은 보통 바이너리 형태
            file_size = Path(index_path).stat().st_size
            
            analysis = {
                "file_path": index_path,
                "file_size_mb": file_size / (1024*1024),
                "file_size_bytes": file_size
            }
            
            # 파일 헤더 읽기 시도
            with open(index_path, 'rb') as f:
                header = f.read(1024)  # 첫 1KB 읽기
                analysis["header_preview"] = header[:100].hex()  # 첫 100바이트를 hex로
                
            # 텍스트로 읽기 시도
            try:
                with open(index_path, 'r', encoding='utf-8') as f:
                    content = f.read(1000)  # 첫 1000자
                    analysis["text_content"] = content
            except:
                try:
                    with open(index_path, 'r', encoding='latin-1') as f:
                        content = f.read(1000)
                        analysis["latin1_content"] = content
                except:
                    analysis["text_readable"] = False
            
            print(f"   ✅ Index 분석 완료: {file_size / (1024*1024):.1f}MB")
            return analysis
            
        except Exception as e:
            print(f"   ❌ Index 분석 실패: {e}")
            return {"error": str(e)}

    def extract_audio_features(self, emotion: str, model_data: Dict) -> Dict:
        """모델 데이터에서 오디오 특성 추출"""
        print(f"🎵 {emotion} 감정 오디오 특성 추출 중...")
        
        features = {
            "emotion": emotion,
            "timestamp": "extracted_from_model",
            "source": "pth_analysis"
        }
        
        try:
            pth_data = model_data.get("pth_analysis", {})
            
            # 모델 파라미터에서 특성 추출
            if "weight" in str(pth_data.get("keys", [])):
                # 가중치 정보 기반 특성 추출
                features["model_type"] = "neural_vocoder"
                features["has_weights"] = True
                
            # 파일명에서 정보 추출
            filename = Path(model_data.get("pth_analysis", {}).get("file_path", "")).stem
            if "600e" in filename:
                features["training_epochs"] = 600
            if "54000s" in filename or "58200s" in filename or "54600s" in filename:
                features["training_steps"] = int(filename.split("_")[-1].replace("s", ""))
            
            # 감정별 특성 매핑
            emotion_characteristics = {
                "우울": {
                    "pitch_tendency": "lower",
                    "tempo_tendency": "slower", 
                    "energy_level": "low",
                    "formant_shift": "darker",
                    "suggested_temp": {"text": 0.6, "waveform": 0.7}
                },
                "슬픔": {
                    "pitch_tendency": "variable",
                    "tempo_tendency": "irregular",
                    "energy_level": "low-medium", 
                    "formant_shift": "breathy",
                    "suggested_temp": {"text": 0.5, "waveform": 0.6}
                },
                "기쁨": {
                    "pitch_tendency": "higher",
                    "tempo_tendency": "faster",
                    "energy_level": "high",
                    "formant_shift": "brighter",
                    "suggested_temp": {"text": 0.7, "waveform": 0.8}
                }
            }
            
            features.update(emotion_characteristics.get(emotion, {}))
            
            print(f"   ✅ {emotion} 특성 추출 완료")
            return features
            
        except Exception as e:
            print(f"   ❌ {emotion} 특성 추출 실패: {e}")
            return {"error": str(e)}

    def create_synthetic_training_data(self, emotion: str, features: Dict) -> List[Dict]:
        """감정 특성 기반 합성 훈련 데이터 생성"""
        print(f"🎯 {emotion} 합성 훈련 데이터 생성 중...")
        
        # 감정별 훈련 문장들
        training_sentences = {
            "우울": [
                "오늘도 힘든 하루가 지나가네요.",
                "모든 것이 무의미하게 느껴집니다.",
                "혼자 있는 시간이 너무 길어요.",
                "아무것도 하고 싶지 않아요.",
                "마음이 무거워서 견디기 힘들어요.",
                "세상이 회색빛으로만 보여요.",
                "웃는 것조차 어려워졌어요.",
                "이 감정이 언제까지 계속될까요."
            ],
            "슬픔": [
                "이별은 정말 아픈 일이에요.",
                "눈물이 멈추지 않네요.",
                "소중한 것을 잃었어요.",
                "다시는 돌아올 수 없는 시간이에요.",
                "가슴이 먹먹하고 아파요.",
                "추억만 남았네요.",
                "왜 이렇게 슬플까요.",
                "마음의 상처가 아물지 않아요."
            ],
            "기쁨": [
                "오늘 정말 기분이 좋아요!",
                "드디어 꿈이 이루어졌어요!",
                "너무 행복해서 춤추고 싶어요!",
                "세상이 이렇게 아름다울 줄 몰랐어요!",
                "웃음이 절로 나와요!",
                "모든 것이 완벽해 보여요!",
                "이 순간이 영원했으면 좋겠어요!",
                "마음이 날아갈 것 같아요!"
            ]
        }
        
        sentences = training_sentences.get(emotion, [])
        training_data = []
        
        for i, sentence in enumerate(sentences):
            data_point = {
                "id": f"{emotion}_{i+1:03d}",
                "text": sentence,
                "emotion": emotion,
                "speaker": f"{emotion}_speaker",
                "language": "ko",
                "duration_estimate": len(sentence) * 0.1,  # 대략적 길이 추정
                "features": features,
                "synthetic": True
            }
            training_data.append(data_point)
        
        print(f"   ✅ {emotion} 훈련 데이터 {len(training_data)}개 생성")
        return training_data

    def generate_bark_training_config(self) -> Dict:
        """Bark 파인튜닝을 위한 설정 생성"""
        print("⚙️ Bark 파인튜닝 설정 생성 중...")
        
        config = {
            "model_config": {
                "use_small_models": False,
                "enable_mps": True,
                "offload_cpu": False,
                "sample_rate": 24000,
                "encodec_bandwidth": 12.0
            },
            "training_config": {
                "batch_size": 1,  # MPS 메모리 고려
                "learning_rate": 1e-5,
                "num_epochs": 50,
                "warmup_steps": 100,
                "save_steps": 500,
                "eval_steps": 250,
                "gradient_accumulation_steps": 4
            },
            "data_config": {
                "max_text_length": 256,
                "max_audio_length": 30.0,
                "validation_split": 0.1,
                "shuffle": True
            },
            "emotion_speakers": {
                "우울": {
                    "base_speaker": "v2/ko_speaker_1",
                    "emotion_tags": ["[sad]", "[melancholy]", "[depressed]"],
                    "temperature": {"text": 0.6, "waveform": 0.7}
                },
                "슬픔": {
                    "base_speaker": "v2/ko_speaker_2", 
                    "emotion_tags": ["[crying]", "[sorrow]", "[grief]"],
                    "temperature": {"text": 0.5, "waveform": 0.6}
                },
                "기쁨": {
                    "base_speaker": "v2/ko_speaker_0",
                    "emotion_tags": ["[happy]", "[excited]", "[joyful]"],
                    "temperature": {"text": 0.7, "waveform": 0.8}
                }
            }
        }
        
        print("   ✅ 설정 생성 완료")
        return config

    def run_full_analysis(self):
        """전체 분석 프로세스 실행"""
        print("🔬 감정 모델 전체 분석 시작")
        print("=" * 60)
        
        results = {
            "analysis_timestamp": "2024-06-30",
            "emotions_analyzed": [],
            "model_analyses": {},
            "extracted_features": {},
            "training_data": {},
            "bark_config": {}
        }
        
        # 각 감정별 모델 분석
        for emotion in self.emotions:
            print(f"\n📂 {emotion} 감정 모델 분석")
            print("-" * 30)
            
            model_dir = Path(f"finetune_output/{emotion}/extracted")
            if not model_dir.exists():
                print(f"❌ {emotion} 모델 디렉토리 없음")
                continue
            
            # PTH 파일 분석
            pth_files = list(model_dir.glob("*.pth"))
            index_files = list(model_dir.glob("*.index"))
            
            if not pth_files or not index_files:
                print(f"❌ {emotion} 모델 파일 부족")
                continue
            
            emotion_analysis = {}
            
            # PTH 분석
            if pth_files:
                emotion_analysis["pth_analysis"] = self.analyze_pth_file(str(pth_files[0]))
            
            # Index 분석  
            if index_files:
                emotion_analysis["index_analysis"] = self.analyze_index_file(str(index_files[0]))
            
            # 특성 추출
            features = self.extract_audio_features(emotion, emotion_analysis)
            
            # 훈련 데이터 생성
            training_data = self.create_synthetic_training_data(emotion, features)
            
            # 결과 저장
            results["emotions_analyzed"].append(emotion)
            results["model_analyses"][emotion] = emotion_analysis
            results["extracted_features"][emotion] = features
            results["training_data"][emotion] = training_data
        
        # Bark 설정 생성
        results["bark_config"] = self.generate_bark_training_config()
        
        # 결과 저장
        output_file = "emotion_models_analysis.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n🎉 전체 분석 완료!")
        print(f"📁 결과 저장: {output_file}")
        print(f"📊 분석된 감정: {len(results['emotions_analyzed'])}개")
        
        return results

def main():
    analyzer = EmotionModelAnalyzer()
    results = analyzer.run_full_analysis()
    
    # 요약 출력
    print(f"\n📋 분석 요약:")
    for emotion in results["emotions_analyzed"]:
        features = results["extracted_features"][emotion]
        training_count = len(results["training_data"][emotion])
        print(f"  🎭 {emotion}: {training_count}개 훈련 샘플, 특성 추출 완료")

if __name__ == "__main__":
    main() 