#!/bin/bash

# 🎭 감정별 Bark TTS 파인튜닝 실행 스크립트

echo "🎭 감정별 Bark TTS 파인튜닝 시작"
echo "=================================="

# 가상환경 확인
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  가상환경이 활성화되지 않았습니다."
    echo "다음 명령어로 가상환경을 만들고 활성화하세요:"
    echo "python -m venv venv_emotion"
    echo "source venv_emotion/bin/activate"
    echo "pip install -r requirements_emotion.txt"
    exit 1
fi

# 의존성 확인
echo "📦 의존성 확인 중..."
python -c "import torch, bark, librosa, soundfile" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ 필요한 패키지가 설치되지 않았습니다."
    echo "다음 명령어로 설치하세요: pip install -r requirements_emotion.txt"
    exit 1
fi

# MPS 지원 확인
echo "🔧 MPS 지원 확인 중..."
python -c "import torch; print('✅ MPS 사용 가능' if torch.backends.mps.is_available() else '⚠️ MPS 사용 불가 - CPU 모드 사용')"

# 데이터 폴더 확인
if [ ! -d "일어" ]; then
    echo "❌ '일어' 폴더를 찾을 수 없습니다."
    echo "감정별 데이터가 포함된 폴더가 있는지 확인하세요."
    exit 1
fi

echo "📁 감정 데이터 확인:"
for emotion in "우울" "슬픔" "기쁨"; do
    if [ -d "일어/$emotion" ]; then
        echo "  ✅ $emotion 데이터 발견"
    else
        echo "  ❌ $emotion 데이터 없음"
    fi
done

# 사용자 선택
echo ""
echo "어떤 감정을 파인튜닝하시겠습니까?"
echo "1) 모든 감정 (우울, 슬픔, 기쁨)"
echo "2) 우울만"
echo "3) 슬픔만" 
echo "4) 기쁨만"
echo "5) 테스트만 실행"
echo "6) 종료"

read -p "선택하세요 (1-6): " choice

case $choice in
    1)
        echo "🚀 모든 감정 파인튜닝 시작..."
        python finetune_bark_emotions.py --emotion all
        ;;
    2)
        echo "🚀 우울 감정 파인튜닝 시작..."
        python finetune_bark_emotions.py --emotion 우울
        ;;
    3)
        echo "🚀 슬픔 감정 파인튜닝 시작..."
        python finetune_bark_emotions.py --emotion 슬픔
        ;;
    4)
        echo "🚀 기쁨 감정 파인튜닝 시작..."
        python finetune_bark_emotions.py --emotion 기쁨
        ;;
    5)
        echo "🧪 테스트 실행..."
        python test_emotion_voices.py
        ;;
    6)
        echo "👋 종료합니다."
        exit 0
        ;;
    *)
        echo "❌ 잘못된 선택입니다."
        exit 1
        ;;
esac

# 완료 후 테스트 실행 여부 확인
if [ $choice -ne 5 ]; then
    echo ""
    read -p "🧪 파인튜닝이 완료되었습니다. 테스트를 실행하시겠습니까? (y/n): " test_choice
    if [[ $test_choice == "y" || $test_choice == "Y" ]]; then
        echo "🧪 테스트 실행 중..."
        python test_emotion_voices.py
    fi
fi

echo ""
echo "🎉 작업이 완료되었습니다!"
echo "📁 결과물 확인:"
echo "  - 파인튜닝 결과: ./finetune_output/"
echo "  - 테스트 결과: ./emotion_test_results/" 