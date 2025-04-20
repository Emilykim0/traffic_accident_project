# 🚦 교통사고 예측 머신러닝 프로젝트

> 교통사고 데이터를 기반으로 부상 여부를 예측하는 머신러닝 모델 구축  
> 예측 결과를 CLI 기반 인터페이스로 사용자에게 안내

---

## 📁 폴더 구조

```
traffic_accident_project/
├── data/
│   └── traffic_accidents.csv                  # 원본 데이터
├── models/
│   ├── lightgbm_tuned.pkl                     # 최종 예측 모델 (LightGBM)
│   ├── label_encoder_most_severe_injury.pkl  # 부상도 인코딩용 LabelEncoder
│   └── feature_names.pkl                      # 훈련된 모델 기준 feature 순서
├── ml_utils.py                                # 모델 평가 및 CLI 예측 함수
└── accident_crush.ipynb                       # 전체 분석 및 학습 코드
```

---

## 🧠 프로젝트 개요

- **목표:** 사고 관련 정보로부터 **사고 심각도 예측**
- **데이터:** Kaggle - [Traffic Accidents Dataset](https://www.kaggle.com/datasets/oktayrdeki/traffic-accidents)
- **모델:** LightGBM (튜닝된 버전 사용)
- **입력 방식:** CLI로 사용자 입력을 받아 예측 결과를 제공

---

## 🔧 실행 방법

```bash
python ml_utils.py
```
실행 시, CLI 환경에서 사고 조건을 선택하여 부상 사고 발생 여부를 예측할 수 있습니다.

---

## ✅ 주요 기능

- 사고 조건 입력 시 실시간 예측
- LightGBM 기반 분류 모델 활용
- 부상 사고 확률 및 안전/위험 메시지 출력
- LabelEncoder 및 Feature 순서 복원 포함

---

## 📌 프로젝트 회고

- 하이퍼파라미터 튜닝을 통해 일반화 성능 향상
- Stacking, VotingClassifier 등 앙상블도 실험했으나 성능 차이 미미 → LightGBM 단일 모델 채택
- CLI 구현을 통해 머신러닝 모델을 실제 서비스 형태로 응용해본 경험이 유익했음
- 한정된 시간 내 데이터 전처리 → 모델 개발 → 서비스 구현까지 전체 사이클을 경험

---

## 👩‍💻 제작자

- **이름:** Emily  
- **역할:** 전처리, 모델링, CLI 구현, 튜닝, 리포트 작성

---

## 🔗 프로젝트 링크
🧾 [프로젝트 상세 노션 페이지](https://yeonghyekim.notion.site/1b9e2859370c80ce97e4cdd5dceeaf74?pvs=4)  