# 4. 통합 리스크 스코어링 시스템 (Integrated Risk Scoring)

## 개요
3가지 이상 거래 패턴을 종합하여 최종 리스크 점수를 산출하고, 고급 퀀트 피처를 추가 분석합니다.

## 핵심 특징
- 3가지 패턴 점수 통합
- 퀀트 기반 고급 지표 분석
- 다중 패턴 탐지
- 최종 피처 정의 및 시각화

## 코드 파일

### final_integrated_system.py
- 3가지 패턴 점수 통합
- 최종 리스크 점수 산출
- 고위험 계정 식별
- 종합 리포트 생성

### advanced_feature_engineering.py
- Sharpe Ratio 계산
- Win Rate, Profit Factor
- Time Entropy
- Position Concentration (HHI)
- Kelly Criterion Deviation

### generate_final_features.py
- 패턴별 핵심 피처 추출
- 피처 정의 문서 생성 (FEATURES_DEFINITION.md)
- 피처 요약 시각화

### eda_analysis.ipynb
- 전체 데이터 탐색적 분석
- Jupyter Notebook 기반 인터랙티브 분석

## 최종 통합 공식
```
FinalRiskScore = 0.30 * FundingScore
               + 0.35 * OrganizedScore
               + 0.20 * BonusScore
               + 0.15 * QuantScore
```

## 주요 출력물
- **FEATURES_DEFINITION.md**: 전체 피처 정의 및 설명
- **pattern{1,2,3}_detailed_distribution.png**: 패턴별 상세 분포 시각화
- **final_integrated_risk_scores.csv**: 최종 통합 리스크 점수
- **multi_pattern_accounts.csv**: 복수 패턴 탐지 계정
- **critical_high_risk_accounts.csv**: 최고위험 계정

## 출력 결과
결과는 `output/4_integrated/` 폴더에 저장됩니다.
