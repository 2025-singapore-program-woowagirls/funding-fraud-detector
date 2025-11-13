# 1. 펀딩피 차익거래 탐지 (Funding Fee Arbitrage Detection)

## 개요
펀딩 시각 전후 단기 포지션 집중을 통해 펀딩비를 노리는 거래 패턴을 탐지합니다.

## 핵심 특징
- 펀딩 시각(0,4,8,12,16,20시) ±30분 거래 집중
- 짧은 포지션 보유시간 (중앙값 10.2분)
- 수익의 대부분이 펀딩피에서 발생

## 코드 파일

### analyze_funding_distribution.py
- 사용자별 펀딩피 분포 분석
- 이상치 탐지 (2σ, 3σ 기준)
- 시각화: 히스토그램, 박스플롯

### analyze_funding_arbitrage.py
- 포지션 보유시간 계산
- 펀딩 시각 거래 집중도 분석
- 펀딩비 수익 비중 계산
- FundingScore 산출

### funding_fee_rule_filter.py
- 규칙 기반 펀딩피 어뷰저 필터링
- 임계값 기반 탐지

### visualize_features_detail.py
- 펀딩피 관련 피처 상세 시각화
- Pattern 1 상세 분포 차트 생성

## 주요 피처
1. **funding_fee_abs**: 펀딩피 절댓값
2. **mean_holding_minutes**: 평균 포지션 보유시간
3. **funding_timing_ratio**: 펀딩 시각(±30분) 거래 비율
4. **funding_profit_ratio**: 펀딩피 / 총수익 비율

## 출력 결과
결과는 `output/1_funding_arbitrage/` 폴더에 저장됩니다.
