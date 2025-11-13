# 2. 조직적 거래 탐지 (Organized Trading Detection)

## 개요
다계정이 동일 시간대, 심볼, 가격에서 거래하는 조직적 거래 패턴을 탐지합니다.

## 핵심 특징
- 동일 IP에서 여러 계정 운영
- 동시간대 동일 심볼 거래
- 유사한 가격대 거래 집중
- 높은 레버리지 사용

## 코드 파일

### analyze_organized_trading.py
- IP 공유 패턴 분석
- 시간대별 동시 거래 탐지
- 가격 유사도 계산
- OrganizedScore 산출

## 주요 피처
1. **ip_shared_ratio**: IP 공유 비율
2. **concurrent_trading_ratio**: 동시 거래 비율
3. **price_similarity**: 가격 유사도
4. **mean_leverage**: 평균 레버리지

## 가중치 공식
```
OrganizedScore = 0.35 * ip_score
               + 0.30 * concurrent_score
               + 0.20 * price_score
               + 0.15 * leverage_score
```

## 출력 결과
결과는 `output/2_organized_trading/` 폴더에 저장됩니다.
