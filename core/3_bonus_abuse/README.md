# 3. 보너스 악용 탐지 (Bonus Abuse Detection)

## 개요
동일 IP에서 다수 계정을 생성하여 보너스를 반복 수령하는 패턴을 탐지합니다.

## 핵심 특징
- 동일 IP에서 여러 계정이 보너스 수령
- 보너스 수령 후 거래 활동 미미
- 높은 Reward-to-Volume 비율

## 코드 파일

### analyze_bonus_abuse.py
- 보너스 수령 패턴 분석
- IP 공유 계정 탐지
- 보너스 수령 후 활동 분석
- BonusScore 산출

## 주요 피처
1. **total_reward**: 총 보너스 수령액
2. **shared_ip**: IP 공유 여부
3. **has_trades**: 거래 활동 여부
4. **reward_to_volume_ratio (RVR)**: 보너스/거래량 비율

## 가중치 공식
```
BonusScore = 0.4 * duplicate_reward_score
           + 0.3 * inactive_ratio_score
           + 0.3 * rvr_score
```

## 출력 결과
결과는 `output/3_bonus_abuse/` 폴더에 저장됩니다.
