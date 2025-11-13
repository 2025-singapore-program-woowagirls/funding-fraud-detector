import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 120)
print("이상거래 패턴별 핵심 피처 추출 및 정리")
print("=" * 120)

# ================================================================================
# 기존 분석 결과 로드
# ================================================================================
print("\n[1] 기존 분석 결과 로드")

funding_arb_df = pd.read_csv('output/funding_analysis/funding_arbitrage_scores_all.csv')
organized_df = pd.read_csv('output/organized_trading/organized_scores_all.csv')
bonus_df = pd.read_csv('output/bonus_abuse/bonus_abuse_scores_all.csv')
quant_df = pd.read_csv('output/funding_analysis/quant_features_all.csv')
funding_raw = pd.read_csv('data/Funding.csv')
trade_raw = pd.read_csv('data/Trade.csv')

print(f"✓ 데이터 로드 완료")

# ================================================================================
# Pattern 1: 펀딩피 차익거래 (Funding Fee Arbitrage) 피처
# ================================================================================
print("\n" + "=" * 120)
print("Pattern 1: 펀딩피 차익거래 (Funding Fee Arbitrage) - 핵심 피처 추출")
print("=" * 120)

# 1.1 펀딩피 절댓값 분포
funding_raw['funding_fee_abs'] = funding_raw['funding_fee'].abs()

funding_features_p1 = []

print("\n[Feature 1-1] 펀딩피 절댓값 (funding_fee_abs)")
print("-" * 120)

# 전체 분포
print(f"  전체 분포:")
print(f"    - 평균: ${funding_raw['funding_fee_abs'].mean():.4f}")
print(f"    - 중앙값: ${funding_raw['funding_fee_abs'].median():.4f} ← 0에 가까움 = 정상")
print(f"    - 95th percentile: ${funding_raw['funding_fee_abs'].quantile(0.95):.4f}")
print(f"    - 99th percentile: ${funding_raw['funding_fee_abs'].quantile(0.99):.4f}")

# 0 근처 vs 큰 값
near_zero = len(funding_raw[funding_raw['funding_fee_abs'] < 1])
large_values = len(funding_raw[funding_raw['funding_fee_abs'] > 10])
print(f"\n  분포 특성:")
print(f"    - 펀딩피 < $1 (0 근처): {near_zero:,}건 ({near_zero/len(funding_raw)*100:.1f}%) ← 정상 거래")
print(f"    - 펀딩피 > $10 (큰 값): {large_values:,}건 ({large_values/len(funding_raw)*100:.1f}%) ← 주목!")

print(f"\n  ✅ 탐지 기준:")
print(f"    - 정상: 평균 펀딩피 절댓값 < $5")
print(f"    - 의심: $5 ~ $30")
print(f"    - 고위험: > $30 (95th percentile)")

funding_features_p1.append({
    'feature_name': 'funding_fee_abs',
    'description': '펀딩피 절댓값',
    'normal_range': '< $5',
    'suspicious_range': '$5 ~ $30',
    'high_risk_range': '> $30',
    'data_median': funding_raw['funding_fee_abs'].median(),
    'data_95th': funding_raw['funding_fee_abs'].quantile(0.95),
    'interpretation': '0 근처가 정상, 큰 값은 펀딩피를 노린 거래'
})

# 1.2 포지션 보유시간
print(f"\n[Feature 1-2] 포지션 보유시간 (holding_minutes)")
print("-" * 120)

holding_times = funding_arb_df['mean_holding_minutes'].dropna()
print(f"  전체 분포:")
print(f"    - 평균: {holding_times.mean():.2f}분")
print(f"    - 중앙값: {holding_times.median():.2f}분")
print(f"    - 5th percentile: {holding_times.quantile(0.05):.2f}분")
print(f"    - 10th percentile: {holding_times.quantile(0.10):.2f}분")

short_holding = len(funding_arb_df[funding_arb_df['mean_holding_minutes'] < 30])
print(f"\n  분포 특성:")
print(f"    - < 30분: {short_holding}개 계정 ({short_holding/len(funding_arb_df)*100:.1f}%) ← 펀딩 차익 의심")
print(f"    - < 10분: {len(funding_arb_df[funding_arb_df['mean_holding_minutes'] < 10])}개 계정")

print(f"\n  ✅ 탐지 기준:")
print(f"    - 정상: 평균 보유시간 > 60분 (1시간)")
print(f"    - 의심: 30~60분")
print(f"    - 고위험: < 30분 (펀딩만 받고 빠른 청산)")

funding_features_p1.append({
    'feature_name': 'mean_holding_minutes',
    'description': '평균 포지션 보유시간',
    'normal_range': '> 60분',
    'suspicious_range': '30~60분',
    'high_risk_range': '< 30분',
    'data_median': holding_times.median(),
    'data_10th': holding_times.quantile(0.10),
    'interpretation': '짧을수록 펀딩피만 노리는 패턴'
})

# 1.3 펀딩 시각 거래 집중도
print(f"\n[Feature 1-3] 펀딩 시각 거래 집중도 (funding_timing_ratio)")
print("-" * 120)

timing_ratio = funding_arb_df['funding_timing_ratio'].dropna()
print(f"  전체 분포:")
print(f"    - 평균: {timing_ratio.mean()*100:.2f}%")
print(f"    - 중앙값: {timing_ratio.median()*100:.2f}%")
print(f"    - 95th percentile: {timing_ratio.quantile(0.95)*100:.2f}%")

high_concentration = len(funding_arb_df[funding_arb_df['funding_timing_ratio'] > 0.5])
print(f"\n  분포 특성:")
print(f"    - > 50% 집중: {high_concentration}개 계정 ({high_concentration/len(funding_arb_df)*100:.1f}%) ← 펀딩 시각만 노림")

print(f"\n  ✅ 탐지 기준:")
print(f"    - 정상: < 30% (우연히 겹치는 수준)")
print(f"    - 의심: 30~50%")
print(f"    - 고위험: > 50% (펀딩 시각(0,4,8,12,16,20시)에만 거래)")

funding_features_p1.append({
    'feature_name': 'funding_timing_ratio',
    'description': '펀딩 시각(±30분) 거래 비율',
    'normal_range': '< 30%',
    'suspicious_range': '30~50%',
    'high_risk_range': '> 50%',
    'data_mean': timing_ratio.mean(),
    'data_95th': timing_ratio.quantile(0.95),
    'interpretation': '펀딩 시각에만 집중하면 차익거래 의심'
})

# 1.4 펀딩피 수익 비중
print(f"\n[Feature 1-4] 펀딩피 수익 비중 (funding_profit_ratio)")
print("-" * 120)

profit_ratio = funding_arb_df['funding_profit_ratio'].dropna()
valid_profit_ratio = profit_ratio[(profit_ratio >= 0) & (profit_ratio <= 1)]

print(f"  전체 분포 (0~100% 범위):")
print(f"    - 평균: {valid_profit_ratio.mean()*100:.2f}%")
print(f"    - 중앙값: {valid_profit_ratio.median()*100:.2f}%")
print(f"    - 95th percentile: {valid_profit_ratio.quantile(0.95)*100:.2f}%")

high_funding_profit = len(funding_arb_df[funding_arb_df['funding_profit_ratio'] > 0.7])
print(f"\n  분포 특성:")
print(f"    - 펀딩피 > 70% 수익: {high_funding_profit}개 계정 ← 거래 수익보다 펀딩피가 주 수익원")

print(f"\n  ✅ 탐지 기준:")
print(f"    - 정상: < 30% (주 수익은 거래 차익)")
print(f"    - 의심: 30~70%")
print(f"    - 고위험: > 70% (펀딩피가 수익의 대부분)")

funding_features_p1.append({
    'feature_name': 'funding_profit_ratio',
    'description': '펀딩피 / 총수익 비율',
    'normal_range': '< 30%',
    'suspicious_range': '30~70%',
    'high_risk_range': '> 70%',
    'data_median': valid_profit_ratio.median(),
    'data_95th': valid_profit_ratio.quantile(0.95),
    'interpretation': '비중이 높을수록 펀딩피 의존형 거래'
})

# DataFrame 저장
funding_p1_df = pd.DataFrame(funding_features_p1)
funding_p1_df.to_csv('output/final_features/pattern1_funding_arbitrage_features.csv', index=False)
print(f"\n✓ Pattern 1 피처 저장: output/final_features/pattern1_funding_arbitrage_features.csv")

# ================================================================================
# Pattern 2: 조직적 거래 (Organized Trading) 피처
# ================================================================================
print("\n" + "=" * 120)
print("Pattern 2: 조직적 거래 (Organized Trading) - 핵심 피처 추출")
print("=" * 120)

organized_features_p2 = []

# 2.1 IP 공유 비율
print(f"\n[Feature 2-1] IP 공유 비율 (ip_shared_ratio)")
print("-" * 120)

ip_shared = organized_df['ip_shared_ratio'].dropna()
print(f"  전체 분포:")
print(f"    - 평균: {ip_shared.mean()*100:.2f}%")
print(f"    - 중앙값: {ip_shared.median()*100:.2f}% ← 대부분 0% (정상)")
print(f"    - 95th percentile: {ip_shared.quantile(0.95)*100:.2f}%")

shared_accounts = len(organized_df[organized_df['ip_shared_ratio'] > 0])
high_shared = len(organized_df[organized_df['ip_shared_ratio'] > 0.5])

print(f"\n  분포 특성:")
print(f"    - IP 공유 있음: {shared_accounts}개 계정 ({shared_accounts/len(organized_df)*100:.1f}%)")
print(f"    - > 50% 공유: {high_shared}개 계정 ← 다계정 의심")

print(f"\n  ✅ 탐지 기준:")
print(f"    - 정상: 0% (단일 IP 또는 다른 IP)")
print(f"    - 의심: > 30% (여러 계정이 같은 IP 사용)")
print(f"    - 고위험: > 50% (명백한 다계정 운영)")

organized_features_p2.append({
    'feature_name': 'ip_shared_ratio',
    'description': 'IP 공유 비율 (다른 계정과 IP 중복)',
    'normal_range': '0%',
    'suspicious_range': '> 30%',
    'high_risk_range': '> 50%',
    'data_mean': ip_shared.mean(),
    'data_median': ip_shared.median(),
    'interpretation': '공유 비율 높으면 다계정 운영 가능성'
})

# 2.2 동시 거래 비율
print(f"\n[Feature 2-2] 동시 거래 비율 (concurrent_trading_ratio)")
print("-" * 120)

concurrent = organized_df['concurrent_trading_ratio'].dropna()
print(f"  전체 분포:")
print(f"    - 평균: {concurrent.mean()*100:.2f}%")
print(f"    - 중앙값: {concurrent.median()*100:.2f}%")
print(f"    - 95th percentile: {concurrent.quantile(0.95)*100:.2f}%")

high_concurrent = len(organized_df[organized_df['concurrent_trading_ratio'] > 0.5])
very_high = len(organized_df[organized_df['concurrent_trading_ratio'] > 0.7])

print(f"\n  분포 특성:")
print(f"    - > 50% 동시 거래: {high_concurrent}개 계정 ({high_concurrent/len(organized_df)*100:.1f}%)")
print(f"    - > 70% 동시 거래: {very_high}개 계정 ← 조직적 거래 확실")

print(f"\n  ✅ 탐지 기준:")
print(f"    - 정상: < 30% (우연히 겹침)")
print(f"    - 의심: 30~50%")
print(f"    - 고위험: > 50% (같은 시간(분)에 같은 심볼 거래)")

organized_features_p2.append({
    'feature_name': 'concurrent_trading_ratio',
    'description': '동시 거래 비율 (같은 시간·심볼 거래)',
    'normal_range': '< 30%',
    'suspicious_range': '30~50%',
    'high_risk_range': '> 50%',
    'data_mean': concurrent.mean(),
    'data_95th': concurrent.quantile(0.95),
    'interpretation': '비율 높으면 조직적으로 거래하는 패턴'
})

# 2.3 가격 유사도
print(f"\n[Feature 2-3] 가격 유사도 (price_similarity_ratio)")
print("-" * 120)

price_sim = organized_df['price_similarity_ratio'].dropna()
price_sim_valid = price_sim[price_sim > 0]

if len(price_sim_valid) > 0:
    print(f"  전체 분포 (유사 거래 있는 계정만):")
    print(f"    - 평균: {price_sim_valid.mean()*100:.2f}%")
    print(f"    - 중앙값: {price_sim_valid.median()*100:.2f}%")

    high_similarity = len(organized_df[organized_df['price_similarity_ratio'] > 0.8])
    print(f"\n  분포 특성:")
    print(f"    - > 80% 유사: {high_similarity}개 계정 ← 거의 동일 가격 거래")

    print(f"\n  ✅ 탐지 기준:")
    print(f"    - 정상: < 60% (다양한 가격대)")
    print(f"    - 의심: 60~80%")
    print(f"    - 고위험: > 80% (동시 거래 중 가격도 거의 동일)")

    organized_features_p2.append({
        'feature_name': 'price_similarity_ratio',
        'description': '동시 거래 중 가격 유사 비율 (CV<1%)',
        'normal_range': '< 60%',
        'suspicious_range': '60~80%',
        'high_risk_range': '> 80%',
        'data_mean': price_sim_valid.mean(),
        'data_median': price_sim_valid.median(),
        'interpretation': '동일 가격대 거래는 조직적 패턴'
    })

# 2.4 평균 레버리지
print(f"\n[Feature 2-4] 평균 레버리지 (mean_leverage)")
print("-" * 120)

leverage = organized_df['mean_leverage'].dropna()
print(f"  전체 분포:")
print(f"    - 평균: {leverage.mean():.2f}x")
print(f"    - 중앙값: {leverage.median():.2f}x")
print(f"    - 95th percentile: {leverage.quantile(0.95):.2f}x")

high_leverage = len(organized_df[organized_df['mean_leverage'] > 30])
very_high_lev = len(organized_df[organized_df['mean_leverage'] > 50])

print(f"\n  분포 특성:")
print(f"    - > 30x: {high_leverage}개 계정 ({high_leverage/len(organized_df)*100:.1f}%)")
print(f"    - > 50x: {very_high_lev}개 계정")

print(f"\n  ✅ 탐지 기준:")
print(f"    - 정상: < 30x")
print(f"    - 의심: 30~50x")
print(f"    - 고위험: > 50x (극단적 고레버리지)")

organized_features_p2.append({
    'feature_name': 'mean_leverage',
    'description': '평균 레버리지',
    'normal_range': '< 30x',
    'suspicious_range': '30~50x',
    'high_risk_range': '> 50x',
    'data_mean': leverage.mean(),
    'data_median': leverage.median(),
    'interpretation': '고레버리지는 공격적 거래 전략'
})

# DataFrame 저장
organized_p2_df = pd.DataFrame(organized_features_p2)
organized_p2_df.to_csv('output/final_features/pattern2_organized_trading_features.csv', index=False)
print(f"\n✓ Pattern 2 피처 저장: output/final_features/pattern2_organized_trading_features.csv")

# ================================================================================
# Pattern 3: 보너스 악용 (Bonus Abuse) 피처
# ================================================================================
print("\n" + "=" * 120)
print("Pattern 3: 보너스 악용 (Bonus Abuse) - 핵심 피처 추출")
print("=" * 120)

bonus_features_p3 = []

# 3.1 총 보너스 금액
print(f"\n[Feature 3-1] 총 보너스 금액 (total_reward)")
print("-" * 120)

rewards = bonus_df['total_reward'].dropna()
print(f"  전체 분포:")
print(f"    - 평균: ${rewards.mean():.2f}")
print(f"    - 중앙값: ${rewards.median():.2f}")
print(f"    - 95th percentile: ${rewards.quantile(0.95):.2f}")

moderate_reward = len(bonus_df[(bonus_df['total_reward'] > 50) & (bonus_df['total_reward'] <= 100)])
high_reward = len(bonus_df[bonus_df['total_reward'] > 100])

print(f"\n  분포 특성:")
print(f"    - $50~$100: {moderate_reward}개 계정 ({moderate_reward/len(bonus_df)*100:.1f}%)")
print(f"    - > $100: {high_reward}개 계정 ({high_reward/len(bonus_df)*100:.1f}%) ← 다계정 의심")

print(f"\n  ✅ 탐지 기준:")
print(f"    - 정상: < $50 (1~2회 정상 보너스)")
print(f"    - 의심: $50~$100 (다수 수령)")
print(f"    - 고위험: > $100 (명백한 다계정 악용)")

bonus_features_p3.append({
    'feature_name': 'total_reward',
    'description': '총 보너스 수령액',
    'normal_range': '< $50',
    'suspicious_range': '$50~$100',
    'high_risk_range': '> $100',
    'data_median': rewards.median(),
    'data_95th': rewards.quantile(0.95),
    'interpretation': '금액 높으면 다계정으로 반복 수령'
})

# 3.2 공유 IP 사용 여부
print(f"\n[Feature 3-2] 공유 IP 사용 (shared_ip)")
print("-" * 120)

shared_ip_count = bonus_df['shared_ip'].sum()
print(f"  분포:")
print(f"    - 공유 IP 사용: {shared_ip_count}개 계정 ({shared_ip_count/len(bonus_df)*100:.1f}%)")
print(f"    - 단일 IP 사용: {len(bonus_df) - shared_ip_count}개 계정")

print(f"\n  ✅ 탐지 기준:")
print(f"    - 정상: False (고유 IP)")
print(f"    - 고위험: True (다른 보너스 수령 계정과 IP 공유)")

bonus_features_p3.append({
    'feature_name': 'shared_ip',
    'description': '보너스 수령 시 IP 공유 여부',
    'normal_range': 'False',
    'suspicious_range': '-',
    'high_risk_range': 'True',
    'data_count_shared': shared_ip_count,
    'data_count_unique': len(bonus_df) - shared_ip_count,
    'interpretation': 'IP 공유는 다계정 생성의 강력한 신호'
})

# 3.3 거래 활동 여부
print(f"\n[Feature 3-3] 거래 활동 여부 (has_trades)")
print("-" * 120)

has_trades = bonus_df['has_trades'].sum()
no_trades = len(bonus_df) - has_trades

print(f"  분포:")
print(f"    - 거래 있음: {has_trades}개 계정 ({has_trades/len(bonus_df)*100:.1f}%)")
print(f"    - 거래 없음: {no_trades}개 계정 ({no_trades/len(bonus_df)*100:.1f}%)")

print(f"\n  ✅ 탐지 기준:")
print(f"    - 정상: True (보너스 후 활발한 거래)")
print(f"    - 의심: 거래 < 10회")
print(f"    - 고위험: False (보너스만 받고 비활성)")

bonus_features_p3.append({
    'feature_name': 'has_trades',
    'description': '보너스 수령 후 거래 활동 여부',
    'normal_range': 'True (활발한 거래)',
    'suspicious_range': '거래 < 10회',
    'high_risk_range': 'False (비활성)',
    'data_count_active': has_trades,
    'data_count_inactive': no_trades,
    'interpretation': '보너스만 받고 거래 안 하면 악용'
})

# 3.4 Reward-to-Volume Ratio
print(f"\n[Feature 3-4] Reward-to-Volume Ratio (RVR)")
print("-" * 120)

rvr = bonus_df['reward_to_volume_ratio'].dropna()
rvr_valid = rvr[rvr < 10]  # 극단값 제외

if len(rvr_valid) > 0:
    print(f"  전체 분포 (극단값 제외):")
    print(f"    - 평균: {rvr_valid.mean():.6f}")
    print(f"    - 중앙값: {rvr_valid.median():.6f}")
    print(f"    - 95th percentile: {rvr_valid.quantile(0.95):.6f}")

    high_rvr = len(bonus_df[bonus_df['reward_to_volume_ratio'] > 0.001])
    print(f"\n  분포 특성:")
    print(f"    - RVR > 0.001: {high_rvr}개 계정 ← 보너스 비해 거래량 적음")

    print(f"\n  ✅ 탐지 기준:")
    print(f"    - 정상: < 0.0001 (보너스 << 거래량)")
    print(f"    - 의심: 0.0001~0.001")
    print(f"    - 고위험: > 0.001 (보너스 > 거래량)")

    bonus_features_p3.append({
        'feature_name': 'reward_to_volume_ratio',
        'description': '보너스 / 거래량 비율',
        'normal_range': '< 0.0001',
        'suspicious_range': '0.0001~0.001',
        'high_risk_range': '> 0.001',
        'data_median': rvr_valid.median(),
        'data_95th': rvr_valid.quantile(0.95),
        'interpretation': '비율 높으면 보너스만 목적'
    })

# DataFrame 저장
bonus_p3_df = pd.DataFrame(bonus_features_p3)
bonus_p3_df.to_csv('output/final_features/pattern3_bonus_abuse_features.csv', index=False)
print(f"\n✓ Pattern 3 피처 저장: output/final_features/pattern3_bonus_abuse_features.csv")

# ================================================================================
# 통합 요약 문서 생성
# ================================================================================
print("\n" + "=" * 120)
print("통합 피처 요약 문서 생성")
print("=" * 120)

summary_md = f"""# 이상거래 패턴별 핵심 피처 정의서

## 📊 데이터 개요
- **분석 기간**: 2025-03-01 ~ 2025-10-31 (8개월)
- **총 계정 수**: 63개
- **총 거래 수**: 52,953건
- **총 펀딩피 기록**: 52,694건

---

## Pattern 1: 펀딩피 차익거래 (Funding Fee Arbitrage)

### 개념
펀딩 시각(0시, 4시, 8시, 12시, 16시, 20시) 전후에만 포지션을 유지하고, 펀딩피 수령 직후 청산하는 패턴

### 핵심 피처

#### 1. 펀딩피 절댓값 (funding_fee_abs)
**실제 데이터 분포:**
- 중앙값: ${funding_raw['funding_fee_abs'].median():.4f} ← **0에 매우 가까움** (정상)
- 95th percentile: ${funding_raw['funding_fee_abs'].quantile(0.95):.4f}
- 펀딩피 < $1: {near_zero:,}건 ({near_zero/len(funding_raw)*100:.1f}%) ← 대부분
- 펀딩피 > $10: {large_values:,}건 ({large_values/len(funding_raw)*100:.1f}%) ← 주목!

**왜 중요한가?**
- 정상 거래자는 펀딩피가 **0 근처**에 분포 (부수적 수익)
- 이상 거래자는 **큰 펀딩피**를 지속적으로 수령 (주 수익원)
- 펀딩피 절댓값이 클수록 "펀딩을 노린 거래" 가능성 ↑

**탐지 기준:**
| 구분 | 범위 | 해석 |
|------|------|------|
| 정상 | 평균 < $5 | 펀딩피는 부수적 |
| 의심 | $5 ~ $30 | 펀딩피 의존도 상승 |
| 고위험 | > $30 | 펀딩피가 주 목적 |

---

#### 2. 포지션 보유시간 (mean_holding_minutes)
**실제 데이터 분포:**
- 중앙값: {holding_times.median():.2f}분 ← **매우 짧음!**
- 10th percentile: {holding_times.quantile(0.10):.2f}분
- < 30분: {short_holding}개 계정 ({short_holding/len(funding_arb_df)*100:.1f}%)

**왜 중요한가?**
- 펀딩피는 8시간마다(또는 4시간) 지급
- 정상: 펀딩 시각과 무관하게 포지션 유지 (수 시간~일)
- 이상: 펀딩 직전 진입 → 펀딩 수령 → 즉시 청산 (**수 분**)

**탐지 기준:**
| 구분 | 범위 | 해석 |
|------|------|------|
| 정상 | > 60분 | 일반적인 거래 |
| 의심 | 30~60분 | 짧은 편 |
| 고위험 | < 30분 | 펀딩만 받고 청산 |

---

#### 3. 펀딩 시각 거래 집중도 (funding_timing_ratio)
**실제 데이터 분포:**
- 평균: {timing_ratio.mean()*100:.2f}%
- 95th percentile: {timing_ratio.quantile(0.95)*100:.2f}%
- > 50% 집중: {high_concentration}개 계정

**왜 중요한가?**
- 펀딩 시각: 0시, 4시, 8시, 12시, 16시, 20시 (±30분)
- 정상: 거래가 시간대별로 **고르게 분포** (< 30%)
- 이상: 거래의 **50% 이상**이 펀딩 시각에만 집중

**탐지 기준:**
| 구분 | 범위 | 해석 |
|------|------|------|
| 정상 | < 30% | 자연스러운 분포 |
| 의심 | 30~50% | 펀딩 시각 선호 |
| 고위험 | > 50% | 펀딩 시각만 노림 |

---

#### 4. 펀딩피 수익 비중 (funding_profit_ratio)
**실제 데이터 분포:**
- 중앙값: {valid_profit_ratio.median()*100:.2f}% ← 거의 0%
- 95th percentile: {valid_profit_ratio.quantile(0.95)*100:.2f}%

**왜 중요한가?**
- 정상: 수익의 대부분은 **거래 차익** (가격 변동)
- 이상: 수익의 **70% 이상**이 펀딩피

**탐지 기준:**
| 구분 | 범위 | 해석 |
|------|------|------|
| 정상 | < 30% | 거래 차익이 주 수익 |
| 의심 | 30~70% | 펀딩피 의존도 상승 |
| 고위험 | > 70% | 펀딩피가 주 수익원 |

---

## Pattern 2: 조직적 거래 (Organized Trading)

### 개념
다계정을 운영하여 동일 시간대, 동일 심볼, 유사 가격대에서 거래하는 패턴

### 핵심 피처

#### 1. IP 공유 비율 (ip_shared_ratio)
**실제 데이터 분포:**
- 평균: {ip_shared.mean()*100:.2f}%
- 중앙값: {ip_shared.median()*100:.2f}% ← **대부분 0%**
- IP 공유 있음: {shared_accounts}개 계정

**왜 중요한가?**
- 1인 1계정 원칙: 정상 사용자는 **고유 IP** 사용
- 다계정 악용자는 **동일 IP**에서 여러 계정 접속
- VPN 사용해도 일부 패턴 탐지 가능

**탐지 기준:**
| 구분 | 범위 | 해석 |
|------|------|------|
| 정상 | 0% | 고유 IP |
| 의심 | > 30% | IP 중복 의심 |
| 고위험 | > 50% | 명백한 다계정 |

---

#### 2. 동시 거래 비율 (concurrent_trading_ratio)
**실제 데이터 분포:**
- 평균: {concurrent.mean()*100:.2f}%
- 95th percentile: {concurrent.quantile(0.95)*100:.2f}%
- > 50% 동시: {high_concurrent}개 계정

**왜 중요한가?**
- 우연히 겹치는 경우: < 30%
- 조직적 거래: **같은 시간(1분 단위) + 같은 심볼**에서 거래
- 자동화된 봇 또는 신호 공유 가능성

**탐지 기준:**
| 구분 | 범위 | 해석 |
|------|------|------|
| 정상 | < 30% | 우연히 겹침 |
| 의심 | 30~50% | 타이밍 의심스러움 |
| 고위험 | > 50% | 조직적 거래 확실 |

---

#### 3. 가격 유사도 (price_similarity_ratio)
**실제 데이터 분포 (유사 거래 있는 계정만):**
- 평균: {price_sim_valid.mean()*100:.2f}% if len(price_sim_valid) > 0 else 'N/A'
- > 80% 유사: {high_similarity}개 계정

**왜 중요한가?**
- 동시 거래 중 가격까지 **거의 동일** (CV < 1%)
- 같은 신호로 거래하거나, 봇 사용 가능성
- 단독으로는 약하지만 **동시 거래와 결합 시** 강력한 신호

**탐지 기준:**
| 구분 | 범위 | 해석 |
|------|------|------|
| 정상 | < 60% | 다양한 가격대 |
| 의심 | 60~80% | 유사 가격 선호 |
| 고위험 | > 80% | 거의 동일 가격 |

---

#### 4. 평균 레버리지 (mean_leverage)
**실제 데이터 분포:**
- 평균: {leverage.mean():.2f}x
- 중앙값: {leverage.median():.2f}x
- > 30x: {high_leverage}개 계정

**왜 중요한가?**
- 레버리지 = 리스크 감수 정도
- 극단적 고레버리지 (> 50x)는 **공격적 전략**
- 다계정과 결합 시 시장 조작 가능성

**탐지 기준:**
| 구분 | 범위 | 해석 |
|------|------|------|
| 정상 | < 30x | 일반적인 레버리지 |
| 의심 | 30~50x | 공격적 |
| 고위험 | > 50x | 극단적 고위험 |

---

## Pattern 3: 보너스 악용 (Bonus Abuse)

### 개념
동일 IP에서 다수 계정을 생성하여 보너스만 수령 후 출금/비활성화

### 핵심 피처

#### 1. 총 보너스 금액 (total_reward)
**실제 데이터 분포:**
- 중앙값: ${rewards.median():.2f}
- 95th percentile: ${rewards.quantile(0.95):.2f}
- > $100: {high_reward}개 계정

**왜 중요한가?**
- 일반적인 보너스 정책: 가입 $5~10, 첫 입금 $10~50
- 정상 사용자: 총 $10~50 정도
- 다계정 악용: 반복 수령으로 **$100+**

**탐지 기준:**
| 구분 | 범위 | 해석 |
|------|------|------|
| 정상 | < $50 | 1~2회 정상 수령 |
| 의심 | $50~$100 | 다수 수령 |
| 고위험 | > $100 | 다계정 악용 확실 |

---

#### 2. 공유 IP 사용 (shared_ip)
**실제 데이터 분포:**
- 공유 IP 사용: {shared_ip_count}개 계정 ({shared_ip_count/len(bonus_df)*100:.1f}%)

**왜 중요한가?**
- **가장 강력한 신호**
- 동일 IP에서 **여러 계정이 보너스 수령**
- Sybil Attack의 핵심 지표

**탐지 기준:**
| 구분 | 범위 | 해석 |
|------|------|------|
| 정상 | False | 고유 IP |
| 고위험 | True | 다계정 생성 확실 |

---

#### 3. 거래 활동 여부 (has_trades)
**실제 데이터 분포:**
- 거래 있음: {has_trades}개 계정 ({has_trades/len(bonus_df)*100:.1f}%)
- 거래 없음: {no_trades}개 계정

**왜 중요한가?**
- 정상: 보너스 후 **활발한 거래** (거래소 목적 달성)
- 악용: 보너스만 받고 **거래 없음/최소화**

**탐지 기준:**
| 구분 | 범위 | 해석 |
|------|------|------|
| 정상 | True, 활발한 거래 | 정상 사용자 |
| 의심 | 거래 < 10회 | 최소 거래 |
| 고위험 | False | 보너스만 받고 비활성 |

---

#### 4. Reward-to-Volume Ratio (RVR)
**실제 데이터 분포:**
- 중앙값: {rvr_valid.median():.6f} if len(rvr_valid) > 0 else 'N/A'
- > 0.001: {high_rvr}개 계정

**왜 중요한가?**
- RVR = 보너스 / 거래량
- 정상: 보너스 << 거래량 (RVR < 0.0001)
- 악용: 보너스만 목적 (RVR > 0.001)

**탐지 기준:**
| 구분 | 범위 | 해석 |
|------|------|------|
| 정상 | < 0.0001 | 거래량이 보너스의 1000배+ |
| 의심 | 0.0001~0.001 | 보너스 비해 거래 적음 |
| 고위험 | > 0.001 | 보너스 > 거래량 |

---

## 💡 피처 활용 가이드

### 1. 단독 사용 가능한 강력한 피처
- **펀딩 시각 집중도** (> 70%): 펀딩 차익 거의 확실
- **공유 IP** (True): 다계정 생성 확실
- **포지션 보유시간** (< 10분): 펀딩만 노림

### 2. 조합 시 강력한 피처
- **동시 거래** + **가격 유사도**: 조직적 거래 신호
- **보너스 금액** + **공유 IP**: 보너스 악용 신호
- **짧은 보유시간** + **펀딩 시각 집중**: 펀딩 차익 신호

### 3. 보조 지표
- **레버리지**: 단독으로는 약하지만 다른 패턴과 결합 시 유용
- **RVR**: 거래 활동 정도 보조 지표

---

## 📌 실무 적용 팁

### 즉시 Alert 조건 (고위험)
```
펀딩피 차익거래:
  - 펀딩 시각 집중 > 70% AND 보유시간 < 20분

조직적 거래:
  - 동시 거래 > 70% AND 가격 유사 > 90%
  - 또는 IP 공유 > 50%

보너스 악용:
  - 공유 IP = True AND 보너스 > $100
  - 또는 RVR > 0.01
```

### 주간 모니터링 (의심)
- 위 고위험 기준의 70% 수준
- 추세 변화 관찰

---

*본 문서는 실제 데이터 분석을 기반으로 작성되었습니다.*
*마지막 업데이트: 2025-03*
"""

with open('output/final_features/FEATURES_DEFINITION.md', 'w', encoding='utf-8') as f:
    f.write(summary_md)

print(f"✓ 통합 문서 저장: output/final_features/FEATURES_DEFINITION.md")

# ================================================================================
# 간단한 시각화 생성
# ================================================================================
print("\n" + "=" * 120)
print("피처 요약 시각화 생성")
print("=" * 120)

fig, axes = plt.subplots(3, 3, figsize=(20, 16))
fig.suptitle('이상거래 패턴별 핵심 피처 분포', fontsize=18, fontweight='bold', y=0.998)

# Pattern 1 - 4개 피처
ax1 = axes[0, 0]
ax1.hist(funding_raw['funding_fee_abs'], bins=100, edgecolor='black', alpha=0.7, color='steelblue', range=(0, 50))
ax1.axvline(30.88, color='red', linestyle='--', linewidth=2, label='95th ($30.88)')
ax1.set_xlabel('Funding Fee (abs) ($)', fontsize=10)
ax1.set_ylabel('Count', fontsize=10)
ax1.set_title('[P1-1] 펀딩피 절댓값', fontsize=11, fontweight='bold')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

ax2 = axes[0, 1]
ax2.hist(holding_times, bins=50, edgecolor='black', alpha=0.7, color='green', range=(0, 300))
ax2.axvline(30, color='orange', linestyle='--', linewidth=2, label='의심 (30분)')
ax2.axvline(60, color='red', linestyle='--', linewidth=2, label='정상 (60분)')
ax2.set_xlabel('Holding Time (min)', fontsize=10)
ax2.set_ylabel('Count', fontsize=10)
ax2.set_title('[P1-2] 포지션 보유시간', fontsize=11, fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

ax3 = axes[0, 2]
ax3.hist(timing_ratio*100, bins=30, edgecolor='black', alpha=0.7, color='coral')
ax3.axvline(30, color='orange', linestyle='--', linewidth=2, label='의심 (30%)')
ax3.axvline(50, color='red', linestyle='--', linewidth=2, label='고위험 (50%)')
ax3.set_xlabel('Funding Timing Ratio (%)', fontsize=10)
ax3.set_ylabel('Count', fontsize=10)
ax3.set_title('[P1-3] 펀딩 시각 집중도', fontsize=11, fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Pattern 2 - 3개 피처
ax4 = axes[1, 0]
ax4.hist(ip_shared*100, bins=30, edgecolor='black', alpha=0.7, color='purple')
ax4.axvline(30, color='orange', linestyle='--', linewidth=2, label='의심 (30%)')
ax4.axvline(50, color='red', linestyle='--', linewidth=2, label='고위험 (50%)')
ax4.set_xlabel('IP Shared Ratio (%)', fontsize=10)
ax4.set_ylabel('Count', fontsize=10)
ax4.set_title('[P2-1] IP 공유 비율', fontsize=11, fontweight='bold')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

ax5 = axes[1, 1]
ax5.hist(concurrent*100, bins=30, edgecolor='black', alpha=0.7, color='orange')
ax5.axvline(30, color='orange', linestyle='--', linewidth=2, label='의심 (30%)')
ax5.axvline(50, color='red', linestyle='--', linewidth=2, label='고위험 (50%)')
ax5.set_xlabel('Concurrent Trading (%)', fontsize=10)
ax5.set_ylabel('Count', fontsize=10)
ax5.set_title('[P2-2] 동시 거래 비율', fontsize=11, fontweight='bold')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

ax6 = axes[1, 2]
ax6.hist(leverage, bins=30, edgecolor='black', alpha=0.7, color='gold')
ax6.axvline(30, color='orange', linestyle='--', linewidth=2, label='의심 (30x)')
ax6.axvline(50, color='red', linestyle='--', linewidth=2, label='고위험 (50x)')
ax6.set_xlabel('Mean Leverage (x)', fontsize=10)
ax6.set_ylabel('Count', fontsize=10)
ax6.set_title('[P2-3] 평균 레버리지', fontsize=11, fontweight='bold')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

# Pattern 3 - 2개 피처
ax7 = axes[2, 0]
ax7.hist(rewards, bins=30, edgecolor='black', alpha=0.7, color='crimson')
ax7.axvline(50, color='orange', linestyle='--', linewidth=2, label='의심 ($50)')
ax7.axvline(100, color='red', linestyle='--', linewidth=2, label='고위험 ($100)')
ax7.set_xlabel('Total Reward ($)', fontsize=10)
ax7.set_ylabel('Count', fontsize=10)
ax7.set_title('[P3-1] 총 보너스 금액', fontsize=11, fontweight='bold')
ax7.legend(fontsize=8)
ax7.grid(True, alpha=0.3)

ax8 = axes[2, 1]
shared_counts = [len(bonus_df) - shared_ip_count, shared_ip_count]
labels = ['고유 IP', '공유 IP']
colors_pie = ['green', 'red']
ax8.pie(shared_counts, labels=labels, colors=colors_pie, autopct='%1.1f%%',
        startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
ax8.set_title('[P3-2] IP 공유 여부', fontsize=11, fontweight='bold')

ax9 = axes[2, 2]
trade_counts = [has_trades, no_trades]
labels2 = ['거래 있음', '거래 없음']
colors_pie2 = ['green', 'red']
ax9.pie(trade_counts, labels=labels2, colors=colors_pie2, autopct='%1.1f%%',
        startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
ax9.set_title('[P3-3] 거래 활동 여부', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('output/final_features/features_visualization.png', dpi=300, bbox_inches='tight')
print(f"✓ 시각화 저장: output/final_features/features_visualization.png")

print("\n" + "=" * 120)
print("✅ 모든 피처 정리 완료!")
print("=" * 120)
print("\n생성된 파일:")
print("  1. output/final_features/pattern1_funding_arbitrage_features.csv")
print("  2. output/final_features/pattern2_organized_trading_features.csv")
print("  3. output/final_features/pattern3_bonus_abuse_features.csv")
print("  4. output/final_features/FEATURES_DEFINITION.md (통합 문서)")
print("  5. output/final_features/features_visualization.png")
