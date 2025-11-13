"""
통계적 임계값 계산 및 피처 유의미성 검증
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 120)
print("통계적 임계값 도출 및 피처 유의미성 검증")
print("=" * 120)

# 데이터 로드
funding_df = pd.read_csv('data/Funding.csv')
trade_df = pd.read_csv('data/Trade.csv')
reward_df = pd.read_csv('data/Reward.csv')
ip_df = pd.read_csv('data/IP.csv')

funding_df['ts'] = pd.to_datetime(funding_df['ts'])
trade_df['ts'] = pd.to_datetime(trade_df['ts'])

print("\n[데이터 로드 완료]")
print(f"  - Funding: {len(funding_df):,} rows")
print(f"  - Trade: {len(trade_df):,} rows")
print(f"  - Reward: {len(reward_df):,} rows")
print(f"  - IP: {len(ip_df):,} rows")

# ============================================================================
# Pattern 1: 펀딩피 차익거래
# ============================================================================
print("\n" + "=" * 120)
print("Pattern 1: 펀딩피 차익거래 임계값 도출")
print("=" * 120)

# 1-1. 펀딩피 절댓값
funding_raw = funding_df.copy()
funding_raw['funding_fee_abs'] = funding_raw['funding_fee'].abs()

print("\n[1-1. 펀딩피 절댓값 (funding_fee_abs)]")
print(f"  평균: ${funding_raw['funding_fee_abs'].mean():.2f}")
print(f"  중앙값: ${funding_raw['funding_fee_abs'].median():.2f}")
print(f"  표준편차: ${funding_raw['funding_fee_abs'].std():.2f}")
print(f"  Q1 (25%): ${funding_raw['funding_fee_abs'].quantile(0.25):.2f}")
print(f"  Q3 (75%): ${funding_raw['funding_fee_abs'].quantile(0.75):.2f}")
print(f"  90th percentile: ${funding_raw['funding_fee_abs'].quantile(0.90):.2f}")
print(f"  95th percentile: ${funding_raw['funding_fee_abs'].quantile(0.95):.2f}")
print(f"  99th percentile: ${funding_raw['funding_fee_abs'].quantile(0.99):.2f}")

# IQR method
Q1 = funding_raw['funding_fee_abs'].quantile(0.25)
Q3 = funding_raw['funding_fee_abs'].quantile(0.75)
IQR = Q3 - Q1
outlier_threshold_mild = Q3 + 1.5 * IQR
outlier_threshold_extreme = Q3 + 3 * IQR
print(f"\n  IQR method:")
print(f"    - 경미한 이상치 (Q3 + 1.5*IQR): ${outlier_threshold_mild:.2f}")
print(f"    - 극단적 이상치 (Q3 + 3*IQR): ${outlier_threshold_extreme:.2f}")

# 금융권 기준 참고
print(f"\n  금융권 기준 참고:")
print(f"    - 일반적 펀딩비: ±0.01% ~ ±0.05% (시가 대비)")
print(f"    - $10 이상은 상당히 큰 펀딩비로 간주 가능")
print(f"\n  권장 임계값:")
print(f"    - 의심 (Suspicious): ${funding_raw['funding_fee_abs'].quantile(0.90):.2f} (90th percentile)")
print(f"    - 고위험 (High Risk): ${funding_raw['funding_fee_abs'].quantile(0.95):.2f} (95th percentile)")

# 유의미성 검증 - 상위 5%와 하위 95% 비교
top_5_pct = funding_raw['funding_fee_abs'] >= funding_raw['funding_fee_abs'].quantile(0.95)
normal_95_pct = funding_raw['funding_fee_abs'] < funding_raw['funding_fee_abs'].quantile(0.95)
print(f"\n  유의미성 검증:")
print(f"    - 상위 5%: 평균 ${funding_raw[top_5_pct]['funding_fee_abs'].mean():.2f}")
print(f"    - 하위 95%: 평균 ${funding_raw[normal_95_pct]['funding_fee_abs'].mean():.2f}")
print(f"    - 차이 배수: {funding_raw[top_5_pct]['funding_fee_abs'].mean() / funding_raw[normal_95_pct]['funding_fee_abs'].mean():.1f}x")

# 1-2. 포지션 보유시간
print("\n[1-2. 포지션 보유시간 (mean_holding_minutes)]")
positions = []
for account_id in trade_df['account_id'].unique():
    acc_trades = trade_df[trade_df['account_id'] == account_id].sort_values('ts')

    for symbol in acc_trades['symbol'].unique():
        sym_trades = acc_trades[acc_trades['symbol'] == symbol].copy()
        position = 0
        open_time = None

        for idx, row in sym_trades.iterrows():
            if row['openclose'] == 'open':
                if position == 0:
                    open_time = row['ts']
                position += row['qty'] if row['side'] == 'buy' else -row['qty']
            else:
                position += -row['qty'] if row['side'] == 'buy' else row['qty']

                if abs(position) < 0.0001 and open_time is not None:
                    holding_minutes = (row['ts'] - open_time).total_seconds() / 60
                    positions.append({
                        'account_id': account_id,
                        'holding_minutes': holding_minutes
                    })
                    open_time = None

if len(positions) > 0:
    position_df = pd.DataFrame(positions)
    user_holding = position_df.groupby('account_id')['holding_minutes'].mean().reset_index()

    print(f"  평균: {user_holding['holding_minutes'].mean():.1f}분")
    print(f"  중앙값: {user_holding['holding_minutes'].median():.1f}분")
    print(f"  표준편차: {user_holding['holding_minutes'].std():.1f}분")
    print(f"  10th percentile: {user_holding['holding_minutes'].quantile(0.10):.1f}분")
    print(f"  25th percentile: {user_holding['holding_minutes'].quantile(0.25):.1f}분")
    print(f"  75th percentile: {user_holding['holding_minutes'].quantile(0.75):.1f}분")

    print(f"\n  금융권 기준 참고:")
    print(f"    - 펀딩비 수령 주기: 8시간마다 (480분)")
    print(f"    - 30분 이하: 펀딩비만 노리는 초단타")
    print(f"    - 1시간 이하: 의심스러운 단기 거래")

    print(f"\n  권장 임계값:")
    print(f"    - 고위험 (High Risk): {user_holding['holding_minutes'].quantile(0.10):.1f}분 (10th percentile, 짧을수록 위험)")
    print(f"    - 의심 (Suspicious): {user_holding['holding_minutes'].quantile(0.25):.1f}분 (25th percentile)")

    # 유의미성 검증
    bottom_10_pct = user_holding['holding_minutes'] <= user_holding['holding_minutes'].quantile(0.10)
    normal_90_pct = user_holding['holding_minutes'] > user_holding['holding_minutes'].quantile(0.10)
    print(f"\n  유의미성 검증:")
    print(f"    - 하위 10%: 평균 {user_holding[bottom_10_pct]['holding_minutes'].mean():.1f}분")
    print(f"    - 상위 90%: 평균 {user_holding[normal_90_pct]['holding_minutes'].mean():.1f}분")
    print(f"    - 차이: {user_holding[normal_90_pct]['holding_minutes'].mean() / user_holding[bottom_10_pct]['holding_minutes'].mean():.1f}x 더 김")
else:
    print("  경고: 포지션 데이터가 없습니다.")

# 1-3. 펀딩 시각 거래 집중도
print("\n[1-3. 펀딩 시각 거래 집중도 (funding_timing_ratio)]")
trade_df['hour'] = trade_df['ts'].dt.hour
funding_hours = [0, 4, 8, 12, 16, 20]

user_timing = []
for account_id in trade_df['account_id'].unique():
    acc_trades = trade_df[trade_df['account_id'] == account_id]
    total_trades = len(acc_trades)

    if total_trades == 0:
        continue

    funding_trades = 0
    for hour in funding_hours:
        hour_trades = acc_trades[
            ((acc_trades['hour'] >= hour - 0.5) & (acc_trades['hour'] <= hour + 0.5)) |
            ((hour == 0) & (acc_trades['hour'] >= 23.5)) |
            ((hour == 0) & (acc_trades['hour'] <= 0.5))
        ]
        funding_trades += len(hour_trades)

    ratio = funding_trades / total_trades
    user_timing.append({
        'account_id': account_id,
        'funding_timing_ratio': ratio
    })

timing_df = pd.DataFrame(user_timing)

print(f"  평균: {timing_df['funding_timing_ratio'].mean():.2%}")
print(f"  중앙값: {timing_df['funding_timing_ratio'].median():.2%}")
print(f"  표준편차: {timing_df['funding_timing_ratio'].std():.2%}")
print(f"  75th percentile: {timing_df['funding_timing_ratio'].quantile(0.75):.2%}")
print(f"  90th percentile: {timing_df['funding_timing_ratio'].quantile(0.90):.2%}")
print(f"  95th percentile: {timing_df['funding_timing_ratio'].quantile(0.95):.2%}")

print(f"\n  이론적 기준:")
print(f"    - 완전 랜덤: ~25% (펀딩 시각 ±30분 = 6시간 / 24시간)")
print(f"    - 50% 이상: 명확한 펀딩 시각 집중")
print(f"    - 70% 이상: 극단적인 펀딩 거래 편향")

print(f"\n  권장 임계값:")
print(f"    - 의심 (Suspicious): {timing_df['funding_timing_ratio'].quantile(0.75):.2%} (75th percentile)")
print(f"    - 고위험 (High Risk): {timing_df['funding_timing_ratio'].quantile(0.90):.2%} (90th percentile)")

# 유의미성 검증
top_10_pct = timing_df['funding_timing_ratio'] >= timing_df['funding_timing_ratio'].quantile(0.90)
normal_90_pct = timing_df['funding_timing_ratio'] < timing_df['funding_timing_ratio'].quantile(0.90)
print(f"\n  유의미성 검증:")
print(f"    - 상위 10%: 평균 {timing_df[top_10_pct]['funding_timing_ratio'].mean():.2%}")
print(f"    - 하위 90%: 평균 {timing_df[normal_90_pct]['funding_timing_ratio'].mean():.2%}")
print(f"    - 차이: {timing_df[top_10_pct]['funding_timing_ratio'].mean() / timing_df[normal_90_pct]['funding_timing_ratio'].mean():.1f}x")

# 1-4. 펀딩피 수익 비중
print("\n[1-4. 펀딩피 수익 비중 (funding_profit_ratio)]")
user_funding = funding_df.groupby('account_id')['funding_fee'].sum().reset_index()
user_funding['funding_fee_abs_sum'] = user_funding['funding_fee'].abs()
user_volume = trade_df.groupby('account_id')['amount'].sum().reset_index()
merged = pd.merge(user_funding, user_volume, on='account_id', how='outer').fillna(0)
merged['funding_profit_ratio'] = merged.apply(
    lambda x: min(x['funding_fee_abs_sum'] / (x['amount'] + 0.001), 1.0) if x['amount'] > 0 else 0, axis=1
)
merged['funding_profit_ratio'] = merged['funding_profit_ratio'].clip(0, 1)

print(f"  평균: {merged['funding_profit_ratio'].mean():.2%}")
print(f"  중앙값: {merged['funding_profit_ratio'].median():.2%}")
print(f"  75th percentile: {merged['funding_profit_ratio'].quantile(0.75):.2%}")
print(f"  90th percentile: {merged['funding_profit_ratio'].quantile(0.90):.2%}")
print(f"  95th percentile: {merged['funding_profit_ratio'].quantile(0.95):.2%}")

print(f"\n  권장 임계값:")
print(f"    - 의심 (Suspicious): {merged['funding_profit_ratio'].quantile(0.75):.2%} (75th percentile)")
print(f"    - 고위험 (High Risk): {merged['funding_profit_ratio'].quantile(0.90):.2%} (90th percentile)")

# ============================================================================
# Pattern 2: 조직적 거래
# ============================================================================
print("\n" + "=" * 120)
print("Pattern 2: 조직적 거래 임계값 도출")
print("=" * 120)

# 2-1. IP 공유 비율
print("\n[2-1. IP 공유 비율 (ip_shared_ratio)]")
ip_counts = ip_df.groupby('ip').size().reset_index(name='account_count')
ip_users = ip_df.merge(ip_counts, on='ip')
ip_users['ip_shared_ratio'] = (ip_users['account_count'] - 1) / ip_users['account_count']

print(f"  평균: {ip_users['ip_shared_ratio'].mean():.2%}")
print(f"  중앙값: {ip_users['ip_shared_ratio'].median():.2%}")
print(f"  75th percentile: {ip_users['ip_shared_ratio'].quantile(0.75):.2%}")
print(f"  90th percentile: {ip_users['ip_shared_ratio'].quantile(0.90):.2%}")

print(f"\n  실제 분포:")
print(f"    - 단독 IP: {(ip_users['account_count'] == 1).sum():,} ({(ip_users['account_count'] == 1).sum() / len(ip_users) * 100:.1f}%)")
print(f"    - 공유 IP (2개 이상): {(ip_users['account_count'] > 1).sum():,} ({(ip_users['account_count'] > 1).sum() / len(ip_users) * 100:.1f}%)")

print(f"\n  권장 임계값:")
print(f"    - 의심 (Suspicious): 2개 계정 공유 (ratio ≈ 50%)")
print(f"    - 고위험 (High Risk): 3개 이상 계정 공유 (ratio ≥ 67%)")

# 2-2. 평균 레버리지
print("\n[2-2. 평균 레버리지 (mean_leverage)]")
user_leverage = trade_df.groupby('account_id')['leverage'].mean().reset_index(name='mean_leverage')

print(f"  평균: {user_leverage['mean_leverage'].mean():.1f}배")
print(f"  중앙값: {user_leverage['mean_leverage'].median():.1f}배")
print(f"  75th percentile: {user_leverage['mean_leverage'].quantile(0.75):.1f}배")
print(f"  90th percentile: {user_leverage['mean_leverage'].quantile(0.90):.1f}배")
print(f"  95th percentile: {user_leverage['mean_leverage'].quantile(0.95):.1f}배")

print(f"\n  금융권 기준:")
print(f"    - 일반 투자자: 10-20배")
print(f"    - 공격적 투자자: 20-50배")
print(f"    - 극단적 고위험: 50배 이상")

print(f"\n  권장 임계값:")
print(f"    - 의심 (Suspicious): {user_leverage['mean_leverage'].quantile(0.75):.1f}배 (75th percentile)")
print(f"    - 고위험 (High Risk): {user_leverage['mean_leverage'].quantile(0.90):.1f}배 (90th percentile)")

# ============================================================================
# Pattern 3: 보너스 악용
# ============================================================================
print("\n" + "=" * 120)
print("Pattern 3: 보너스 악용 임계값 도출")
print("=" * 120)

# 3-1. 총 보너스 수령액
print("\n[3-1. 총 보너스 수령액 (total_reward)]")
user_reward = reward_df.groupby('account_id')['reward_amount'].sum().reset_index(name='total_reward')

print(f"  평균: ${user_reward['total_reward'].mean():.2f}")
print(f"  중앙값: ${user_reward['total_reward'].median():.2f}")
print(f"  75th percentile: ${user_reward['total_reward'].quantile(0.75):.2f}")
print(f"  90th percentile: ${user_reward['total_reward'].quantile(0.90):.2f}")
print(f"  95th percentile: ${user_reward['total_reward'].quantile(0.95):.2f}")

print(f"\n  거래소 정책 참고:")
print(f"    - 일반 가입 보너스: $10-$50")
print(f"    - 다중 수령은 명백한 악용")

print(f"\n  권장 임계값:")
print(f"    - 의심 (Suspicious): ${user_reward['total_reward'].quantile(0.75):.2f} (75th percentile)")
print(f"    - 고위험 (High Risk): ${user_reward['total_reward'].quantile(0.90):.2f} (90th percentile)")

# 3-2. IP 공유 (보너스 계정)
print("\n[3-2. IP 공유 (보너스 계정)]")
reward_accounts = set(reward_df['account_id'].unique())
reward_ip = ip_df[ip_df['account_id'].isin(reward_accounts)].copy()
ip_counts_reward = reward_ip.groupby('ip').size().reset_index(name='account_count')
reward_ip = reward_ip.merge(ip_counts_reward, on='ip')

print(f"  단독 IP: {(reward_ip['account_count'] == 1).sum():,} ({(reward_ip['account_count'] == 1).sum() / len(reward_ip) * 100:.1f}%)")
print(f"  2개 계정 공유: {(reward_ip['account_count'] == 2).sum():,} ({(reward_ip['account_count'] == 2).sum() / len(reward_ip) * 100:.1f}%)")
print(f"  3개 이상 공유: {(reward_ip['account_count'] >= 3).sum():,} ({(reward_ip['account_count'] >= 3).sum() / len(reward_ip) * 100:.1f}%)")

print(f"\n  권장 기준:")
print(f"    - 의심 (Suspicious): 2개 계정 공유")
print(f"    - 고위험 (High Risk): 3개 이상 계정 공유")

# 3-3. 거래 활동 여부
print("\n[3-3. 거래 활동 여부]")
all_accounts = set(trade_df['account_id'].unique())
trading_reward = reward_accounts.intersection(all_accounts)
no_trading_reward = reward_accounts - all_accounts

print(f"  거래 있음: {len(trading_reward):,} ({len(trading_reward) / len(reward_accounts) * 100:.1f}%)")
print(f"  거래 없음: {len(no_trading_reward):,} ({len(no_trading_reward) / len(reward_accounts) * 100:.1f}%)")

print(f"\n  권장 기준:")
print(f"    - 거래 없으면서 보너스만 수령: 고위험")

print("\n" + "=" * 120)
print("✅ 통계적 임계값 도출 완료")
print("=" * 120)