#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
실제 데이터로 피처 분포 검증
"""

import pandas as pd
import numpy as np
from pathlib import Path

# 경로 설정
BASE_DIR = Path(__file__).resolve().parent
data_dir = BASE_DIR / 'data'

print("=" * 80)
print("실제 데이터 피처 분포 검증")
print("=" * 80)

# 데이터 로드
print("\n[1] 데이터 로드")
trade_df = pd.read_csv(data_dir / 'Trade.csv')
funding_df = pd.read_csv(data_dir / 'Funding.csv')
reward_df = pd.read_csv(data_dir / 'Reward.csv')
ip_df = pd.read_csv(data_dir / 'IP.csv')

print(f"  ✓ Trade: {len(trade_df):,}건")
print(f"  ✓ Funding: {len(funding_df):,}건")
print(f"  ✓ Reward: {len(reward_df):,}건")
print(f"  ✓ IP: {len(ip_df):,}건")

print("\n" + "=" * 80)
print("Pattern 1: 펀딩피 차익거래")
print("=" * 80)

# 1. funding_fee_abs
print("\n[Feature 1] 펀딩피 절댓값 (funding_fee_abs)")
funding_fee_abs = funding_df['funding_fee'].abs()
print(f"  - 평균: ${funding_fee_abs.mean():.2f}")
print(f"  - 중앙값: ${funding_fee_abs.median():.2f}")
print(f"  - 표준편차: ${funding_fee_abs.std():.2f}")
print(f"  - 90th percentile: ${funding_fee_abs.quantile(0.90):.2f}")
print(f"  - 95th percentile: ${funding_fee_abs.quantile(0.95):.2f}")
print(f"  - 99th percentile: ${funding_fee_abs.quantile(0.99):.2f}")

# 2. mean_holding_minutes
print("\n[Feature 2] 포지션 보유시간 (mean_holding_minutes)")
positions = []
for (account_id, position_id), group in trade_df.groupby(['account_id', 'position_id']):
    open_trades = group[group['openclose'] == 'OPEN']
    close_trades = group[group['openclose'] == 'CLOSE']

    if len(open_trades) > 0 and len(close_trades) > 0:
        open_time = pd.to_datetime(open_trades['ts'].iloc[0])
        close_time = pd.to_datetime(close_trades['ts'].iloc[-1])
        holding_minutes = (close_time - open_time).total_seconds() / 60
        positions.append({
            'account_id': account_id,
            'holding_minutes': holding_minutes
        })

position_df = pd.DataFrame(positions)
mean_holding = position_df.groupby('account_id')['holding_minutes'].mean()

print(f"  - 평균: {mean_holding.mean():.1f}분")
print(f"  - 중앙값: {mean_holding.median():.1f}분")
print(f"  - 10th percentile: {mean_holding.quantile(0.10):.1f}분")
print(f"  - 25th percentile: {mean_holding.quantile(0.25):.1f}분")

# 3. funding_timing_ratio
print("\n[Feature 3] 펀딩 시각 거래 집중도 (funding_timing_ratio)")
trade_df['ts'] = pd.to_datetime(trade_df['ts'])
trade_df['hour'] = trade_df['ts'].dt.hour
trade_df['minute'] = trade_df['ts'].dt.minute

def is_funding_time(row):
    funding_hours = [0, 4, 8, 12, 16, 20]
    hour = row['hour']
    minute = row['minute']

    for fh in funding_hours:
        if hour == fh and minute <= 30:
            return True
        if hour == (fh - 1) % 24 and minute >= 30:
            return True
    return False

trade_df['is_funding'] = trade_df.apply(is_funding_time, axis=1)
funding_trades = trade_df.groupby('account_id')['is_funding'].sum()
total_trades = trade_df.groupby('account_id').size()
timing_ratio = (funding_trades / total_trades)

print(f"  - 평균: {timing_ratio.mean()*100:.2f}%")
print(f"  - 중앙값: {timing_ratio.median()*100:.2f}%")
print(f"  - 75th percentile: {timing_ratio.quantile(0.75)*100:.2f}%")
print(f"  - 90th percentile: {timing_ratio.quantile(0.90)*100:.2f}%")

# 4. funding_profit_ratio
print("\n[Feature 4] 펀딩피 수익 비중 (funding_profit_ratio)")
# PnL 계산
pnl_list = []
for (account_id, position_id), group in trade_df.groupby(['account_id', 'position_id']):
    open_trades = group[group['openclose'] == 'OPEN']
    close_trades = group[group['openclose'] == 'CLOSE']

    if len(open_trades) > 0 and len(close_trades) > 0:
        open_amount = open_trades['amount'].sum()
        close_amount = close_trades['amount'].sum()
        pnl = close_amount - open_amount
        pnl_list.append({'account_id': account_id, 'pnl': pnl})

pnl_df = pd.DataFrame(pnl_list)
account_pnl = pnl_df.groupby('account_id')['pnl'].sum().reset_index()
account_funding = funding_df.groupby('account_id')['funding_fee'].sum().reset_index()
account_funding['funding_abs'] = account_funding['funding_fee'].abs()

merged = pd.merge(account_pnl, account_funding[['account_id', 'funding_abs']], on='account_id', how='outer').fillna(0)
merged['pnl_abs'] = merged['pnl'].abs()
merged['funding_profit_ratio'] = merged.apply(
    lambda row: row['funding_abs'] / (row['funding_abs'] + row['pnl_abs'] + 0.001)
    if (row['funding_abs'] + row['pnl_abs']) > 0 else 0,
    axis=1
)

print(f"  - 평균: {merged['funding_profit_ratio'].mean()*100:.2f}%")
print(f"  - 중앙값: {merged['funding_profit_ratio'].median()*100:.2f}%")
print(f"  - 표준편차: {merged['funding_profit_ratio'].std()*100:.2f}%")
print(f"  - 75th percentile: {merged['funding_profit_ratio'].quantile(0.75)*100:.2f}%")
print(f"  - 90th percentile: {merged['funding_profit_ratio'].quantile(0.90)*100:.2f}%")
print(f"  - 95th percentile: {merged['funding_profit_ratio'].quantile(0.95)*100:.2f}%")

print("\n" + "=" * 80)
print("Pattern 2: 조직적 거래")
print("=" * 80)

# 1. IP 공유 비율
print("\n[Feature 1] IP 공유 비율 (ip_shared_ratio)")
ip_counts = ip_df.groupby('ip')['account_id'].nunique()
total_ips = len(ip_counts)
single_ip = (ip_counts == 1).sum()
two_accounts = (ip_counts == 2).sum()
three_plus = (ip_counts >= 3).sum()

print(f"  - 총 IP 수: {total_ips:,}개")
print(f"  - 단독 IP: {single_ip:,}개 ({single_ip/total_ips*100:.1f}%)")
print(f"  - 2개 계정 공유: {two_accounts:,}개 ({two_accounts/total_ips*100:.1f}%)")
print(f"  - 3개 이상 공유: {three_plus:,}개 ({three_plus/total_ips*100:.1f}%)")

# 2. 동시 거래율
print("\n[Feature 2] 동시 거래율 (concurrent_trading_ratio)")
# 1분 단위로 거래 그룹화
trade_df['ts_minute'] = trade_df['ts'].dt.floor('1min')
concurrent_trades = trade_df.groupby('ts_minute')['account_id'].nunique()

# 계정별 동시 거래율 계산 (간단한 근사)
# 각 거래의 시간대에 몇 명이 같이 거래했는지
trade_with_concurrent = trade_df.merge(
    concurrent_trades.rename('concurrent_count'),
    left_on='ts_minute',
    right_index=True
)
account_concurrent = trade_with_concurrent.groupby('account_id')['concurrent_count'].apply(
    lambda x: (x > 1).sum() / len(x)
)

print(f"  - 평균: {account_concurrent.mean()*100:.1f}%")
print(f"  - 중앙값: {account_concurrent.median()*100:.1f}%")
print(f"  - 75th percentile: {account_concurrent.quantile(0.75)*100:.1f}%")
print(f"  - 90th percentile: {account_concurrent.quantile(0.90)*100:.1f}%")

# 3. 평균 레버리지
print("\n[Feature 3] 평균 레버리지 (mean_leverage)")
leverage_data = trade_df[trade_df['openclose'] == 'OPEN'].groupby('account_id')['leverage'].mean()

print(f"  - 평균: {leverage_data.mean():.1f}배")
print(f"  - 중앙값: {leverage_data.median():.1f}배")
print(f"  - 75th percentile: {leverage_data.quantile(0.75):.1f}배")
print(f"  - 90th percentile: {leverage_data.quantile(0.90):.1f}배")
print(f"  - 95th percentile: {leverage_data.quantile(0.95):.1f}배")

print("\n" + "=" * 80)
print("Pattern 3: 보너스 악용")
print("=" * 80)

# 1. 총 보너스 수령액
print("\n[Feature 1] 총 보너스 수령액 (total_reward)")
total_rewards = reward_df.groupby('account_id')['reward_amount'].sum()

print(f"  - 평균: ${total_rewards.mean():.2f}")
print(f"  - 중앙값: ${total_rewards.median():.2f}")
print(f"  - 75th percentile: ${total_rewards.quantile(0.75):.2f}")
print(f"  - 90th percentile: ${total_rewards.quantile(0.90):.2f}")
print(f"  - 95th percentile: ${total_rewards.quantile(0.95):.2f}")

# 2. 보너스 수령 빈도
print("\n[Feature 2] 보너스 수령 빈도 (reward_frequency)")
reward_counts = reward_df.groupby('account_id').size()

print(f"  - 평균: {reward_counts.mean():.1f}회")
print(f"  - 중앙값: {reward_counts.median():.1f}회")
print(f"  - 75th percentile: {reward_counts.quantile(0.75):.1f}회")
print(f"  - 90th percentile: {reward_counts.quantile(0.90):.1f}회")
print(f"  - 최대: {reward_counts.max():.0f}회")

# 3. 보너스 출금 속도 (데이터에서 출금 정보를 찾을 수 없으므로 스킵)
print("\n[Feature 3] 보너스 출금 속도 (withdrawal_speed)")
print("  ※ 주의: 현재 데이터셋에는 출금 정보가 없어 계산 불가")
print("  - 이 피처는 실제 출금 데이터가 필요함")

# 4. 거래량 대비 보너스 비율
print("\n[Feature 4] 거래량 대비 보너스 비율 (bonus_to_volume_ratio)")
# 계정별 총 거래량
account_volume = trade_df.groupby('account_id')['amount'].sum().reset_index()
account_volume.columns = ['account_id', 'total_volume']

# 보너스와 거래량 병합
bonus_volume = pd.merge(
    reward_df.groupby('account_id')['reward_amount'].sum().reset_index(),
    account_volume,
    on='account_id',
    how='inner'
)
bonus_volume['ratio'] = bonus_volume['reward_amount'] / bonus_volume['total_volume']

print(f"  - 평균: {bonus_volume['ratio'].mean()*100:.2f}%")
print(f"  - 중앙값: {bonus_volume['ratio'].median()*100:.2f}%")
print(f"  - 75th percentile: {bonus_volume['ratio'].quantile(0.75)*100:.2f}%")
print(f"  - 90th percentile: {bonus_volume['ratio'].quantile(0.90)*100:.2f}%")

print("\n" + "=" * 80)
print("검증 완료")
print("=" * 80)
