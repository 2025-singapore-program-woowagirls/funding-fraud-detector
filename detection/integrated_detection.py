"""
통합 이상 거래 탐지 시스템
FEATURE_ANALYSIS_REPORT.md 기반 임계값 if문 구현
"""
import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================
# 데이터 로딩
# ============================================================
print("=" * 100)
print("데이터 로딩 중...")
print("=" * 100)

funding_df = pd.read_csv('data/Funding.csv')
trade_df = pd.read_csv('data/Trade.csv')
reward_df = pd.read_csv('data/Reward.csv')
ip_df = pd.read_csv('data/IP.csv')

funding_df['ts'] = pd.to_datetime(funding_df['ts'])
trade_df['ts'] = pd.to_datetime(trade_df['ts'])
reward_df['ts'] = pd.to_datetime(reward_df['ts'])

# 현재 시점 설정 (2025-10-31 20:00:38.588185)
CURRENT_TIME = pd.to_datetime('2025-10-31 20:00:38.588185')

print(f"데이터 로딩 완료")
print(f"현재 시점: {CURRENT_TIME}")
print(f"Funding: {len(funding_df)} 건")
print(f"Trade: {len(trade_df)} 건")
print(f"Reward: {len(reward_df)} 건")
print(f"IP: {len(ip_df)} 건")
print()

# ============================================================
# Pattern 1: 펀딩피 차익거래 피처 계산
# ============================================================
print("=" * 100)
print("Pattern 1: 펀딩피 차익거래 피처 계산 중...")
print("=" * 100)

# 1. 펀딩피 절댓값
funding_fee_abs = funding_df.groupby('account_id')['funding_fee'].apply(lambda x: abs(x).mean())

# 2. 포지션 보유시간
holding_times = []
for (account_id, symbol, position_id), group in trade_df.groupby(['account_id', 'symbol', 'position_id']):
    open_trades = group[group['openclose'] == 'OPEN']
    close_trades = group[group['openclose'] == 'CLOSE']
    if len(open_trades) > 0 and len(close_trades) > 0:
        open_time = open_trades['ts'].min()
        close_time = close_trades['ts'].max()
        holding_minutes = (close_time - open_time).total_seconds() / 60
        holding_times.append({'account_id': account_id, 'holding_minutes': holding_minutes})

holding_df = pd.DataFrame(holding_times)
mean_holding = holding_df.groupby('account_id')['holding_minutes'].mean()

# 3. 펀딩 시각 거래 집중도
def is_funding_time(timestamp):
    hour = timestamp.hour
    minute = timestamp.minute
    funding_hours = [0, 4, 8, 12, 16, 20]
    for fh in funding_hours:
        if fh == 0:
            if (hour == 23 and minute >= 30) or (hour == 0 and minute <= 30):
                return True
        else:
            if (hour == fh - 1 and minute >= 30) or (hour == fh and minute <= 30):
                return True
    return False

trade_df['is_funding_time'] = trade_df['ts'].apply(is_funding_time)
funding_timing_ratio = trade_df.groupby('account_id')['is_funding_time'].mean() * 100  # 백분율

# 4. 펀딩피 수익 비중
account_funding = funding_df.groupby('account_id')['funding_fee'].apply(lambda x: abs(x).sum())

pnl_list = []
for (account_id, position_id), group in trade_df.groupby(['account_id', 'position_id']):
    open_trades = group[group['openclose'] == 'OPEN']
    close_trades = group[group['openclose'] == 'CLOSE']
    if len(open_trades) > 0 and len(close_trades) > 0:
        pnl = close_trades['amount'].sum() - open_trades['amount'].sum()
        pnl_list.append({'account_id': account_id, 'pnl': abs(pnl)})

pnl_df = pd.DataFrame(pnl_list)
account_pnl = pnl_df.groupby('account_id')['pnl'].sum()

funding_profit_ratio = (account_funding / (account_funding + account_pnl + 1e-9)) * 100  # 백분율

print(f"펀딩피 차익거래 피처 계산 완료")
print()

# ============================================================
# Pattern 2: 조직적 거래 피처 계산
# ============================================================
print("=" * 100)
print("Pattern 2: 조직적 거래 피처 계산 중...")
print("=" * 100)

# 1. IP 공유 수
ip_counts = ip_df.groupby('ip')['account_id'].nunique()
account_to_shared_count = {}
for ip, count in ip_counts.items():
    if count > 1:
        accounts = ip_df[ip_df['ip'] == ip]['account_id'].unique()
        for acc in accounts:
            account_to_shared_count[acc] = count

# 2. 평균 레버리지
mean_leverage = trade_df.groupby('account_id')['leverage'].mean()

print(f"조직적 거래 피처 계산 완료")
print()

# ============================================================
# Pattern 3: 보너스 악용 피처 계산
# ============================================================
print("=" * 100)
print("Pattern 3: 보너스 악용 피처 계산 중...")
print("=" * 100)

# 1. 총 보너스 수령액
total_reward = reward_df.groupby('account_id')['reward_amount'].sum()

# 2. 보너스 계정 IP 공유
reward_accounts = reward_df['account_id'].unique()
reward_ip_df = ip_df[ip_df['account_id'].isin(reward_accounts)]
reward_ip_counts = reward_ip_df.groupby('ip')['account_id'].nunique()
reward_account_to_shared_count = {}
for ip, count in reward_ip_counts.items():
    if count > 1:
        accounts = reward_ip_df[reward_ip_df['ip'] == ip]['account_id'].unique()
        for acc in accounts:
            reward_account_to_shared_count[acc] = count

print(f"보너스 악용 피처 계산 완료")
print()

# ============================================================
# 모든 계정 ID 수집
# ============================================================
all_accounts = set()
all_accounts.update(funding_df['account_id'].unique())
all_accounts.update(trade_df['account_id'].unique())
all_accounts.update(reward_df['account_id'].unique())
all_accounts = sorted(list(all_accounts))

print(f"총 계정 수: {len(all_accounts)}")
print()

# ============================================================
# 통합 데이터프레임 생성
# ============================================================
print("=" * 100)
print("통합 데이터프레임 생성 중...")
print("=" * 100)

results = []

for account_id in all_accounts:
    row = {'account_id': account_id}

    # ===== Pattern 1: 펀딩피 차익거래 =====
    # 1. 펀딩피 절댓값
    fee_abs = funding_fee_abs.get(account_id, 0)
    row['funding_fee_abs'] = fee_abs
    if fee_abs >= 30.88:
        row['funding_fee_abs_risk'] = 'HIGH'
    elif fee_abs >= 11.16:
        row['funding_fee_abs_risk'] = 'SUSPICIOUS'
    else:
        row['funding_fee_abs_risk'] = 'NORMAL'

    # 2. 포지션 보유시간 (짧을수록 위험)
    holding = mean_holding.get(account_id, 999999)
    row['mean_holding_minutes'] = holding if holding != 999999 else None
    if holding <= 10.8:
        row['mean_holding_risk'] = 'HIGH'
    elif holding <= 59.3:
        row['mean_holding_risk'] = 'SUSPICIOUS'
    else:
        row['mean_holding_risk'] = 'NORMAL'

    # 3. 펀딩 시각 거래 집중도
    timing = funding_timing_ratio.get(account_id, 0)
    row['funding_timing_ratio'] = timing
    if timing >= 36.73:
        row['funding_timing_risk'] = 'HIGH'
    elif timing >= 27.73:
        row['funding_timing_risk'] = 'SUSPICIOUS'
    else:
        row['funding_timing_risk'] = 'NORMAL'

    # 4. 펀딩피 수익 비중
    profit_ratio = funding_profit_ratio.get(account_id, 0)
    row['funding_profit_ratio'] = profit_ratio
    if profit_ratio >= 37.38:
        row['funding_profit_ratio_risk'] = 'HIGH'
    elif profit_ratio >= 10.05:
        row['funding_profit_ratio_risk'] = 'SUSPICIOUS'
    else:
        row['funding_profit_ratio_risk'] = 'NORMAL'

    # Pattern 1 종합 리스크
    p1_risks = [
        row['funding_fee_abs_risk'],
        row['mean_holding_risk'],
        row['funding_timing_risk'],
        row['funding_profit_ratio_risk']
    ]
    high_count = p1_risks.count('HIGH')
    suspicious_count = p1_risks.count('SUSPICIOUS')

    if high_count >= 2:
        row['pattern1_risk'] = 'HIGH'
    elif high_count >= 1 or suspicious_count >= 3:
        row['pattern1_risk'] = 'SUSPICIOUS'
    else:
        row['pattern1_risk'] = 'NORMAL'

    # ===== Pattern 2: 조직적 거래 =====
    # 1. IP 공유 수
    ip_shared = account_to_shared_count.get(account_id, 1)
    row['ip_shared_count'] = ip_shared
    if ip_shared >= 3:
        row['ip_shared_risk'] = 'HIGH'
    elif ip_shared >= 2:
        row['ip_shared_risk'] = 'SUSPICIOUS'
    else:
        row['ip_shared_risk'] = 'NORMAL'

    # 2. 평균 레버리지
    leverage = mean_leverage.get(account_id, 0)
    row['mean_leverage'] = leverage
    if leverage >= 31.3:
        row['mean_leverage_risk'] = 'HIGH'
    elif leverage >= 14.1:
        row['mean_leverage_risk'] = 'SUSPICIOUS'
    else:
        row['mean_leverage_risk'] = 'NORMAL'

    # Pattern 2 종합 리스크
    p2_risks = [row['ip_shared_risk'], row['mean_leverage_risk']]
    if 'HIGH' in p2_risks:
        row['pattern2_risk'] = 'HIGH'
    elif 'SUSPICIOUS' in p2_risks:
        row['pattern2_risk'] = 'SUSPICIOUS'
    else:
        row['pattern2_risk'] = 'NORMAL'

    # ===== Pattern 3: 보너스 악용 =====
    # 1. 총 보너스 수령액
    reward = total_reward.get(account_id, 0)
    row['total_reward'] = reward
    if reward >= 534.90:
        row['total_reward_risk'] = 'HIGH'
    elif reward >= 159.99:
        row['total_reward_risk'] = 'SUSPICIOUS'
    elif reward > 0:
        row['total_reward_risk'] = 'NORMAL'
    else:
        row['total_reward_risk'] = 'NO_REWARD'

    # 2. 보너스 계정 IP 공유
    reward_ip_shared = reward_account_to_shared_count.get(account_id, 1)
    row['reward_ip_shared_count'] = reward_ip_shared if reward > 0 else 0
    if reward > 0 and reward_ip_shared >= 2:
        row['reward_ip_shared_risk'] = 'HIGH'
    elif reward > 0:
        row['reward_ip_shared_risk'] = 'NORMAL'
    else:
        row['reward_ip_shared_risk'] = 'NO_REWARD'

    # Pattern 3 종합 리스크
    if reward == 0:
        row['pattern3_risk'] = 'NO_REWARD'
    else:
        p3_risks = [row['total_reward_risk'], row['reward_ip_shared_risk']]
        if 'HIGH' in p3_risks:
            row['pattern3_risk'] = 'HIGH'
        elif 'SUSPICIOUS' in p3_risks:
            row['pattern3_risk'] = 'SUSPICIOUS'
        else:
            row['pattern3_risk'] = 'NORMAL'

    # ===== 통합 리스크 레벨 =====
    pattern_risks = [row['pattern1_risk'], row['pattern2_risk'], row['pattern3_risk']]
    pattern_high = sum([1 for r in pattern_risks if r == 'HIGH'])
    pattern_suspicious = sum([1 for r in pattern_risks if r == 'SUSPICIOUS'])

    if pattern_high >= 2:
        row['final_risk'] = 'CRITICAL'
    elif pattern_high >= 1:
        row['final_risk'] = 'HIGH'
    elif pattern_suspicious >= 2:
        row['final_risk'] = 'SUSPICIOUS'
    else:
        row['final_risk'] = 'NORMAL'

    # 추가 정보
    row['total_profit_usd'] = account_pnl.get(account_id, 0) + account_funding.get(account_id, 0)
    row['detection_time'] = CURRENT_TIME
    row['reviewed'] = False  # 기본값: 미확인

    results.append(row)

df = pd.DataFrame(results)

print(f"통합 데이터프레임 생성 완료: {len(df)} 계정")
print()

# ============================================================
# 결과 저장
# ============================================================
print("=" * 100)
print("결과 저장 중...")
print("=" * 100)

# 1. 전체 결과 저장
output_file = 'output/final/integrated_detection_results.csv'
df.to_csv(output_file, index=False)
print(f"✅ 전체 결과 저장: {output_file}")

# 2. 고위험 계정만 저장
high_risk_df = df[df['final_risk'].isin(['CRITICAL', 'HIGH', 'SUSPICIOUS'])]
high_risk_file = 'output/final/high_risk_accounts_detected.csv'
high_risk_df.to_csv(high_risk_file, index=False)
print(f"✅ 고위험 계정 저장: {high_risk_file} ({len(high_risk_df)} 계정)")

# 3. Grafana용 시계열 데이터 생성
grafana_data = []
for _, row in df.iterrows():
    base_record = {
        'timestamp': row['detection_time'],
        'account_id': row['account_id'],
        'reviewed': row['reviewed']
    }

    # Pattern 1
    if row['pattern1_risk'] in ['HIGH', 'SUSPICIOUS']:
        grafana_data.append({
            **base_record,
            'pattern': 'Pattern1_FundingFeeArbitrage',
            'risk_level': row['pattern1_risk'],
            'metric_value': row['funding_profit_ratio'],
            'metric_name': 'funding_profit_ratio',
            'description': f"펀딩피 수익 비중: {row['funding_profit_ratio']:.2f}%"
        })

    # Pattern 2
    if row['pattern2_risk'] in ['HIGH', 'SUSPICIOUS']:
        grafana_data.append({
            **base_record,
            'pattern': 'Pattern2_OrganizedTrading',
            'risk_level': row['pattern2_risk'],
            'metric_value': row['ip_shared_count'],
            'metric_name': 'ip_shared_count',
            'description': f"IP 공유 계정 수: {row['ip_shared_count']}"
        })

    # Pattern 3
    if row['pattern3_risk'] in ['HIGH', 'SUSPICIOUS'] and row['total_reward'] > 0:
        grafana_data.append({
            **base_record,
            'pattern': 'Pattern3_BonusAbuse',
            'risk_level': row['pattern3_risk'],
            'metric_value': row['total_reward'],
            'metric_name': 'total_reward',
            'description': f"총 보너스 수령액: ${row['total_reward']:.2f}"
        })

grafana_df = pd.DataFrame(grafana_data)
grafana_file = 'output/final/grafana_detection_alerts.csv'
grafana_df.to_csv(grafana_file, index=False)
print(f"✅ Grafana용 데이터 저장: {grafana_file} ({len(grafana_df)} 알림)")

# 4. 요약 통계
print()
print("=" * 100)
print("탐지 결과 요약")
print("=" * 100)
print(f"총 계정 수: {len(df)}")
print(f"CRITICAL: {len(df[df['final_risk'] == 'CRITICAL'])} 계정")
print(f"HIGH: {len(df[df['final_risk'] == 'HIGH'])} 계정")
print(f"SUSPICIOUS: {len(df[df['final_risk'] == 'SUSPICIOUS'])} 계정")
print(f"NORMAL: {len(df[df['final_risk'] == 'NORMAL'])} 계정")
print()
print(f"Pattern 1 (펀딩피 차익거래) 고위험: {len(df[df['pattern1_risk'] == 'HIGH'])} 계정")
print(f"Pattern 2 (조직적 거래) 고위험: {len(df[df['pattern2_risk'] == 'HIGH'])} 계정")
print(f"Pattern 3 (보너스 악용) 고위험: {len(df[df['pattern3_risk'] == 'HIGH'])} 계정")
print()
print("=" * 100)
print("완료!")
print("=" * 100)
