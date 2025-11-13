import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 100)
print("보너스 악용(Bonus Abuse) 이상거래 탐지 분석")
print("=" * 100)
print("\n[연구 배경]")
print("거래소는 신규 고객 유치를 위해 가입 보너스, 입금 보너스, 거래량 인센티브 등을 제공합니다.")
print("악의적 사용자는 다계정을 생성하여 보너스만 수령 후 출금하거나 비활성화되는 패턴을 보입니다.")
print("본 분석에서는 Reward 데이터와 IP, Trade 패턴을 결합하여 보너스 악용 패턴을 탐지합니다.")

# ================================================================================
# 1. 데이터 로드
# ================================================================================
print("\n" + "=" * 100)
print("[1] 데이터 로딩 및 기초 통계")
print("=" * 100)

reward_df = pd.read_csv('data/Reward.csv')
trade_df = pd.read_csv('data/Trade.csv')
ip_df = pd.read_csv('data/IP.csv')
funding_df = pd.read_csv('data/Funding.csv')

# 타임스탬프 변환
reward_df['ts'] = pd.to_datetime(reward_df['ts'])
trade_df['ts'] = pd.to_datetime(trade_df['ts'])
funding_df['ts'] = pd.to_datetime(funding_df['ts'])

print(f"\n✓ Reward: {len(reward_df):,} rows, {reward_df['account_id'].nunique()} unique accounts")
print(f"✓ Trade: {len(trade_df):,} rows, {trade_df['account_id'].nunique()} unique accounts")
print(f"✓ IP: {len(ip_df):,} rows")
print(f"✓ Funding: {len(funding_df):,} rows")

# ================================================================================
# 2. 보너스 수령 패턴 탐색적 분석 (EDA)
# ================================================================================
print("\n" + "=" * 100)
print("[2] 보너스 수령 패턴 탐색적 분석 (Exploratory Data Analysis)")
print("=" * 100)

print("\n[연구 질문 1] 보너스 수령 행태의 정상/이상 경계는?")

# 2.1 계정별 보너스 집계
account_reward_stats = reward_df.groupby('account_id').agg({
    'reward_amount': ['sum', 'mean', 'count', 'std'],
    'ts': ['min', 'max']
}).reset_index()

account_reward_stats.columns = ['account_id', 'total_reward', 'mean_reward',
                                 'reward_count', 'std_reward', 'first_reward', 'last_reward']

# 보너스 수령 기간
account_reward_stats['reward_period_days'] = (
    account_reward_stats['last_reward'] - account_reward_stats['first_reward']
).dt.total_seconds() / 86400

print(f"\n[보너스 수령 통계]")
print(f"  총 보너스 지급액: ${reward_df['reward_amount'].sum():,.2f}")
print(f"  보너스 받은 계정 수: {len(account_reward_stats)}")
print(f"  평균 보너스 수령액: ${account_reward_stats['total_reward'].mean():,.2f}")
print(f"  중앙값: ${account_reward_stats['total_reward'].median():,.2f}")
print(f"  표준편차: ${account_reward_stats['total_reward'].std():,.2f}")

print(f"\n[보너스 수령 횟수 분포]")
count_dist = account_reward_stats['reward_count'].value_counts().sort_index()
for count, freq in count_dist.items():
    print(f"  {int(count)}회: {freq}개 계정 ({freq/len(account_reward_stats)*100:.1f}%)")

print(f"\n[보너스 금액 분포 Percentiles]")
for p in [10, 25, 50, 75, 90, 95, 99]:
    val = account_reward_stats['total_reward'].quantile(p/100)
    print(f"  {p}th: ${val:,.2f}")

# 도메인 지식: 거래소에서 일반적인 보너스 정책
# - 신규 가입: $5-10
# - 첫 입금: $10-50
# - 거래량 달성: $10-100
# → 정상 사용자는 총 $10-50 정도
# → 다계정 악용 시 $50+ 반복 수령

print(f"\n[도메인 지식 기반 정상 범위 설정]")
print(f"  정상: 총 보너스 < $50 (1-2회 수령)")
print(f"  의심: $50-100 (다수 수령)")
print(f"  고위험: $100+ (명백한 다계정 악용)")

high_reward_accounts = account_reward_stats[account_reward_stats['total_reward'] > 50]
print(f"\n  $50 초과 계정: {len(high_reward_accounts)} ({len(high_reward_accounts)/len(account_reward_stats)*100:.1f}%)")

# 2.2 시간대별 보너스 수령 패턴
print(f"\n[연구 질문 2] 단시간 내 집중 수령 패턴이 있는가?")

reward_df['hour'] = reward_df['ts'].dt.hour
reward_df['date'] = reward_df['ts'].dt.date

# 같은 날짜 내 여러 계정이 보너스 받은 경우
daily_reward_accounts = reward_df.groupby('date')['account_id'].apply(list).reset_index()
daily_reward_accounts['account_count'] = daily_reward_accounts['account_id'].apply(len)

print(f"\n[일별 보너스 수령 계정 수]")
print(f"  평균: {daily_reward_accounts['account_count'].mean():.1f}개/일")
print(f"  최대: {daily_reward_accounts['account_count'].max()}개/일")

high_activity_days = daily_reward_accounts[daily_reward_accounts['account_count'] >= 5]
print(f"  하루 5개 이상 계정 수령: {len(high_activity_days)}일")

# ================================================================================
# 3. IP 공유 패턴 분석 (실제 거래소 리스크 팀 지표)
# ================================================================================
print("\n" + "=" * 100)
print("[3] IP 기반 다계정 탐지 (Multi-Account Detection via IP)")
print("=" * 100)

print("\n[연구 질문 3] 동일 IP에서 여러 계정이 보너스를 수령했는가?")
print("[실무 참고] 거래소는 IP, Device Fingerprint, Browser Session을 추적하여 Sybil Attack 탐지")

# IP별 계정 매핑
ip_to_accounts = ip_df.groupby('ip')['account_id'].apply(set).to_dict()

# 보너스 받은 계정의 IP 정보
reward_accounts = set(reward_df['account_id'].unique())
reward_account_ips = ip_df[ip_df['account_id'].isin(reward_accounts)].copy()

# IP별 보너스 받은 계정 수
ip_reward_mapping = reward_account_ips.groupby('ip')['account_id'].apply(set).reset_index()
ip_reward_mapping['reward_account_count'] = ip_reward_mapping['account_id'].apply(len)
ip_reward_mapping = ip_reward_mapping.sort_values('reward_account_count', ascending=False)

print(f"\n[IP 공유 통계]")
print(f"  보너스 받은 계정이 사용한 총 IP 수: {len(ip_reward_mapping)}")

shared_reward_ips = ip_reward_mapping[ip_reward_mapping['reward_account_count'] >= 2]
print(f"  2개 이상 계정이 보너스 받은 IP: {len(shared_reward_ips)} ({len(shared_reward_ips)/len(ip_reward_mapping)*100:.1f}%)")

if len(shared_reward_ips) > 0:
    print(f"\n[공유 IP 상세]")
    for _, row in shared_reward_ips.head(10).iterrows():
        accounts = list(row['account_id'])
        rewards = reward_df[reward_df['account_id'].isin(accounts)].groupby('account_id')['reward_amount'].sum()
        print(f"  IP {row['ip']}: {row['reward_account_count']}개 계정")
        for acc in accounts:
            print(f"    - {acc}: ${rewards.get(acc, 0):.2f}")

# 계정별 공유 IP 사용 여부
account_shared_ip_status = {}
for account in reward_accounts:
    account_ips = ip_df[ip_df['account_id'] == account]['ip'].tolist()
    shared_ips = [ip for ip in account_ips
                  if ip in shared_reward_ips['ip'].values]
    account_shared_ip_status[account] = {
        'total_ips': len(account_ips),
        'shared_ips': len(shared_ips),
        'is_shared': len(shared_ips) > 0
    }

shared_count = sum(1 for v in account_shared_ip_status.values() if v['is_shared'])
print(f"\n  공유 IP 사용 계정: {shared_count}/{len(reward_accounts)} ({shared_count/len(reward_accounts)*100:.1f}%)")

# ================================================================================
# 4. 보너스 수령 후 활동성 분석 (핵심 지표!)
# ================================================================================
print("\n" + "=" * 100)
print("[4] 보너스 수령 후 거래 활동성 분석 (Post-Reward Activity)")
print("=" * 100)

print("\n[연구 질문 4] 보너스 받고 나서 실제로 거래하는가?")
print("[실무 참고] 정상 사용자: 보너스 후 활발한 거래 | 악용: 보너스만 받고 비활성")

# 계정별 보너스 최초/최종 수령 시각
reward_timeline = reward_df.groupby('account_id').agg({
    'ts': ['min', 'max'],
    'reward_amount': 'sum'
}).reset_index()
reward_timeline.columns = ['account_id', 'first_reward_time', 'last_reward_time', 'total_reward']

# 계정별 거래 활동
trade_activity = trade_df.groupby('account_id').agg({
    'ts': ['min', 'max', 'count']
}).reset_index()
trade_activity.columns = ['account_id', 'first_trade_time', 'last_trade_time', 'trade_count']

# 병합
activity_analysis = reward_timeline.merge(trade_activity, on='account_id', how='left')

# 보너스 이후 거래 여부
activity_analysis['has_trades'] = ~activity_analysis['first_trade_time'].isna()
activity_analysis['trade_count'] = activity_analysis['trade_count'].fillna(0)

# 보너스 이후 얼마나 거래했는지
activity_analysis['days_after_reward'] = (
    activity_analysis['last_trade_time'] - activity_analysis['last_reward_time']
).dt.total_seconds() / 86400

# 보너스 전후 거래 비율
activity_analysis['trades_after_reward'] = activity_analysis.apply(
    lambda row: len(trade_df[
        (trade_df['account_id'] == row['account_id']) &
        (trade_df['ts'] > row['last_reward_time'])
    ]) if pd.notna(row['last_reward_time']) else 0,
    axis=1
)

activity_analysis['trade_ratio_after_reward'] = activity_analysis.apply(
    lambda row: row['trades_after_reward'] / row['trade_count']
    if row['trade_count'] > 0 else 0,
    axis=1
)

print(f"\n[보너스 수령 후 활동성 통계]")
has_trades = activity_analysis[activity_analysis['has_trades']].shape[0]
no_trades = activity_analysis[~activity_analysis['has_trades']].shape[0]
print(f"  거래 활동 있음: {has_trades} ({has_trades/len(activity_analysis)*100:.1f}%)")
print(f"  거래 활동 없음: {no_trades} ({no_trades/len(activity_analysis)*100:.1f}%) 🚨")

low_activity = activity_analysis[
    (activity_analysis['has_trades']) &
    (activity_analysis['trade_count'] < 10)
]
print(f"  거래 10회 미만: {len(low_activity)} ({len(low_activity)/len(activity_analysis)*100:.1f}%)")

# 보너스 이후 거래 비율
print(f"\n[보너스 이후 거래 비율]")
print(f"  평균: {activity_analysis['trade_ratio_after_reward'].mean()*100:.1f}%")
print(f"  중앙값: {activity_analysis['trade_ratio_after_reward'].median()*100:.1f}%")

inactive_after_reward = activity_analysis[activity_analysis['trade_ratio_after_reward'] < 0.1]
print(f"  보너스 이후 거래 < 10%: {len(inactive_after_reward)} 🚨")

# ================================================================================
# 5. 보너스 대비 거래량 비율 (Reward-to-Volume Ratio)
# ================================================================================
print("\n" + "=" * 100)
print("[5] 보너스 대비 거래량 비율 분석 (핵심 퀀트 지표)")
print("=" * 100)

print("\n[연구 질문 5] 받은 보너스 대비 실제 거래 규모는?")
print("[실무 참고] 정상: 보너스 << 거래량 | 악용: 보너스만 받고 거래 없음")

# 계정별 거래 금액
account_trade_volume = trade_df.groupby('account_id')['amount'].sum().reset_index()
account_trade_volume.columns = ['account_id', 'total_volume']

# 보너스와 거래량 비교
volume_analysis = account_reward_stats[['account_id', 'total_reward', 'reward_count']].merge(
    account_trade_volume, on='account_id', how='left'
)
volume_analysis['total_volume'] = volume_analysis['total_volume'].fillna(0)

# Reward-to-Volume Ratio (RVR)
# RVR이 높을수록 의심 (보너스만 많고 거래는 적음)
volume_analysis['reward_to_volume_ratio'] = volume_analysis.apply(
    lambda row: row['total_reward'] / row['total_volume']
    if row['total_volume'] > 0 else np.inf,
    axis=1
)

volume_analysis['has_volume'] = volume_analysis['total_volume'] > 0

print(f"\n[거래량 통계]")
print(f"  거래 활동 있음: {volume_analysis['has_volume'].sum()}")
print(f"  거래 활동 없음: {(~volume_analysis['has_volume']).sum()} 🚨")

# 거래 있는 계정의 RVR
active_accounts = volume_analysis[volume_analysis['has_volume']].copy()
active_accounts = active_accounts[active_accounts['reward_to_volume_ratio'] != np.inf]

if len(active_accounts) > 0:
    print(f"\n[Reward-to-Volume Ratio (거래 있는 계정)]")
    print(f"  평균: {active_accounts['reward_to_volume_ratio'].mean():.6f}")
    print(f"  중앙값: {active_accounts['reward_to_volume_ratio'].median():.6f}")
    print(f"  표준편차: {active_accounts['reward_to_volume_ratio'].std():.6f}")

    # 도메인 지식: RVR이 0.001 이상이면 의심
    # (예: 보너스 $10, 거래량 $10,000 → RVR = 0.001)
    high_rvr = active_accounts[active_accounts['reward_to_volume_ratio'] > 0.001]
    print(f"\n  고위험 (RVR > 0.001): {len(high_rvr)} 🚨")

# ================================================================================
# 6. 종합 위험 점수 산출 (Bonus Abuse Score)
# ================================================================================
print("\n" + "=" * 100)
print("[6] 보너스 악용 종합 위험 점수 산출")
print("=" * 100)

bonus_abuse_scores = []

for account in reward_accounts:
    # 1. 보너스 수령 금액 점수 (많을수록 의심)
    reward_data = account_reward_stats[account_reward_stats['account_id'] == account]
    if len(reward_data) > 0:
        total_reward = reward_data['total_reward'].values[0]
        reward_count = reward_data['reward_count'].values[0]
        # $100 이상이면 만점
        reward_amount_score = min(total_reward / 100, 1.0)
    else:
        total_reward = 0
        reward_count = 0
        reward_amount_score = 0

    # 2. 공유 IP 점수
    ip_status = account_shared_ip_status.get(account, {'is_shared': False})
    shared_ip_score = 1.0 if ip_status['is_shared'] else 0.0

    # 3. 보너스 후 비활성 점수
    activity_data = activity_analysis[activity_analysis['account_id'] == account]
    if len(activity_data) > 0:
        has_trades = activity_data['has_trades'].values[0]
        trade_count = activity_data['trade_count'].values[0]
        trade_ratio = activity_data['trade_ratio_after_reward'].values[0]

        if not has_trades:
            inactive_score = 1.0  # 거래 없음 = 만점
        elif trade_count < 10:
            inactive_score = 0.8  # 거래 매우 적음
        elif trade_ratio < 0.1:
            inactive_score = 0.9  # 보너스 후 거의 안 함
        else:
            inactive_score = max(1 - trade_ratio, 0)  # 거래 많을수록 낮음
    else:
        inactive_score = 0
        has_trades = False
        trade_count = 0
        trade_ratio = 0

    # 4. Reward-to-Volume Ratio 점수
    volume_data = volume_analysis[volume_analysis['account_id'] == account]
    if len(volume_data) > 0:
        rvr = volume_data['reward_to_volume_ratio'].values[0]
        total_volume = volume_data['total_volume'].values[0]

        if rvr == np.inf:
            rvr_score = 1.0  # 거래 없음
        else:
            rvr_score = min(rvr / 0.001, 1.0)  # 0.001 이상이면 만점
    else:
        rvr = 0
        total_volume = 0
        rvr_score = 0

    # 가중치: w1=0.25 (보너스 금액), w2=0.30 (공유 IP), w3=0.30 (비활성), w4=0.15 (RVR)
    total_score = (0.25 * reward_amount_score +
                   0.30 * shared_ip_score +
                   0.30 * inactive_score +
                   0.15 * rvr_score)

    bonus_abuse_scores.append({
        'account_id': account,
        'total_reward': total_reward,
        'reward_count': reward_count,
        'reward_amount_score': reward_amount_score,
        'shared_ip': ip_status['is_shared'],
        'shared_ip_score': shared_ip_score,
        'has_trades': has_trades,
        'trade_count': trade_count,
        'trade_ratio_after_reward': trade_ratio,
        'inactive_score': inactive_score,
        'total_volume': total_volume,
        'reward_to_volume_ratio': rvr if rvr != np.inf else 999,
        'rvr_score': rvr_score,
        'bonus_abuse_score': total_score
    })

bonus_df = pd.DataFrame(bonus_abuse_scores)
bonus_df = bonus_df.sort_values('bonus_abuse_score', ascending=False)

print(f"\n[종합 점수 분포]")
print(f"  평균: {bonus_df['bonus_abuse_score'].mean():.4f}")
print(f"  중앙값: {bonus_df['bonus_abuse_score'].median():.4f}")
print(f"  표준편차: {bonus_df['bonus_abuse_score'].std():.4f}")
print(f"  95th: {bonus_df['bonus_abuse_score'].quantile(0.95):.4f}")
print(f"  99th: {bonus_df['bonus_abuse_score'].quantile(0.99):.4f}")

# 위험도 분류
bonus_df['risk_level'] = pd.cut(
    bonus_df['bonus_abuse_score'],
    bins=[-np.inf, 0.3, 0.6, np.inf],
    labels=['Low', 'Medium', 'High']
)

risk_counts = bonus_df['risk_level'].value_counts()
print(f"\n[위험도 분류]")
for level in ['High', 'Medium', 'Low']:
    count = risk_counts.get(level, 0)
    print(f"  {level} Risk: {count}개 ({count/len(bonus_df)*100:.2f}%)")

high_risk = bonus_df[bonus_df['risk_level'] == 'High']
print(f"\n[고위험 계정 Top 10]")
for i, row in high_risk.head(10).iterrows():
    print(f"  {row['account_id']}: Score={row['bonus_abuse_score']:.4f}")
    print(f"    보너스=${row['total_reward']:.2f} ({int(row['reward_count'])}회), "
          f"공유IP={row['shared_ip']}, 거래={int(row['trade_count'])}회, RVR={row['reward_to_volume_ratio']:.6f}")

# ================================================================================
# 7. 결과 저장
# ================================================================================
print("\n" + "=" * 100)
print("[7] 결과 저장")
print("=" * 100)

bonus_df.to_csv('output/bonus_abuse/bonus_abuse_scores_all.csv', index=False)
print(f"✓ 전체 계정: output/bonus_abuse/bonus_abuse_scores_all.csv ({len(bonus_df)}개)")

high_risk.to_csv('output/bonus_abuse/bonus_abuse_high_risk.csv', index=False)
print(f"✓ 고위험 계정: output/bonus_abuse/bonus_abuse_high_risk.csv ({len(high_risk)}개)")

if len(shared_reward_ips) > 0:
    shared_reward_ips.to_csv('output/bonus_abuse/shared_reward_ips.csv', index=False)
    print(f"✓ 공유 IP: output/bonus_abuse/shared_reward_ips.csv ({len(shared_reward_ips)}개)")

activity_analysis.to_csv('output/bonus_abuse/reward_activity_analysis.csv', index=False)
print(f"✓ 활동성 분석: output/bonus_abuse/reward_activity_analysis.csv")

summary = pd.DataFrame({
    'Metric': [
        '보너스 받은 계정 수',
        '고위험 계정 수',
        '중위험 계정 수',
        '저위험 계정 수',
        '평균 점수',
        '총 보너스 지급액',
        '거래 없는 계정',
        '공유 IP 사용 계정',
        '보너스 후 비활성 계정'
    ],
    'Value': [
        len(bonus_df),
        risk_counts.get('High', 0),
        risk_counts.get('Medium', 0),
        risk_counts.get('Low', 0),
        f"{bonus_df['bonus_abuse_score'].mean():.4f}",
        f"${bonus_df['total_reward'].sum():.2f}",
        f"{(~bonus_df['has_trades']).sum()}",
        f"{bonus_df['shared_ip'].sum()}",
        f"{len(inactive_after_reward)}"
    ]
})
summary.to_csv('output/bonus_abuse/summary_statistics.csv', index=False)
print(f"✓ 요약 통계: output/bonus_abuse/summary_statistics.csv")

# ================================================================================
# 8. 시각화
# ================================================================================
print("\n" + "=" * 100)
print("[8] 시각화 생성")
print("=" * 100)

fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# 1. 보너스 금액 분포
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(account_reward_stats['total_reward'], bins=30, edgecolor='black', alpha=0.7, color='gold')
ax1.axvline(50, color='orange', linestyle='--', linewidth=2, label='의심 ($50)')
ax1.axvline(100, color='red', linestyle='--', linewidth=2, label='고위험 ($100)')
ax1.set_xlabel('총 보너스 금액 ($)', fontsize=11)
ax1.set_ylabel('계정 수', fontsize=11)
ax1.set_title('1. 보너스 금액 분포', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 보너스 수령 횟수
ax2 = fig.add_subplot(gs[0, 1])
count_dist = account_reward_stats['reward_count'].value_counts().sort_index()
ax2.bar(count_dist.index, count_dist.values, color='skyblue', edgecolor='black')
ax2.set_xlabel('보너스 수령 횟수', fontsize=11)
ax2.set_ylabel('계정 수', fontsize=11)
ax2.set_title('2. 보너스 수령 횟수 분포', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# 3. 거래 활동 여부
ax3 = fig.add_subplot(gs[0, 2])
trade_status = ['거래 있음', '거래 없음']
trade_counts = [has_trades, no_trades]
colors = ['green', 'red']
bars = ax3.bar(trade_status, trade_counts, color=colors, edgecolor='black')
ax3.set_ylabel('계정 수', fontsize=11)
ax3.set_title('3. 보너스 후 거래 활동', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
for bar, count in zip(bars, trade_counts):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{count}\n({count/(has_trades+no_trades)*100:.1f}%)',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# 4. 거래 횟수 분포
ax4 = fig.add_subplot(gs[1, 0])
active = activity_analysis[activity_analysis['has_trades']]
ax4.hist(active['trade_count'], bins=50, edgecolor='black', alpha=0.7, color='lightgreen', range=(0, 500))
ax4.axvline(10, color='orange', linestyle='--', linewidth=2, label='10회')
ax4.set_xlabel('거래 횟수', fontsize=11)
ax4.set_ylabel('계정 수', fontsize=11)
ax4.set_title('4. 거래 횟수 분포 (거래 있는 계정)', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Reward-to-Volume Ratio
ax5 = fig.add_subplot(gs[1, 1])
rvr_plot = active_accounts[active_accounts['reward_to_volume_ratio'] < 0.01]['reward_to_volume_ratio']
ax5.hist(rvr_plot, bins=50, edgecolor='black', alpha=0.7, color='coral')
ax5.axvline(0.001, color='red', linestyle='--', linewidth=2, label='위험 기준 (0.001)')
ax5.set_xlabel('Reward-to-Volume Ratio', fontsize=11)
ax5.set_ylabel('계정 수', fontsize=11)
ax5.set_title('5. 보너스/거래량 비율', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. 종합 점수 분포
ax6 = fig.add_subplot(gs[1, 2])
ax6.hist(bonus_df['bonus_abuse_score'], bins=30, edgecolor='black', alpha=0.7, color='crimson')
ax6.axvline(0.3, color='orange', linestyle='--', linewidth=2, label='Medium')
ax6.axvline(0.6, color='darkred', linestyle='--', linewidth=2, label='High')
ax6.set_xlabel('Bonus Abuse Score', fontsize=11)
ax6.set_ylabel('계정 수', fontsize=11)
ax6.set_title('6. 보너스 악용 종합 점수', fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 7. 위험도 분류
ax7 = fig.add_subplot(gs[2, 0])
risk_sorted = risk_counts.reindex(['High', 'Medium', 'Low'])
colors = ['red', 'orange', 'green']
bars = ax7.bar(range(len(risk_sorted)), risk_sorted.values, color=colors, edgecolor='black')
ax7.set_xticks(range(len(risk_sorted)))
ax7.set_xticklabels(risk_sorted.index, fontsize=11)
ax7.set_ylabel('계정 수', fontsize=11)
ax7.set_title('7. 위험도 분류', fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.3, axis='y')
for bar, count in zip(bars, risk_sorted.values):
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{count}\n({count/len(bonus_df)*100:.1f}%)',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# 8. 고위험 계정 Top 10
ax8 = fig.add_subplot(gs[2, 1:])
top_10 = bonus_df.head(10)
y_pos = np.arange(len(top_10))
bars = ax8.barh(y_pos, top_10['bonus_abuse_score'], color='darkred', edgecolor='black')
ax8.set_yticks(y_pos)
ax8.set_yticklabels([f"{acc[:15]}..." for acc in top_10['account_id']], fontsize=9)
ax8.set_xlabel('Bonus Abuse Score', fontsize=11)
ax8.set_title('8. 보너스 악용 의심 계정 Top 10', fontsize=12, fontweight='bold')
ax8.invert_yaxis()
ax8.grid(True, alpha=0.3, axis='x')
for i, (idx, row) in enumerate(top_10.iterrows()):
    ax8.text(row['bonus_abuse_score'], i, f" {row['bonus_abuse_score']:.3f}",
             va='center', fontsize=8, fontweight='bold')

fig.suptitle('보너스 악용(Bonus Abuse) 이상거래 탐지 분석',
             fontsize=18, fontweight='bold', y=0.998)

plt.savefig('output/bonus_abuse/bonus_abuse_visualization.png', dpi=300, bbox_inches='tight')
print(f"✓ 시각화: output/bonus_abuse/bonus_abuse_visualization.png")

print("\n" + "=" * 100)
print("분석 완료!")
print("=" * 100)
print("\n[핵심 공식]")
print("BonusAbuseScore = 0.25 × RewardAmount_Score + 0.30 × SharedIP_Score + 0.30 × Inactive_Score + 0.15 × RVR_Score")
print("\n임계값:")
print("  - Low Risk: Score < 0.3")
print("  - Medium Risk: 0.3 ≤ Score < 0.6")
print("  - High Risk: Score ≥ 0.6")
