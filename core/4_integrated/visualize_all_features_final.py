"""
모든 피처 시각화 (사용/미사용 포함)
통계적 임계값 기반
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 120)
print("전체 피처 시각화 및 통계 분석")
print("=" * 120)

# 데이터 로드
print("\n[1] 데이터 로드")
funding_df = pd.read_csv('data/Funding.csv')
trade_df = pd.read_csv('data/Trade.csv')
reward_df = pd.read_csv('data/Reward.csv')
ip_df = pd.read_csv('data/IP.csv')

funding_df['ts'] = pd.to_datetime(funding_df['ts'])
trade_df['ts'] = pd.to_datetime(trade_df['ts'])

print("✓ 데이터 로드 완료")

# 출력 디렉토리
output_dir = Path('output/final')

# 통계 저장용
stats_report = []

# ============================================================================
# Pattern 1: 펀딩피 차익거래
# ============================================================================
print("\n[2] Pattern 1: 펀딩피 차익거래 - 4개 피처 시각화")

# 1-1. 펀딩피 절댓값
funding_raw = funding_df.copy()
funding_raw['funding_fee_abs'] = funding_raw['funding_fee'].abs()

suspicious_threshold = funding_raw['funding_fee_abs'].quantile(0.90)
high_risk_threshold = funding_raw['funding_fee_abs'].quantile(0.95)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('1. 펀딩피 절댓값 (funding_fee_abs)', fontsize=16, fontweight='bold', y=1.02)

ax1 = axes[0]
counts, bins, patches = ax1.hist(funding_raw['funding_fee_abs'], bins=100, range=(0, 100),
                                  color='skyblue', edgecolor='black', alpha=0.7)
for i, patch in enumerate(patches):
    bin_center = (bins[i] + bins[i+1]) / 2
    if bin_center >= high_risk_threshold:
        patch.set_facecolor('red')
        patch.set_alpha(0.8)
    elif bin_center >= suspicious_threshold:
        patch.set_facecolor('orange')
        patch.set_alpha(0.8)

ax1.axvline(suspicious_threshold, color='orange', linestyle='--', linewidth=2.5,
            label=f'의심 (90th): ${suspicious_threshold:.2f}')
ax1.axvline(high_risk_threshold, color='red', linestyle='--', linewidth=2.5,
            label=f'고위험 (95th): ${high_risk_threshold:.2f}')

median_val = funding_raw['funding_fee_abs'].median()
mean_val = funding_raw['funding_fee_abs'].mean()
ax1.text(0.98, 0.97, f'중앙값: ${median_val:.2f}\n평균: ${mean_val:.2f}\n90th: ${suspicious_threshold:.2f}\n95th: ${high_risk_threshold:.2f}',
         transform=ax1.transAxes, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9), fontsize=11)
ax1.set_xlabel('펀딩피 절댓값 ($)', fontsize=12, fontweight='bold')
ax1.set_ylabel('거래 건수', fontsize=12, fontweight='bold')
ax1.set_title('전체 분포 (0~$100)', fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

ax2 = axes[1]
counts2, bins2, patches2 = ax2.hist(funding_raw['funding_fee_abs'], bins=100, range=(0, 50),
                                     color='lightblue', edgecolor='black', alpha=0.7)
for i, patch in enumerate(patches2):
    bin_center = (bins2[i] + bins2[i+1]) / 2
    if bin_center >= high_risk_threshold:
        patch.set_facecolor('crimson')
        patch.set_alpha(0.9)
    elif bin_center >= suspicious_threshold:
        patch.set_facecolor('darkorange')
        patch.set_alpha(0.9)
    else:
        patch.set_facecolor('mediumseagreen')
        patch.set_alpha(0.7)

ax2.axvline(suspicious_threshold, color='darkorange', linestyle='--', linewidth=3)
ax2.axvline(high_risk_threshold, color='crimson', linestyle='--', linewidth=3)

normal_pct = (funding_raw['funding_fee_abs'] < suspicious_threshold).mean() * 100
suspicious_pct = ((funding_raw['funding_fee_abs'] >= suspicious_threshold) &
                  (funding_raw['funding_fee_abs'] < high_risk_threshold)).mean() * 100
high_risk_pct = (funding_raw['funding_fee_abs'] >= high_risk_threshold).mean() * 100

ax2.text(0.02, 0.97, f'정상: {normal_pct:.1f}%\n의심: {suspicious_pct:.1f}%\n고위험: {high_risk_pct:.1f}%',
         transform=ax2.transAxes, verticalalignment='top', horizontalalignment='left',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9), fontsize=11, fontweight='bold')
ax2.set_xlabel('펀딩피 절댓값 ($)', fontsize=12, fontweight='bold')
ax2.set_ylabel('거래 건수', fontsize=12, fontweight='bold')
ax2.set_title('확대 (0~$50)', fontsize=13)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(output_dir / 'pattern1' / '1_funding_fee_abs.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 1_funding_fee_abs.png")

stats_report.append({
    'pattern': 'Pattern 1',
    'feature': '1. 펀딩피 절댓값',
    'feature_eng': 'funding_fee_abs',
    'data_source': 'Funding.csv의 funding_fee 절댓값',
    'mean': mean_val,
    'median': median_val,
    'suspicious': suspicious_threshold,
    'high_risk': high_risk_threshold,
    'status': '✓ 사용'
})

# 1-2. 포지션 보유시간 (대문자 수정)
print("\n  포지션 보유시간 계산 중...")
positions = []
for account_id in trade_df['account_id'].unique():
    acc_trades = trade_df[trade_df['account_id'] == account_id].sort_values('ts')

    for symbol in acc_trades['symbol'].unique():
        sym_trades = acc_trades[acc_trades['symbol'] == symbol].copy()
        position = 0
        open_time = None

        for idx, row in sym_trades.iterrows():
            if row['openclose'] == 'OPEN':  # 대문자로 수정
                if position == 0:
                    open_time = row['ts']
                position += row['qty'] if row['side'] == 'LONG' else -row['qty']
            else:  # CLOSE
                position += -row['qty'] if row['side'] == 'LONG' else row['qty']

                if abs(position) < 0.0001 and open_time is not None:
                    holding_minutes = (row['ts'] - open_time).total_seconds() / 60
                    if holding_minutes > 0:  # 유효한 값만
                        positions.append({
                            'account_id': account_id,
                            'holding_minutes': holding_minutes
                        })
                    open_time = None

if len(positions) > 0:
    position_df = pd.DataFrame(positions)
    user_holding = position_df.groupby('account_id')['holding_minutes'].mean().reset_index()

    suspicious_hold = user_holding['holding_minutes'].quantile(0.25)  # 짧을수록 위험
    high_risk_hold = user_holding['holding_minutes'].quantile(0.10)

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    fig.suptitle('2. 포지션 보유시간 (mean_holding_minutes)', fontsize=16, fontweight='bold', y=0.98)

    counts, bins, patches = ax.hist(user_holding['holding_minutes'], bins=100, range=(0, 500),
                                     color='lightblue', edgecolor='black', alpha=0.7)

    for i, patch in enumerate(patches):
        bin_center = (bins[i] + bins[i+1]) / 2
        if bin_center <= high_risk_hold:
            patch.set_facecolor('red')
            patch.set_alpha(0.8)
        elif bin_center <= suspicious_hold:
            patch.set_facecolor('orange')
            patch.set_alpha(0.8)
        else:
            patch.set_facecolor('green')
            patch.set_alpha(0.6)

    ax.axvline(high_risk_hold, color='red', linestyle='--', linewidth=2.5,
               label=f'고위험 (10th): {high_risk_hold:.1f}분')
    ax.axvline(suspicious_hold, color='orange', linestyle='--', linewidth=2.5,
               label=f'의심 (25th): {suspicious_hold:.1f}분')
    ax.axvline(480, color='blue', linestyle=':', linewidth=2,
               label='펀딩비 주기 (480분)', alpha=0.7)

    median_hold = user_holding['holding_minutes'].median()
    mean_hold = user_holding['holding_minutes'].mean()

    high_risk_pct = (user_holding['holding_minutes'] <= high_risk_hold).mean() * 100
    suspicious_pct = ((user_holding['holding_minutes'] > high_risk_hold) &
                      (user_holding['holding_minutes'] <= suspicious_hold)).mean() * 100
    normal_pct = (user_holding['holding_minutes'] > suspicious_hold).mean() * 100

    ax.text(0.98, 0.97,
            f'중앙값: {median_hold:.1f}분\n평균: {mean_hold:.1f}분\n10th: {high_risk_hold:.1f}분\n25th: {suspicious_hold:.1f}분\n\n'
            f'고위험: {high_risk_pct:.1f}%\n의심: {suspicious_pct:.1f}%\n정상: {normal_pct:.1f}%',
            transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9), fontsize=11, fontweight='bold')

    ax.set_xlabel('평균 보유시간 (분)', fontsize=12, fontweight='bold')
    ax.set_ylabel('계정 수', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_dir / 'pattern1' / '2_mean_holding_minutes.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 2_mean_holding_minutes.png")

    stats_report.append({
        'pattern': 'Pattern 1',
        'feature': '2. 포지션 보유시간',
        'feature_eng': 'mean_holding_minutes',
        'data_source': 'Trade.csv의 OPEN/CLOSE 타임스탬프 차이',
        'mean': mean_hold,
        'median': median_hold,
        'suspicious': suspicious_hold,
        'high_risk': high_risk_hold,
        'status': '✓ 사용'
    })
else:
    print("  ⚠️  포지션 데이터 없음")

# 1-3. 펀딩 시각 거래 집중도
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

suspicious_timing = timing_df['funding_timing_ratio'].quantile(0.75)
high_risk_timing = timing_df['funding_timing_ratio'].quantile(0.90)

fig, ax = plt.subplots(1, 1, figsize=(12, 7))
fig.suptitle('3. 펀딩 시각 거래 집중도 (funding_timing_ratio)', fontsize=16, fontweight='bold', y=0.98)

ax_hist = ax
ax_cdf = ax.twinx()

counts, bins, patches = ax_hist.hist(timing_df['funding_timing_ratio'], bins=50,
                                      color='lightblue', edgecolor='black', alpha=0.6)

for i, patch in enumerate(patches):
    bin_center = (bins[i] + bins[i+1]) / 2
    if bin_center >= high_risk_timing:
        patch.set_facecolor('red')
        patch.set_alpha(0.8)
    elif bin_center >= suspicious_timing:
        patch.set_facecolor('orange')
        patch.set_alpha(0.8)
    else:
        patch.set_facecolor('green')
        patch.set_alpha(0.6)

sorted_data = np.sort(timing_df['funding_timing_ratio'])
cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
ax_cdf.plot(sorted_data, cdf, linewidth=3, color='navy', label='CDF', alpha=0.8)

ax_hist.axvline(0.25, color='gray', linestyle=':', linewidth=2, label='이론적 랜덤 (25%)')
ax_hist.axvline(suspicious_timing, color='orange', linestyle='--', linewidth=2.5,
                label=f'의심 (75th): {suspicious_timing:.1%}')
ax_hist.axvline(high_risk_timing, color='red', linestyle='--', linewidth=2.5,
                label=f'고위험 (90th): {high_risk_timing:.1%}')

median_timing = timing_df['funding_timing_ratio'].median()
mean_timing = timing_df['funding_timing_ratio'].mean()
ax_hist.text(0.02, 0.97,
             f'평균: {mean_timing:.1%}\n중앙값: {median_timing:.1%}\n이론적: 25.0%\n75th: {suspicious_timing:.1%}\n90th: {high_risk_timing:.1%}',
             transform=ax_hist.transAxes, verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9), fontsize=11, fontweight='bold')

ax_hist.set_xlabel('펀딩 시각(±30분) 거래 비율', fontsize=12, fontweight='bold')
ax_hist.set_ylabel('계정 수', fontsize=12, fontweight='bold')
ax_cdf.set_ylabel('누적 확률', fontsize=12, fontweight='bold', color='navy')
ax_cdf.tick_params(axis='y', labelcolor='navy')

lines1, labels1 = ax_hist.get_legend_handles_labels()
lines2, labels2 = ax_cdf.get_legend_handles_labels()
ax_hist.legend(lines1 + lines2, labels1 + labels2, fontsize=10)
ax_hist.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(output_dir / 'pattern1' / '3_funding_timing_ratio.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 3_funding_timing_ratio.png")

stats_report.append({
    'pattern': 'Pattern 1',
    'feature': '3. 펀딩 시각 거래 집중도',
    'feature_eng': 'funding_timing_ratio',
    'data_source': 'Trade.csv의 ts 시각대(0,4,8,12,16,20시 ±30분) 거래 비율',
    'mean': mean_timing,
    'median': median_timing,
    'suspicious': suspicious_timing,
    'high_risk': high_risk_timing,
    'status': '✓ 사용'
})

# 1-4. 펀딩피 수익 비중
user_funding = funding_df.groupby('account_id')['funding_fee'].sum().reset_index()
user_funding['funding_fee_abs_sum'] = user_funding['funding_fee'].abs()
user_volume = trade_df.groupby('account_id')['amount'].sum().reset_index()
merged = pd.merge(user_funding, user_volume, on='account_id', how='outer').fillna(0)
merged['funding_profit_ratio'] = merged.apply(
    lambda x: min(x['funding_fee_abs_sum'] / (x['amount'] + 0.001), 1.0) if x['amount'] > 0 else 0, axis=1
)
merged['funding_profit_ratio'] = merged['funding_profit_ratio'].clip(0, 1)

suspicious_fpr = merged['funding_profit_ratio'].quantile(0.75)
high_risk_fpr = merged['funding_profit_ratio'].quantile(0.90)

fig, ax = plt.subplots(1, 1, figsize=(12, 7))
fig.suptitle('4. 펀딩피 수익 비중 (funding_profit_ratio) - 미사용', fontsize=16, fontweight='bold', y=0.98)

counts, bins, patches = ax.hist(merged['funding_profit_ratio'], bins=50,
                                 color='lightgray', edgecolor='black', alpha=0.7)

median_fpr = merged['funding_profit_ratio'].median()
mean_fpr = merged['funding_profit_ratio'].mean()

ax.axvline(suspicious_fpr, color='orange', linestyle='--', linewidth=2.5, alpha=0.5)
ax.axvline(high_risk_fpr, color='red', linestyle='--', linewidth=2.5, alpha=0.5)

ax.text(0.98, 0.97,
        f'평균: {mean_fpr:.2%}\n중앙값: {median_fpr:.2%}\n75th: {suspicious_fpr:.2%}\n90th: {high_risk_fpr:.2%}\n\n'
        f'❌ 미사용 이유:\n값이 너무 작고\n변별력 부족',
        transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.9), fontsize=11, fontweight='bold')

ax.set_xlabel('펀딩피 / 거래금액 비율', fontsize=12, fontweight='bold')
ax.set_ylabel('계정 수', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(output_dir / 'pattern1' / '4_funding_profit_ratio.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 4_funding_profit_ratio.png (미사용)")

stats_report.append({
    'pattern': 'Pattern 1',
    'feature': '4. 펀딩피 수익 비중',
    'feature_eng': 'funding_profit_ratio',
    'data_source': 'Funding.csv의 funding_fee 절댓값 / Trade.csv의 amount 합계',
    'mean': mean_fpr,
    'median': median_fpr,
    'suspicious': suspicious_fpr,
    'high_risk': high_risk_fpr,
    'status': '❌ 미사용 (변별력 부족)'
})

# ============================================================================
# Pattern 2: 조직적 거래
# ============================================================================
print("\n[3] Pattern 2: 조직적 거래 - 2개 피처 시각화")

# 2-1. IP 공유 비율
ip_counts = ip_df.groupby('ip').size().reset_index(name='account_count')
single_ip = (ip_counts['account_count'] == 1).sum()
two_accounts = (ip_counts['account_count'] == 2).sum()
three_plus_accounts = (ip_counts['account_count'] >= 3).sum()

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
fig.suptitle('1. IP 공유 비율 (ip_shared_ratio)', fontsize=16, fontweight='bold', y=0.95)

categories = ['단독 IP\n(정상)', '2개 계정 공유\n(의심)', '3개 이상 공유\n(고위험)']
counts = [single_ip, two_accounts, three_plus_accounts]
colors = ['mediumseagreen', 'orange', 'crimson']

bars = ax.bar(categories, counts, color=colors, edgecolor='black', linewidth=2, alpha=0.8)

total = sum(counts)
for i, (bar, count) in enumerate(zip(bars, counts)):
    height = bar.get_height()
    pct = count / total * 100
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{count:,}개\n({pct:.1f}%)',
            ha='center', va='bottom', fontsize=13, fontweight='bold')

ax.set_ylabel('IP 개수', fontsize=13, fontweight='bold')
ax.set_title(f'총 {total:,}개 IP 분석', fontsize=12)
ax.grid(axis='y', alpha=0.3, linestyle='--')

ax.text(0.5, 0.95,
        f'✓ {single_ip}개 ({single_ip/total*100:.1f}%)가 정상\n'
        f'⚠ {two_accounts + three_plus_accounts}개 ({(two_accounts + three_plus_accounts)/total*100:.1f}%)가 의심',
        transform=ax.transAxes, verticalalignment='top', horizontalalignment='center',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9), fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'pattern2' / '1_ip_shared_ratio.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 1_ip_shared_ratio.png")

stats_report.append({
    'pattern': 'Pattern 2',
    'feature': '1. IP 공유 비율',
    'feature_eng': 'ip_shared_ratio',
    'data_source': 'IP.csv의 ip별 account_count',
    'mean': (two_accounts + three_plus_accounts) / total,
    'median': 0,
    'suspicious': '2개 계정',
    'high_risk': '3개 이상',
    'status': '✓ 사용'
})

# 2-2. 평균 레버리지
user_leverage = trade_df.groupby('account_id')['leverage'].mean().reset_index(name='mean_leverage')

suspicious_lev = user_leverage['mean_leverage'].quantile(0.75)
high_risk_lev = user_leverage['mean_leverage'].quantile(0.90)

fig, ax = plt.subplots(1, 1, figsize=(12, 7))
fig.suptitle('2. 평균 레버리지 (mean_leverage)', fontsize=16, fontweight='bold', y=0.98)

counts, bins, patches = ax.hist(user_leverage['mean_leverage'], bins=50, range=(0, 60),
                                 color='lightblue', edgecolor='black', alpha=0.7)

for i, patch in enumerate(patches):
    bin_center = (bins[i] + bins[i+1]) / 2
    if bin_center >= high_risk_lev:
        patch.set_facecolor('red')
        patch.set_alpha(0.8)
    elif bin_center >= suspicious_lev:
        patch.set_facecolor('orange')
        patch.set_alpha(0.8)
    else:
        patch.set_facecolor('green')
        patch.set_alpha(0.6)

ax.axvline(20, color='blue', linestyle=':', linewidth=2, label='금융권 기준 (20배)', alpha=0.7)
ax.axvline(suspicious_lev, color='orange', linestyle='--', linewidth=2.5,
           label=f'의심 (75th): {suspicious_lev:.1f}배')
ax.axvline(high_risk_lev, color='red', linestyle='--', linewidth=2.5,
           label=f'고위험 (90th): {high_risk_lev:.1f}배')

median_lev = user_leverage['mean_leverage'].median()
mean_lev = user_leverage['mean_leverage'].mean()
normal_pct = (user_leverage['mean_leverage'] < suspicious_lev).mean() * 100
suspicious_pct = ((user_leverage['mean_leverage'] >= suspicious_lev) &
                  (user_leverage['mean_leverage'] < high_risk_lev)).mean() * 100
high_risk_pct = (user_leverage['mean_leverage'] >= high_risk_lev).mean() * 100

ax.text(0.98, 0.97,
        f'중앙값: {median_lev:.1f}배\n평균: {mean_lev:.1f}배\n\n'
        f'정상: {normal_pct:.1f}%\n의심: {suspicious_pct:.1f}%\n고위험: {high_risk_pct:.1f}%',
        transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9), fontsize=11, fontweight='bold')

ax.set_xlabel('평균 레버리지 (배)', fontsize=12, fontweight='bold')
ax.set_ylabel('계정 수', fontsize=12, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(output_dir / 'pattern2' / '2_mean_leverage.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 2_mean_leverage.png")

stats_report.append({
    'pattern': 'Pattern 2',
    'feature': '2. 평균 레버리지',
    'feature_eng': 'mean_leverage',
    'data_source': 'Trade.csv의 leverage 평균',
    'mean': mean_lev,
    'median': median_lev,
    'suspicious': suspicious_lev,
    'high_risk': high_risk_lev,
    'status': '✓ 사용'
})

# ============================================================================
# Pattern 3: 보너스 악용
# ============================================================================
print("\n[4] Pattern 3: 보너스 악용 - 3개 피처 시각화")

# 3-1. 총 보너스 수령액
user_reward = reward_df.groupby('account_id')['reward_amount'].sum().reset_index(name='total_reward')

suspicious_reward = user_reward['total_reward'].quantile(0.75)
high_risk_reward = user_reward['total_reward'].quantile(0.90)

fig, ax = plt.subplots(1, 1, figsize=(12, 7))
fig.suptitle('1. 총 보너스 수령액 (total_reward)', fontsize=16, fontweight='bold', y=0.98)

counts, bins, patches = ax.hist(user_reward['total_reward'], bins=50,
                                 color='lightblue', edgecolor='black', alpha=0.7)

for i, patch in enumerate(patches):
    bin_center = (bins[i] + bins[i+1]) / 2
    if bin_center >= high_risk_reward:
        patch.set_facecolor('red')
        patch.set_alpha(0.8)
    elif bin_center >= suspicious_reward:
        patch.set_facecolor('orange')
        patch.set_alpha(0.8)
    else:
        patch.set_facecolor('green')
        patch.set_alpha(0.6)

ax.axvline(suspicious_reward, color='orange', linestyle='--', linewidth=2.5,
           label=f'의심 (75th): ${suspicious_reward:.2f}')
ax.axvline(high_risk_reward, color='red', linestyle='--', linewidth=2.5,
           label=f'고위험 (90th): ${high_risk_reward:.2f}')
ax.axvline(40, color='gray', linestyle=':', linewidth=2, label='중앙값 ($40)', alpha=0.7)

median_reward = user_reward['total_reward'].median()
mean_reward = user_reward['total_reward'].mean()

ax.text(0.98, 0.97,
        f'중앙값: ${median_reward:.2f}\n평균: ${mean_reward:.2f}\n75th: ${suspicious_reward:.2f}\n90th: ${high_risk_reward:.2f}',
        transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9), fontsize=11, fontweight='bold')

ax.set_xlabel('총 보너스 수령액 ($)', fontsize=12, fontweight='bold')
ax.set_ylabel('계정 수', fontsize=12, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(output_dir / 'pattern3' / '1_total_reward.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 1_total_reward.png")

stats_report.append({
    'pattern': 'Pattern 3',
    'feature': '1. 총 보너스 수령액',
    'feature_eng': 'total_reward',
    'data_source': 'Reward.csv의 reward_amount 합계',
    'mean': mean_reward,
    'median': median_reward,
    'suspicious': suspicious_reward,
    'high_risk': high_risk_reward,
    'status': '✓ 사용'
})

# 3-2. IP 공유 (보너스 계정)
reward_accounts = set(reward_df['account_id'].unique())
reward_ip = ip_df[ip_df['account_id'].isin(reward_accounts)].copy()
ip_counts_reward = reward_ip.groupby('ip').size().reset_index(name='account_count')

single_reward = (ip_counts_reward['account_count'] == 1).sum()
two_reward = (ip_counts_reward['account_count'] == 2).sum()
three_plus_reward = (ip_counts_reward['account_count'] >= 3).sum()

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
fig.suptitle('2. 보너스 계정 IP 공유 (shared_ip)', fontsize=16, fontweight='bold', y=0.95)

categories = ['단독 IP\n(정상)', '2개 계정 공유\n(의심)', '3개 이상 공유\n(고위험)']
counts = [single_reward, two_reward, three_plus_reward]
colors = ['mediumseagreen', 'orange', 'crimson']

bars = ax.bar(categories, counts, color=colors, edgecolor='black', linewidth=2, alpha=0.8)

total_reward_ips = sum(counts)
for i, (bar, count) in enumerate(zip(bars, counts)):
    height = bar.get_height()
    pct = count / total_reward_ips * 100 if total_reward_ips > 0 else 0
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{count:,}개\n({pct:.1f}%)',
            ha='center', va='bottom', fontsize=13, fontweight='bold')

ax.set_ylabel('IP 개수', fontsize=13, fontweight='bold')
ax.set_title(f'총 {total_reward_ips:,}개 보너스 계정 IP', fontsize=12)
ax.grid(axis='y', alpha=0.3, linestyle='--')

if two_reward + three_plus_reward > 0:
    ax.text(0.5, 0.95,
            f'⚠ {two_reward + three_plus_reward}개 IP가 다중 계정으로 보너스 수령',
            transform=ax.transAxes, verticalalignment='top', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9), fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'pattern3' / '2_shared_ip.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 2_shared_ip.png")

stats_report.append({
    'pattern': 'Pattern 3',
    'feature': '2. IP 공유 (보너스)',
    'feature_eng': 'shared_ip',
    'data_source': 'Reward.csv와 IP.csv 결합 (IP별 보너스 계정 수)',
    'mean': (two_reward + three_plus_reward) / total_reward_ips if total_reward_ips > 0 else 0,
    'median': 0,
    'suspicious': '2개 계정',
    'high_risk': '3개 이상',
    'status': '✓ 사용'
})

# 3-3. 거래 활동 여부 - 미사용
all_accounts = set(trade_df['account_id'].unique())
trading_reward = reward_accounts.intersection(all_accounts)
no_trading_reward = reward_accounts - all_accounts

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
fig.suptitle('3. 거래 활동 여부 (has_trades) - 미사용', fontsize=16, fontweight='bold', y=0.95)

categories = ['거래 있음', '거래 없음']
counts = [len(trading_reward), len(no_trading_reward)]
colors = ['mediumseagreen', 'crimson']

bars = ax.bar(categories, counts, color=colors, edgecolor='black', linewidth=2, alpha=0.8)

total_reward_acc = len(reward_accounts)
for i, (bar, count) in enumerate(zip(bars, counts)):
    height = bar.get_height()
    pct = count / total_reward_acc * 100
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{count:,}개\n({pct:.1f}%)',
            ha='center', va='bottom', fontsize=13, fontweight='bold')

ax.set_ylabel('계정 수', fontsize=13, fontweight='bold')
ax.set_title(f'총 {total_reward_acc:,}개 보너스 계정', fontsize=12)
ax.grid(axis='y', alpha=0.3, linestyle='--')

ax.text(0.5, 0.95,
        f'❌ 미사용 이유:\n100%가 거래 있음\n변별력 없음',
        transform=ax.transAxes, verticalalignment='top', horizontalalignment='center',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.9), fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'pattern3' / '3_has_trades.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 3_has_trades.png (미사용)")

stats_report.append({
    'pattern': 'Pattern 3',
    'feature': '3. 거래 활동 여부',
    'feature_eng': 'has_trades',
    'data_source': 'Reward.csv와 Trade.csv 계정 매칭',
    'mean': len(trading_reward) / total_reward_acc,
    'median': 1,
    'suspicious': 'N/A',
    'high_risk': 'N/A',
    'status': '❌ 미사용 (변별력 없음, 100% 거래 있음)'
})

print("\n" + "=" * 120)
print("✅ 전체 피처 시각화 완료!")
print("=" * 120)

# 통계 리포트 저장
stats_df = pd.DataFrame(stats_report)
stats_df.to_csv(output_dir / 'feature_statistics_summary.csv', index=False, encoding='utf-8-sig')
print(f"\n통계 요약 저장: {output_dir / 'feature_statistics_summary.csv'}")
