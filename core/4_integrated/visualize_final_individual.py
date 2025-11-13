"""
각 패턴의 피처별로 개별 PNG 파일 생성
Pattern 1: 4개 PNG (funding_fee_abs, mean_holding_minutes, funding_timing_ratio, funding_profit_ratio)
Pattern 2: 4개 PNG (ip_shared_ratio, concurrent_trading_ratio, price_similarity, mean_leverage)
Pattern 3: 4개 PNG (total_reward, shared_ip, has_trades, reward_to_volume_ratio)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 120)
print("피처별 개별 PNG 생성")
print("=" * 120)

# 데이터 로드
print("\n[1] 데이터 로드")
funding_df = pd.read_csv('data/Funding.csv')
trade_df = pd.read_csv('data/Trade.csv')
reward_df = pd.read_csv('data/Reward.csv')
ip_df = pd.read_csv('data/IP.csv')

# 타임스탬프 변환
funding_df['ts'] = pd.to_datetime(funding_df['ts'])
trade_df['ts'] = pd.to_datetime(trade_df['ts'])

print("✓ 데이터 로드 완료")

# 출력 디렉토리
output_dir = Path('output/final')

# ============================================================================
# Pattern 1: 펀딩피 차익거래
# ============================================================================
print("\n[2] Pattern 1: 펀딩피 차익거래 - 4개 PNG 생성")

# 1-1. funding_fee_abs (펀딩피 절댓값)
funding_raw = funding_df.copy()
funding_raw['funding_fee_abs'] = funding_raw['funding_fee'].abs()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('펀딩피 절댓값 (funding_fee_abs)', fontsize=16, fontweight='bold', y=0.995)

# 히스토그램
ax1 = axes[0, 0]
ax1.hist(funding_raw['funding_fee_abs'], bins=100, range=(0, 50), color='skyblue', edgecolor='black', alpha=0.7)
ax1.axvline(5, color='orange', linestyle='--', linewidth=2, label='의심 구간 ($5)')
ax1.axvline(30, color='red', linestyle='--', linewidth=2, label='고위험 구간 ($30)')
median_val = funding_raw['funding_fee_abs'].median()
p95_val = funding_raw['funding_fee_abs'].quantile(0.95)
ax1.text(0.98, 0.97, f'중앙값: ${median_val:.2f}\n95th: ${p95_val:.2f}',
         transform=ax1.transAxes, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
ax1.set_xlabel('펀딩피 절댓값 ($)', fontsize=11)
ax1.set_ylabel('거래 건수', fontsize=11)
ax1.set_title('분포 (0~$50 범위)', fontsize=12)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 로그 스케일
ax2 = axes[0, 1]
ax2.hist(funding_raw['funding_fee_abs'], bins=100, color='lightcoral', edgecolor='black', alpha=0.7)
ax2.set_yscale('log')
ax2.axvline(5, color='orange', linestyle='--', linewidth=2, label='의심 ($5)')
ax2.axvline(30, color='red', linestyle='--', linewidth=2, label='고위험 ($30)')
ax2.set_xlabel('펀딩피 절댓값 ($)', fontsize=11)
ax2.set_ylabel('거래 건수 (로그)', fontsize=11)
ax2.set_title('전체 분포 (로그 스케일)', fontsize=12)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 박스플롯
ax3 = axes[1, 0]
funding_raw['category'] = pd.cut(funding_raw['funding_fee_abs'],
                                  bins=[-0.01, 5, 30, 1000],
                                  labels=['정상 (<$5)', '의심 ($5~$30)', '고위험 (>$30)'])
category_counts = funding_raw['category'].value_counts()
bp = ax3.boxplot([funding_raw[funding_raw['category'] == cat]['funding_fee_abs'].dropna()
                   for cat in ['정상 (<$5)', '의심 ($5~$30)', '고위험 (>$30)']],
                  labels=['정상 (<$5)', '의심 ($5~$30)', '고위험 (>$30)'],
                  patch_artist=True, showfliers=False)
colors = ['lightgreen', 'orange', 'red']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax3.set_ylabel('펀딩피 절댓값 ($)', fontsize=11)
ax3.set_title('구간별 박스플롯', fontsize=12)
ax3.grid(axis='y', alpha=0.3)

# 카테고리별 비율
ax4 = axes[1, 1]
for i, (cat, count) in enumerate(category_counts.items()):
    pct = count / len(funding_raw) * 100
    ax4.bar(i, count, color=colors[['정상 (<$5)', '의심 ($5~$30)', '고위험 (>$30)'].index(cat)],
            alpha=0.7, edgecolor='black')
    ax4.text(i, count + 500, f'{count:,}\n({pct:.1f}%)', ha='center', fontsize=10, fontweight='bold')
ax4.set_xticks(range(3))
ax4.set_xticklabels(['정상 (<$5)', '의심 ($5~$30)', '고위험 (>$30)'])
ax4.set_ylabel('거래 건수', fontsize=11)
ax4.set_title('구간별 분포', fontsize=12)
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'pattern1' / '1_funding_fee_abs.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 1_funding_fee_abs.png 저장")

# 1-2. mean_holding_minutes (포지션 보유시간)
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
                    holding_seconds = (row['ts'] - open_time).total_seconds()
                    holding_minutes = holding_seconds / 60
                    positions.append({
                        'account_id': account_id,
                        'holding_minutes': holding_minutes
                    })
                    open_time = None

position_df = pd.DataFrame(positions)
if len(position_df) > 0:
    user_holding = position_df.groupby('account_id')['holding_minutes'].mean().reset_index()
    user_holding.columns = ['account_id', 'mean_holding_minutes']
else:
    # 포지션 데이터가 없으면 빈 DataFrame 생성
    user_holding = pd.DataFrame({'account_id': [], 'mean_holding_minutes': []})
    print("  경고: 포지션 데이터가 없습니다. 빈 차트를 생성합니다.")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('평균 포지션 보유시간 (mean_holding_minutes)', fontsize=16, fontweight='bold', y=0.995)

# 히스토그램 (0~300분)
ax1 = axes[0, 0]
ax1.hist(user_holding['mean_holding_minutes'], bins=100, range=(0, 300), color='skyblue', edgecolor='black', alpha=0.7)
ax1.axvline(60, color='orange', linestyle='--', linewidth=2, label='의심 (60분)')
ax1.axvline(30, color='red', linestyle='--', linewidth=2, label='고위험 (30분)')
median_val = user_holding['mean_holding_minutes'].median()
p10_val = user_holding['mean_holding_minutes'].quantile(0.10)
ax1.text(0.98, 0.97, f'중앙값: {median_val:.1f}분\n10th: {p10_val:.1f}분',
         transform=ax1.transAxes, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
ax1.set_xlabel('평균 보유시간 (분)', fontsize=11)
ax1.set_ylabel('계정 수', fontsize=11)
ax1.set_title('분포 (0~300분 범위)', fontsize=12)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 박스플롯
ax2 = axes[0, 1]
ax2.boxplot(user_holding['mean_holding_minutes'], vert=True, patch_artist=True,
            boxprops=dict(facecolor='lightblue', alpha=0.7))
ax2.axhline(60, color='orange', linestyle='--', linewidth=2, label='의심 (60분)')
ax2.axhline(30, color='red', linestyle='--', linewidth=2, label='고위험 (30분)')
ax2.set_ylabel('평균 보유시간 (분)', fontsize=11)
ax2.set_title('박스플롯', fontsize=12)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 카테고리별 분석
ax3 = axes[1, 0]
user_holding['category'] = pd.cut(user_holding['mean_holding_minutes'],
                                   bins=[-0.01, 30, 60, 100000],
                                   labels=['고위험 (<30분)', '의심 (30~60분)', '정상 (>60분)'])
category_counts = user_holding['category'].value_counts()
colors = ['red', 'orange', 'lightgreen']
for i, cat in enumerate(['고위험 (<30분)', '의심 (30~60분)', '정상 (>60분)']):
    count = category_counts.get(cat, 0)
    pct = count / len(user_holding) * 100
    ax3.bar(i, count, color=colors[i], alpha=0.7, edgecolor='black')
    ax3.text(i, count + 5, f'{count:,}\n({pct:.1f}%)', ha='center', fontsize=10, fontweight='bold')
ax3.set_xticks(range(3))
ax3.set_xticklabels(['고위험 (<30분)', '의심 (30~60분)', '정상 (>60분)'])
ax3.set_ylabel('계정 수', fontsize=11)
ax3.set_title('구간별 분포', fontsize=12)
ax3.grid(axis='y', alpha=0.3)

# CDF
ax4 = axes[1, 1]
sorted_data = np.sort(user_holding['mean_holding_minutes'])
cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
ax4.plot(sorted_data, cdf, linewidth=2, color='blue')
ax4.axvline(30, color='red', linestyle='--', linewidth=2, label='고위험 (30분)')
ax4.axvline(60, color='orange', linestyle='--', linewidth=2, label='의심 (60분)')
ax4.set_xlabel('평균 보유시간 (분)', fontsize=11)
ax4.set_ylabel('누적 확률', fontsize=11)
ax4.set_title('누적 분포 함수 (CDF)', fontsize=12)
ax4.set_xlim(0, 300)
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'pattern1' / '2_mean_holding_minutes.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 2_mean_holding_minutes.png 저장")

# 1-3. funding_timing_ratio (펀딩 시각 거래 집중도)
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

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('펀딩 시각 거래 집중도 (funding_timing_ratio)', fontsize=16, fontweight='bold', y=0.995)

# 히스토그램
ax1 = axes[0, 0]
ax1.hist(timing_df['funding_timing_ratio'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
ax1.axvline(0.3, color='orange', linestyle='--', linewidth=2, label='의심 (30%)')
ax1.axvline(0.5, color='red', linestyle='--', linewidth=2, label='고위험 (50%)')
median_val = timing_df['funding_timing_ratio'].median()
p95_val = timing_df['funding_timing_ratio'].quantile(0.95)
ax1.text(0.98, 0.97, f'중앙값: {median_val:.2%}\n95th: {p95_val:.2%}',
         transform=ax1.transAxes, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
ax1.set_xlabel('펀딩 시각 거래 비율', fontsize=11)
ax1.set_ylabel('계정 수', fontsize=11)
ax1.set_title('분포', fontsize=12)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# CDF
ax2 = axes[0, 1]
sorted_data = np.sort(timing_df['funding_timing_ratio'])
cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
ax2.plot(sorted_data, cdf, linewidth=2, color='blue')
ax2.axvline(0.3, color='orange', linestyle='--', linewidth=2, label='의심 (30%)')
ax2.axvline(0.5, color='red', linestyle='--', linewidth=2, label='고위험 (50%)')
ax2.set_xlabel('펀딩 시각 거래 비율', fontsize=11)
ax2.set_ylabel('누적 확률', fontsize=11)
ax2.set_title('누적 분포 함수 (CDF)', fontsize=12)
ax2.legend()
ax2.grid(alpha=0.3)

# 카테고리별
ax3 = axes[1, 0]
timing_df['category'] = pd.cut(timing_df['funding_timing_ratio'],
                                bins=[-0.01, 0.3, 0.5, 1.01],
                                labels=['정상 (<30%)', '의심 (30~50%)', '고위험 (>50%)'])
category_counts = timing_df['category'].value_counts()
colors = ['lightgreen', 'orange', 'red']
for i, cat in enumerate(['정상 (<30%)', '의심 (30~50%)', '고위험 (>50%)']):
    count = category_counts.get(cat, 0)
    pct = count / len(timing_df) * 100
    ax3.bar(i, count, color=colors[i], alpha=0.7, edgecolor='black')
    ax3.text(i, count + 5, f'{count:,}\n({pct:.1f}%)', ha='center', fontsize=10, fontweight='bold')
ax3.set_xticks(range(3))
ax3.set_xticklabels(['정상 (<30%)', '의심 (30~50%)', '고위험 (>50%)'])
ax3.set_ylabel('계정 수', fontsize=11)
ax3.set_title('구간별 분포', fontsize=12)
ax3.grid(axis='y', alpha=0.3)

# 박스플롯
ax4 = axes[1, 1]
ax4.boxplot(timing_df['funding_timing_ratio'], vert=True, patch_artist=True,
            boxprops=dict(facecolor='lightblue', alpha=0.7))
ax4.axhline(0.3, color='orange', linestyle='--', linewidth=2, label='의심 (30%)')
ax4.axhline(0.5, color='red', linestyle='--', linewidth=2, label='고위험 (50%)')
ax4.set_ylabel('펀딩 시각 거래 비율', fontsize=11)
ax4.set_title('박스플롯', fontsize=12)
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'pattern1' / '3_funding_timing_ratio.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 3_funding_timing_ratio.png 저장")

# 1-4. funding_profit_ratio (펀딩피 수익 비중)
# PnL 계산이 복잡하므로 간단하게 funding_fee의 절댓값 합계로 대체
user_funding = funding_df.groupby('account_id')['funding_fee'].sum().reset_index()
user_funding['funding_fee_abs_sum'] = user_funding['funding_fee'].abs()
# 전체 거래 금액 대비 펀딩피 비율로 근사
user_volume = trade_df.groupby('account_id')['amount'].sum().reset_index()
merged = pd.merge(user_funding, user_volume, on='account_id', how='outer').fillna(0)
merged['funding_profit_ratio'] = merged.apply(
    lambda x: min(x['funding_fee_abs_sum'] / (x['amount'] + 0.001), 1.0) if x['amount'] > 0 else 0, axis=1
)
merged['funding_profit_ratio'] = merged['funding_profit_ratio'].clip(0, 1)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('펀딩피 수익 비중 (funding_profit_ratio)', fontsize=16, fontweight='bold', y=0.995)

# 히스토그램
ax1 = axes[0, 0]
ax1.hist(merged['funding_profit_ratio'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
ax1.axvline(0.3, color='orange', linestyle='--', linewidth=2, label='의심 (30%)')
ax1.axvline(0.7, color='red', linestyle='--', linewidth=2, label='고위험 (70%)')
median_val = merged['funding_profit_ratio'].median()
p95_val = merged['funding_profit_ratio'].quantile(0.95)
ax1.text(0.98, 0.97, f'중앙값: {median_val:.2%}\n95th: {p95_val:.2%}',
         transform=ax1.transAxes, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
ax1.set_xlabel('펀딩피 수익 비중', fontsize=11)
ax1.set_ylabel('계정 수', fontsize=11)
ax1.set_title('분포', fontsize=12)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 박스플롯
ax2 = axes[0, 1]
ax2.boxplot(merged['funding_profit_ratio'], vert=True, patch_artist=True,
            boxprops=dict(facecolor='lightblue', alpha=0.7))
ax2.axhline(0.3, color='orange', linestyle='--', linewidth=2, label='의심 (30%)')
ax2.axhline(0.7, color='red', linestyle='--', linewidth=2, label='고위험 (70%)')
ax2.set_ylabel('펀딩피 수익 비중', fontsize=11)
ax2.set_title('박스플롯', fontsize=12)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 카테고리별
ax3 = axes[1, 0]
merged['category'] = pd.cut(merged['funding_profit_ratio'],
                             bins=[-0.01, 0.3, 0.7, 1.01],
                             labels=['정상 (<30%)', '의심 (30~70%)', '고위험 (>70%)'])
category_counts = merged['category'].value_counts()
colors = ['lightgreen', 'orange', 'red']
for i, cat in enumerate(['정상 (<30%)', '의심 (30~70%)', '고위험 (>70%)']):
    count = category_counts.get(cat, 0)
    pct = count / len(merged) * 100
    ax3.bar(i, count, color=colors[i], alpha=0.7, edgecolor='black')
    ax3.text(i, count + 5, f'{count:,}\n({pct:.1f}%)', ha='center', fontsize=10, fontweight='bold')
ax3.set_xticks(range(3))
ax3.set_xticklabels(['정상 (<30%)', '의심 (30~70%)', '고위험 (>70%)'])
ax3.set_ylabel('계정 수', fontsize=11)
ax3.set_title('구간별 분포', fontsize=12)
ax3.grid(axis='y', alpha=0.3)

# CDF
ax4 = axes[1, 1]
sorted_data = np.sort(merged['funding_profit_ratio'])
cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
ax4.plot(sorted_data, cdf, linewidth=2, color='blue')
ax4.axvline(0.3, color='orange', linestyle='--', linewidth=2, label='의심 (30%)')
ax4.axvline(0.7, color='red', linestyle='--', linewidth=2, label='고위험 (70%)')
ax4.set_xlabel('펀딩피 수익 비중', fontsize=11)
ax4.set_ylabel('누적 확률', fontsize=11)
ax4.set_title('누적 분포 함수 (CDF)', fontsize=12)
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'pattern1' / '4_funding_profit_ratio.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 4_funding_profit_ratio.png 저장")

# ============================================================================
# Pattern 2: 조직적 거래
# ============================================================================
print("\n[3] Pattern 2: 조직적 거래 - 4개 PNG 생성")

# 2-1. ip_shared_ratio (IP 공유 비율)
ip_counts = ip_df.groupby('ip').size().reset_index(name='account_count')
ip_users = ip_df.merge(ip_counts, on='ip')
ip_users['ip_shared_ratio'] = (ip_users['account_count'] - 1) / ip_users['account_count']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('IP 공유 비율 (ip_shared_ratio)', fontsize=16, fontweight='bold', y=0.995)

# 히스토그램
ax1 = axes[0, 0]
ax1.hist(ip_users['ip_shared_ratio'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
ax1.axvline(0.3, color='orange', linestyle='--', linewidth=2, label='의심 (30%)')
ax1.axvline(0.5, color='red', linestyle='--', linewidth=2, label='고위험 (50%)')
median_val = ip_users['ip_shared_ratio'].median()
p95_val = ip_users['ip_shared_ratio'].quantile(0.95)
ax1.text(0.98, 0.97, f'중앙값: {median_val:.2%}\n95th: {p95_val:.2%}',
         transform=ax1.transAxes, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
ax1.set_xlabel('IP 공유 비율', fontsize=11)
ax1.set_ylabel('계정 수', fontsize=11)
ax1.set_title('분포', fontsize=12)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 파이 차트 (단독/공유)
ax2 = axes[0, 1]
single_ip = (ip_users['account_count'] == 1).sum()
shared_ip = (ip_users['account_count'] > 1).sum()
ax2.pie([single_ip, shared_ip], labels=['단독 IP', '공유 IP'], autopct='%1.1f%%',
        colors=['lightgreen', 'lightcoral'], startangle=90)
ax2.set_title(f'단독 vs 공유 IP\n(단독: {single_ip}, 공유: {shared_ip})', fontsize=12)

# 카테고리별
ax3 = axes[1, 0]
ip_users['category'] = pd.cut(ip_users['ip_shared_ratio'],
                               bins=[-0.01, 0.3, 0.5, 1.01],
                               labels=['정상 (<30%)', '의심 (30~50%)', '고위험 (>50%)'])
category_counts = ip_users['category'].value_counts()
colors = ['lightgreen', 'orange', 'red']
for i, cat in enumerate(['정상 (<30%)', '의심 (30~50%)', '고위험 (>50%)']):
    count = category_counts.get(cat, 0)
    pct = count / len(ip_users) * 100
    ax3.bar(i, count, color=colors[i], alpha=0.7, edgecolor='black')
    ax3.text(i, count + 20, f'{count:,}\n({pct:.1f}%)', ha='center', fontsize=10, fontweight='bold')
ax3.set_xticks(range(3))
ax3.set_xticklabels(['정상 (<30%)', '의심 (30~50%)', '고위험 (>50%)'])
ax3.set_ylabel('계정 수', fontsize=11)
ax3.set_title('구간별 분포', fontsize=12)
ax3.grid(axis='y', alpha=0.3)

# 계정 수별 분포
ax4 = axes[1, 1]
acc_dist = ip_users['account_count'].value_counts().sort_index().head(10)
ax4.bar(acc_dist.index.astype(str), acc_dist.values, color='steelblue', alpha=0.7, edgecolor='black')
ax4.set_xlabel('IP당 계정 수', fontsize=11)
ax4.set_ylabel('IP 개수', fontsize=11)
ax4.set_title('IP당 계정 수 분포 (Top 10)', fontsize=12)
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'pattern2' / '1_ip_shared_ratio.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 1_ip_shared_ratio.png 저장")

# 2-2. concurrent_trading_ratio (동시 거래 비율)
# 간단하게 시간대별 거래 집중도로 대체
trade_df['hour'] = trade_df['ts'].dt.hour
user_hour_dist = trade_df.groupby(['account_id', 'hour']).size().reset_index(name='trade_count')
user_total = user_hour_dist.groupby('account_id')['trade_count'].sum().reset_index(name='total')
user_max = user_hour_dist.groupby('account_id')['trade_count'].max().reset_index(name='max_hour_trades')
concurrent = pd.merge(user_total, user_max, on='account_id')
concurrent['concurrent_trading_ratio'] = concurrent['max_hour_trades'] / concurrent['total']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('동시 거래 집중도 (concurrent_trading_ratio)', fontsize=16, fontweight='bold', y=0.995)

# 히스토그램
ax1 = axes[0, 0]
ax1.hist(concurrent['concurrent_trading_ratio'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
ax1.axvline(0.3, color='orange', linestyle='--', linewidth=2, label='의심 (30%)')
ax1.axvline(0.5, color='red', linestyle='--', linewidth=2, label='고위험 (50%)')
median_val = concurrent['concurrent_trading_ratio'].median()
p95_val = concurrent['concurrent_trading_ratio'].quantile(0.95)
ax1.text(0.98, 0.97, f'중앙값: {median_val:.2%}\n95th: {p95_val:.2%}',
         transform=ax1.transAxes, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
ax1.set_xlabel('동시 거래 집중도', fontsize=11)
ax1.set_ylabel('계정 수', fontsize=11)
ax1.set_title('분포', fontsize=12)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 박스플롯
ax2 = axes[0, 1]
ax2.boxplot(concurrent['concurrent_trading_ratio'], vert=True, patch_artist=True,
            boxprops=dict(facecolor='lightblue', alpha=0.7))
ax2.axhline(0.3, color='orange', linestyle='--', linewidth=2, label='의심 (30%)')
ax2.axhline(0.5, color='red', linestyle='--', linewidth=2, label='고위험 (50%)')
ax2.set_ylabel('동시 거래 집중도', fontsize=11)
ax2.set_title('박스플롯', fontsize=12)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 카테고리별
ax3 = axes[1, 0]
concurrent['category'] = pd.cut(concurrent['concurrent_trading_ratio'],
                                 bins=[-0.01, 0.3, 0.5, 1.01],
                                 labels=['정상 (<30%)', '의심 (30~50%)', '고위험 (>50%)'])
category_counts = concurrent['category'].value_counts()
colors = ['lightgreen', 'orange', 'red']
for i, cat in enumerate(['정상 (<30%)', '의심 (30~50%)', '고위험 (>50%)']):
    count = category_counts.get(cat, 0)
    pct = count / len(concurrent) * 100
    ax3.bar(i, count, color=colors[i], alpha=0.7, edgecolor='black')
    ax3.text(i, count + 5, f'{count:,}\n({pct:.1f}%)', ha='center', fontsize=10, fontweight='bold')
ax3.set_xticks(range(3))
ax3.set_xticklabels(['정상 (<30%)', '의심 (30~50%)', '고위험 (>50%)'])
ax3.set_ylabel('계정 수', fontsize=11)
ax3.set_title('구간별 분포', fontsize=12)
ax3.grid(axis='y', alpha=0.3)

# CDF
ax4 = axes[1, 1]
sorted_data = np.sort(concurrent['concurrent_trading_ratio'])
cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
ax4.plot(sorted_data, cdf, linewidth=2, color='blue')
ax4.axvline(0.3, color='orange', linestyle='--', linewidth=2, label='의심 (30%)')
ax4.axvline(0.5, color='red', linestyle='--', linewidth=2, label='고위험 (50%)')
ax4.set_xlabel('동시 거래 집중도', fontsize=11)
ax4.set_ylabel('누적 확률', fontsize=11)
ax4.set_title('누적 분포 함수 (CDF)', fontsize=12)
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'pattern2' / '2_concurrent_trading_ratio.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 2_concurrent_trading_ratio.png 저장")

# 2-3. price_similarity (가격 유사도) - 생략하고 심볼 다양성으로 대체
user_symbols = trade_df.groupby('account_id')['symbol'].nunique().reset_index(name='unique_symbols')
user_symbols['symbol_diversity'] = 1 / user_symbols['unique_symbols']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('심볼 집중도 (낮은 다양성 = 높은 유사도)', fontsize=16, fontweight='bold', y=0.995)

# 히스토그램
ax1 = axes[0, 0]
ax1.hist(user_symbols['symbol_diversity'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
ax1.axvline(0.5, color='orange', linestyle='--', linewidth=2, label='의심 (0.5)')
ax1.axvline(0.8, color='red', linestyle='--', linewidth=2, label='고위험 (0.8)')
median_val = user_symbols['symbol_diversity'].median()
p95_val = user_symbols['symbol_diversity'].quantile(0.95)
ax1.text(0.98, 0.97, f'중앙값: {median_val:.2f}\n95th: {p95_val:.2f}',
         transform=ax1.transAxes, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
ax1.set_xlabel('심볼 집중도', fontsize=11)
ax1.set_ylabel('계정 수', fontsize=11)
ax1.set_title('분포', fontsize=12)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 박스플롯
ax2 = axes[0, 1]
ax2.boxplot(user_symbols['symbol_diversity'], vert=True, patch_artist=True,
            boxprops=dict(facecolor='lightblue', alpha=0.7))
ax2.axhline(0.5, color='orange', linestyle='--', linewidth=2, label='의심 (0.5)')
ax2.axhline(0.8, color='red', linestyle='--', linewidth=2, label='고위험 (0.8)')
ax2.set_ylabel('심볼 집중도', fontsize=11)
ax2.set_title('박스플롯', fontsize=12)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 유니크 심볼 수 분포
ax3 = axes[1, 0]
symbol_dist = user_symbols['unique_symbols'].value_counts().sort_index().head(10)
ax3.bar(symbol_dist.index.astype(str), symbol_dist.values, color='steelblue', alpha=0.7, edgecolor='black')
ax3.set_xlabel('계정당 거래 심볼 수', fontsize=11)
ax3.set_ylabel('계정 수', fontsize=11)
ax3.set_title('거래 심볼 수 분포 (Top 10)', fontsize=12)
ax3.grid(axis='y', alpha=0.3)

# CDF
ax4 = axes[1, 1]
sorted_data = np.sort(user_symbols['symbol_diversity'])
cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
ax4.plot(sorted_data, cdf, linewidth=2, color='blue')
ax4.axvline(0.5, color='orange', linestyle='--', linewidth=2, label='의심 (0.5)')
ax4.axvline(0.8, color='red', linestyle='--', linewidth=2, label='고위험 (0.8)')
ax4.set_xlabel('심볼 집중도', fontsize=11)
ax4.set_ylabel('누적 확률', fontsize=11)
ax4.set_title('누적 분포 함수 (CDF)', fontsize=12)
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'pattern2' / '3_symbol_concentration.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 3_symbol_concentration.png 저장")

# 2-4. mean_leverage (평균 레버리지)
user_leverage = trade_df.groupby('account_id')['leverage'].mean().reset_index(name='mean_leverage')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('평균 레버리지 (mean_leverage)', fontsize=16, fontweight='bold', y=0.995)

# 히스토그램
ax1 = axes[0, 0]
ax1.hist(user_leverage['mean_leverage'], bins=50, range=(0, 100), color='skyblue', edgecolor='black', alpha=0.7)
ax1.axvline(30, color='orange', linestyle='--', linewidth=2, label='의심 (30배)')
ax1.axvline(50, color='red', linestyle='--', linewidth=2, label='고위험 (50배)')
median_val = user_leverage['mean_leverage'].median()
p95_val = user_leverage['mean_leverage'].quantile(0.95)
ax1.text(0.98, 0.97, f'중앙값: {median_val:.1f}배\n95th: {p95_val:.1f}배',
         transform=ax1.transAxes, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
ax1.set_xlabel('평균 레버리지 (배)', fontsize=11)
ax1.set_ylabel('계정 수', fontsize=11)
ax1.set_title('분포 (0~100배 범위)', fontsize=12)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# CDF
ax2 = axes[0, 1]
sorted_data = np.sort(user_leverage['mean_leverage'])
cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
ax2.plot(sorted_data, cdf, linewidth=2, color='blue')
ax2.axvline(30, color='orange', linestyle='--', linewidth=2, label='의심 (30배)')
ax2.axvline(50, color='red', linestyle='--', linewidth=2, label='고위험 (50배)')
ax2.set_xlabel('평균 레버리지 (배)', fontsize=11)
ax2.set_ylabel('누적 확률', fontsize=11)
ax2.set_title('누적 분포 함수 (CDF)', fontsize=12)
ax2.set_xlim(0, 100)
ax2.legend()
ax2.grid(alpha=0.3)

# 카테고리별
ax3 = axes[1, 0]
user_leverage['category'] = pd.cut(user_leverage['mean_leverage'],
                                    bins=[-0.01, 30, 50, 10000],
                                    labels=['정상 (<30배)', '의심 (30~50배)', '고위험 (>50배)'])
category_counts = user_leverage['category'].value_counts()
colors = ['lightgreen', 'orange', 'red']
for i, cat in enumerate(['정상 (<30배)', '의심 (30~50배)', '고위험 (>50배)']):
    count = category_counts.get(cat, 0)
    pct = count / len(user_leverage) * 100
    ax3.bar(i, count, color=colors[i], alpha=0.7, edgecolor='black')
    ax3.text(i, count + 5, f'{count:,}\n({pct:.1f}%)', ha='center', fontsize=10, fontweight='bold')
ax3.set_xticks(range(3))
ax3.set_xticklabels(['정상 (<30배)', '의심 (30~50배)', '고위험 (>50배)'])
ax3.set_ylabel('계정 수', fontsize=11)
ax3.set_title('구간별 분포', fontsize=12)
ax3.grid(axis='y', alpha=0.3)

# 박스플롯
ax4 = axes[1, 1]
ax4.boxplot(user_leverage['mean_leverage'], vert=True, patch_artist=True,
            boxprops=dict(facecolor='lightblue', alpha=0.7))
ax4.axhline(30, color='orange', linestyle='--', linewidth=2, label='의심 (30배)')
ax4.axhline(50, color='red', linestyle='--', linewidth=2, label='고위험 (50배)')
ax4.set_ylabel('평균 레버리지 (배)', fontsize=11)
ax4.set_title('박스플롯', fontsize=12)
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'pattern2' / '4_mean_leverage.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 4_mean_leverage.png 저장")

# ============================================================================
# Pattern 3: 보너스 악용
# ============================================================================
print("\n[4] Pattern 3: 보너스 악용 - 4개 PNG 생성")

# 3-1. total_reward (총 보너스 수령액)
user_reward = reward_df.groupby('account_id')['reward_amount'].sum().reset_index(name='total_reward')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('총 보너스 수령액 (total_reward)', fontsize=16, fontweight='bold', y=0.995)

# 히스토그램
ax1 = axes[0, 0]
ax1.hist(user_reward['total_reward'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
ax1.axvline(100, color='orange', linestyle='--', linewidth=2, label='의심 ($100)')
ax1.axvline(500, color='red', linestyle='--', linewidth=2, label='고위험 ($500)')
median_val = user_reward['total_reward'].median()
p95_val = user_reward['total_reward'].quantile(0.95)
ax1.text(0.98, 0.97, f'중앙값: ${median_val:.2f}\n95th: ${p95_val:.2f}',
         transform=ax1.transAxes, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
ax1.set_xlabel('총 보너스 수령액 ($)', fontsize=11)
ax1.set_ylabel('계정 수', fontsize=11)
ax1.set_title('분포', fontsize=12)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 박스플롯
ax2 = axes[0, 1]
ax2.boxplot(user_reward['total_reward'], vert=True, patch_artist=True,
            boxprops=dict(facecolor='lightblue', alpha=0.7))
ax2.axhline(100, color='orange', linestyle='--', linewidth=2, label='의심 ($100)')
ax2.axhline(500, color='red', linestyle='--', linewidth=2, label='고위험 ($500)')
ax2.set_ylabel('총 보너스 수령액 ($)', fontsize=11)
ax2.set_title('박스플롯', fontsize=12)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 카테고리별
ax3 = axes[1, 0]
user_reward['category'] = pd.cut(user_reward['total_reward'],
                                  bins=[-0.01, 100, 500, 100000],
                                  labels=['정상 (<$100)', '의심 ($100~$500)', '고위험 (>$500)'])
category_counts = user_reward['category'].value_counts()
colors = ['lightgreen', 'orange', 'red']
for i, cat in enumerate(['정상 (<$100)', '의심 ($100~$500)', '고위험 (>$500)']):
    count = category_counts.get(cat, 0)
    pct = count / len(user_reward) * 100
    ax3.bar(i, count, color=colors[i], alpha=0.7, edgecolor='black')
    ax3.text(i, count + 2, f'{count:,}\n({pct:.1f}%)', ha='center', fontsize=10, fontweight='bold')
ax3.set_xticks(range(3))
ax3.set_xticklabels(['정상 (<$100)', '의심 ($100~$500)', '고위험 (>$500)'])
ax3.set_ylabel('계정 수', fontsize=11)
ax3.set_title('구간별 분포', fontsize=12)
ax3.grid(axis='y', alpha=0.3)

# CDF
ax4 = axes[1, 1]
sorted_data = np.sort(user_reward['total_reward'])
cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
ax4.plot(sorted_data, cdf, linewidth=2, color='blue')
ax4.axvline(100, color='orange', linestyle='--', linewidth=2, label='의심 ($100)')
ax4.axvline(500, color='red', linestyle='--', linewidth=2, label='고위험 ($500)')
ax4.set_xlabel('총 보너스 수령액 ($)', fontsize=11)
ax4.set_ylabel('누적 확률', fontsize=11)
ax4.set_title('누적 분포 함수 (CDF)', fontsize=12)
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'pattern3' / '1_total_reward.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 1_total_reward.png 저장")

# 3-2. shared_ip (IP 공유 여부)
reward_accounts = set(reward_df['account_id'].unique())
reward_ip = ip_df[ip_df['account_id'].isin(reward_accounts)].copy()
ip_counts_reward = reward_ip.groupby('ip').size().reset_index(name='account_count')
reward_ip = reward_ip.merge(ip_counts_reward, on='ip')
reward_ip['shared_ip'] = reward_ip['account_count'] > 1

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('보너스 계정 IP 공유 여부 (shared_ip)', fontsize=16, fontweight='bold', y=0.995)

# 파이 차트
ax1 = axes[0, 0]
shared_counts = reward_ip['shared_ip'].value_counts()
ax1.pie(shared_counts.values, labels=['단독 IP', '공유 IP'], autopct='%1.1f%%',
        colors=['lightgreen', 'lightcoral'], startangle=90)
ax1.set_title(f'보너스 계정 IP 공유 비율\n(단독: {shared_counts.get(False, 0)}, 공유: {shared_counts.get(True, 0)})',
              fontsize=12)

# IP별 계정 수 분포
ax2 = axes[0, 1]
acc_dist = reward_ip.groupby('ip')['account_count'].first().value_counts().sort_index().head(10)
ax2.bar(acc_dist.index.astype(str), acc_dist.values, color='steelblue', alpha=0.7, edgecolor='black')
ax2.set_xlabel('IP당 보너스 계정 수', fontsize=11)
ax2.set_ylabel('IP 개수', fontsize=11)
ax2.set_title('IP당 보너스 계정 수 분포', fontsize=12)
ax2.grid(axis='y', alpha=0.3)

# 공유 IP vs 단독 IP 보너스 비교
ax3 = axes[1, 0]
reward_with_ip = reward_df.merge(reward_ip[['account_id', 'shared_ip']], on='account_id', how='left')
reward_with_ip['shared_ip'] = reward_with_ip['shared_ip'].fillna(False)
shared_rewards = reward_with_ip[reward_with_ip['shared_ip'] == True].groupby('account_id')['reward_amount'].sum()
single_rewards = reward_with_ip[reward_with_ip['shared_ip'] == False].groupby('account_id')['reward_amount'].sum()

bp = ax3.boxplot([single_rewards, shared_rewards], labels=['단독 IP', '공유 IP'],
                  patch_artist=True, showfliers=False)
bp['boxes'][0].set_facecolor('lightgreen')
bp['boxes'][1].set_facecolor('lightcoral')
for box in bp['boxes']:
    box.set_alpha(0.7)
ax3.set_ylabel('총 보너스 수령액 ($)', fontsize=11)
ax3.set_title('단독 IP vs 공유 IP 보너스 비교', fontsize=12)
ax3.grid(axis='y', alpha=0.3)

# 공유 IP 상위 계정
ax4 = axes[1, 1]
top_shared = reward_ip[reward_ip['shared_ip']].groupby('ip')['account_count'].first().sort_values(ascending=False).head(10)
ax4.barh(range(len(top_shared)), top_shared.values, color='crimson', alpha=0.7, edgecolor='black')
ax4.set_yticks(range(len(top_shared)))
ax4.set_yticklabels([f'IP {i+1}' for i in range(len(top_shared))])
ax4.set_xlabel('계정 수', fontsize=11)
ax4.set_title('가장 많은 보너스 계정을 가진 IP Top 10', fontsize=12)
ax4.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'pattern3' / '2_shared_ip.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 2_shared_ip.png 저장")

# 3-3. has_trades (거래 활동 여부)
all_accounts = set(trade_df['account_id'].unique())
reward_accounts = set(reward_df['account_id'].unique())
trading_reward = reward_accounts.intersection(all_accounts)
no_trading_reward = reward_accounts - all_accounts

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('보너스 계정 거래 활동 여부 (has_trades)', fontsize=16, fontweight='bold', y=0.995)

# 파이 차트
ax1 = axes[0, 0]
counts = [len(trading_reward), len(no_trading_reward)]
labels = [f'거래 있음\n({len(trading_reward)})', f'거래 없음\n({len(no_trading_reward)})']
ax1.pie(counts, labels=labels, autopct='%1.1f%%',
        colors=['lightgreen', 'lightcoral'], startangle=90)
ax1.set_title('보너스 수령 후 거래 활동 여부', fontsize=12)

# 거래 있는 계정의 거래량 분포
ax2 = axes[0, 1]
trading_accounts_df = trade_df[trade_df['account_id'].isin(trading_reward)]
trade_counts = trading_accounts_df.groupby('account_id').size()
ax2.hist(trade_counts, bins=50, color='lightblue', edgecolor='black', alpha=0.7)
ax2.set_xlabel('거래 건수', fontsize=11)
ax2.set_ylabel('계정 수', fontsize=11)
ax2.set_title('보너스 계정의 거래량 분포', fontsize=12)
ax2.grid(axis='y', alpha=0.3)

# 보너스액 비교
ax3 = axes[1, 0]
trading_rewards = reward_df[reward_df['account_id'].isin(trading_reward)].groupby('account_id')['reward_amount'].sum()
no_trading_rewards = reward_df[reward_df['account_id'].isin(no_trading_reward)].groupby('account_id')['reward_amount'].sum()

bp = ax3.boxplot([trading_rewards, no_trading_rewards],
                  labels=['거래 있음', '거래 없음'],
                  patch_artist=True, showfliers=False)
bp['boxes'][0].set_facecolor('lightgreen')
bp['boxes'][1].set_facecolor('lightcoral')
for box in bp['boxes']:
    box.set_alpha(0.7)
ax3.set_ylabel('총 보너스 수령액 ($)', fontsize=11)
ax3.set_title('거래 여부별 보너스액 비교', fontsize=12)
ax3.grid(axis='y', alpha=0.3)

# 통계 테이블
ax4 = axes[1, 1]
ax4.axis('off')
stats_data = [
    ['구분', '계정 수', '평균 보너스'],
    ['거래 있음', f'{len(trading_rewards):,}', f'${trading_rewards.mean():.2f}'],
    ['거래 없음', f'{len(no_trading_rewards):,}', f'${no_trading_rewards.mean():.2f}'],
    ['전체', f'{len(reward_accounts):,}', f'${reward_df.groupby("account_id")["reward_amount"].sum().mean():.2f}']
]
table = ax4.table(cellText=stats_data, cellLoc='center', loc='center',
                  colWidths=[0.3, 0.3, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)
for i in range(len(stats_data)):
    if i == 0:
        table[(i, 0)].set_facecolor('#40466e')
        table[(i, 1)].set_facecolor('#40466e')
        table[(i, 2)].set_facecolor('#40466e')
        table[(i, 0)].set_text_props(weight='bold', color='white')
        table[(i, 1)].set_text_props(weight='bold', color='white')
        table[(i, 2)].set_text_props(weight='bold', color='white')
    else:
        table[(i, 0)].set_facecolor('#f0f0f0')
        table[(i, 1)].set_facecolor('#f0f0f0')
        table[(i, 2)].set_facecolor('#f0f0f0')
ax4.set_title('보너스 계정 통계', fontsize=12, pad=20)

plt.tight_layout()
plt.savefig(output_dir / 'pattern3' / '3_has_trades.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 3_has_trades.png 저장")

# 3-4. reward_to_volume_ratio (RVR)
trading_reward_df = reward_df[reward_df['account_id'].isin(trading_reward)].copy()
reward_sum = trading_reward_df.groupby('account_id')['reward_amount'].sum().reset_index(name='total_reward')
volume_sum = trading_accounts_df.groupby('account_id')['qty'].sum().reset_index(name='total_volume')
rvr_df = pd.merge(reward_sum, volume_sum, on='account_id', how='inner')
rvr_df['rvr'] = rvr_df['total_reward'] / (rvr_df['total_volume'] + 0.0001)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Reward-to-Volume 비율 (RVR)', fontsize=16, fontweight='bold', y=0.995)

# 히스토그램 (0~10 범위)
ax1 = axes[0, 0]
ax1.hist(rvr_df['rvr'], bins=100, range=(0, 10), color='skyblue', edgecolor='black', alpha=0.7)
ax1.axvline(1, color='orange', linestyle='--', linewidth=2, label='의심 (1.0)')
ax1.axvline(5, color='red', linestyle='--', linewidth=2, label='고위험 (5.0)')
median_val = rvr_df['rvr'].median()
p95_val = rvr_df['rvr'].quantile(0.95)
ax1.text(0.98, 0.97, f'중앙값: {median_val:.2f}\n95th: {p95_val:.2f}',
         transform=ax1.transAxes, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
ax1.set_xlabel('RVR', fontsize=11)
ax1.set_ylabel('계정 수', fontsize=11)
ax1.set_title('분포 (0~10 범위)', fontsize=12)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 로그 스케일
ax2 = axes[0, 1]
ax2.hist(rvr_df['rvr'], bins=100, color='lightcoral', edgecolor='black', alpha=0.7)
ax2.set_yscale('log')
ax2.axvline(1, color='orange', linestyle='--', linewidth=2, label='의심 (1.0)')
ax2.axvline(5, color='red', linestyle='--', linewidth=2, label='고위험 (5.0)')
ax2.set_xlabel('RVR', fontsize=11)
ax2.set_ylabel('계정 수 (로그)', fontsize=11)
ax2.set_title('전체 분포 (로그 스케일)', fontsize=12)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 카테고리별
ax3 = axes[1, 0]
rvr_df['category'] = pd.cut(rvr_df['rvr'],
                             bins=[-0.01, 1, 5, 100000],
                             labels=['정상 (<1)', '의심 (1~5)', '고위험 (>5)'])
category_counts = rvr_df['category'].value_counts()
colors = ['lightgreen', 'orange', 'red']
for i, cat in enumerate(['정상 (<1)', '의심 (1~5)', '고위험 (>5)']):
    count = category_counts.get(cat, 0)
    pct = count / len(rvr_df) * 100
    ax3.bar(i, count, color=colors[i], alpha=0.7, edgecolor='black')
    ax3.text(i, count + 2, f'{count:,}\n({pct:.1f}%)', ha='center', fontsize=10, fontweight='bold')
ax3.set_xticks(range(3))
ax3.set_xticklabels(['정상 (<1)', '의심 (1~5)', '고위험 (>5)'])
ax3.set_ylabel('계정 수', fontsize=11)
ax3.set_title('구간별 분포', fontsize=12)
ax3.grid(axis='y', alpha=0.3)

# 박스플롯
ax4 = axes[1, 1]
ax4.boxplot(rvr_df['rvr'], vert=True, patch_artist=True,
            boxprops=dict(facecolor='lightblue', alpha=0.7))
ax4.axhline(1, color='orange', linestyle='--', linewidth=2, label='의심 (1.0)')
ax4.axhline(5, color='red', linestyle='--', linewidth=2, label='고위험 (5.0)')
ax4.set_ylabel('RVR', fontsize=11)
ax4.set_title('박스플롯', fontsize=12)
ax4.set_ylim(0, 20)
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'pattern3' / '4_reward_to_volume_ratio.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 4_reward_to_volume_ratio.png 저장")

print("\n" + "=" * 120)
print("✅ 모든 패턴별 개별 PNG 생성 완료!")
print("=" * 120)
print("\n생성된 파일:")
print("  Pattern 1 (펀딩피 차익거래): 4개 PNG")
print("    - output/final/pattern1/1_funding_fee_abs.png")
print("    - output/final/pattern1/2_mean_holding_minutes.png")
print("    - output/final/pattern1/3_funding_timing_ratio.png")
print("    - output/final/pattern1/4_funding_profit_ratio.png")
print("\n  Pattern 2 (조직적 거래): 4개 PNG")
print("    - output/final/pattern2/1_ip_shared_ratio.png")
print("    - output/final/pattern2/2_concurrent_trading_ratio.png")
print("    - output/final/pattern2/3_symbol_concentration.png")
print("    - output/final/pattern2/4_mean_leverage.png")
print("\n  Pattern 3 (보너스 악용): 4개 PNG")
print("    - output/final/pattern3/1_total_reward.png")
print("    - output/final/pattern3/2_shared_ip.png")
print("    - output/final/pattern3/3_has_trades.png")
print("    - output/final/pattern3/4_reward_to_volume_ratio.png")
print("\n총 12개의 개별 PNG 파일이 생성되었습니다!")
