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
print("피처별 상세 데이터 분포 시각화")
print("=" * 120)

# ================================================================================
# 데이터 로드
# ================================================================================
print("\n[1] 데이터 로드")

funding_arb_df = pd.read_csv('output/funding_analysis/funding_arbitrage_scores_all.csv')
organized_df = pd.read_csv('output/organized_trading/organized_scores_all.csv')
bonus_df = pd.read_csv('output/bonus_abuse/bonus_abuse_scores_all.csv')
funding_raw = pd.read_csv('data/Funding.csv')
trade_raw = pd.read_csv('data/Trade.csv')

funding_raw['funding_fee_abs'] = funding_raw['funding_fee'].abs()

print(f"✓ 데이터 로드 완료")

# ================================================================================
# Pattern 1: 펀딩피 차익거래 - 4개 피처 상세 시각화
# ================================================================================
print("\n[2] Pattern 1: 펀딩피 차익거래 시각화")

fig1 = plt.figure(figsize=(20, 12))
gs1 = fig1.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# ============================================================
# Feature 1-1: 펀딩피 절댓값 (funding_fee_abs)
# ============================================================

# 1-1-1: 전체 분포 (히스토그램)
ax1 = fig1.add_subplot(gs1[0, 0])
ax1.hist(funding_raw['funding_fee_abs'], bins=100, edgecolor='black', alpha=0.7,
         color='steelblue', range=(0, 50))
ax1.axvline(5, color='orange', linestyle='--', linewidth=2, label='의심 ($5)')
ax1.axvline(30, color='red', linestyle='--', linewidth=2, label='고위험 ($30)')
ax1.set_xlabel('Funding Fee (abs) ($)', fontsize=11)
ax1.set_ylabel('발생 횟수', fontsize=11)
ax1.set_title('[1-1-1] 펀딩피 절댓값 - 전체 분포', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
# 통계 추가
median_val = funding_raw['funding_fee_abs'].median()
p95_val = funding_raw['funding_fee_abs'].quantile(0.95)
ax1.text(0.98, 0.97, f'중앙값: ${median_val:.2f}\n95th: ${p95_val:.2f}',
         transform=ax1.transAxes, fontsize=9, verticalalignment='top',
         horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 1-1-2: 로그 스케일
ax2 = fig1.add_subplot(gs1[0, 1])
positive_funding = funding_raw[funding_raw['funding_fee_abs'] > 0]['funding_fee_abs']
ax2.hist(np.log10(positive_funding + 0.01), bins=100, edgecolor='black', alpha=0.7, color='teal')
ax2.set_xlabel('Log10(Funding Fee abs + 0.01)', fontsize=11)
ax2.set_ylabel('발생 횟수', fontsize=11)
ax2.set_title('[1-1-2] 펀딩피 절댓값 - 로그 스케일', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 1-1-3: 범주별 비율 (0 근처 vs 큰 값)
ax3 = fig1.add_subplot(gs1[0, 2])
near_zero = len(funding_raw[funding_raw['funding_fee_abs'] < 1])
small = len(funding_raw[(funding_raw['funding_fee_abs'] >= 1) & (funding_raw['funding_fee_abs'] < 5)])
medium = len(funding_raw[(funding_raw['funding_fee_abs'] >= 5) & (funding_raw['funding_fee_abs'] < 30)])
large = len(funding_raw[funding_raw['funding_fee_abs'] >= 30])

categories = ['<$1\n(0 근처)', '$1~$5\n(정상)', '$5~$30\n(의심)', '≥$30\n(고위험)']
counts = [near_zero, small, medium, large]
colors = ['green', 'lightgreen', 'orange', 'red']
bars = ax3.bar(categories, counts, color=colors, edgecolor='black')
ax3.set_ylabel('발생 횟수', fontsize=11)
ax3.set_title('[1-1-3] 펀딩피 범주별 분포', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
for bar, count in zip(bars, counts):
    height = bar.get_height()
    pct = count / len(funding_raw) * 100
    ax3.text(bar.get_x() + bar.get_width()/2, height,
             f'{count:,}\n({pct:.1f}%)',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# ============================================================
# Feature 1-2: 포지션 보유시간 (mean_holding_minutes)
# ============================================================

holding_times = funding_arb_df['mean_holding_minutes'].dropna()

# 1-2-1: 전체 분포
ax4 = fig1.add_subplot(gs1[1, 0])
ax4.hist(holding_times, bins=50, edgecolor='black', alpha=0.7, color='green', range=(0, 500))
ax4.axvline(30, color='red', linestyle='--', linewidth=2, label='고위험 (30분)')
ax4.axvline(60, color='orange', linestyle='--', linewidth=2, label='정상 (60분)')
ax4.set_xlabel('평균 보유시간 (분)', fontsize=11)
ax4.set_ylabel('계정 수', fontsize=11)
ax4.set_title('[1-2-1] 포지션 보유시간 - 전체 분포', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)
# 통계
median_hold = holding_times.median()
p10_hold = holding_times.quantile(0.10)
ax4.text(0.98, 0.97, f'중앙값: {median_hold:.1f}분\n10th: {p10_hold:.1f}분',
         transform=ax4.transAxes, fontsize=9, verticalalignment='top',
         horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 1-2-2: BoxPlot
ax5 = fig1.add_subplot(gs1[1, 1])
bp = ax5.boxplot([holding_times], vert=True, patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor('lightgreen')
ax5.axhline(30, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax5.axhline(60, color='orange', linestyle='--', linewidth=2, alpha=0.7)
ax5.set_ylabel('평균 보유시간 (분)', fontsize=11)
ax5.set_title('[1-2-2] 보유시간 BoxPlot', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)

# 1-2-3: 범주별 비율
ax6 = fig1.add_subplot(gs1[1, 2])
very_short = len(funding_arb_df[funding_arb_df['mean_holding_minutes'] < 10])
short = len(funding_arb_df[(funding_arb_df['mean_holding_minutes'] >= 10) &
                            (funding_arb_df['mean_holding_minutes'] < 30)])
medium_hold = len(funding_arb_df[(funding_arb_df['mean_holding_minutes'] >= 30) &
                                   (funding_arb_df['mean_holding_minutes'] < 60)])
long = len(funding_arb_df[funding_arb_df['mean_holding_minutes'] >= 60])

categories2 = ['<10분\n(확실)', '10~30분\n(고위험)', '30~60분\n(의심)', '≥60분\n(정상)']
counts2 = [very_short, short, medium_hold, long]
colors2 = ['darkred', 'red', 'orange', 'green']
bars2 = ax6.bar(categories2, counts2, color=colors2, edgecolor='black')
ax6.set_ylabel('계정 수', fontsize=11)
ax6.set_title('[1-2-3] 보유시간 범주별 분포', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')
for bar, count in zip(bars2, counts2):
    height = bar.get_height()
    pct = count / len(funding_arb_df) * 100
    ax6.text(bar.get_x() + bar.get_width()/2, height,
             f'{count}\n({pct:.1f}%)',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# ============================================================
# Feature 1-3: 펀딩 시각 거래 집중도 (funding_timing_ratio)
# ============================================================

timing_ratio = funding_arb_df['funding_timing_ratio'].dropna()

# 1-3-1: 전체 분포
ax7 = fig1.add_subplot(gs1[2, 0])
ax7.hist(timing_ratio*100, bins=30, edgecolor='black', alpha=0.7, color='coral')
ax7.axvline(30, color='orange', linestyle='--', linewidth=2, label='의심 (30%)')
ax7.axvline(50, color='red', linestyle='--', linewidth=2, label='고위험 (50%)')
ax7.set_xlabel('펀딩 시각 거래 비율 (%)', fontsize=11)
ax7.set_ylabel('계정 수', fontsize=11)
ax7.set_title('[1-3-1] 펀딩 시각 집중도 - 전체 분포', fontsize=12, fontweight='bold')
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3)
# 통계
mean_timing = timing_ratio.mean() * 100
p95_timing = timing_ratio.quantile(0.95) * 100
ax7.text(0.98, 0.97, f'평균: {mean_timing:.1f}%\n95th: {p95_timing:.1f}%',
         transform=ax7.transAxes, fontsize=9, verticalalignment='top',
         horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 1-3-2: CDF (누적분포)
ax8 = fig1.add_subplot(gs1[2, 1])
sorted_timing = np.sort(timing_ratio)
cdf = np.arange(1, len(sorted_timing) + 1) / len(sorted_timing)
ax8.plot(sorted_timing * 100, cdf, linewidth=2, color='navy')
ax8.axvline(30, color='orange', linestyle='--', linewidth=2, alpha=0.7)
ax8.axvline(50, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax8.set_xlabel('펀딩 시각 거래 비율 (%)', fontsize=11)
ax8.set_ylabel('누적 확률', fontsize=11)
ax8.set_title('[1-3-2] 펀딩 시각 집중도 CDF', fontsize=12, fontweight='bold')
ax8.grid(True, alpha=0.3)

# 1-3-3: 범주별 비율
ax9 = fig1.add_subplot(gs1[2, 2])
low_conc = len(funding_arb_df[funding_arb_df['funding_timing_ratio'] < 0.3])
med_conc = len(funding_arb_df[(funding_arb_df['funding_timing_ratio'] >= 0.3) &
                               (funding_arb_df['funding_timing_ratio'] < 0.5)])
high_conc = len(funding_arb_df[funding_arb_df['funding_timing_ratio'] >= 0.5])

categories3 = ['<30%\n(정상)', '30~50%\n(의심)', '≥50%\n(고위험)']
counts3 = [low_conc, med_conc, high_conc]
colors3 = ['green', 'orange', 'red']
bars3 = ax9.bar(categories3, counts3, color=colors3, edgecolor='black')
ax9.set_ylabel('계정 수', fontsize=11)
ax9.set_title('[1-3-3] 펀딩 시각 집중도 범주별', fontsize=12, fontweight='bold')
ax9.grid(True, alpha=0.3, axis='y')
for bar, count in zip(bars3, counts3):
    height = bar.get_height()
    pct = count / len(funding_arb_df) * 100
    ax9.text(bar.get_x() + bar.get_width()/2, height,
             f'{count}\n({pct:.1f}%)',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

fig1.suptitle('Pattern 1: 펀딩피 차익거래 - 피처별 상세 분포',
              fontsize=18, fontweight='bold', y=0.998)

plt.savefig('output/final_features/pattern1_detailed_distribution.png', dpi=300, bbox_inches='tight')
print(f"✓ Pattern 1 상세 시각화 저장: output/final_features/pattern1_detailed_distribution.png")

# ================================================================================
# Pattern 2: 조직적 거래 - 4개 피처 상세 시각화
# ================================================================================
print("\n[3] Pattern 2: 조직적 거래 시각화")

fig2 = plt.figure(figsize=(20, 12))
gs2 = fig2.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# ============================================================
# Feature 2-1: IP 공유 비율 (ip_shared_ratio)
# ============================================================

ip_shared = organized_df['ip_shared_ratio'].dropna()

# 2-1-1: 전체 분포
ax1 = fig2.add_subplot(gs2[0, 0])
ax1.hist(ip_shared*100, bins=30, edgecolor='black', alpha=0.7, color='purple')
ax1.axvline(30, color='orange', linestyle='--', linewidth=2, label='의심 (30%)')
ax1.axvline(50, color='red', linestyle='--', linewidth=2, label='고위험 (50%)')
ax1.set_xlabel('IP 공유 비율 (%)', fontsize=11)
ax1.set_ylabel('계정 수', fontsize=11)
ax1.set_title('[2-1-1] IP 공유 비율 - 전체 분포', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
# 통계
mean_ip = ip_shared.mean() * 100
median_ip = ip_shared.median() * 100
ax1.text(0.98, 0.97, f'평균: {mean_ip:.2f}%\n중앙값: {median_ip:.2f}%',
         transform=ax1.transAxes, fontsize=9, verticalalignment='top',
         horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 2-1-2: 0 vs 공유 있음
ax2 = fig2.add_subplot(gs2[0, 1])
no_shared = len(organized_df[organized_df['ip_shared_ratio'] == 0])
has_shared = len(organized_df[organized_df['ip_shared_ratio'] > 0])
labels = ['공유 없음\n(0%)', '공유 있음\n(>0%)']
sizes = [no_shared, has_shared]
colors_pie = ['green', 'orange']
ax2.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
        startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
ax2.set_title('[2-1-2] IP 공유 여부', fontsize=12, fontweight='bold')

# 2-1-3: 범주별 비율
ax3 = fig2.add_subplot(gs2[0, 2])
no_share = len(organized_df[organized_df['ip_shared_ratio'] == 0])
low_share = len(organized_df[(organized_df['ip_shared_ratio'] > 0) &
                              (organized_df['ip_shared_ratio'] < 0.3)])
med_share = len(organized_df[(organized_df['ip_shared_ratio'] >= 0.3) &
                              (organized_df['ip_shared_ratio'] < 0.5)])
high_share = len(organized_df[organized_df['ip_shared_ratio'] >= 0.5])

categories = ['0%\n(정상)', '0~30%\n(주의)', '30~50%\n(의심)', '≥50%\n(고위험)']
counts = [no_share, low_share, med_share, high_share]
colors = ['green', 'yellow', 'orange', 'red']
bars = ax3.bar(categories, counts, color=colors, edgecolor='black')
ax3.set_ylabel('계정 수', fontsize=11)
ax3.set_title('[2-1-3] IP 공유 범주별 분포', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
for bar, count in zip(bars, counts):
    height = bar.get_height()
    pct = count / len(organized_df) * 100
    ax3.text(bar.get_x() + bar.get_width()/2, height,
             f'{count}\n({pct:.1f}%)',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# ============================================================
# Feature 2-2: 동시 거래 비율 (concurrent_trading_ratio)
# ============================================================

concurrent = organized_df['concurrent_trading_ratio'].dropna()

# 2-2-1: 전체 분포
ax4 = fig2.add_subplot(gs2[1, 0])
ax4.hist(concurrent*100, bins=30, edgecolor='black', alpha=0.7, color='orange')
ax4.axvline(30, color='orange', linestyle='--', linewidth=2, label='의심 (30%)')
ax4.axvline(50, color='red', linestyle='--', linewidth=2, label='고위험 (50%)')
ax4.set_xlabel('동시 거래 비율 (%)', fontsize=11)
ax4.set_ylabel('계정 수', fontsize=11)
ax4.set_title('[2-2-1] 동시 거래 비율 - 전체 분포', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)
# 통계
mean_conc = concurrent.mean() * 100
p95_conc = concurrent.quantile(0.95) * 100
ax4.text(0.98, 0.97, f'평균: {mean_conc:.1f}%\n95th: {p95_conc:.1f}%',
         transform=ax4.transAxes, fontsize=9, verticalalignment='top',
         horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 2-2-2: BoxPlot
ax5 = fig2.add_subplot(gs2[1, 1])
bp = ax5.boxplot([concurrent*100], vert=True, patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor('lightsalmon')
ax5.axhline(30, color='orange', linestyle='--', linewidth=2, alpha=0.7)
ax5.axhline(50, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax5.set_ylabel('동시 거래 비율 (%)', fontsize=11)
ax5.set_title('[2-2-2] 동시 거래 BoxPlot', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)

# 2-2-3: 범주별
ax6 = fig2.add_subplot(gs2[1, 2])
low_conc2 = len(organized_df[organized_df['concurrent_trading_ratio'] < 0.3])
med_conc2 = len(organized_df[(organized_df['concurrent_trading_ratio'] >= 0.3) &
                              (organized_df['concurrent_trading_ratio'] < 0.5)])
high_conc2 = len(organized_df[organized_df['concurrent_trading_ratio'] >= 0.5])

categories2 = ['<30%\n(정상)', '30~50%\n(의심)', '≥50%\n(고위험)']
counts2 = [low_conc2, med_conc2, high_conc2]
colors2 = ['green', 'orange', 'red']
bars2 = ax6.bar(categories2, counts2, color=colors2, edgecolor='black')
ax6.set_ylabel('계정 수', fontsize=11)
ax6.set_title('[2-2-3] 동시 거래 범주별', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')
for bar, count in zip(bars2, counts2):
    height = bar.get_height()
    pct = count / len(organized_df) * 100
    ax6.text(bar.get_x() + bar.get_width()/2, height,
             f'{count}\n({pct:.1f}%)',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# ============================================================
# Feature 2-3: 평균 레버리지 (mean_leverage)
# ============================================================

leverage = organized_df['mean_leverage'].dropna()

# 2-3-1: 전체 분포
ax7 = fig2.add_subplot(gs2[2, 0])
ax7.hist(leverage, bins=30, edgecolor='black', alpha=0.7, color='gold')
ax7.axvline(30, color='orange', linestyle='--', linewidth=2, label='의심 (30x)')
ax7.axvline(50, color='red', linestyle='--', linewidth=2, label='고위험 (50x)')
ax7.set_xlabel('평균 레버리지 (x)', fontsize=11)
ax7.set_ylabel('계정 수', fontsize=11)
ax7.set_title('[2-3-1] 평균 레버리지 - 전체 분포', fontsize=12, fontweight='bold')
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3)
# 통계
mean_lev = leverage.mean()
median_lev = leverage.median()
ax7.text(0.98, 0.97, f'평균: {mean_lev:.1f}x\n중앙값: {median_lev:.1f}x',
         transform=ax7.transAxes, fontsize=9, verticalalignment='top',
         horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 2-3-2: CDF
ax8 = fig2.add_subplot(gs2[2, 1])
sorted_lev = np.sort(leverage)
cdf = np.arange(1, len(sorted_lev) + 1) / len(sorted_lev)
ax8.plot(sorted_lev, cdf, linewidth=2, color='darkgoldenrod')
ax8.axvline(30, color='orange', linestyle='--', linewidth=2, alpha=0.7)
ax8.axvline(50, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax8.set_xlabel('평균 레버리지 (x)', fontsize=11)
ax8.set_ylabel('누적 확률', fontsize=11)
ax8.set_title('[2-3-2] 레버리지 CDF', fontsize=12, fontweight='bold')
ax8.grid(True, alpha=0.3)

# 2-3-3: 범주별
ax9 = fig2.add_subplot(gs2[2, 2])
low_lev = len(organized_df[organized_df['mean_leverage'] < 30])
med_lev = len(organized_df[(organized_df['mean_leverage'] >= 30) &
                            (organized_df['mean_leverage'] < 50)])
high_lev = len(organized_df[organized_df['mean_leverage'] >= 50])

categories3 = ['<30x\n(정상)', '30~50x\n(의심)', '≥50x\n(고위험)']
counts3 = [low_lev, med_lev, high_lev]
colors3 = ['green', 'orange', 'red']
bars3 = ax9.bar(categories3, counts3, color=colors3, edgecolor='black')
ax9.set_ylabel('계정 수', fontsize=11)
ax9.set_title('[2-3-3] 레버리지 범주별', fontsize=12, fontweight='bold')
ax9.grid(True, alpha=0.3, axis='y')
for bar, count in zip(bars3, counts3):
    height = bar.get_height()
    pct = count / len(organized_df) * 100
    ax9.text(bar.get_x() + bar.get_width()/2, height,
             f'{count}\n({pct:.1f}%)',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

fig2.suptitle('Pattern 2: 조직적 거래 - 피처별 상세 분포',
              fontsize=18, fontweight='bold', y=0.998)

plt.savefig('output/final_features/pattern2_detailed_distribution.png', dpi=300, bbox_inches='tight')
print(f"✓ Pattern 2 상세 시각화 저장: output/final_features/pattern2_detailed_distribution.png")

# ================================================================================
# Pattern 3: 보너스 악용 - 4개 피처 상세 시각화
# ================================================================================
print("\n[4] Pattern 3: 보너스 악용 시각화")

fig3 = plt.figure(figsize=(20, 12))
gs3 = fig3.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# ============================================================
# Feature 3-1: 총 보너스 금액 (total_reward)
# ============================================================

rewards = bonus_df['total_reward'].dropna()

# 3-1-1: 전체 분포
ax1 = fig3.add_subplot(gs3[0, 0])
ax1.hist(rewards, bins=30, edgecolor='black', alpha=0.7, color='crimson')
ax1.axvline(50, color='orange', linestyle='--', linewidth=2, label='의심 ($50)')
ax1.axvline(100, color='red', linestyle='--', linewidth=2, label='고위험 ($100)')
ax1.set_xlabel('총 보너스 금액 ($)', fontsize=11)
ax1.set_ylabel('계정 수', fontsize=11)
ax1.set_title('[3-1-1] 총 보너스 금액 - 전체 분포', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
# 통계
median_reward = rewards.median()
p95_reward = rewards.quantile(0.95)
ax1.text(0.98, 0.97, f'중앙값: ${median_reward:.2f}\n95th: ${p95_reward:.2f}',
         transform=ax1.transAxes, fontsize=9, verticalalignment='top',
         horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 3-1-2: BoxPlot
ax2 = fig3.add_subplot(gs3[0, 1])
bp = ax2.boxplot([rewards], vert=True, patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor('lightcoral')
ax2.axhline(50, color='orange', linestyle='--', linewidth=2, alpha=0.7)
ax2.axhline(100, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax2.set_ylabel('총 보너스 금액 ($)', fontsize=11)
ax2.set_title('[3-1-2] 보너스 금액 BoxPlot', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3-1-3: 범주별
ax3 = fig3.add_subplot(gs3[0, 2])
low_reward = len(bonus_df[bonus_df['total_reward'] < 50])
med_reward = len(bonus_df[(bonus_df['total_reward'] >= 50) &
                          (bonus_df['total_reward'] < 100)])
high_reward = len(bonus_df[bonus_df['total_reward'] >= 100])

categories = ['<$50\n(정상)', '$50~$100\n(의심)', '≥$100\n(고위험)']
counts = [low_reward, med_reward, high_reward]
colors = ['green', 'orange', 'red']
bars = ax3.bar(categories, counts, color=colors, edgecolor='black')
ax3.set_ylabel('계정 수', fontsize=11)
ax3.set_title('[3-1-3] 보너스 금액 범주별', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
for bar, count in zip(bars, counts):
    height = bar.get_height()
    pct = count / len(bonus_df) * 100
    ax3.text(bar.get_x() + bar.get_width()/2, height,
             f'{count}\n({pct:.1f}%)',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# ============================================================
# Feature 3-2: 공유 IP 사용 (shared_ip)
# ============================================================

# 3-2-1: 공유 IP 여부 파이 차트
ax4 = fig3.add_subplot(gs3[1, 0])
shared_ip_count = bonus_df['shared_ip'].sum()
unique_ip_count = len(bonus_df) - shared_ip_count
labels = ['고유 IP', '공유 IP']
sizes = [unique_ip_count, shared_ip_count]
colors_pie = ['green', 'red']
explode = (0, 0.1)  # 공유 IP 강조
ax4.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
        startangle=90, explode=explode,
        textprops={'fontsize': 11, 'fontweight': 'bold'})
ax4.set_title('[3-2-1] IP 공유 여부', fontsize=12, fontweight='bold')

# 3-2-2: 공유 IP 계정의 보너스 금액
ax5 = fig3.add_subplot(gs3[1, 1])
shared_rewards = bonus_df[bonus_df['shared_ip'] == True]['total_reward']
unique_rewards = bonus_df[bonus_df['shared_ip'] == False]['total_reward']

if len(shared_rewards) > 0 and len(unique_rewards) > 0:
    bp = ax5.boxplot([unique_rewards, shared_rewards],
                      labels=['고유 IP', '공유 IP'],
                      patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax5.set_ylabel('총 보너스 금액 ($)', fontsize=11)
    ax5.set_title('[3-2-2] IP 유형별 보너스 비교', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)

# 3-2-3: 바 차트 비교
ax6 = fig3.add_subplot(gs3[1, 2])
categories2 = ['고유 IP', '공유 IP']
counts2 = [unique_ip_count, shared_ip_count]
colors2 = ['green', 'red']
bars2 = ax6.bar(categories2, counts2, color=colors2, edgecolor='black', width=0.5)
ax6.set_ylabel('계정 수', fontsize=11)
ax6.set_title('[3-2-3] IP 유형별 계정 수', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')
for bar, count in zip(bars2, counts2):
    height = bar.get_height()
    pct = count / len(bonus_df) * 100
    ax6.text(bar.get_x() + bar.get_width()/2, height,
             f'{count}\n({pct:.1f}%)',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# ============================================================
# Feature 3-3: 거래 활동 여부 (has_trades)
# ============================================================

# 3-3-1: 거래 활동 파이 차트
ax7 = fig3.add_subplot(gs3[2, 0])
has_trades = bonus_df['has_trades'].sum()
no_trades = len(bonus_df) - has_trades
labels3 = ['거래 있음', '거래 없음']
sizes3 = [has_trades, no_trades]
colors3 = ['green', 'red']
ax7.pie(sizes3, labels=labels3, colors=colors3, autopct='%1.1f%%',
        startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
ax7.set_title('[3-3-1] 보너스 후 거래 활동', fontsize=12, fontweight='bold')

# 3-3-2: 거래 횟수 분포
ax8 = fig3.add_subplot(gs3[2, 1])
trade_counts = bonus_df['trade_count'].dropna()
ax8.hist(trade_counts, bins=30, edgecolor='black', alpha=0.7, color='teal', range=(0, 1500))
ax8.axvline(10, color='red', linestyle='--', linewidth=2, label='최소 기준 (10회)')
ax8.set_xlabel('거래 횟수', fontsize=11)
ax8.set_ylabel('계정 수', fontsize=11)
ax8.set_title('[3-3-2] 거래 횟수 분포', fontsize=12, fontweight='bold')
ax8.legend(fontsize=9)
ax8.grid(True, alpha=0.3)

# 3-3-3: RVR 분포 (극단값 제외)
ax9 = fig3.add_subplot(gs3[2, 2])
rvr = bonus_df['reward_to_volume_ratio'].dropna()
rvr_valid = rvr[rvr < 0.01]  # 1% 미만만

if len(rvr_valid) > 0:
    ax9.hist(rvr_valid*1000, bins=30, edgecolor='black', alpha=0.7, color='violet')
    ax9.axvline(1, color='red', linestyle='--', linewidth=2, label='고위험 (0.001)')
    ax9.set_xlabel('RVR (×1000)', fontsize=11)
    ax9.set_ylabel('계정 수', fontsize=11)
    ax9.set_title('[3-3-3] Reward-to-Volume Ratio', fontsize=12, fontweight='bold')
    ax9.legend(fontsize=9)
    ax9.grid(True, alpha=0.3)
    # 통계
    median_rvr = rvr_valid.median() * 1000
    p95_rvr = rvr_valid.quantile(0.95) * 1000
    ax9.text(0.98, 0.97, f'중앙값: {median_rvr:.3f}\n95th: {p95_rvr:.3f}',
             transform=ax9.transAxes, fontsize=9, verticalalignment='top',
             horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

fig3.suptitle('Pattern 3: 보너스 악용 - 피처별 상세 분포',
              fontsize=18, fontweight='bold', y=0.998)

plt.savefig('output/final_features/pattern3_detailed_distribution.png', dpi=300, bbox_inches='tight')
print(f"✓ Pattern 3 상세 시각화 저장: output/final_features/pattern3_detailed_distribution.png")

print("\n" + "=" * 120)
print("✅ 모든 패턴 상세 시각화 완료!")
print("=" * 120)
print("\n생성된 파일:")
print("  1. output/final_features/pattern1_detailed_distribution.png (9개 차트)")
print("  2. output/final_features/pattern2_detailed_distribution.png (9개 차트)")
print("  3. output/final_features/pattern3_detailed_distribution.png (9개 차트)")
print("\n총 27개의 상세 분포 차트가 생성되었습니다!")
