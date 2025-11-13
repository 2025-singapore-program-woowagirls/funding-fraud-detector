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
print("조직적 이상거래(Organized Trading) 탐지 분석")
print("=" * 100)

# ================================================================================
# 1. 데이터 로드
# ================================================================================
print("\n[1] 데이터 로딩 중...")

trade_df = pd.read_csv('data/Trade.csv')
ip_df = pd.read_csv('data/IP.csv')
funding_df = pd.read_csv('data/Funding.csv')

# 타임스탬프 변환
trade_df['ts'] = pd.to_datetime(trade_df['ts'])
funding_df['ts'] = pd.to_datetime(funding_df['ts'])

print(f"✓ Trade: {len(trade_df):,} rows, {trade_df['account_id'].nunique()} accounts")
print(f"✓ IP: {len(ip_df):,} rows, {ip_df['ip'].nunique()} unique IPs")
print(f"✓ Funding: {len(funding_df):,} rows")

# ================================================================================
# 2. IP 공유 패턴 분석 (Multi-Account Detection)
# ================================================================================
print("\n" + "=" * 100)
print("[2] IP 공유 패턴 분석 (다계정 탐지)")
print("=" * 100)

# IP별 계정 수 집계
ip_account_counts = ip_df.groupby('ip')['account_id'].agg(['count', lambda x: list(x.unique())]).reset_index()
ip_account_counts.columns = ['ip', 'account_count', 'accounts']

# 다계정 IP 필터링 (2개 이상)
shared_ips = ip_account_counts[ip_account_counts['account_count'] >= 2].copy()
shared_ips = shared_ips.sort_values('account_count', ascending=False)

print(f"\n총 IP 수: {len(ip_account_counts):,}")
print(f"다계정 IP 수: {len(shared_ips):,} ({len(shared_ips)/len(ip_account_counts)*100:.2f}%)")
print(f"최대 계정 수 (단일 IP): {shared_ips['account_count'].max()}")

# 계정별 공유 IP 수 계산
account_ip_map = ip_df.groupby('account_id')['ip'].apply(list).to_dict()
account_shared_ip_count = {}
for account, ips in account_ip_map.items():
    shared_count = sum(1 for ip in ips if ip in shared_ips['ip'].values)
    account_shared_ip_count[account] = {
        'total_ips': len(ips),
        'shared_ips': shared_count,
        'shared_ratio': shared_count / len(ips) if len(ips) > 0 else 0
    }

# 통계
shared_ratios = [v['shared_ratio'] for v in account_shared_ip_count.values()]
print(f"\n[정상 거래 기준선 - IP 공유]")
print(f"  평균 공유 IP 비율: {np.mean(shared_ratios)*100:.2f}%")
print(f"  중앙값: {np.median(shared_ratios)*100:.2f}%")
print(f"  표준편차: {np.std(shared_ratios)*100:.2f}%")
print(f"  95th percentile: {np.percentile(shared_ratios, 95)*100:.2f}%")
print(f"  99th percentile: {np.percentile(shared_ratios, 99)*100:.2f}%")

# 이상치 기준: 도메인 지식 + 통계
# - 정상: 단일 IP 또는 2개 이하의 다른 IP (공유 비율 < 50%)
# - 의심: 공유 IP 비율 > 50%
# - 고위험: 공유 IP 비율 > 80%
threshold_shared_ip_ratio_suspicious = 0.5
threshold_shared_ip_ratio_high_risk = 0.8

suspicious_accounts_ip = [k for k, v in account_shared_ip_count.items()
                          if v['shared_ratio'] > threshold_shared_ip_ratio_suspicious]

print(f"\n[이상치 탐지 결과]")
print(f"  의심 계정 (공유 IP 비율 > 50%): {len(suspicious_accounts_ip)}개")

# ================================================================================
# 3. 시간대별 동시 거래 패턴 분석
# ================================================================================
print("\n" + "=" * 100)
print("[3] 시간대별 동시 거래 패턴 분석")
print("=" * 100)

# 1분 단위로 그룹핑하여 동시 거래 탐지
trade_df['ts_minute'] = trade_df['ts'].dt.floor('1min')

# 같은 시간(1분 단위) + 같은 심볼에서 거래한 계정 쌍 찾기
concurrent_trades = trade_df.groupby(['ts_minute', 'symbol']).agg({
    'account_id': lambda x: list(x.unique()),
    'price': ['mean', 'std', 'count']
}).reset_index()

concurrent_trades.columns = ['ts_minute', 'symbol', 'accounts', 'price_mean', 'price_std', 'trade_count']
concurrent_trades['account_count'] = concurrent_trades['accounts'].apply(len)

# 2명 이상 동시 거래
concurrent_multi = concurrent_trades[concurrent_trades['account_count'] >= 2].copy()

print(f"\n총 거래 시간-심볼 조합: {len(concurrent_trades):,}")
print(f"다계정 동시 거래 발생: {len(concurrent_multi):,} ({len(concurrent_multi)/len(concurrent_trades)*100:.2f}%)")

# 계정별 동시 거래 참여 횟수
account_concurrent_count = defaultdict(int)
for _, row in concurrent_multi.iterrows():
    for account in row['accounts']:
        account_concurrent_count[account] += 1

# 각 계정의 총 거래 횟수 대비 동시 거래 비율
account_total_trades = trade_df.groupby('account_id').size().to_dict()
account_concurrent_ratio = {}
for account in account_total_trades.keys():
    total = account_total_trades.get(account, 0)
    concurrent = account_concurrent_count.get(account, 0)
    account_concurrent_ratio[account] = concurrent / total if total > 0 else 0

concurrent_ratios = list(account_concurrent_ratio.values())
print(f"\n[정상 거래 기준선 - 동시 거래]")
print(f"  평균 동시 거래 비율: {np.mean(concurrent_ratios)*100:.2f}%")
print(f"  중앙값: {np.median(concurrent_ratios)*100:.2f}%")
print(f"  표준편차: {np.std(concurrent_ratios)*100:.2f}%")
print(f"  95th percentile: {np.percentile(concurrent_ratios, 95)*100:.2f}%")
print(f"  99th percentile: {np.percentile(concurrent_ratios, 99)*100:.2f}%")

# 이상치 기준: 도메인 지식
# - 정상: 우연히 겹칠 수 있는 수준 (< 30%)
# - 의심: 동시 거래 비율 > 50%
# - 고위험: 동시 거래 비율 > 70%
threshold_concurrent_ratio_suspicious = 0.5
threshold_concurrent_ratio_high_risk = 0.7

suspicious_accounts_concurrent = [k for k, v in account_concurrent_ratio.items()
                                  if v > threshold_concurrent_ratio_suspicious]

print(f"\n[이상치 탐지 결과]")
print(f"  의심 계정 (동시 거래 비율 > 50%): {len(suspicious_accounts_concurrent)}개")

# ================================================================================
# 4. 가격 유사도 분석 (Price Similarity)
# ================================================================================
print("\n" + "=" * 100)
print("[4] 거래 가격 유사도 분석")
print("=" * 100)

# 같은 시간대(1분) + 같은 심볼에서 가격 편차 분석
concurrent_multi['price_cv'] = concurrent_multi['price_std'] / concurrent_multi['price_mean']  # 변동계수
concurrent_multi['price_cv'] = concurrent_multi['price_cv'].fillna(0)

# 가격이 거의 동일한 경우 (CV < 0.01, 즉 1% 미만 차이)
similar_price_trades = concurrent_multi[concurrent_multi['price_cv'] < 0.01]

print(f"\n다계정 동시 거래 중:")
print(f"  가격 유사도 높은 거래 (CV < 1%): {len(similar_price_trades):,} ({len(similar_price_trades)/len(concurrent_multi)*100:.2f}%)")

# 계정별 가격 유사 거래 비율
account_similar_price_count = defaultdict(int)
for _, row in similar_price_trades.iterrows():
    for account in row['accounts']:
        account_similar_price_count[account] += 1

account_similar_price_ratio = {}
for account in account_total_trades.keys():
    concurrent = account_concurrent_count.get(account, 0)
    similar = account_similar_price_count.get(account, 0)
    account_similar_price_ratio[account] = similar / concurrent if concurrent > 0 else 0

similar_price_ratios = [v for v in account_similar_price_ratio.values() if v > 0]

if len(similar_price_ratios) > 0:
    print(f"\n[정상 거래 기준선 - 가격 유사도]")
    print(f"  평균 유사 가격 거래 비율: {np.mean(similar_price_ratios)*100:.2f}%")
    print(f"  중앙값: {np.median(similar_price_ratios)*100:.2f}%")
    print(f"  표준편차: {np.std(similar_price_ratios)*100:.2f}%")

    # 이상치 기준: 동시 거래 중 80% 이상이 유사 가격이면 의심
    threshold_similar_price_ratio = 0.8
    suspicious_accounts_price = [k for k, v in account_similar_price_ratio.items()
                                 if v > threshold_similar_price_ratio and account_concurrent_count.get(k, 0) > 5]

    print(f"\n[이상치 탐지 결과]")
    print(f"  의심 계정 (유사가격 비율 > 80%): {len(suspicious_accounts_price)}개")
else:
    suspicious_accounts_price = []

# ================================================================================
# 5. 레버리지 패턴 분석
# ================================================================================
print("\n" + "=" * 100)
print("[5] 레버리지 패턴 분석")
print("=" * 100)

# OPEN 거래만 필터링 (레버리지는 오픈 시에만 의미 있음)
open_trades = trade_df[trade_df['openclose'] == 'OPEN'].copy()

# 계정별 평균 레버리지
account_leverage = open_trades.groupby('account_id')['leverage'].agg(['mean', 'std', 'max', 'count']).reset_index()
account_leverage.columns = ['account_id', 'mean_leverage', 'std_leverage', 'max_leverage', 'trade_count']

print(f"\n[정상 거래 기준선 - 레버리지]")
print(f"  평균 레버리지: {account_leverage['mean_leverage'].mean():.2f}x")
print(f"  중앙값: {account_leverage['mean_leverage'].median():.2f}x")
print(f"  표준편차: {account_leverage['mean_leverage'].std():.2f}x")
print(f"  95th percentile: {account_leverage['mean_leverage'].quantile(0.95):.2f}x")
print(f"  99th percentile: {account_leverage['mean_leverage'].quantile(0.99):.2f}x")

# 이상치 기준: 도메인 지식 (일반적으로 50배 이상은 고위험)
# - 정상: 평균 레버리지 < 30x
# - 의심: 평균 레버리지 30-50x
# - 고위험: 평균 레버리지 > 50x
threshold_leverage_suspicious = 30
threshold_leverage_high_risk = 50

suspicious_accounts_leverage = account_leverage[
    account_leverage['mean_leverage'] > threshold_leverage_suspicious
]['account_id'].tolist()

print(f"\n[이상치 탐지 결과]")
print(f"  의심 계정 (평균 레버리지 > 30x): {len(suspicious_accounts_leverage)}개")

# ================================================================================
# 6. 종합 위험 점수 계산 (Organized Trading Score)
# ================================================================================
print("\n" + "=" * 100)
print("[6] 조직적 거래 종합 위험 점수 산출")
print("=" * 100)

# 모든 계정 리스트
all_accounts = list(set(trade_df['account_id'].unique()))

# 각 계정의 피처 수집
organized_scores = []

for account in all_accounts:
    # 1. IP 공유 점수 (0-1)
    ip_data = account_shared_ip_count.get(account, {'shared_ratio': 0})
    ip_score = min(ip_data['shared_ratio'] / threshold_shared_ip_ratio_high_risk, 1.0)

    # 2. 동시 거래 점수 (0-1)
    concurrent_ratio = account_concurrent_ratio.get(account, 0)
    concurrent_score = min(concurrent_ratio / threshold_concurrent_ratio_high_risk, 1.0)

    # 3. 가격 유사도 점수 (0-1)
    price_sim_ratio = account_similar_price_ratio.get(account, 0)
    price_score = min(price_sim_ratio / threshold_similar_price_ratio, 1.0) if account_concurrent_count.get(account, 0) > 5 else 0

    # 4. 레버리지 점수 (0-1)
    lev_data = account_leverage[account_leverage['account_id'] == account]
    if len(lev_data) > 0:
        mean_lev = lev_data['mean_leverage'].values[0]
        leverage_score = min(max(mean_lev - 20, 0) / 30, 1.0)  # 20x 이하는 0점, 50x 이상은 1점
    else:
        leverage_score = 0

    # 가중치 적용한 종합 점수
    # w1=0.35 (IP 공유), w2=0.30 (동시 거래), w3=0.20 (가격 유사), w4=0.15 (레버리지)
    total_score = (0.35 * ip_score +
                   0.30 * concurrent_score +
                   0.20 * price_score +
                   0.15 * leverage_score)

    organized_scores.append({
        'account_id': account,
        'ip_shared_ratio': ip_data['shared_ratio'],
        'ip_score': ip_score,
        'concurrent_trading_ratio': concurrent_ratio,
        'concurrent_score': concurrent_score,
        'price_similarity_ratio': price_sim_ratio,
        'price_score': price_score,
        'mean_leverage': lev_data['mean_leverage'].values[0] if len(lev_data) > 0 else 0,
        'leverage_score': leverage_score,
        'organized_score': total_score,
        'total_trades': account_total_trades.get(account, 0)
    })

organized_df = pd.DataFrame(organized_scores)
organized_df = organized_df.sort_values('organized_score', ascending=False)

print(f"\n[종합 점수 분포]")
print(f"  평균: {organized_df['organized_score'].mean():.4f}")
print(f"  중앙값: {organized_df['organized_score'].median():.4f}")
print(f"  표준편차: {organized_df['organized_score'].std():.4f}")
print(f"  95th percentile: {organized_df['organized_score'].quantile(0.95):.4f}")
print(f"  99th percentile: {organized_df['organized_score'].quantile(0.99):.4f}")

# 위험도 분류
# - Low Risk: score < 0.3
# - Medium Risk: 0.3 <= score < 0.6
# - High Risk: score >= 0.6
organized_df['risk_level'] = pd.cut(
    organized_df['organized_score'],
    bins=[-np.inf, 0.3, 0.6, np.inf],
    labels=['Low', 'Medium', 'High']
)

risk_counts = organized_df['risk_level'].value_counts()
print(f"\n[위험도 분류]")
for level in ['High', 'Medium', 'Low']:
    count = risk_counts.get(level, 0)
    print(f"  {level} Risk: {count}개 ({count/len(organized_df)*100:.2f}%)")

# 고위험 계정
high_risk_accounts = organized_df[organized_df['risk_level'] == 'High']
print(f"\n[고위험 계정 Top 10]")
for i, row in high_risk_accounts.head(10).iterrows():
    print(f"  {row['account_id']}: Score={row['organized_score']:.4f} "
          f"(IP={row['ip_score']:.2f}, Concurrent={row['concurrent_score']:.2f}, "
          f"Price={row['price_score']:.2f}, Lev={row['leverage_score']:.2f})")

# ================================================================================
# 7. 결과 저장
# ================================================================================
print("\n" + "=" * 100)
print("[7] 결과 저장")
print("=" * 100)

# 전체 결과
organized_df.to_csv('output/organized_trading/organized_scores_all.csv', index=False)
print(f"✓ 전체 계정 점수: output/organized_trading/organized_scores_all.csv ({len(organized_df)}개)")

# 고위험 계정만
high_risk_accounts.to_csv('output/organized_trading/organized_high_risk.csv', index=False)
print(f"✓ 고위험 계정: output/organized_trading/organized_high_risk.csv ({len(high_risk_accounts)}개)")

# 공유 IP 상세
shared_ips.to_csv('output/organized_trading/shared_ips_detail.csv', index=False)
print(f"✓ 공유 IP 상세: output/organized_trading/shared_ips_detail.csv ({len(shared_ips)}개)")

# 요약 통계
summary_stats = pd.DataFrame({
    'Metric': [
        '총 계정 수',
        '고위험 계정 수',
        '중위험 계정 수',
        '저위험 계정 수',
        '평균 점수',
        '중앙값 점수',
        '95th percentile',
        '다계정 IP 수',
        '동시 거래 발생 횟수',
        'IP 공유 의심 계정',
        '동시 거래 의심 계정',
        '고레버리지 의심 계정'
    ],
    'Value': [
        len(organized_df),
        risk_counts.get('High', 0),
        risk_counts.get('Medium', 0),
        risk_counts.get('Low', 0),
        f"{organized_df['organized_score'].mean():.4f}",
        f"{organized_df['organized_score'].median():.4f}",
        f"{organized_df['organized_score'].quantile(0.95):.4f}",
        len(shared_ips),
        len(concurrent_multi),
        len(suspicious_accounts_ip),
        len(suspicious_accounts_concurrent),
        len(suspicious_accounts_leverage)
    ]
})
summary_stats.to_csv('output/organized_trading/summary_statistics.csv', index=False)
print(f"✓ 요약 통계: output/organized_trading/summary_statistics.csv")

# ================================================================================
# 8. 시각화
# ================================================================================
print("\n" + "=" * 100)
print("[8] 시각화 생성")
print("=" * 100)

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. 종합 점수 분포
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(organized_df['organized_score'], bins=50, edgecolor='black', alpha=0.7, color='coral')
ax1.axvline(0.3, color='orange', linestyle='--', linewidth=2, label='Medium threshold')
ax1.axvline(0.6, color='red', linestyle='--', linewidth=2, label='High threshold')
ax1.set_xlabel('Organized Trading Score', fontsize=11)
ax1.set_ylabel('계정 수', fontsize=11)
ax1.set_title('1. 조직적 거래 종합 점수 분포', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 위험도별 계정 수
ax2 = fig.add_subplot(gs[0, 1])
risk_counts_sorted = risk_counts.reindex(['High', 'Medium', 'Low'])
colors = ['red', 'orange', 'green']
bars = ax2.bar(range(len(risk_counts_sorted)), risk_counts_sorted.values, color=colors, edgecolor='black')
ax2.set_xticks(range(len(risk_counts_sorted)))
ax2.set_xticklabels(risk_counts_sorted.index, fontsize=11)
ax2.set_ylabel('계정 수', fontsize=11)
ax2.set_title('2. 위험도별 분류', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
for i, (bar, count) in enumerate(zip(bars, risk_counts_sorted.values)):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{count}\n({count/len(organized_df)*100:.1f}%)',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# 3. IP 공유 비율 분포
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(organized_df['ip_shared_ratio']*100, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
ax3.axvline(50, color='orange', linestyle='--', linewidth=2, label='의심 기준 (50%)')
ax3.axvline(80, color='red', linestyle='--', linewidth=2, label='고위험 기준 (80%)')
ax3.set_xlabel('IP 공유 비율 (%)', fontsize=11)
ax3.set_ylabel('계정 수', fontsize=11)
ax3.set_title('3. IP 공유 비율 분포', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. 동시 거래 비율 분포
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(organized_df['concurrent_trading_ratio']*100, bins=50, edgecolor='black', alpha=0.7, color='lightgreen')
ax4.axvline(50, color='orange', linestyle='--', linewidth=2, label='의심 기준 (50%)')
ax4.axvline(70, color='red', linestyle='--', linewidth=2, label='고위험 기준 (70%)')
ax4.set_xlabel('동시 거래 비율 (%)', fontsize=11)
ax4.set_ylabel('계정 수', fontsize=11)
ax4.set_title('4. 동시 거래 비율 분포', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. 레버리지 분포
ax5 = fig.add_subplot(gs[1, 1])
lev_data = organized_df[organized_df['mean_leverage'] > 0]['mean_leverage']
ax5.hist(lev_data, bins=50, edgecolor='black', alpha=0.7, color='plum')
ax5.axvline(30, color='orange', linestyle='--', linewidth=2, label='의심 기준 (30x)')
ax5.axvline(50, color='red', linestyle='--', linewidth=2, label='고위험 기준 (50x)')
ax5.set_xlabel('평균 레버리지 (x)', fontsize=11)
ax5.set_ylabel('계정 수', fontsize=11)
ax5.set_title('5. 평균 레버리지 분포', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. 점수별 계정 수 (누적)
ax6 = fig.add_subplot(gs[1, 2])
sorted_scores = np.sort(organized_df['organized_score'])
cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
ax6.plot(sorted_scores, cdf, linewidth=2, color='navy')
ax6.axvline(0.3, color='orange', linestyle='--', linewidth=2, alpha=0.7)
ax6.axvline(0.6, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax6.axhline(0.95, color='gray', linestyle=':', linewidth=1)
ax6.set_xlabel('Organized Trading Score', fontsize=11)
ax6.set_ylabel('누적 확률', fontsize=11)
ax6.set_title('6. 점수 누적 분포 (CDF)', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)

# 7. 고위험 계정 Top 15
ax7 = fig.add_subplot(gs[2, :2])
top_15 = organized_df.head(15)
y_pos = np.arange(len(top_15))
bars = ax7.barh(y_pos, top_15['organized_score'], color='crimson', edgecolor='black')
ax7.set_yticks(y_pos)
ax7.set_yticklabels([f"{acc[:15]}..." for acc in top_15['account_id']], fontsize=9)
ax7.set_xlabel('Organized Trading Score', fontsize=11)
ax7.set_title('7. 조직적 거래 의심 계정 Top 15', fontsize=12, fontweight='bold')
ax7.invert_yaxis()
ax7.grid(True, alpha=0.3, axis='x')
for i, (idx, row) in enumerate(top_15.iterrows()):
    ax7.text(row['organized_score'], i, f" {row['organized_score']:.3f}",
             va='center', fontsize=8, fontweight='bold')

# 8. 점수 요소 상관관계
ax8 = fig.add_subplot(gs[2, 2])
score_components = organized_df[['ip_score', 'concurrent_score', 'price_score', 'leverage_score']].corr()
im = ax8.imshow(score_components, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
ax8.set_xticks(range(4))
ax8.set_yticks(range(4))
ax8.set_xticklabels(['IP', 'Concurrent', 'Price', 'Leverage'], fontsize=9, rotation=45)
ax8.set_yticklabels(['IP', 'Concurrent', 'Price', 'Leverage'], fontsize=9)
ax8.set_title('8. 점수 요소 상관관계', fontsize=12, fontweight='bold')
for i in range(4):
    for j in range(4):
        text = ax8.text(j, i, f'{score_components.iloc[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=9)
plt.colorbar(im, ax=ax8)

fig.suptitle('조직적 이상거래(Organized Trading) 탐지 분석 결과',
             fontsize=18, fontweight='bold', y=0.998)

plt.savefig('output/organized_trading/analysis_visualization.png', dpi=300, bbox_inches='tight')
print(f"✓ 시각화: output/organized_trading/analysis_visualization.png")

print("\n" + "=" * 100)
print("분석 완료!")
print("=" * 100)
print("\n[핵심 공식]")
print("OrganizedScore = 0.35 × IP_Score + 0.30 × Concurrent_Score + 0.20 × Price_Score + 0.15 × Leverage_Score")
print("\n각 점수는 0~1로 정규화되며, 임계값:")
print("  - Low Risk: Score < 0.3")
print("  - Medium Risk: 0.3 ≤ Score < 0.6")
print("  - High Risk: Score ≥ 0.6")
