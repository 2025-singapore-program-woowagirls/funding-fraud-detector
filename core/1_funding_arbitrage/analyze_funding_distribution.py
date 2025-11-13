import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
print("=" * 80)
print("데이터 로딩 중...")
print("=" * 80)

funding_df = pd.read_csv('data/Funding.csv')
trade_df = pd.read_csv('data/Trade.csv')
reward_df = pd.read_csv('data/Reward.csv')
ip_df = pd.read_csv('data/IP.csv')
spec_df = pd.read_csv('data/Spec.csv')

print(f"\n✓ Funding 데이터: {len(funding_df):,} rows")
print(f"✓ Trade 데이터: {len(trade_df):,} rows")
print(f"✓ Reward 데이터: {len(reward_df):,} rows")
print(f"✓ IP 데이터: {len(ip_df):,} rows")
print(f"✓ Spec 데이터: {len(spec_df):,} rows")

# 데이터 구조 확인
print("\n" + "=" * 80)
print("데이터 구조 요약")
print("=" * 80)

print("\n[Funding.csv]")
print(f"  컬럼: {list(funding_df.columns)}")
print(f"  기간: {funding_df['ts'].min()} ~ {funding_df['ts'].max()}")
print(f"  고유 계정 수: {funding_df['account_id'].nunique():,}")

print("\n[Trade.csv]")
print(f"  컬럼: {list(trade_df.columns)}")
print(f"  기간: {trade_df['ts'].min()} ~ {trade_df['ts'].max()}")
print(f"  고유 계정 수: {trade_df['account_id'].nunique():,}")

print("\n[Reward.csv]")
print(f"  컬럼: {list(reward_df.columns)}")
print(f"  고유 계정 수: {reward_df['account_id'].nunique():,}")

print("\n[IP.csv]")
print(f"  컬럼: {list(ip_df.columns)}")
print(f"  고유 계정 수: {ip_df['account_id'].nunique():,}")
print(f"  고유 IP 수: {ip_df['ip'].nunique():,}")

print("\n[Spec.csv]")
print(f"  컬럼: {list(spec_df.columns)}")
print(f"  고유 심볼 수: {spec_df['symbol'].nunique():,}")

# 펀딩피 사용자별 집계
print("\n" + "=" * 80)
print("펀딩피 사용자별 분석")
print("=" * 80)

funding_by_user = funding_df.groupby('account_id').agg({
    'funding_fee': ['sum', 'mean', 'std', 'count', 'min', 'max'],
    'fee_rate': ['mean', 'std']
}).reset_index()

funding_by_user.columns = ['account_id', 'total_funding_fee', 'mean_funding_fee',
                            'std_funding_fee', 'funding_count', 'min_funding_fee',
                            'max_funding_fee', 'mean_fee_rate', 'std_fee_rate']

# 통계량 출력
print(f"\n전체 통계:")
print(f"  총 펀딩피 합계: ${funding_by_user['total_funding_fee'].sum():,.2f}")
print(f"  사용자 수: {len(funding_by_user):,}")
print(f"  사용자당 평균 펀딩피: ${funding_by_user['total_funding_fee'].mean():,.2f}")
print(f"  사용자당 펀딩피 중앙값: ${funding_by_user['total_funding_fee'].median():,.2f}")
print(f"  사용자당 펀딩피 표준편차: ${funding_by_user['total_funding_fee'].std():,.2f}")

print(f"\n분포 통계 (Total Funding Fee by User):")
percentiles = [10, 25, 50, 75, 90, 95, 99]
for p in percentiles:
    val = funding_by_user['total_funding_fee'].quantile(p/100)
    print(f"  {p}th percentile: ${val:,.2f}")

# 이상치 탐지를 위한 기준선
mean_funding = funding_by_user['total_funding_fee'].mean()
std_funding = funding_by_user['total_funding_fee'].std()
threshold_2sigma = mean_funding + 2 * std_funding
threshold_3sigma = mean_funding + 3 * std_funding

print(f"\n이상치 기준선:")
print(f"  평균 (μ): ${mean_funding:,.2f}")
print(f"  표준편차 (σ): ${std_funding:,.2f}")
print(f"  μ + 2σ (95% CI): ${threshold_2sigma:,.2f}")
print(f"  μ + 3σ (99.7% CI): ${threshold_3sigma:,.2f}")

outliers_2sigma = funding_by_user[funding_by_user['total_funding_fee'] > threshold_2sigma]
outliers_3sigma = funding_by_user[funding_by_user['total_funding_fee'] > threshold_3sigma]

print(f"\n이상 사용자 수:")
print(f"  2σ 초과: {len(outliers_2sigma)} ({len(outliers_2sigma)/len(funding_by_user)*100:.2f}%)")
print(f"  3σ 초과: {len(outliers_3sigma)} ({len(outliers_3sigma)/len(funding_by_user)*100:.2f}%)")

# 시각화
print("\n" + "=" * 80)
print("시각화 생성 중...")
print("=" * 80)

fig, axes = plt.subplots(3, 2, figsize=(16, 18))
fig.suptitle('펀딩피 사용자별 분포 분석', fontsize=20, fontweight='bold', y=0.995)

# 1. 전체 분포 (히스토그램)
ax1 = axes[0, 0]
ax1.hist(funding_by_user['total_funding_fee'], bins=100, edgecolor='black', alpha=0.7)
ax1.axvline(mean_funding, color='red', linestyle='--', linewidth=2, label=f'평균: ${mean_funding:.2f}')
ax1.axvline(threshold_2sigma, color='orange', linestyle='--', linewidth=2, label=f'μ+2σ: ${threshold_2sigma:.2f}')
ax1.axvline(threshold_3sigma, color='darkred', linestyle='--', linewidth=2, label=f'μ+3σ: ${threshold_3sigma:.2f}')
ax1.set_xlabel('Total Funding Fee ($)', fontsize=12)
ax1.set_ylabel('사용자 수', fontsize=12)
ax1.set_title('1. 펀딩피 전체 분포 (히스토그램)', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 로그 스케일 분포
ax2 = axes[0, 1]
positive_funding = funding_by_user[funding_by_user['total_funding_fee'] > 0]['total_funding_fee']
ax2.hist(np.log10(positive_funding + 1), bins=100, edgecolor='black', alpha=0.7, color='green')
ax2.set_xlabel('Log10(Total Funding Fee + 1)', fontsize=12)
ax2.set_ylabel('사용자 수', fontsize=12)
ax2.set_title('2. 로그 스케일 분포 (양수만)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. 박스플롯
ax3 = axes[1, 0]
bp = ax3.boxplot([funding_by_user['total_funding_fee']],
                  vert=True,
                  patch_artist=True,
                  widths=0.5)
bp['boxes'][0].set_facecolor('lightblue')
ax3.set_ylabel('Total Funding Fee ($)', fontsize=12)
ax3.set_title('3. 박스플롯 (이상치 확인)', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. CDF (누적분포함수)
ax4 = axes[1, 1]
sorted_funding = np.sort(funding_by_user['total_funding_fee'])
cdf = np.arange(1, len(sorted_funding) + 1) / len(sorted_funding)
ax4.plot(sorted_funding, cdf, linewidth=2)
ax4.axhline(0.95, color='red', linestyle='--', label='95th percentile')
ax4.axhline(0.99, color='orange', linestyle='--', label='99th percentile')
ax4.set_xlabel('Total Funding Fee ($)', fontsize=12)
ax4.set_ylabel('누적 확률', fontsize=12)
ax4.set_title('4. 누적 분포 함수 (CDF)', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. 상위 20명 사용자
ax5 = axes[2, 0]
top_20 = funding_by_user.nlargest(20, 'total_funding_fee')
bars = ax5.barh(range(len(top_20)), top_20['total_funding_fee'], color='coral')
ax5.set_yticks(range(len(top_20)))
ax5.set_yticklabels([f"{acc[:12]}..." for acc in top_20['account_id']], fontsize=9)
ax5.set_xlabel('Total Funding Fee ($)', fontsize=12)
ax5.set_title('5. 펀딩피 수령 상위 20명 사용자', fontsize=14, fontweight='bold')
ax5.invert_yaxis()
ax5.grid(True, alpha=0.3, axis='x')

# 값 표시
for i, (idx, row) in enumerate(top_20.iterrows()):
    ax5.text(row['total_funding_fee'], i, f" ${row['total_funding_fee']:,.0f}",
             va='center', fontsize=8)

# 6. 펀딩 횟수 vs 총 펀딩피
ax6 = axes[2, 1]
scatter = ax6.scatter(funding_by_user['funding_count'],
                     funding_by_user['total_funding_fee'],
                     alpha=0.5, s=30, c=funding_by_user['mean_funding_fee'],
                     cmap='viridis')
ax6.set_xlabel('펀딩 횟수', fontsize=12)
ax6.set_ylabel('Total Funding Fee ($)', fontsize=12)
ax6.set_title('6. 펀딩 횟수 vs 총 펀딩피', fontsize=14, fontweight='bold')
ax6.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax6, label='평균 펀딩피')

plt.tight_layout()
plt.savefig('output/funding_distribution_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ 시각화 저장: output/funding_distribution_analysis.png")

# 상세 통계 테이블 저장
print("\n" + "=" * 80)
print("상세 분석 결과 저장 중...")
print("=" * 80)

# 이상 사용자 리스트 저장
outliers_2sigma_sorted = outliers_2sigma.sort_values('total_funding_fee', ascending=False)
outliers_2sigma_sorted.to_csv('output/funding_outliers_2sigma.csv', index=False)
print(f"\n✓ 2σ 이상치 사용자 저장: output/funding_outliers_2sigma.csv ({len(outliers_2sigma_sorted)}명)")

# 전체 사용자 통계 저장
funding_by_user_sorted = funding_by_user.sort_values('total_funding_fee', ascending=False)
funding_by_user_sorted.to_csv('output/funding_by_user_full.csv', index=False)
print(f"✓ 전체 사용자 펀딩피 통계 저장: output/funding_by_user_full.csv ({len(funding_by_user_sorted)}명)")

# 요약 통계 리포트
summary_stats = pd.DataFrame({
    'Metric': [
        '총 사용자 수',
        '총 펀딩피 합계',
        '평균 (μ)',
        '중앙값',
        '표준편차 (σ)',
        '최소값',
        '최대값',
        '25th percentile',
        '75th percentile',
        '90th percentile',
        '95th percentile',
        '99th percentile',
        'μ + 2σ (임계값)',
        'μ + 3σ (임계값)',
        '2σ 초과 사용자 수',
        '3σ 초과 사용자 수'
    ],
    'Value': [
        f"{len(funding_by_user):,}",
        f"${funding_by_user['total_funding_fee'].sum():,.2f}",
        f"${mean_funding:,.2f}",
        f"${funding_by_user['total_funding_fee'].median():,.2f}",
        f"${std_funding:,.2f}",
        f"${funding_by_user['total_funding_fee'].min():,.2f}",
        f"${funding_by_user['total_funding_fee'].max():,.2f}",
        f"${funding_by_user['total_funding_fee'].quantile(0.25):,.2f}",
        f"${funding_by_user['total_funding_fee'].quantile(0.75):,.2f}",
        f"${funding_by_user['total_funding_fee'].quantile(0.90):,.2f}",
        f"${funding_by_user['total_funding_fee'].quantile(0.95):,.2f}",
        f"${funding_by_user['total_funding_fee'].quantile(0.99):,.2f}",
        f"${threshold_2sigma:,.2f}",
        f"${threshold_3sigma:,.2f}",
        f"{len(outliers_2sigma)} ({len(outliers_2sigma)/len(funding_by_user)*100:.2f}%)",
        f"{len(outliers_3sigma)} ({len(outliers_3sigma)/len(funding_by_user)*100:.2f}%)"
    ]
})

summary_stats.to_csv('output/funding_summary_statistics.csv', index=False)
print(f"✓ 요약 통계 저장: output/funding_summary_statistics.csv")

print("\n" + "=" * 80)
print("피처 탐색을 위한 추가 분석")
print("=" * 80)

print("\n[잠재적 이상거래 탐지 피처]")
print("\n1. Funding Fee 관련:")
print("   - total_funding_fee: 총 펀딩피 수령액 (μ+2σ 초과 시 의심)")
print("   - funding_profit_ratio: 펀딩피 수익 / 총 거래 수익 비율")
print("   - funding_count: 펀딩 발생 횟수")
print("   - mean_funding_fee: 평균 펀딩피")
print("   - std_funding_fee: 펀딩피 변동성")

print("\n2. Trading Pattern 관련:")
print("   - holding_time: 포지션 유지 시간 (짧을수록 의심)")
print("   - leverage: 평균 레버리지 (높을수록 의심)")
print("   - trade_frequency: 거래 빈도")
print("   - position_size_consistency: 포지션 크기 일관성")

print("\n3. Network 관련:")
print("   - shared_ip_count: IP 공유 계정 수")
print("   - concurrent_trading: 동시 거래 발생 빈도")

print("\n4. Reward 관련:")
print("   - reward_count: 보너스 수령 횟수")
print("   - reward_total: 총 보너스 금액")
print("   - inactive_after_reward: 보너스 후 비활성 여부")

print("\n다음 단계:")
print("  → Trade 데이터와 결합하여 holding_time, leverage 분석")
print("  → IP 데이터로 다계정 패턴 분석")
print("  → 펀딩 시각 전후 거래 집중도 분석")

print("\n" + "=" * 80)
print("분석 완료!")
print("=" * 80)
