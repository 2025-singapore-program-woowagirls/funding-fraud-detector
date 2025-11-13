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
print("최종 통합 이상거래 탐지 시스템 (Integrated Anomaly Detection System)")
print("=" * 120)

# ================================================================================
# 1. 이전 분석 결과 로드
# ================================================================================
print("\n[1] 이전 분석 결과 통합")
print("=" * 120)

# 1.1 펀딩피 차익거래
funding_arb_df = pd.read_csv('output/funding_analysis/funding_arbitrage_scores_all.csv')
print(f"✓ 펀딩피 차익거래 분석: {len(funding_arb_df)}개 계정")

# 1.2 조직적 거래
organized_df = pd.read_csv('output/organized_trading/organized_scores_all.csv')
print(f"✓ 조직적 거래 분석: {len(organized_df)}개 계정")

# 1.3 보너스 악용
bonus_df = pd.read_csv('output/bonus_abuse/bonus_abuse_scores_all.csv')
print(f"✓ 보너스 악용 분석: {len(bonus_df)}개 계정")

# 1.4 퀀트 피처
quant_df = pd.read_csv('output/funding_analysis/quant_features_all.csv')
print(f"✓ 퀀트 기반 분석: {len(quant_df)}개 계정")

# ================================================================================
# 2. 계정별 통합 데이터 구축
# ================================================================================
print("\n[2] 계정별 통합 데이터 구축")
print("=" * 120)

# 모든 계정 리스트
all_accounts = set()
all_accounts.update(funding_arb_df['account_id'].tolist())
all_accounts.update(organized_df['account_id'].tolist())
all_accounts.update(bonus_df['account_id'].tolist())
all_accounts.update(quant_df['account_id'].tolist())

print(f"총 분석 대상 계정: {len(all_accounts)}개")

# 통합 데이터프레임 구축
integrated_data = []

for account in all_accounts:
    # 펀딩피 차익거래 점수
    funding_data = funding_arb_df[funding_arb_df['account_id'] == account]
    if len(funding_data) > 0:
        funding_score = funding_data['funding_arbitrage_score'].values[0]
        mean_holding = funding_data['mean_holding_minutes'].values[0]
        funding_timing_ratio = funding_data['funding_timing_ratio'].values[0]
    else:
        funding_score = 0
        mean_holding = 0
        funding_timing_ratio = 0

    # 조직적 거래 점수
    organized_data = organized_df[organized_df['account_id'] == account]
    if len(organized_data) > 0:
        organized_score = organized_data['organized_score'].values[0]
        concurrent_ratio = organized_data['concurrent_trading_ratio'].values[0]
        ip_shared = organized_data['ip_shared_ratio'].values[0]
    else:
        organized_score = 0
        concurrent_ratio = 0
        ip_shared = 0

    # 보너스 악용 점수
    bonus_data = bonus_df[bonus_df['account_id'] == account]
    if len(bonus_data) > 0:
        bonus_score = bonus_data['bonus_abuse_score'].values[0]
        total_reward = bonus_data['total_reward'].values[0]
    else:
        bonus_score = 0
        total_reward = 0

    # 퀀트 피처 점수
    quant_data = quant_df[quant_df['account_id'] == account]
    if len(quant_data) > 0:
        quant_score = quant_data['quant_anomaly_score'].values[0]
        sharpe = quant_data['sharpe_ratio'].values[0]
        win_rate = quant_data['win_rate'].values[0]
    else:
        quant_score = 0
        sharpe = 0
        win_rate = 0

    integrated_data.append({
        'account_id': account,
        'funding_arbitrage_score': funding_score,
        'organized_trading_score': organized_score,
        'bonus_abuse_score': bonus_score,
        'quant_anomaly_score': quant_score,
        'mean_holding_minutes': mean_holding,
        'funding_timing_ratio': funding_timing_ratio,
        'concurrent_trading_ratio': concurrent_ratio,
        'ip_shared_ratio': ip_shared,
        'total_reward': total_reward,
        'sharpe_ratio': sharpe,
        'win_rate': win_rate
    })

integrated_df = pd.DataFrame(integrated_data)

# ================================================================================
# 3. 최종 통합 위험 점수 산출
# ================================================================================
print("\n[3] 최종 통합 위험 점수 산출")
print("=" * 120)

print("\n[가중치 설계]")
print("펀딩피 차익거래의 위험도와 조직적 거래의 위험도를 달리 평가:")
print("  - 펀딩피 차익거래: 0.30 (시장 왜곡도 중간)")
print("  - 조직적 거래: 0.35 (다계정 악용, 고위험)")
print("  - 보너스 악용: 0.20 (금액 손실, 중위험)")
print("  - 퀀트 이상: 0.15 (보조 지표)")

# 최종 통합 점수 계산
integrated_df['final_risk_score'] = (
    0.30 * integrated_df['funding_arbitrage_score'] +
    0.35 * integrated_df['organized_trading_score'] +
    0.20 * integrated_df['bonus_abuse_score'] +
    0.15 * integrated_df['quant_anomaly_score']
)

integrated_df = integrated_df.sort_values('final_risk_score', ascending=False)

print(f"\n[통합 점수 분포]")
print(f"  평균: {integrated_df['final_risk_score'].mean():.4f}")
print(f"  중앙값: {integrated_df['final_risk_score'].median():.4f}")
print(f"  표준편차: {integrated_df['final_risk_score'].std():.4f}")
print(f"  95th: {integrated_df['final_risk_score'].quantile(0.95):.4f}")
print(f"  99th: {integrated_df['final_risk_score'].quantile(0.99):.4f}")

# 위험도 등급 분류
integrated_df['final_risk_level'] = pd.cut(
    integrated_df['final_risk_score'],
    bins=[-np.inf, 0.3, 0.5, 0.7, np.inf],
    labels=['Low', 'Medium', 'High', 'Critical']
)

risk_counts = integrated_df['final_risk_level'].value_counts()
print(f"\n[최종 위험도 분류]")
for level in ['Critical', 'High', 'Medium', 'Low']:
    count = risk_counts.get(level, 0)
    print(f"  {level}: {count}개 ({count/len(integrated_df)*100:.2f}%)")

# ================================================================================
# 4. 패턴별 교차 분석
# ================================================================================
print("\n[4] 패턴 교차 분석 (Cross-Pattern Analysis)")
print("=" * 120)

# 4.1 다중 패턴 감지 계정
multi_pattern_threshold = 0.4

integrated_df['pattern_count'] = (
    (integrated_df['funding_arbitrage_score'] > multi_pattern_threshold).astype(int) +
    (integrated_df['organized_trading_score'] > multi_pattern_threshold).astype(int) +
    (integrated_df['bonus_abuse_score'] > multi_pattern_threshold).astype(int) +
    (integrated_df['quant_anomaly_score'] > multi_pattern_threshold).astype(int)
)

multi_pattern = integrated_df[integrated_df['pattern_count'] >= 2]
print(f"\n다중 이상 패턴 (2개 이상): {len(multi_pattern)}개 계정 🚨")

if len(multi_pattern) > 0:
    print(f"\n[다중 패턴 계정 상세]")
    for _, row in multi_pattern.head(10).iterrows():
        print(f"\n  {row['account_id']}: 최종점수={row['final_risk_score']:.4f}, 패턴수={int(row['pattern_count'])}")
        print(f"    Funding={row['funding_arbitrage_score']:.3f}, Organized={row['organized_trading_score']:.3f}, "
              f"Bonus={row['bonus_abuse_score']:.3f}, Quant={row['quant_anomaly_score']:.3f}")

# 4.2 패턴 간 상관관계
print(f"\n[패턴 간 상관계수]")
correlation_matrix = integrated_df[[
    'funding_arbitrage_score',
    'organized_trading_score',
    'bonus_abuse_score',
    'quant_anomaly_score'
]].corr()

print(correlation_matrix.to_string())

# ================================================================================
# 5. 고위험 계정 프로파일링
# ================================================================================
print("\n" + "=" * 120)
print("[5] 고위험 계정 프로파일링")
print("=" * 120)

critical_accounts = integrated_df[integrated_df['final_risk_level'] == 'Critical']
high_accounts = integrated_df[integrated_df['final_risk_level'] == 'High']

print(f"\n🚨 Critical 등급: {len(critical_accounts)}개")
print(f"⚠️  High 등급: {len(high_accounts)}개")

print(f"\n[Top 20 고위험 계정 프로파일]")
print("-" * 120)
top_20 = integrated_df.head(20)

for i, row in top_20.iterrows():
    print(f"\n{i+1}. {row['account_id']} | 최종점수: {row['final_risk_score']:.4f} | 등급: {row['final_risk_level']}")
    print(f"   └─ 펀딩피차익: {row['funding_arbitrage_score']:.3f} (보유시간={row['mean_holding_minutes']:.1f}분)")
    print(f"   └─ 조직거래: {row['organized_trading_score']:.3f} (동시거래={row['concurrent_trading_ratio']*100:.1f}%, IP공유={row['ip_shared_ratio']*100:.1f}%)")
    print(f"   └─ 보너스악용: {row['bonus_abuse_score']:.3f} (총보너스=${row['total_reward']:.2f})")
    print(f"   └─ 퀀트이상: {row['quant_anomaly_score']:.3f} (Sharpe={row['sharpe_ratio']:.2f}, 승률={row['win_rate']*100:.1f}%)")

# ================================================================================
# 6. 결과 저장
# ================================================================================
print("\n" + "=" * 120)
print("[6] 최종 결과 저장")
print("=" * 120)

# 전체 통합 데이터
integrated_df.to_csv('output/final_integrated_risk_scores.csv', index=False)
print(f"✓ 통합 점수: output/final_integrated_risk_scores.csv ({len(integrated_df)}개)")

# Critical/High 등급만
critical_high = integrated_df[integrated_df['final_risk_level'].isin(['Critical', 'High'])]
critical_high.to_csv('output/critical_high_risk_accounts.csv', index=False)
print(f"✓ Critical/High 등급: output/critical_high_risk_accounts.csv ({len(critical_high)}개)")

# 다중 패턴 계정
if len(multi_pattern) > 0:
    multi_pattern.to_csv('output/multi_pattern_accounts.csv', index=False)
    print(f"✓ 다중 패턴: output/multi_pattern_accounts.csv ({len(multi_pattern)}개)")

# 요약 통계
summary = pd.DataFrame({
    'Metric': [
        '총 분석 계정',
        'Critical 등급',
        'High 등급',
        'Medium 등급',
        'Low 등급',
        '다중 패턴 계정 (2+)',
        '평균 최종 점수',
        '중앙값 최종 점수',
        '최고 위험 점수',
        '펀딩피 고위험(>0.6)',
        '조직거래 고위험(>0.6)',
        '보너스 고위험(>0.6)',
        '퀀트 고위험(>0.6)'
    ],
    'Value': [
        len(integrated_df),
        risk_counts.get('Critical', 0),
        risk_counts.get('High', 0),
        risk_counts.get('Medium', 0),
        risk_counts.get('Low', 0),
        len(multi_pattern),
        f"{integrated_df['final_risk_score'].mean():.4f}",
        f"{integrated_df['final_risk_score'].median():.4f}",
        f"{integrated_df['final_risk_score'].max():.4f}",
        len(integrated_df[integrated_df['funding_arbitrage_score'] > 0.6]),
        len(integrated_df[integrated_df['organized_trading_score'] > 0.6]),
        len(integrated_df[integrated_df['bonus_abuse_score'] > 0.6]),
        len(integrated_df[integrated_df['quant_anomaly_score'] > 0.6])
    ]
})

summary.to_csv('output/final_summary_statistics.csv', index=False)
print(f"✓ 요약 통계: output/final_summary_statistics.csv")

# ================================================================================
# 7. 최종 시각화
# ================================================================================
print("\n" + "=" * 120)
print("[7] 최종 통합 시각화")
print("=" * 120)

fig = plt.figure(figsize=(24, 16))
gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)

# 1. 최종 위험 점수 분포
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(integrated_df['final_risk_score'], bins=50, edgecolor='black', alpha=0.7, color='crimson')
ax1.axvline(0.3, color='orange', linestyle='--', linewidth=2, label='Medium (0.3)')
ax1.axvline(0.5, color='red', linestyle='--', linewidth=2, label='High (0.5)')
ax1.axvline(0.7, color='darkred', linestyle='--', linewidth=2, label='Critical (0.7)')
ax1.set_xlabel('Final Risk Score', fontsize=11)
ax1.set_ylabel('계정 수', fontsize=11)
ax1.set_title('1. 최종 위험 점수 분포', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# 2. 위험도 등급 분포
ax2 = fig.add_subplot(gs[0, 1])
risk_sorted = risk_counts.reindex(['Critical', 'High', 'Medium', 'Low'])
colors_map = {'Critical': 'darkred', 'High': 'red', 'Medium': 'orange', 'Low': 'green'}
colors = [colors_map.get(level, 'gray') for level in risk_sorted.index]
bars = ax2.bar(range(len(risk_sorted)), risk_sorted.values, color=colors, edgecolor='black')
ax2.set_xticks(range(len(risk_sorted)))
ax2.set_xticklabels(risk_sorted.index, fontsize=10)
ax2.set_ylabel('계정 수', fontsize=11)
ax2.set_title('2. 최종 위험도 등급', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
for bar, count in zip(bars, risk_sorted.values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, height,
             f'{count}\n({count/len(integrated_df)*100:.1f}%)',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# 3. 패턴별 점수 분포 (바이올린 플롯)
ax3 = fig.add_subplot(gs[0, 2:])
pattern_scores = pd.DataFrame({
    'Funding Arb': integrated_df['funding_arbitrage_score'],
    'Organized': integrated_df['organized_trading_score'],
    'Bonus Abuse': integrated_df['bonus_abuse_score'],
    'Quant Anomaly': integrated_df['quant_anomaly_score']
})
parts = ax3.violinplot([pattern_scores[col].values for col in pattern_scores.columns],
                        positions=range(4), showmeans=True, showmedians=True)
ax3.set_xticks(range(4))
ax3.set_xticklabels(pattern_scores.columns, fontsize=10, rotation=15)
ax3.set_ylabel('Score', fontsize=11)
ax3.set_title('3. 패턴별 점수 분포 (Violin Plot)', fontsize=12, fontweight='bold')
ax3.axhline(0.6, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax3.grid(True, alpha=0.3, axis='y')

# 4. 패턴 간 상관관계 히트맵
ax4 = fig.add_subplot(gs[1, 0])
im = ax4.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
ax4.set_xticks(range(4))
ax4.set_yticks(range(4))
labels = ['Funding', 'Organized', 'Bonus', 'Quant']
ax4.set_xticklabels(labels, fontsize=9, rotation=45)
ax4.set_yticklabels(labels, fontsize=9)
ax4.set_title('4. 패턴 간 상관관계', fontsize=12, fontweight='bold')
for i in range(4):
    for j in range(4):
        text = ax4.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                        ha="center", va="center", color="black", fontsize=9)
plt.colorbar(im, ax=ax4, fraction=0.046)

# 5. 펀딩 vs 조직 산점도
ax5 = fig.add_subplot(gs[1, 1])
scatter = ax5.scatter(integrated_df['funding_arbitrage_score'],
                      integrated_df['organized_trading_score'],
                      c=integrated_df['final_risk_score'],
                      cmap='Reds', s=50, alpha=0.6, edgecolors='black')
ax5.axvline(0.6, color='gray', linestyle='--', alpha=0.5)
ax5.axhline(0.6, color='gray', linestyle='--', alpha=0.5)
ax5.set_xlabel('Funding Arbitrage Score', fontsize=11)
ax5.set_ylabel('Organized Trading Score', fontsize=11)
ax5.set_title('5. 펀딩피 vs 조직거래', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax5, label='Final Risk')

# 6. 보너스 vs 퀀트 산점도
ax6 = fig.add_subplot(gs[1, 2])
scatter2 = ax6.scatter(integrated_df['bonus_abuse_score'],
                       integrated_df['quant_anomaly_score'],
                       c=integrated_df['final_risk_score'],
                       cmap='Reds', s=50, alpha=0.6, edgecolors='black')
ax6.axvline(0.6, color='gray', linestyle='--', alpha=0.5)
ax6.axhline(0.6, color='gray', linestyle='--', alpha=0.5)
ax6.set_xlabel('Bonus Abuse Score', fontsize=11)
ax6.set_ylabel('Quant Anomaly Score', fontsize=11)
ax6.set_title('6. 보너스 vs 퀀트', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=ax6, label='Final Risk')

# 7. 다중 패턴 분포
ax7 = fig.add_subplot(gs[1, 3])
pattern_count_dist = integrated_df['pattern_count'].value_counts().sort_index()
bars = ax7.bar(pattern_count_dist.index, pattern_count_dist.values,
               color=['green', 'yellow', 'orange', 'red', 'darkred'][:len(pattern_count_dist)],
               edgecolor='black')
ax7.set_xlabel('패턴 수 (>0.4 기준)', fontsize=11)
ax7.set_ylabel('계정 수', fontsize=11)
ax7.set_title('7. 다중 패턴 분포', fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax7.text(bar.get_x() + bar.get_width()/2, height,
             f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 8-10. Top 20 고위험 계정 (3개 차트로 분할)
top_20 = integrated_df.head(20)

# 8. 1-7위
ax8 = fig.add_subplot(gs[2, :2])
top_7 = top_20.head(7)
y_pos = np.arange(len(top_7))
bars = ax8.barh(y_pos, top_7['final_risk_score'],
                color=['darkred' if level == 'Critical' else 'red'
                       for level in top_7['final_risk_level']],
                edgecolor='black')
ax8.set_yticks(y_pos)
ax8.set_yticklabels([f"{acc[:18]}..." for acc in top_7['account_id']], fontsize=9)
ax8.set_xlabel('Final Risk Score', fontsize=11)
ax8.set_title('8. 고위험 계정 Top 1-7', fontsize=12, fontweight='bold')
ax8.invert_yaxis()
ax8.grid(True, alpha=0.3, axis='x')
for i, (idx, row) in enumerate(top_7.iterrows()):
    ax8.text(row['final_risk_score'], i,
             f" {row['final_risk_score']:.4f} [{row['final_risk_level']}]",
             va='center', fontsize=8, fontweight='bold')

# 9. 8-14위
ax9 = fig.add_subplot(gs[2, 2:])
top_8_14 = top_20.iloc[7:14]
y_pos = np.arange(len(top_8_14))
bars = ax9.barh(y_pos, top_8_14['final_risk_score'],
                color=['darkred' if level == 'Critical' else 'red' if level == 'High' else 'orange'
                       for level in top_8_14['final_risk_level']],
                edgecolor='black')
ax9.set_yticks(y_pos)
ax9.set_yticklabels([f"{acc[:18]}..." for acc in top_8_14['account_id']], fontsize=9)
ax9.set_xlabel('Final Risk Score', fontsize=11)
ax9.set_title('9. 고위험 계정 Top 8-14', fontsize=12, fontweight='bold')
ax9.invert_yaxis()
ax9.grid(True, alpha=0.3, axis='x')
for i, (idx, row) in enumerate(top_8_14.iterrows()):
    ax9.text(row['final_risk_score'], i,
             f" {row['final_risk_score']:.4f} [{row['final_risk_level']}]",
             va='center', fontsize=8, fontweight='bold')

# 10. 15-20위
ax10 = fig.add_subplot(gs[3, :])
top_15_20 = top_20.iloc[14:20]
y_pos = np.arange(len(top_15_20))
bars = ax10.barh(y_pos, top_15_20['final_risk_score'],
                 color=['red' if level == 'High' else 'orange'
                        for level in top_15_20['final_risk_level']],
                 edgecolor='black')
ax10.set_yticks(y_pos)
ax10.set_yticklabels([f"{acc[:18]}..." for acc in top_15_20['account_id']], fontsize=9)
ax10.set_xlabel('Final Risk Score', fontsize=11)
ax10.set_title('10. 고위험 계정 Top 15-20', fontsize=12, fontweight='bold')
ax10.invert_yaxis()
ax10.grid(True, alpha=0.3, axis='x')
for i, (idx, row) in enumerate(top_15_20.iterrows()):
    ax10.text(row['final_risk_score'], i,
              f" {row['final_risk_score']:.4f} [{row['final_risk_level']}]",
              va='center', fontsize=8, fontweight='bold')

fig.suptitle('최종 통합 이상거래 탐지 시스템 (Final Integrated Anomaly Detection)',
             fontsize=20, fontweight='bold', y=0.998)

plt.savefig('output/final_integrated_visualization.png', dpi=300, bbox_inches='tight')
print(f"✓ 시각화: output/final_integrated_visualization.png")

print("\n" + "=" * 120)
print("최종 통합 분석 완료!")
print("=" * 120)

print("\n[핵심 공식]")
print("FinalRiskScore = 0.30 × FundingArb + 0.35 × Organized + 0.20 × BonusAbuse + 0.15 × QuantAnomaly")
print("\n[위험도 등급]")
print("  - Low: Score < 0.3")
print("  - Medium: 0.3 ≤ Score < 0.5")
print("  - High: 0.5 ≤ Score < 0.7")
print("  - Critical: Score ≥ 0.7 🚨")
