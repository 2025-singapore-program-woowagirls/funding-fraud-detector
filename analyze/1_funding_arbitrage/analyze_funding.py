import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 한글 폰트 설정
# plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['font.family'] = 'AppleGothic'  # Mac
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
funding_df = pd.read_csv('data/Funding.csv')
trade_df = pd.read_csv('data/Trade.csv')

print("=" * 80)
print("패턴 1: 펀딩피 차익거래 피처 분포 분석")
print("=" * 80)

# ============================================================================
# 1. 펀딩피 절댓값 (funding_fee_abs)
# ============================================================================
print("\n[1] 펀딩피 절댓값 분포")
print("-" * 80)

funding_fee_abs = funding_df['funding_fee'].abs()

stats_1 = {
    '평균': funding_fee_abs.mean(),
    '중앙값': funding_fee_abs.median(),
    '표준편차': funding_fee_abs.std(),
    '최솟값': funding_fee_abs.min(),
    '최댓값': funding_fee_abs.max(),
    '10th percentile': funding_fee_abs.quantile(0.10),
    '25th percentile': funding_fee_abs.quantile(0.25),
    '50th percentile': funding_fee_abs.quantile(0.50),
    '75th percentile': funding_fee_abs.quantile(0.75),
    '90th percentile': funding_fee_abs.quantile(0.90),
    '95th percentile': funding_fee_abs.quantile(0.95),
    '99th percentile': funding_fee_abs.quantile(0.99)
}

for key, value in stats_1.items():
    print(f"{key:20s}: ${value:>10.2f}")

# ============================================================================
# 2. 포지션 보유시간 (mean_holding_minutes)
# ============================================================================
print("\n[2] 포지션 보유시간 분포")
print("-" * 80)

# 타임스탬프 변환
trade_df['ts'] = pd.to_datetime(trade_df['ts'])

# 계정별, 심볼별, position_id별로 그룹화
holding_times = []

for (account_id, symbol, position_id), group in trade_df.groupby(['account_id', 'symbol', 'position_id']):
    open_trades = group[group['openclose'] == 'OPEN']
    close_trades = group[group['openclose'] == 'CLOSE']
    
    if len(open_trades) > 0 and len(close_trades) > 0:
        open_time = open_trades['ts'].min()
        close_time = close_trades['ts'].max()
        holding_minutes = (close_time - open_time).total_seconds() / 60
        holding_times.append({
            'account_id': account_id,
            'holding_minutes': holding_minutes
        })

holding_df = pd.DataFrame(holding_times)
mean_holding = holding_df.groupby('account_id')['holding_minutes'].mean()

stats_2 = {
    '평균': mean_holding.mean(),
    '중앙값': mean_holding.median(),
    '표준편차': mean_holding.std(),
    '최솟값': mean_holding.min(),
    '최댓값': mean_holding.max(),
    '10th percentile': mean_holding.quantile(0.10),
    '25th percentile': mean_holding.quantile(0.25),
    '50th percentile': mean_holding.quantile(0.50),
    '75th percentile': mean_holding.quantile(0.75),
    '90th percentile': mean_holding.quantile(0.90),
    '95th percentile': mean_holding.quantile(0.95)
}

for key, value in stats_2.items():
    print(f"{key:20s}: {value:>10.2f}분 ({value/60:>6.2f}시간)")

# ============================================================================
# 3. 펀딩 시각 거래 집중도 (funding_timing_ratio)
# ============================================================================
print("\n[3] 펀딩 시각 거래 집중도 분포")
print("-" * 80)

# 펀딩 시각 정의 (0시, 4시, 8시, 12시, 16시, 20시 ±30분)
def is_funding_time(timestamp):
    hour = timestamp.hour
    minute = timestamp.minute
    
    funding_hours = [0, 4, 8, 12, 16, 20]
    
    for fh in funding_hours:
        # 펀딩 시각 ±30분 window
        if fh == 0:
            if (hour == 23 and minute >= 30) or (hour == 0 and minute <= 30):
                return True
        else:
            if (hour == fh - 1 and minute >= 30) or (hour == fh and minute <= 30):
                return True
    return False

trade_df['is_funding_time'] = trade_df['ts'].apply(is_funding_time)

funding_timing_ratios = []
for account_id, group in trade_df.groupby('account_id'):
    total_trades = len(group)
    funding_trades = group['is_funding_time'].sum()
    ratio = funding_trades / total_trades if total_trades > 0 else 0
    funding_timing_ratios.append(ratio * 100)  # 퍼센트로 변환

funding_timing_series = pd.Series(funding_timing_ratios)

stats_3 = {
    '평균': funding_timing_series.mean(),
    '중앙값': funding_timing_series.median(),
    '표준편차': funding_timing_series.std(),
    '최솟값': funding_timing_series.min(),
    '최댓값': funding_timing_series.max(),
    '25th percentile': funding_timing_series.quantile(0.25),
    '50th percentile': funding_timing_series.quantile(0.50),
    '75th percentile': funding_timing_series.quantile(0.75),
    '90th percentile': funding_timing_series.quantile(0.90),
    '95th percentile': funding_timing_series.quantile(0.95)
}

for key, value in stats_3.items():
    print(f"{key:20s}: {value:>10.2f}%")

print(f"\n이론적 랜덤 비율: 25.00% (6시간 / 24시간)")

# ============================================================================
# 4. 펀딩피 수익 비중 (funding_profit_ratio)
# ============================================================================
print("\n[4] 펀딩피 수익 비중 분포")
print("-" * 80)

# 계정별 총 펀딩비
account_funding = funding_df.groupby('account_id')['funding_fee'].apply(lambda x: abs(x).sum())

# 계정별 실현손익 (PnL) 계산
pnl_list = []
for (account_id, position_id), group in trade_df.groupby(['account_id', 'position_id']):
    open_trades = group[group['openclose'] == 'OPEN']
    close_trades = group[group['openclose'] == 'CLOSE']
    
    if len(open_trades) > 0 and len(close_trades) > 0:
        open_amount = open_trades['amount'].sum()
        close_amount = close_trades['amount'].sum()
        pnl = close_amount - open_amount
        pnl_list.append({
            'account_id': account_id,
            'pnl': pnl
        })

pnl_df = pd.DataFrame(pnl_list)
account_pnl = pnl_df.groupby('account_id')['pnl'].apply(lambda x: abs(x).sum())

# 펀딩피 수익 비중 계산
funding_profit_ratios = []
for account_id in account_funding.index:
    if account_id in account_pnl.index:
        total_funding = account_funding[account_id]
        total_pnl = account_pnl[account_id]
        
        if total_pnl + total_funding > 0:
            ratio = total_funding / (total_pnl + total_funding)
            funding_profit_ratios.append(ratio * 100)  # 퍼센트로 변환

funding_profit_series = pd.Series(funding_profit_ratios)

stats_4 = {
    '평균': funding_profit_series.mean(),
    '중앙값': funding_profit_series.median(),
    '표준편차': funding_profit_series.std(),
    '최솟값': funding_profit_series.min(),
    '최댓값': funding_profit_series.max(),
    '25th percentile': funding_profit_series.quantile(0.25),
    '50th percentile': funding_profit_series.quantile(0.50),
    '75th percentile': funding_profit_series.quantile(0.75),
    '90th percentile': funding_profit_series.quantile(0.90),
    '95th percentile': funding_profit_series.quantile(0.95),
    '99th percentile': funding_profit_series.quantile(0.99)
}

for key, value in stats_4.items():
    print(f"{key:20s}: {value:>10.2f}%")

# ============================================================================
# 시각화
# ============================================================================
print("\n" + "=" * 80)
print("시각화 생성 중...")
print("=" * 80)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('패턴 1: 펀딩피 차익거래 피처 분포 분석', fontsize=16, fontweight='bold')

# 1-1. 펀딩피 절댓값 히스토그램
ax = axes[0, 0]
ax.hist(funding_fee_abs, bins=50, edgecolor='black', alpha=0.7)
ax.axvline(stats_1['90th percentile'], color='orange', linestyle='--', label=f"90th: ${stats_1['90th percentile']:.2f}")
ax.axvline(stats_1['95th percentile'], color='red', linestyle='--', label=f"95th: ${stats_1['95th percentile']:.2f}")
ax.set_xlabel('펀딩피 절댓값 ($)')
ax.set_ylabel('빈도')
ax.set_title('1-1. 펀딩피 절댓값 분포')
ax.legend()
ax.grid(alpha=0.3)

# 1-2. 펀딩피 절댓값 박스플롯
ax = axes[0, 1]
ax.boxplot(funding_fee_abs, vert=True)
ax.set_ylabel('펀딩피 절댓값 ($)')
ax.set_title('1-2. 펀딩피 절댓값 박스플롯')
ax.grid(alpha=0.3)

# 2-1. 포지션 보유시간 히스토그램
ax = axes[0, 2]
ax.hist(mean_holding, bins=30, edgecolor='black', alpha=0.7)
ax.axvline(stats_2['10th percentile'], color='red', linestyle='--', label=f"10th: {stats_2['10th percentile']:.1f}분")
ax.axvline(stats_2['25th percentile'], color='orange', linestyle='--', label=f"25th: {stats_2['25th percentile']:.1f}분")
ax.set_xlabel('평균 보유시간 (분)')
ax.set_ylabel('빈도')
ax.set_title('2-1. 포지션 보유시간 분포')
ax.legend()
ax.grid(alpha=0.3)

# 2-2. 포지션 보유시간 박스플롯
ax = axes[0, 3]
ax.boxplot(mean_holding, vert=True)
ax.set_ylabel('평균 보유시간 (분)')
ax.set_title('2-2. 포지션 보유시간 박스플롯')
ax.grid(alpha=0.3)

# 3-1. 펀딩 시각 집중도 히스토그램
ax = axes[1, 0]
ax.hist(funding_timing_series, bins=30, edgecolor='black', alpha=0.7)
ax.axvline(25, color='green', linestyle='--', linewidth=2, label='이론적 랜덤: 25%')
ax.axvline(stats_3['75th percentile'], color='orange', linestyle='--', label=f"75th: {stats_3['75th percentile']:.1f}%")
ax.axvline(stats_3['90th percentile'], color='red', linestyle='--', label=f"90th: {stats_3['90th percentile']:.1f}%")
ax.set_xlabel('펀딩 시각 거래 비율 (%)')
ax.set_ylabel('빈도')
ax.set_title('3-1. 펀딩 시각 집중도 분포')
ax.legend()
ax.grid(alpha=0.3)

# 3-2. 펀딩 시각 집중도 박스플롯
ax = axes[1, 1]
ax.boxplot(funding_timing_series, vert=True)
ax.axhline(25, color='green', linestyle='--', linewidth=2, label='이론적 랜덤')
ax.set_ylabel('펀딩 시각 거래 비율 (%)')
ax.set_title('3-2. 펀딩 시각 집중도 박스플롯')
ax.legend()
ax.grid(alpha=0.3)

# 4-1. 펀딩피 수익 비중 히스토그램
ax = axes[1, 2]
ax.hist(funding_profit_series, bins=30, edgecolor='black', alpha=0.7)
ax.axvline(stats_4['75th percentile'], color='orange', linestyle='--', label=f"75th: {stats_4['75th percentile']:.1f}%")
ax.axvline(stats_4['90th percentile'], color='red', linestyle='--', label=f"90th: {stats_4['90th percentile']:.1f}%")
ax.set_xlabel('펀딩피 수익 비중 (%)')
ax.set_ylabel('빈도')
ax.set_title('4-1. 펀딩피 수익 비중 분포')
ax.legend()
ax.grid(alpha=0.3)

# 4-2. 펀딩피 수익 비중 박스플롯
ax = axes[1, 3]
ax.boxplot(funding_profit_series, vert=True)
ax.set_ylabel('펀딩피 수익 비중 (%)')
ax.set_title('4-2. 펀딩피 수익 비중 박스플롯')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('pattern1_feature_distribution.png', dpi=300, bbox_inches='tight')
print("\n✅ 시각화 저장 완료: pattern1_feature_distribution.png")

plt.show()

# ============================================================================
# 요약 통계 저장
# ============================================================================
summary = pd.DataFrame({
    '피처': ['펀딩피 절댓값 ($)', '보유시간 (분)', '펀딩시각 집중도 (%)', '펀딩피 수익비중 (%)'],
    '평균': [stats_1['평균'], stats_2['평균'], stats_3['평균'], stats_4['평균']],
    '중앙값': [stats_1['중앙값'], stats_2['중앙값'], stats_3['중앙값'], stats_4['중앙값']],
    '표준편차': [stats_1['표준편차'], stats_2['표준편차'], stats_3['표준편차'], stats_4['표준편차']],
    '10th': [stats_1.get('10th percentile', np.nan), stats_2['10th percentile'], np.nan, np.nan],
    '25th': [stats_1.get('25th percentile', np.nan), stats_2['25th percentile'], stats_3['25th percentile'], stats_4['25th percentile']],
    '75th': [stats_1.get('75th percentile', np.nan), stats_2['75th percentile'], stats_3['75th percentile'], stats_4['75th percentile']],
    '90th': [stats_1['90th percentile'], stats_2['90th percentile'], stats_3['90th percentile'], stats_4['90th percentile']],
    '95th': [stats_1['95th percentile'], stats_2.get('95th percentile', np.nan), stats_3['95th percentile'], stats_4['95th percentile']],
})

summary.to_csv('pattern1_feature_statistics.csv', index=False, encoding='utf-8-sig')
print("✅ 통계 요약 저장 완료: pattern1_feature_statistics.csv")

print("\n" + "=" * 80)
print("분석 완료!")
print("=" * 80)



# ============================================================================
# 이상치 계정 요약 추출
# ============================================================================

print("\n" + "=" * 80)
print("이상치 계정 요약 생성")
print("=" * 80)

# 기준치 설정: 펀딩피 절댓값 95th 이상 or 펀딩 수익비중 90th 이상 or 펀딩시각 집중도 90th 이상
funding_threshold = stats_1['95th percentile']
timing_threshold = stats_3['90th percentile']
profit_threshold = stats_4['90th percentile']

# 계정별 피처 집계
feature_df = pd.DataFrame({
    'account_id': account_funding.index,
    'funding_fee_abs': account_funding.values,
    'funding_profit_ratio': funding_profit_series.values[:len(account_funding)],
})

# 보유시간, 펀딩시각 집중도 추가
feature_df = feature_df.merge(mean_holding.rename('mean_holding_minutes'), on='account_id', how='left')
account_timing = trade_df.groupby('account_id')['is_funding_time'].mean() * 100
feature_df = feature_df.merge(account_timing.rename('funding_timing_ratio'), on='account_id', how='left')

# 이상치 필터
outlier_accounts = feature_df[
    (feature_df['funding_fee_abs'] >= funding_threshold) |
    (feature_df['funding_profit_ratio'] >= profit_threshold) |
    (feature_df['funding_timing_ratio'] >= timing_threshold)
].copy()

# 점수 계산 (정규화 후 평균)
for col in ['funding_fee_abs', 'funding_profit_ratio', 'funding_timing_ratio']:
    feature_df[f'{col}_score'] = (feature_df[col] - feature_df[col].min()) / (feature_df[col].max() - feature_df[col].min())
feature_df['funding_score'] = feature_df[['funding_fee_abs_score', 'funding_profit_ratio_score', 'funding_timing_ratio_score']].mean(axis=1)

# 위험도 등급
def risk_label(x):
    if x > 0.6:
        return "High"
    elif x > 0.4:
        return "Medium"
    else:
        return "Low"

feature_df['risk_level'] = feature_df['funding_score'].apply(risk_label)

# 이상치 상위 2개 추출
top_outliers = feature_df.sort_values('funding_score', ascending=False).head(2)

# 출력
print(top_outliers[['account_id', 'funding_fee_abs', 'mean_holding_minutes',
                    'funding_timing_ratio', 'funding_profit_ratio',
                    'funding_score', 'risk_level']].to_string(index=False))

# CSV 저장
top_outliers[['account_id', 'funding_fee_abs', 'mean_holding_minutes',
               'funding_timing_ratio', 'funding_profit_ratio',
               'funding_score', 'risk_level']].to_csv('pattern1_outlier_accounts.csv',
                                                      index=False, encoding='utf-8-sig')
print("\n✅ 이상치 계정 저장 완료: pattern1_outlier_accounts.csv")
