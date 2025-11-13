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
print("펀딩피 차익거래(Funding Fee Arbitrage) 이상거래 탐지 분석")
print("=" * 100)

# ================================================================================
# 1. 데이터 로드 및 전처리
# ================================================================================
print("\n[1] 데이터 로딩 중...")

trade_df = pd.read_csv('data/Trade.csv')
funding_df = pd.read_csv('data/Funding.csv')
spec_df = pd.read_csv('data/Spec.csv')

# 타임스탬프 변환
trade_df['ts'] = pd.to_datetime(trade_df['ts'])
funding_df['ts'] = pd.to_datetime(funding_df['ts'])
spec_df['day'] = pd.to_datetime(spec_df['day'])

print(f"✓ Trade: {len(trade_df):,} rows")
print(f"✓ Funding: {len(funding_df):,} rows")
print(f"✓ Spec: {len(spec_df):,} rows")

# ================================================================================
# 2. 펀딩피 분포 상세 분석
# ================================================================================
print("\n" + "=" * 100)
print("[2] 펀딩피 분포 상세 분석")
print("=" * 100)

# 양수/음수/0 근처 분포
funding_df['funding_fee_abs'] = funding_df['funding_fee'].abs()
funding_df['funding_sign'] = funding_df['funding_fee'].apply(
    lambda x: 'positive' if x > 0.01 else ('negative' if x < -0.01 else 'near_zero')
)

sign_counts = funding_df['funding_sign'].value_counts()
print(f"\n펀딩피 부호별 분포:")
for sign, count in sign_counts.items():
    pct = count / len(funding_df) * 100
    print(f"  {sign}: {count:,} ({pct:.2f}%)")

# 전체 펀딩피 통계
print(f"\n[전체 펀딩피 통계]")
print(f"  평균: ${funding_df['funding_fee'].mean():.4f}")
print(f"  중앙값: ${funding_df['funding_fee'].median():.4f}")
print(f"  표준편차: ${funding_df['funding_fee'].std():.4f}")
print(f"  최소값: ${funding_df['funding_fee'].min():.4f}")
print(f"  최대값: ${funding_df['funding_fee'].max():.4f}")

percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
print(f"\n분포 Percentiles:")
for p in percentiles:
    val = funding_df['funding_fee'].quantile(p/100)
    print(f"  {p:2d}th: ${val:10.4f}")

# 절댓값 기준 통계 (0 근처 vs 큰 값)
print(f"\n[펀딩피 절댓값 통계]")
print(f"  평균 절댓값: ${funding_df['funding_fee_abs'].mean():.4f}")
print(f"  중앙값 절댓값: ${funding_df['funding_fee_abs'].median():.4f}")

# 큰 펀딩피 (절댓값 기준)
large_funding_threshold = funding_df['funding_fee_abs'].quantile(0.95)
print(f"\n  95th percentile 절댓값: ${large_funding_threshold:.4f}")

large_funding_counts = funding_df[funding_df['funding_fee_abs'] > large_funding_threshold].groupby('account_id').size()
print(f"  큰 펀딩피(>95th) 받은 계정 수: {len(large_funding_counts)}")

# 계정별 펀딩피 집계
account_funding_stats = funding_df.groupby('account_id').agg({
    'funding_fee': ['sum', 'mean', 'std', 'count', 'min', 'max'],
    'funding_fee_abs': ['mean', 'max']
}).reset_index()

account_funding_stats.columns = ['account_id', 'total_funding', 'mean_funding', 'std_funding',
                                   'funding_count', 'min_funding', 'max_funding',
                                   'mean_abs_funding', 'max_abs_funding']

print(f"\n[계정별 펀딩피 통계]")
print(f"  총 계정 수: {len(account_funding_stats)}")
print(f"  평균 펀딩 횟수: {account_funding_stats['funding_count'].mean():.1f}")
print(f"  평균 총 펀딩피: ${account_funding_stats['total_funding'].mean():.2f}")

# ================================================================================
# 3. 포지션 보유 시간 분석 (Trade 데이터)
# ================================================================================
print("\n" + "=" * 100)
print("[3] 포지션 보유 시간 분석")
print("=" * 100)

# 포지션별 OPEN/CLOSE 매칭
positions = {}
for _, row in trade_df.iterrows():
    pos_id = row['position_id']
    if pos_id not in positions:
        positions[pos_id] = {'open': None, 'close': None, 'account_id': row['account_id']}

    if row['openclose'] == 'OPEN':
        if positions[pos_id]['open'] is None:
            positions[pos_id]['open'] = row
        else:
            # 추가 오픈 (평단가 조정) - 첫 오픈 시각 유지
            pass
    elif row['openclose'] == 'CLOSE':
        if positions[pos_id]['close'] is None:
            positions[pos_id]['close'] = row

# 보유 시간 계산
position_holding_times = []
for pos_id, pos_data in positions.items():
    if pos_data['open'] is not None and pos_data['close'] is not None:
        open_time = pos_data['open']['ts']
        close_time = pos_data['close']['ts']
        holding_seconds = (close_time - open_time).total_seconds()
        holding_minutes = holding_seconds / 60
        holding_hours = holding_seconds / 3600

        # PnL 계산
        open_trade = pos_data['open']
        close_trade = pos_data['close']

        if open_trade['side'] == 'LONG':
            pnl = (close_trade['price'] - open_trade['price']) * open_trade['qty']
        else:  # SHORT
            pnl = (open_trade['price'] - close_trade['price']) * open_trade['qty']

        position_holding_times.append({
            'position_id': pos_id,
            'account_id': pos_data['account_id'],
            'open_time': open_time,
            'close_time': close_time,
            'holding_seconds': holding_seconds,
            'holding_minutes': holding_minutes,
            'holding_hours': holding_hours,
            'symbol': open_trade['symbol'],
            'side': open_trade['side'],
            'leverage': open_trade['leverage'],
            'qty': open_trade['qty'],
            'open_price': open_trade['price'],
            'close_price': close_trade['price'],
            'pnl': pnl,
            'open_amount': open_trade['amount'],
        })

holding_df = pd.DataFrame(position_holding_times)

print(f"\n완료된 포지션 수: {len(holding_df):,}")
print(f"\n[포지션 보유 시간 통계]")
print(f"  평균: {holding_df['holding_minutes'].mean():.2f}분 ({holding_df['holding_hours'].mean():.2f}시간)")
print(f"  중앙값: {holding_df['holding_minutes'].median():.2f}분 ({holding_df['holding_hours'].median():.2f}시간)")
print(f"  표준편차: {holding_df['holding_minutes'].std():.2f}분")
print(f"  최소: {holding_df['holding_minutes'].min():.2f}분")
print(f"  최대: {holding_df['holding_minutes'].max():.2f}분")

percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
print(f"\n보유시간 Percentiles (분):")
for p in percentiles:
    val = holding_df['holding_minutes'].quantile(p/100)
    print(f"  {p:2d}th: {val:10.2f}분 ({val/60:.2f}시간)")

# 짧은 보유 시간 (5분 미만, 30분 미만 등)
short_5min = len(holding_df[holding_df['holding_minutes'] < 5])
short_30min = len(holding_df[holding_df['holding_minutes'] < 30])
short_1hour = len(holding_df[holding_df['holding_minutes'] < 60])

print(f"\n짧은 보유시간 포지션:")
print(f"  < 5분: {short_5min} ({short_5min/len(holding_df)*100:.2f}%)")
print(f"  < 30분: {short_30min} ({short_30min/len(holding_df)*100:.2f}%)")
print(f"  < 1시간: {short_1hour} ({short_1hour/len(holding_df)*100:.2f}%)")

# 계정별 평균 보유시간
account_holding = holding_df.groupby('account_id').agg({
    'holding_minutes': ['mean', 'median', 'std', 'count'],
    'pnl': ['sum', 'mean'],
    'leverage': 'mean'
}).reset_index()

account_holding.columns = ['account_id', 'mean_holding_min', 'median_holding_min',
                           'std_holding_min', 'position_count',
                           'total_pnl', 'mean_pnl', 'mean_leverage']

print(f"\n[계정별 보유시간 통계]")
print(f"  평균 보유시간이 가장 짧은 계정: {account_holding.nsmallest(5, 'mean_holding_min')[['account_id', 'mean_holding_min']].to_string(index=False)}")

# ================================================================================
# 4. 펀딩 시각 전후 거래 집중도 분석
# ================================================================================
print("\n" + "=" * 100)
print("[4] 펀딩 시각 전후 거래 집중도 분석")
print("=" * 100)

# Spec에서 펀딩 간격 확인
print(f"\n[펀딩 간격 정보]")
funding_intervals = spec_df['funding_interval'].value_counts().sort_index()
print(funding_intervals)

# 일반적으로 funding_interval=4 -> 4시간마다, 8 -> 8시간마다
# 펀딩 시각: 0시, 4시, 8시, 12시, 16시, 20시 (4시간) 또는 0시, 8시, 16시 (8시간)

# 거래 시각의 시간대 추출
trade_df['hour'] = trade_df['ts'].dt.hour
funding_df['hour'] = funding_df['ts'].dt.hour

# 펀딩 시각 근처 정의: ±30분
def is_near_funding_time(hour, minute, funding_interval=4):
    """펀딩 시각 ±30분 이내인지 확인"""
    if funding_interval == 4:
        funding_hours = [0, 4, 8, 12, 16, 20]
    elif funding_interval == 8:
        funding_hours = [0, 8, 16]
    else:
        funding_hours = [0, 8]  # 기본값

    for fh in funding_hours:
        # 펀딩 시각 전후 30분
        if fh == 0:
            # 자정 전후 처리
            if (23 <= hour <= 23 and minute >= 30) or (hour == 0 and minute <= 30):
                return True
        else:
            if (hour == fh - 1 and minute >= 30) or (hour == fh and minute <= 30):
                return True
    return False

trade_df['minute'] = trade_df['ts'].dt.minute
trade_df['near_funding'] = trade_df.apply(
    lambda row: is_near_funding_time(row['hour'], row['minute']), axis=1
)

# 계정별 펀딩 시각 근처 거래 비율
account_funding_timing = trade_df.groupby('account_id').agg({
    'near_funding': ['sum', 'count']
}).reset_index()
account_funding_timing.columns = ['account_id', 'near_funding_count', 'total_trades']
account_funding_timing['funding_timing_ratio'] = (
    account_funding_timing['near_funding_count'] / account_funding_timing['total_trades']
)

print(f"\n[펀딩 시각 근처 거래 통계]")
print(f"  전체 거래 중 펀딩 시각 근처(±30분): {trade_df['near_funding'].sum()} ({trade_df['near_funding'].mean()*100:.2f}%)")
print(f"\n  계정별 펀딩 시각 거래 비율:")
print(f"    평균: {account_funding_timing['funding_timing_ratio'].mean()*100:.2f}%")
print(f"    중앙값: {account_funding_timing['funding_timing_ratio'].median()*100:.2f}%")
print(f"    95th percentile: {account_funding_timing['funding_timing_ratio'].quantile(0.95)*100:.2f}%")

high_funding_timing = account_funding_timing[account_funding_timing['funding_timing_ratio'] > 0.5]
print(f"\n  펀딩 시각 집중 계정 (>50%): {len(high_funding_timing)}개")

# ================================================================================
# 5. 펀딩피 수익 비중 분석
# ================================================================================
print("\n" + "=" * 100)
print("[5] 펀딩피 수익 비중 분석")
print("=" * 100)

# 계정별 총 PnL vs 펀딩피
merged_stats = account_funding_stats.merge(
    account_holding[['account_id', 'total_pnl', 'mean_holding_min', 'mean_leverage', 'position_count']],
    on='account_id',
    how='outer'
)

# 펀딩피 비중 계산
merged_stats['total_pnl'] = merged_stats['total_pnl'].fillna(0)
merged_stats['total_funding'] = merged_stats['total_funding'].fillna(0)
merged_stats['total_profit'] = merged_stats['total_pnl'] + merged_stats['total_funding']

# 음수 처리 (손실인 경우)
merged_stats['funding_profit_ratio'] = merged_stats.apply(
    lambda row: row['total_funding'] / row['total_profit']
    if row['total_profit'] > 0 else 0,
    axis=1
)

# 절대 펀딩피가 거래 수익보다 큰 경우
merged_stats['funding_dominance'] = merged_stats.apply(
    lambda row: abs(row['total_funding']) / (abs(row['total_pnl']) + 1)
    if abs(row['total_pnl']) > 0 else abs(row['total_funding']),
    axis=1
)

print(f"\n[펀딩피 수익 비중 통계]")
print(f"  평균 펀딩피 비중: {merged_stats['funding_profit_ratio'].mean()*100:.2f}%")
print(f"  중앙값: {merged_stats['funding_profit_ratio'].median()*100:.2f}%")
print(f"  95th percentile: {merged_stats['funding_profit_ratio'].quantile(0.95)*100:.2f}%")

# 펀딩피가 수익의 50% 이상인 계정
high_funding_ratio = merged_stats[merged_stats['funding_profit_ratio'] > 0.5]
print(f"\n  펀딩피 비중 > 50%: {len(high_funding_ratio)}개 ({len(high_funding_ratio)/len(merged_stats)*100:.2f}%)")

very_high_funding_ratio = merged_stats[merged_stats['funding_profit_ratio'] > 0.8]
print(f"  펀딩피 비중 > 80%: {len(very_high_funding_ratio)}개 ({len(very_high_funding_ratio)/len(merged_stats)*100:.2f}%)")

# ================================================================================
# 6. 정상 vs 이상 패턴 정의
# ================================================================================
print("\n" + "=" * 100)
print("[6] 정상 거래 vs 이상 거래 패턴 정의")
print("=" * 100)

print("\n[정상 거래 기준선]")
print("\n1. 포지션 보유시간:")
print(f"   - 정상: 평균 > 60분 (1시간)")
print(f"   - 의심: 30~60분")
print(f"   - 고위험: < 30분 (펀딩만 노리고 빠른 청산)")
print(f"   실제 데이터 - 중앙값: {holding_df['holding_minutes'].median():.2f}분, 평균: {holding_df['holding_minutes'].mean():.2f}분")

print("\n2. 펀딩 시각 거래 집중도:")
print(f"   - 정상: < 30% (자연스러운 거래)")
print(f"   - 의심: 30~50%")
print(f"   - 고위험: > 50% (펀딩 시각에 집중)")
print(f"   실제 데이터 - 평균: {account_funding_timing['funding_timing_ratio'].mean()*100:.2f}%")

print("\n3. 펀딩피 수익 비중:")
print(f"   - 정상: < 30% (주 수익은 거래 차익)")
print(f"   - 의심: 30~70%")
print(f"   - 고위험: > 70% (펀딩피가 주 수익원)")
print(f"   실제 데이터 - 중앙값: {merged_stats['funding_profit_ratio'].median()*100:.2f}%")

print("\n4. 펀딩피 절댓값:")
print(f"   - 정상: 작은 값들 (<${large_funding_threshold:.2f})")
print(f"   - 의심: 큰 값들 지속 (>95th percentile)")

# ================================================================================
# 7. 종합 위험 점수 계산 (Funding Fee Arbitrage Score)
# ================================================================================
print("\n" + "=" * 100)
print("[7] 펀딩피 차익거래 종합 위험 점수 산출")
print("=" * 100)

# 모든 계정 통합
all_account_ids = set(merged_stats['account_id'].dropna()) | set(account_funding_timing['account_id'])

funding_arbitrage_scores = []

for account in all_account_ids:
    # 1. 포지션 보유시간 점수 (짧을수록 높음)
    holding_data = merged_stats[merged_stats['account_id'] == account]
    if len(holding_data) > 0 and pd.notna(holding_data['mean_holding_min'].values[0]):
        mean_hold = holding_data['mean_holding_min'].values[0]
        # 120분(2시간)을 기준으로, 짧을수록 점수 높음
        holding_score = max(1 - mean_hold / 120, 0)
    else:
        holding_score = 0

    # 2. 펀딩 시각 집중도 점수
    timing_data = account_funding_timing[account_funding_timing['account_id'] == account]
    if len(timing_data) > 0:
        timing_ratio = timing_data['funding_timing_ratio'].values[0]
        timing_score = min(timing_ratio / 0.5, 1.0)  # 50% 이상이면 만점
    else:
        timing_score = 0

    # 3. 펀딩피 수익 비중 점수
    if len(holding_data) > 0 and pd.notna(holding_data['funding_profit_ratio'].values[0]):
        funding_ratio = holding_data['funding_profit_ratio'].values[0]
        funding_ratio_score = min(funding_ratio / 0.7, 1.0)  # 70% 이상이면 만점
    else:
        funding_ratio_score = 0

    # 4. 평균 펀딩피 절댓값 점수
    if len(holding_data) > 0 and pd.notna(holding_data['mean_abs_funding'].values[0]):
        mean_abs_fund = holding_data['mean_abs_funding'].values[0]
        funding_size_score = min(mean_abs_fund / large_funding_threshold, 1.0)
    else:
        funding_size_score = 0

    # 가중치 적용 종합 점수
    # w1=0.30 (보유시간), w2=0.25 (펀딩시각), w3=0.30 (펀딩비중), w4=0.15 (펀딩크기)
    total_score = (0.30 * holding_score +
                   0.25 * timing_score +
                   0.30 * funding_ratio_score +
                   0.15 * funding_size_score)

    # 데이터 수집
    if len(holding_data) > 0:
        total_funding = holding_data['total_funding'].values[0] if pd.notna(holding_data['total_funding'].values[0]) else 0
        total_pnl = holding_data['total_pnl'].values[0] if pd.notna(holding_data['total_pnl'].values[0]) else 0
        funding_count = holding_data['funding_count'].values[0] if pd.notna(holding_data['funding_count'].values[0]) else 0
        mean_hold_val = mean_hold if len(holding_data) > 0 else 0
    else:
        total_funding = 0
        total_pnl = 0
        funding_count = 0
        mean_hold_val = 0

    funding_arbitrage_scores.append({
        'account_id': account,
        'mean_holding_minutes': mean_hold_val,
        'holding_score': holding_score,
        'funding_timing_ratio': timing_ratio if len(timing_data) > 0 else 0,
        'timing_score': timing_score,
        'funding_profit_ratio': funding_ratio if len(holding_data) > 0 and pd.notna(holding_data['funding_profit_ratio'].values[0]) else 0,
        'funding_ratio_score': funding_ratio_score,
        'mean_abs_funding': mean_abs_fund if len(holding_data) > 0 and pd.notna(holding_data['mean_abs_funding'].values[0]) else 0,
        'funding_size_score': funding_size_score,
        'funding_arbitrage_score': total_score,
        'total_funding': total_funding,
        'total_pnl': total_pnl,
        'funding_count': funding_count
    })

funding_arb_df = pd.DataFrame(funding_arbitrage_scores)
funding_arb_df = funding_arb_df.sort_values('funding_arbitrage_score', ascending=False)

print(f"\n[종합 점수 분포]")
print(f"  평균: {funding_arb_df['funding_arbitrage_score'].mean():.4f}")
print(f"  중앙값: {funding_arb_df['funding_arbitrage_score'].median():.4f}")
print(f"  표준편차: {funding_arb_df['funding_arbitrage_score'].std():.4f}")
print(f"  95th percentile: {funding_arb_df['funding_arbitrage_score'].quantile(0.95):.4f}")
print(f"  99th percentile: {funding_arb_df['funding_arbitrage_score'].quantile(0.99):.4f}")

# 위험도 분류
funding_arb_df['risk_level'] = pd.cut(
    funding_arb_df['funding_arbitrage_score'],
    bins=[-np.inf, 0.3, 0.6, np.inf],
    labels=['Low', 'Medium', 'High']
)

risk_counts = funding_arb_df['risk_level'].value_counts()
print(f"\n[위험도 분류]")
for level in ['High', 'Medium', 'Low']:
    count = risk_counts.get(level, 0)
    print(f"  {level} Risk: {count}개 ({count/len(funding_arb_df)*100:.2f}%)")

# 고위험 계정
high_risk_accounts = funding_arb_df[funding_arb_df['risk_level'] == 'High']
print(f"\n[고위험 계정 Top 10]")
for i, row in high_risk_accounts.head(10).iterrows():
    print(f"  {row['account_id']}: Score={row['funding_arbitrage_score']:.4f}")
    print(f"    보유시간={row['mean_holding_minutes']:.1f}분, 펀딩시각비율={row['funding_timing_ratio']*100:.1f}%, "
          f"펀딩수익비중={row['funding_profit_ratio']*100:.1f}%")

# ================================================================================
# 8. 결과 저장
# ================================================================================
print("\n" + "=" * 100)
print("[8] 결과 저장")
print("=" * 100)

# 전체 결과
funding_arb_df.to_csv('output/funding_analysis/funding_arbitrage_scores_all.csv', index=False)
print(f"✓ 전체 계정 점수: output/funding_analysis/funding_arbitrage_scores_all.csv ({len(funding_arb_df)}개)")

# 고위험 계정
high_risk_accounts.to_csv('output/funding_analysis/funding_arbitrage_high_risk.csv', index=False)
print(f"✓ 고위험 계정: output/funding_analysis/funding_arbitrage_high_risk.csv ({len(high_risk_accounts)}개)")

# 포지션 상세
holding_df.to_csv('output/funding_analysis/position_holding_details.csv', index=False)
print(f"✓ 포지션 상세: output/funding_analysis/position_holding_details.csv ({len(holding_df)}개)")

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
        '평균 보유시간(분)',
        '펀딩 시각 집중 평균(%)',
        '펀딩피 비중 평균(%)',
        '총 펀딩피 합계',
        '총 거래 PnL 합계'
    ],
    'Value': [
        len(funding_arb_df),
        risk_counts.get('High', 0),
        risk_counts.get('Medium', 0),
        risk_counts.get('Low', 0),
        f"{funding_arb_df['funding_arbitrage_score'].mean():.4f}",
        f"{funding_arb_df['funding_arbitrage_score'].median():.4f}",
        f"{funding_arb_df['funding_arbitrage_score'].quantile(0.95):.4f}",
        f"{holding_df['holding_minutes'].mean():.2f}",
        f"{account_funding_timing['funding_timing_ratio'].mean()*100:.2f}",
        f"{merged_stats['funding_profit_ratio'].mean()*100:.2f}",
        f"${funding_arb_df['total_funding'].sum():.2f}",
        f"${funding_arb_df['total_pnl'].sum():.2f}"
    ]
})
summary_stats.to_csv('output/funding_analysis/funding_arbitrage_summary.csv', index=False)
print(f"✓ 요약 통계: output/funding_analysis/funding_arbitrage_summary.csv")

# ================================================================================
# 9. 시각화
# ================================================================================
print("\n" + "=" * 100)
print("[9] 시각화 생성")
print("=" * 100)

fig = plt.figure(figsize=(22, 16))
gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

# 1. 펀딩피 분포 (전체)
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(funding_df['funding_fee'], bins=100, edgecolor='black', alpha=0.7, color='steelblue')
ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
ax1.set_xlabel('Funding Fee ($)', fontsize=11)
ax1.set_ylabel('발생 횟수', fontsize=11)
ax1.set_title('1. 펀딩피 전체 분포', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 펀딩피 절댓값 분포 (로그)
ax2 = fig.add_subplot(gs[0, 1])
positive_funding = funding_df[funding_df['funding_fee_abs'] > 0]['funding_fee_abs']
ax2.hist(np.log10(positive_funding + 0.001), bins=100, edgecolor='black', alpha=0.7, color='orange')
ax2.axvline(np.log10(large_funding_threshold), color='red', linestyle='--', linewidth=2, label=f'95th: ${large_funding_threshold:.2f}')
ax2.set_xlabel('Log10(|Funding Fee| + 0.001)', fontsize=11)
ax2.set_ylabel('발생 횟수', fontsize=11)
ax2.set_title('2. 펀딩피 절댓값 분포 (로그)', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 포지션 보유시간 분포
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(holding_df['holding_minutes'], bins=100, edgecolor='black', alpha=0.7, color='green', range=(0, 500))
ax3.axvline(30, color='orange', linestyle='--', linewidth=2, label='30��')
ax3.axvline(60, color='red', linestyle='--', linewidth=2, label='60분')
ax3.set_xlabel('보유시간 (분)', fontsize=11)
ax3.set_ylabel('포지션 수', fontsize=11)
ax3.set_title('3. 포지션 보유시간 분포', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. 보유시간 BoxPlot
ax4 = fig.add_subplot(gs[1, 0])
bp = ax4.boxplot([holding_df['holding_minutes']], vert=True, patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor('lightgreen')
ax4.set_ylabel('보유시간 (분)', fontsize=11)
ax4.set_title('4. 보유시간 박스플롯', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

# 5. 펀딩 시각 거래 집중도
ax5 = fig.add_subplot(gs[1, 1])
ax5.hist(account_funding_timing['funding_timing_ratio']*100, bins=50, edgecolor='black', alpha=0.7, color='coral')
ax5.axvline(30, color='orange', linestyle='--', linewidth=2, label='의심 (30%)')
ax5.axvline(50, color='red', linestyle='--', linewidth=2, label='고위험 (50%)')
ax5.set_xlabel('펀딩 시각 거래 비율 (%)', fontsize=11)
ax5.set_ylabel('계정 수', fontsize=11)
ax5.set_title('5. 펀딩 시각 거래 집중도', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. 펀딩피 수익 비중
ax6 = fig.add_subplot(gs[1, 2])
valid_ratios = merged_stats[merged_stats['funding_profit_ratio'].between(0, 2)]['funding_profit_ratio']
ax6.hist(valid_ratios*100, bins=50, edgecolor='black', alpha=0.7, color='purple')
ax6.axvline(30, color='orange', linestyle='--', linewidth=2, label='의심 (30%)')
ax6.axvline(70, color='red', linestyle='--', linewidth=2, label='고위험 (70%)')
ax6.set_xlabel('펀딩피 수익 비중 (%)', fontsize=11)
ax6.set_ylabel('계정 수', fontsize=11)
ax6.set_title('6. 펀딩피 수익 비중 분포', fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 7. 종합 점수 분포
ax7 = fig.add_subplot(gs[2, 0])
ax7.hist(funding_arb_df['funding_arbitrage_score'], bins=50, edgecolor='black', alpha=0.7, color='crimson')
ax7.axvline(0.3, color='orange', linestyle='--', linewidth=2, label='Medium')
ax7.axvline(0.6, color='darkred', linestyle='--', linewidth=2, label='High')
ax7.set_xlabel('Funding Arbitrage Score', fontsize=11)
ax7.set_ylabel('계정 수', fontsize=11)
ax7.set_title('7. 펀딩피 차익거래 종합 점수', fontsize=12, fontweight='bold')
ax7.legend()
ax7.grid(True, alpha=0.3)

# 8. 위험도 분류
ax8 = fig.add_subplot(gs[2, 1])
risk_counts_sorted = risk_counts.reindex(['High', 'Medium', 'Low'])
colors = ['red', 'orange', 'green']
bars = ax8.bar(range(len(risk_counts_sorted)), risk_counts_sorted.values, color=colors, edgecolor='black')
ax8.set_xticks(range(len(risk_counts_sorted)))
ax8.set_xticklabels(risk_counts_sorted.index, fontsize=11)
ax8.set_ylabel('계정 수', fontsize=11)
ax8.set_title('8. 위험도별 분류', fontsize=12, fontweight='bold')
ax8.grid(True, alpha=0.3, axis='y')
for bar, count in zip(bars, risk_counts_sorted.values):
    ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{count}\n({count/len(funding_arb_df)*100:.1f}%)',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# 9. 시간대별 거래 발생 빈도
ax9 = fig.add_subplot(gs[2, 2])
hour_counts = trade_df['hour'].value_counts().sort_index()
ax9.bar(hour_counts.index, hour_counts.values, color='skyblue', edgecolor='black')
# 펀딩 시각 하이라이트 (0, 4, 8, 12, 16, 20시)
funding_hours = [0, 4, 8, 12, 16, 20]
for fh in funding_hours:
    ax9.axvline(fh, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
ax9.set_xlabel('시간대 (Hour)', fontsize=11)
ax9.set_ylabel('거래 수', fontsize=11)
ax9.set_title('9. 시간대별 거래 발생 빈도', fontsize=12, fontweight='bold')
ax9.grid(True, alpha=0.3, axis='y')

# 10. 고위험 계정 Top 15
ax10 = fig.add_subplot(gs[3, :2])
top_15 = funding_arb_df.head(15)
y_pos = np.arange(len(top_15))
bars = ax10.barh(y_pos, top_15['funding_arbitrage_score'], color='darkred', edgecolor='black')
ax10.set_yticks(y_pos)
ax10.set_yticklabels([f"{acc[:15]}..." for acc in top_15['account_id']], fontsize=9)
ax10.set_xlabel('Funding Arbitrage Score', fontsize=11)
ax10.set_title('10. 펀딩피 차익거래 의심 계정 Top 15', fontsize=12, fontweight='bold')
ax10.invert_yaxis()
ax10.grid(True, alpha=0.3, axis='x')
for i, (idx, row) in enumerate(top_15.iterrows()):
    ax10.text(row['funding_arbitrage_score'], i, f" {row['funding_arbitrage_score']:.3f}",
              va='center', fontsize=8, fontweight='bold')

# 11. 점수 요소 상관관계
ax11 = fig.add_subplot(gs[3, 2])
score_components = funding_arb_df[['holding_score', 'timing_score', 'funding_ratio_score', 'funding_size_score']].corr()
im = ax11.imshow(score_components, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
ax11.set_xticks(range(4))
ax11.set_yticks(range(4))
ax11.set_xticklabels(['Holding', 'Timing', 'Ratio', 'Size'], fontsize=9, rotation=45)
ax11.set_yticklabels(['Holding', 'Timing', 'Ratio', 'Size'], fontsize=9)
ax11.set_title('11. 점수 요소 상관관계', fontsize=12, fontweight='bold')
for i in range(4):
    for j in range(4):
        ax11.text(j, i, f'{score_components.iloc[i, j]:.2f}',
                  ha="center", va="center", color="black", fontsize=9)
plt.colorbar(im, ax=ax11)

fig.suptitle('펀딩피 차익거래(Funding Fee Arbitrage) 이상거래 탐지 분석',
             fontsize=18, fontweight='bold', y=0.998)

plt.savefig('output/funding_analysis/funding_arbitrage_visualization.png', dpi=300, bbox_inches='tight')
print(f"✓ 시각화: output/funding_analysis/funding_arbitrage_visualization.png")

print("\n" + "=" * 100)
print("분석 완료!")
print("=" * 100)
print("\n[핵심 공식]")
print("FundingArbitrageScore = 0.30 × Holding_Score + 0.25 × Timing_Score + 0.30 × FundingRatio_Score + 0.15 × Size_Score")
print("\n각 점수는 0~1로 정규화되며, 임계값:")
print("  - Low Risk: Score < 0.3")
print("  - Medium Risk: 0.3 ≤ Score < 0.6")
print("  - High Risk: Score ≥ 0.6")
