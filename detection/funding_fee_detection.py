import pandas as pd
import numpy as np
from scipy.stats import zscore

# ======================================================
# 1. 데이터 로딩
# ======================================================
funding_df = pd.read_csv('data/Funding.csv')
trade_df = pd.read_csv('data/Trade.csv')

funding_df['ts'] = pd.to_datetime(funding_df['ts'])
trade_df['ts'] = pd.to_datetime(trade_df['ts'])

# ======================================================
# 2. 펀딩피 절댓값 (funding_fee_abs)
# ======================================================
funding_fee_abs = funding_df.groupby('account_id')['funding_fee'].apply(lambda x: abs(x).mean())

# ======================================================
# 3. 포지션 보유시간 (mean_holding_minutes)
# ======================================================
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

# ======================================================
# 4. 펀딩 시각 거래 집중도 (funding_timing_ratio)
# ======================================================
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
funding_timing = trade_df.groupby('account_id')['is_funding_time'].mean()

# ======================================================
# 5. 펀딩피 수익 비중 (funding_profit_ratio)
# ======================================================
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

funding_profit_ratio = account_funding / (account_funding + account_pnl + 1e-9)

# ======================================================
# 6. 피처 통합
# ======================================================
merged = pd.concat([
    funding_fee_abs.rename('funding_fee_abs'),
    mean_holding.rename('mean_holding'),
    funding_timing.rename('funding_timing'),
    funding_profit_ratio.rename('funding_profit_ratio')
], axis=1).fillna(0)

# ======================================================
# 7. Z-score 정규화
# ======================================================
features = ['funding_fee_abs', 'mean_holding', 'funding_timing', 'funding_profit_ratio']
merged_z = merged[features].apply(zscore).fillna(0)

# ======================================================
# 8. 가중 평균 기반 Funding_Score
# ======================================================
merged['Funding_Score'] = (
    0.15 * merged_z['funding_fee_abs'] +
    0.30 * merged_z['mean_holding'] +
    0.15 * merged_z['funding_timing'] +
    0.40 * merged_z['funding_profit_ratio']
)

# ======================================================
# 9. 펀딩 참여 없는 계정 감점
# ======================================================
funding_participation = funding_df.groupby('account_id')['funding_fee'].apply(lambda x: abs(x).sum())
participation_aligned = funding_participation.reindex(merged.index).fillna(0)
merged['Funding_Score'] *= np.where(participation_aligned > 0, 1.0, 0.5)

# ======================================================
# 10. Risk Level 분류
# ======================================================
high_thresh = max(1.0, merged['Funding_Score'].quantile(0.85))
medium_thresh = 0

def classify(score):
    if score >= high_thresh:
        return '높음'
    elif score >= medium_thresh:
        return '중간'
    else:
        return '낮음'

merged['Risk_Level'] = merged['Funding_Score'].apply(classify)

# ======================================================
# 11. 계정별 총 수익 계산
# ======================================================
merged['Trading_PnL_USD'] = merged.index.map(account_pnl).fillna(0)
merged['Funding_Profit_USD'] = merged.index.map(account_funding).fillna(0)
merged['Total_Profit_USD'] = merged['Trading_PnL_USD'] + merged['Funding_Profit_USD']

# ======================================================
# 12. 결과 정렬 및 초과 수익 계산
# ======================================================
merged_sorted = merged.sort_values(by='Funding_Score', ascending=False)
merged_sorted['Rank'] = np.arange(1, len(merged_sorted) + 1)
merged_sorted['Excess_Profit'] = zscore(merged_sorted['Total_Profit_USD'].fillna(0))

# ======================================================
# 13. 해석 로직
# ======================================================
def interpret(row):
    score = row['Funding_Score']
    excess = row['Excess_Profit']
    total_profit = row['Total_Profit_USD']

    # 고위험 + 수익이 높은 계정
    if score >= high_thresh and (excess > 0.5 or total_profit > 1e5):
        return "고위험·고수익형 (조작 가능성 높음)"
    # 고위험인데 수익이 낮은 경우
    elif score >= high_thresh and (excess <= 0.5 and total_profit <= 1e5):
        return "고위험·저수익형 (비효율적 조작 시도)"
    # 저위험 + 고수익
    elif score < 0 and (excess > 0.5 or total_profit > 1e5):
        return "저위험·고수익형 (은밀형 조작 가능성)"
    # 저위험 + 중간수익
    elif score < 0 and (excess > 0):
        return "저위험·중간수익형 (정상 혹은 우연형)"
    else:
        return "일반형 (특이행동 없음)"


merged_sorted['해석'] = merged_sorted.apply(interpret, axis=1)

# ======================================================
# 14. 한글 열 이름 및 핵심 열만 출력
# ======================================================
merged_sorted_kor = merged_sorted.rename(columns={
    'Rank': '순위',
    'Funding_Score': '펀딩 점수',
    'Risk_Level': '위험 등급',
    'Total_Profit_USD': '총 수익 (USD)',
    'Excess_Profit': '초과 수익',
    '해석': '거래 유형 해석'
})

core_columns_kor = ['순위', '펀딩 점수', '위험 등급', '총 수익 (USD)', '초과 수익', '거래 유형 해석']

print("=====================================================================================================")
print("이상 거래 탐지 결과 ")
print("=====================================================================================================")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0)
print(merged_sorted_kor[core_columns_kor].head(20))
