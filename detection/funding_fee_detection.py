import pandas as pd
import numpy as np

# ------------------------
# 1. 데이터 로딩
# ------------------------
funding_df = pd.read_csv('data/Funding.csv')
trade_df = pd.read_csv('data/Trade.csv')

funding_df['ts'] = pd.to_datetime(funding_df['ts'])
trade_df['ts'] = pd.to_datetime(trade_df['ts'])

# ------------------------
# 2. 펀딩피 절댓값 (funding_fee_abs)
# ------------------------
funding_fee_abs = funding_df.groupby('account_id')['funding_fee'].apply(lambda x: abs(x).mean())

# ------------------------
# 3. 포지션 보유시간 (mean_holding_minutes)
# ------------------------
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

# ------------------------
# 4. 펀딩 시각 거래 집중도 (funding_timing_ratio)
# ------------------------
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

# ------------------------
# 5. 펀딩피 수익 비중 (funding_profit_ratio)
# ------------------------
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

# ------------------------
# 6. 스코어 함수 정의
# ------------------------
def get_ffa_score(x):
    if x < 5: return 0
    elif x < 30: return 0.5
    else: return 1

def get_hm_score(x):
    if x < 10: return 1
    elif x < 30: return 0.75
    elif x < 120: return 0.5
    elif x < 360: return 0.25
    else: return 0

def get_ftr_score(x):
    if x < 0.3: return 0
    elif x < 0.5: return 0.5
    else: return 1

def get_fpr_score(x):
    if x < 0.3: return 0
    elif x < 0.7: return 0.5
    else: return 1

# ------------------------
# 7. 계정별 피처 통합
# ------------------------
merged = pd.concat([
    funding_fee_abs.rename('funding_fee_abs'),
    mean_holding.rename('mean_holding'),
    funding_timing.rename('funding_timing'),
    funding_profit_ratio.rename('funding_profit_ratio')
], axis=1).fillna(0)

# ------------------------
# 8. 스코어 계산
# ------------------------
merged['S_FFA'] = merged['funding_fee_abs'].apply(get_ffa_score)
merged['S_HM'] = merged['mean_holding'].apply(get_hm_score)
merged['S_FTR'] = merged['funding_timing'].apply(get_ftr_score)
merged['S_FPR'] = merged['funding_profit_ratio'].apply(get_fpr_score)

merged['Funding_Score'] = (
    0.15 * merged['S_FFA'] +
    0.30 * merged['S_HM'] +
    0.15 * merged['S_FTR'] +
    0.40 * merged['S_FPR']
)

merged['Risk_Level'] = pd.cut(
    merged['Funding_Score'],
    bins=[-np.inf, 0.3, 0.6, np.inf],
    labels=['Low', 'Medium', 'High']
)


# ------------------------
# 9. 각 계정별 총 수익 계산
# ------------------------
# Trading 수익 (실현 손익)과 Funding 수익을 합산
merged['Trading_PnL_USD'] = merged.index.map(account_pnl).fillna(0)
merged['Funding_Profit_USD'] = merged.index.map(account_funding).fillna(0)
merged['Total_Profit_USD'] = merged['Trading_PnL_USD'] + merged['Funding_Profit_USD']

# 총 수익 기준으로 내림차순 정렬
merged_sorted = merged.sort_values(by='Total_Profit_USD', ascending=False)

# 출력
print(merged_sorted[['Funding_Score', 'Risk_Level', 'Trading_PnL_USD', 'Funding_Profit_USD', 'Total_Profit_USD']].head(10))
