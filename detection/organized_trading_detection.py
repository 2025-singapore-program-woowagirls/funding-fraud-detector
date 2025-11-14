import pandas as pd
import numpy as np

# 데이터 로딩
trade_df = pd.read_csv('data/Trade.csv')
ip_df = pd.read_csv('data/IP.csv')

# 1. IP 공유 비율 (ip_shared_ratio)
ip_account_counts = ip_df.groupby('ip')['account_id'].nunique()
shared_ips = ip_account_counts[ip_account_counts > 1]

# 2. 동시 거래율 (concurrent_trading_ratio)
trade_df['ts_minute'] = pd.to_datetime(trade_df['ts']).dt.floor('1min')

concurrent_trades = trade_df.groupby(['ts_minute', 'symbol']).agg({
    'account_id': lambda x: list(x.unique()),
    'price': ['mean', 'std', 'count']
}).reset_index()

concurrent_trades.columns = ['ts_minute', 'symbol', 'accounts', 'price_mean', 'price_std', 'trade_count']
concurrent_trades['account_count'] = concurrent_trades['accounts'].apply(len)

# 동시 거래가 2명 이상인 경우
multi_account_trades = concurrent_trades[concurrent_trades['account_count'] >= 2]

# 3. 가격 유사도 (price_similarity)
concurrent_trades['price_cv'] = concurrent_trades['price_std'] / concurrent_trades['price_mean']
concurrent_trades['price_cv'] = concurrent_trades['price_cv'].fillna(0)

# 가격 유사도 1% 미만인 거래
similar_price_trades = concurrent_trades[concurrent_trades['price_cv'] < 0.01]

# 4. 레버리지 분석 (leverage)
trade_df['leverage'] = trade_df['leverage'].fillna(1)  # 레버리지 결측치 처리
account_leverage = trade_df.groupby('account_id')['leverage'].mean()

# 5. 스코어 산출 (각 피처에 대한 점수 계산)
# IP 공유 비율 (S_IP)
def get_ip_score(shared_ratio):
    if shared_ratio < 0.05:
        return 0
    elif 0.05 <= shared_ratio < 0.15:
        return 0.5
    else:
        return 1

# 동시 거래 비율 (S_concurrent)
def get_concurrent_score(concurrent_ratio):
    if concurrent_ratio < 0.3:
        return 0
    elif 0.3 <= concurrent_ratio < 0.5:
        return 0.5
    else:
        return 1

# 가격 유사도 (S_price)
def get_price_score(price_cv):
    if price_cv < 0.01:
        return 1
    elif price_cv < 0.05:
        return 0.5
    else:
        return 0

# 레버리지 (S_leverage)
def get_leverage_score(leverage):
    if leverage < 10:
        return 0
    elif 10 <= leverage < 30:
        return 0.5
    else:
        return 1

# 최종 스코어 계산 (Organized_Score)
organized_score = []
for i in range(len(multi_account_trades)):
    ip_score = get_ip_score(shared_ips.iloc[i])
    concurrent_score = get_concurrent_score(concurrent_trades.iloc[i])
    price_score = get_price_score(concurrent_trades.iloc[i])
    leverage_score = get_leverage_score(account_leverage.iloc[i])
    
    total_score = 0.30 * ip_score + 0.35 * concurrent_score + 0.25 * price_score + 0.10 * leverage_score
    organized_score.append(total_score)

# 최종 스코어 결과
multi_account_trades['final_score'] = organized_score
multi_account_trades['risk_level'] = pd.cut(multi_account_trades['final_score'], bins=[-np.inf, 0.3, 0.6, np.inf], labels=['Low', 'Medium', 'High'])
