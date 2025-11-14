import pandas as pd
import numpy as np

# ============================================================
# 1️⃣ 데이터 로딩
# ============================================================
reward_df = pd.read_csv('data/Reward.csv')
trade_df = pd.read_csv('data/Trade.csv')
ip_df = pd.read_csv('data/IP.csv')

reward_df['ts'] = pd.to_datetime(reward_df['ts'])
trade_df['ts'] = pd.to_datetime(trade_df['ts'])

# ============================================================
# 2️⃣ 보너스 이후 거래 비율 계산
# ============================================================
trade_ratio_after_reward = {}
for acc in reward_df['account_id'].unique():
    user_rewards = reward_df[reward_df['account_id'] == acc]
    user_trades = trade_df[trade_df['account_id'] == acc]
    if len(user_rewards) == 0 or len(user_trades) == 0:
        trade_ratio_after_reward[acc] = 0
        continue
    last_reward_time = user_rewards['ts'].max()
    total_trades = len(user_trades)
    after_trades = len(user_trades[user_trades['ts'] > last_reward_time])
    ratio = after_trades / total_trades if total_trades > 0 else 0
    trade_ratio_after_reward[acc] = ratio

# ============================================================
# 3️⃣ 피처 계산
# ============================================================
total_reward = reward_df.groupby('account_id')['reward_amount'].sum()
reward_count = reward_df.groupby('account_id').size()
trade_count = trade_df.groupby('account_id').size()
total_volume = trade_df.groupby('account_id')['amount'].sum()

# --- IP 공유 여부 ---
shared_ips = ip_df.groupby('ip')['account_id'].nunique()
shared_ip_accounts = ip_df[ip_df['ip'].isin(shared_ips[shared_ips > 1].index)]['account_id'].unique()

# ============================================================
# 4️⃣ 피처 통합
# ============================================================
df = pd.DataFrame({
    "account_id": total_reward.index,
    "total_reward": total_reward,
    "reward_count": reward_count,
    "trade_count": trade_count.reindex(total_reward.index, fill_value=0),
    "trade_ratio_after_reward": pd.Series(trade_ratio_after_reward),
    "shared_ip": total_reward.index.isin(shared_ip_accounts),
    "total_volume": total_volume.reindex(total_reward.index, fill_value=0)
}).reset_index(drop=True)

# ============================================================
# 5️⃣ 세부 스코어 계산
# ============================================================
def reward_amount_score(x): return min(x / 100, 1)
def inactive_score(x): return 1 - x
def rvr_score(r): return min(r * 1000, 1)
def ip_score(shared): return 1.0 if shared else 0.0

df["reward_amount_score"] = df["total_reward"].apply(reward_amount_score)
df["inactive_score"] = df["trade_ratio_after_reward"].apply(inactive_score)
df["shared_ip_score"] = df["shared_ip"].apply(ip_score)
df["reward_to_volume_ratio"] = df["total_reward"] / (df["total_volume"] + 1e-9)
df["rvr_score"] = df["reward_to_volume_ratio"].apply(rvr_score)

# ============================================================
# 6️⃣ 최종 점수 계산
# ============================================================
df["bonus_abuse_score"] = (
    0.25 * df["reward_amount_score"] +
    0.25 * df["shared_ip_score"] +
    0.25 * df["inactive_score"] +
    0.25 * df["rvr_score"]
)

# ============================================================
# 7️⃣ 리스크 등급
# ============================================================
df["risk_level"] = pd.cut(
    df["bonus_abuse_score"],
    bins=[-np.inf, 0.3, 0.6, np.inf],
    labels=["Low", "Medium", "High"]
)

# ============================================================
# 8️⃣ 수익 기반 확장 지표
# ============================================================
df["Excess_Profit"] = np.where(df["reward_to_volume_ratio"] > 0.001, 1, -1)
df["Risk_Reward_Ratio"] = df["reward_to_volume_ratio"] / (df["bonus_abuse_score"] + 1e-6)

# ============================================================
# 9️⃣ 자동 해석 (행동 유형)
# ============================================================
def interpret(row):
    if row["risk_level"] == "High" and row["trade_ratio_after_reward"] < 0.2:
        return "고위험·비활성형 (보너스만 수령 후 거래 중단)"
    elif row["risk_level"] == "High" and row["shared_ip"]:
        return "고위험·공동IP형 (조직적 보너스 수령 의심)"
    elif row["risk_level"] == "Medium" and row["reward_to_volume_ratio"] > 0.001:
        return "중위험·고수익형 (보너스 활용 거래)"
    elif row["risk_level"] == "Low":
        return "일반형 (정상 보너스 이용)"
    else:
        return "기타형 (비정형 행동)"

df["해석"] = df.apply(interpret, axis=1)

# ============================================================
# 🔟 결과 출력
# ============================================================
print("=" * 80)
print("보너스 남용 탐지 결과 (행동 기반 스코어링 + 수익 분석 포함)")
print("=" * 80)
print(df[[
    "account_id", "total_reward", "reward_count",
    "reward_amount_score", "shared_ip", "shared_ip_score",
    "trade_count", "trade_ratio_after_reward", "inactive_score",
    "total_volume", "reward_to_volume_ratio", "rvr_score",
    "bonus_abuse_score", "risk_level", "Excess_Profit", "Risk_Reward_Ratio", "해석"
]].sort_values("bonus_abuse_score", ascending=False).head(20))
