import pandas as pd
import numpy as np

# ============================================================
# 1️⃣ Data Loading
# ============================================================
reward_df = pd.read_csv("data/Reward.csv")
trade_df = pd.read_csv("data/Trade.csv")
ip_df = pd.read_csv("data/IP.csv")

reward_df["ts"] = pd.to_datetime(reward_df["ts"])
trade_df["ts"] = pd.to_datetime(trade_df["ts"])

# ============================================================
# 2️⃣ Trade Ratio After Reward
# ============================================================
trade_ratio_after_reward = {}
for acc in reward_df["account_id"].unique():
    user_rewards = reward_df[reward_df["account_id"] == acc]
    user_trades = trade_df[trade_df["account_id"] == acc]
    if len(user_rewards) == 0 or len(user_trades) == 0:
        trade_ratio_after_reward[acc] = 0
        continue
    last_reward_time = user_rewards["ts"].max()
    total_trades = len(user_trades)
    after_trades = len(user_trades[user_trades["ts"] > last_reward_time])
    ratio = after_trades / total_trades if total_trades > 0 else 0
    trade_ratio_after_reward[acc] = ratio

# ============================================================
# 3️⃣ Feature Engineering
# ============================================================
total_reward = reward_df.groupby("account_id")["reward_amount"].sum()
trade_count = trade_df.groupby("account_id").size()
total_volume = trade_df.groupby("account_id")["amount"].sum()

# Shared IP detection
shared_ips = ip_df.groupby("ip")["account_id"].nunique()
shared_ip_accounts = ip_df[ip_df["ip"].isin(shared_ips[shared_ips > 1].index)]["account_id"].unique()

# ============================================================
# 4️⃣ Merge Features
# ============================================================
df = pd.DataFrame({
    "account_id": total_reward.index,
    "total_reward": total_reward,
    "trade_count": trade_count.reindex(total_reward.index, fill_value=0),
    "trade_ratio_after_reward": pd.Series(trade_ratio_after_reward),
    "shared_ip": total_reward.index.isin(shared_ip_accounts),
    "total_volume": total_volume.reindex(total_reward.index, fill_value=0)
}).reset_index(drop=True)

# ============================================================
# 5️⃣ Scoring Functions
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
# 6️⃣ Final Score
# ============================================================
df["Bonus_Score"] = (
    0.25 * df["reward_amount_score"] +
    0.25 * df["shared_ip_score"] +
    0.25 * df["inactive_score"] +
    0.25 * df["rvr_score"]
)

# ============================================================
# 7️⃣ Risk Level
# ============================================================
df["Risk_Level"] = pd.cut(
    df["Bonus_Score"],
    bins=[-np.inf, 0.3, 0.6, np.inf],
    labels=["Low", "Medium", "High"]
)

# ============================================================
# 8️⃣ Profit Metrics
# ============================================================
df["Excess_Profit"] = np.where(df["reward_to_volume_ratio"] > 0.001, 1, -1)
df["Risk_Reward_Ratio"] = df["reward_to_volume_ratio"] / (df["Bonus_Score"] + 1e-6)

# ============================================================
# 9️⃣ Interpretation
# ============================================================
def interpret(row):
    if row["Risk_Level"] == "High" and row["trade_ratio_after_reward"] < 0.2:
        return "High-Risk Inactive (Claimed bonus then stopped trading)"
    elif row["Risk_Level"] == "High" and row["shared_ip"]:
        return "High-Risk Shared-IP (Coordinated bonus claiming)"
    elif row["Risk_Level"] == "Medium" and row["reward_to_volume_ratio"] > 0.001:
        return "Medium-Risk Profitable (Used bonus effectively)"
    elif row["Risk_Level"] == "Low":
        return "Normal (Regular user behavior)"
    else:
        return "Irregular (Unclassified pattern)"

df["Interpretation"] = df.apply(interpret, axis=1)

# ============================================================
# 🔟 Final Output (Core Columns Only)
# ============================================================
core_columns = [
    "account_id", "Bonus_Score", "Risk_Level",
    "total_reward", "reward_to_volume_ratio",
    "Excess_Profit", "Risk_Reward_Ratio", "Interpretation"
]

print("=" * 100)
print("Bonus Abuse Detection Results")
print("=" * 100)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 0)
print(df[core_columns].sort_values("Bonus_Score", ascending=False).head(20))
