import pandas as pd
import numpy as np
from scipy.stats import zscore

# ============================================================
# 1. 데이터 로딩
# ============================================================
trade_df = pd.read_csv("data/Trade.csv")
ip_df = pd.read_csv("data/IP.csv")

# ============================================================
# 2. 피처 계산
# ============================================================
ip_counts = ip_df.groupby("ip")["account_id"].nunique()
shared_ip = ip_counts[ip_counts > 1].index
account_ip_total = ip_df.groupby("account_id")["ip"].nunique()
account_ip_shared = ip_df[ip_df["ip"].isin(shared_ip)].groupby("account_id")["ip"].nunique()
ip_shared_ratio = (account_ip_shared / account_ip_total).fillna(0)

trade_df["ts_minute"] = pd.to_datetime(trade_df["ts"]).dt.floor("1min")
trade_df["concurrent_count"] = trade_df.groupby(["ts_minute", "symbol"])["account_id"].transform("count")
trade_df["is_concurrent"] = (trade_df["concurrent_count"] > 1).astype(int)
concurrent_ratio = trade_df.groupby("account_id")["is_concurrent"].mean()

price_stats = trade_df.groupby(["ts_minute", "symbol"])["price"].agg(["mean", "std"]).reset_index()
price_stats["price_cv"] = price_stats["std"] / price_stats["mean"]
trade_df = trade_df.merge(price_stats[["ts_minute", "symbol", "price_cv"]], on=["ts_minute", "symbol"], how="left")
price_cv = trade_df.groupby("account_id")["price_cv"].mean().fillna(0)

trade_df["leverage"] = trade_df["leverage"].fillna(1)
leverage_mean = trade_df.groupby("account_id")["leverage"].mean()

S_coordinated = np.sqrt(concurrent_ratio * (1 - price_cv.clip(0, 1)))

# ============================================================
# 4. 피처 통합
# ============================================================
merged = pd.DataFrame({
    "account_id": leverage_mean.index,
    "S_IP": ip_shared_ratio.reindex(leverage_mean.index, fill_value=0),
    "S_concurrent": concurrent_ratio.reindex(leverage_mean.index, fill_value=0),
    "S_price": price_cv.reindex(leverage_mean.index, fill_value=0),
    "S_leverage": leverage_mean,
    "S_coordinated": S_coordinated.reindex(leverage_mean.index, fill_value=0)
})

# ============================================================
# 5~7. 표준화 및 가중합
# ============================================================
for col in ["S_IP", "S_concurrent", "S_price", "S_leverage", "S_coordinated"]:
    merged[col + "_z"] = zscore(merged[col].replace([np.inf, -np.inf], 0))

weights = {
    "S_IP_z": 0.25,
    "S_concurrent_z": 0.25,
    "S_price_z": 0.15,
    "S_leverage_z": 0.10,
    "S_coordinated_z": 0.25
}
merged["Organized_Score_raw"] = sum(merged[k] * w for k, w in weights.items())
merged["Organized_Score"] = np.exp(merged["Organized_Score_raw"] / 2)

# ============================================================
# 8. 리스크 등급
# ============================================================
high_thresh = merged["Organized_Score"].quantile(0.85)
medium_thresh = merged["Organized_Score"].quantile(0.5)
conditions = [
    merged["Organized_Score"] >= high_thresh,
    merged["Organized_Score"] >= medium_thresh
]
choices = ["High", "Medium"]
merged["Risk_Level"] = np.select(conditions, choices, default="Low")

# ============================================================
# 9~13. 수익 계산
# ============================================================
pnl_list = []
for (account_id, position_id), group in trade_df.groupby(["account_id", "position_id"]):
    open_trades = group[group["openclose"] == "OPEN"]
    close_trades = group[group["openclose"] == "CLOSE"]
    if len(open_trades) > 0 and len(close_trades) > 0:
        pnl = close_trades["amount"].sum() - open_trades["amount"].sum()
        pnl_list.append({"account_id": account_id, "pnl": pnl})

pnl_df = pd.DataFrame(pnl_list)
account_pnl = pnl_df.groupby("account_id")["pnl"].sum()

merged["Trading_PnL_USD"] = merged["account_id"].map(account_pnl).fillna(0)
merged["Total_Profit_USD"] = merged["Trading_PnL_USD"]

# ============================================================
# 14. 평균 대비 초과 수익
# ============================================================
merged["Excess_Profit"] = zscore(merged["Total_Profit_USD"].fillna(0))

# ============================================================
# 15. 리스크-수익 비율
# ============================================================
merged["Risk_Reward_Ratio"] = merged["Total_Profit_USD"] / (merged["Organized_Score"].abs() + 1e-6)

# ============================================================
# 16. 자동 해석 (행동 유형)
# ============================================================
def interpret(row):
    if row["Risk_Level"] == "High" and row["Excess_Profit"] < 0:
        return "고위험·저수익형 (비효율적 조작 시도)"
    elif row["Risk_Level"] == "High" and row["Excess_Profit"] >= 0:
        return "고위험·고수익형 (공동조작 성공 가능성)"
    elif row["Risk_Level"] == "Medium" and row["Excess_Profit"] >= 0.5:
        return "중위험·고수익형 (은밀한 협조 가능성)"
    elif row["Risk_Level"] == "Low" and row["Excess_Profit"] > 1:
        return "저위험·고수익형 (은폐형 조직 거래)"
    else:
        return "일반형 (특이행동 없음)"

merged["해석"] = merged.apply(interpret, axis=1)

# ============================================================
# 17. 결과 출력
# ============================================================
merged_sorted = merged.sort_values("Organized_Score", ascending=False).reset_index(drop=True)
print("=" * 80)
print("조직적 거래 탐지 결과 (Z-score + 가중치 + 비선형 강조 + 수익 분석 + 해석)")
print("=" * 80)
print(merged_sorted[[
    "account_id", "Organized_Score", "Risk_Level",
    "Total_Profit_USD", "Excess_Profit", "Risk_Reward_Ratio", "해석"
]].head(20))
