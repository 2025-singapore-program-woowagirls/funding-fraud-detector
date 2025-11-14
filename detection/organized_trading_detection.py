import pandas as pd
import numpy as np
from scipy.stats import zscore

# ============================================================
# 1️⃣ 데이터 로딩
# ============================================================
trade_df = pd.read_csv("data/Trade.csv")
ip_df = pd.read_csv("data/IP.csv")

# ============================================================
# 2️⃣ 피처 계산 (Feature Engineering)
# ============================================================

# --- (1) IP 공유율 ---
ip_counts = ip_df.groupby("ip")["account_id"].nunique()
shared_ip = ip_counts[ip_counts > 1].index
account_ip_total = ip_df.groupby("account_id")["ip"].nunique()
account_ip_shared = ip_df[ip_df["ip"].isin(shared_ip)].groupby("account_id")["ip"].nunique()
ip_shared_ratio = (account_ip_shared / account_ip_total).fillna(0)

# --- (2) 동시 거래율 ---
trade_df["ts_minute"] = pd.to_datetime(trade_df["ts"]).dt.floor("1min")
trade_df["concurrent_count"] = trade_df.groupby(["ts_minute", "symbol"])["account_id"].transform("count")
trade_df["is_concurrent"] = (trade_df["concurrent_count"] > 1).astype(int)
concurrent_ratio = trade_df.groupby("account_id")["is_concurrent"].mean()

# --- (3) 가격 유사도 ---
price_stats = trade_df.groupby(["ts_minute", "symbol"])["price"].agg(["mean", "std"]).reset_index()
price_stats["price_cv"] = price_stats["std"] / price_stats["mean"]
trade_df = trade_df.merge(price_stats[["ts_minute", "symbol", "price_cv"]], on=["ts_minute", "symbol"], how="left")
price_cv = trade_df.groupby("account_id")["price_cv"].mean().fillna(0)

# --- (4) 평균 레버리지 ---
trade_df["leverage"] = trade_df["leverage"].fillna(1)
leverage_mean = trade_df.groupby("account_id")["leverage"].mean()

# --- (5) 조율된 거래 (S_coordinated) ---
S_coordinated = np.sqrt(concurrent_ratio * (1 - price_cv.clip(0, 1)))  # 가격 유사도는 1-CV 형태로 변환

# ============================================================
# 3️⃣ 피처 통합
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
# 4️⃣ 표준화(Z-score) + 가중합
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
# 5️⃣ 리스크 등급 분류
# ============================================================
high_thresh = merged["Organized_Score"].quantile(0.85)
medium_thresh = merged["Organized_Score"].quantile(0.5)
conditions = [
    merged["Organized_Score"] >= high_thresh,
    merged["Organized_Score"] >= medium_thresh
]
choices = ["높음", "중간"]
merged["Risk_Level"] = np.select(conditions, choices, default="낮음")

# ============================================================
# 6️⃣ 거래 수익 계산
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
merged["Total_Profit_USD"] = merged["account_id"].map(account_pnl).fillna(0)

# ============================================================
# 7️⃣ 초과 수익 및 리스크-수익 비율
# ============================================================
merged["Excess_Profit"] = zscore(merged["Total_Profit_USD"].fillna(0))
merged["Risk_Reward_Ratio"] = merged["Total_Profit_USD"] / (merged["Organized_Score"].abs() + 1e-6)

# ============================================================
# 8️⃣ 자동 해석 (행동 유형)
# ============================================================
def interpret(row):
    if row["Risk_Level"] == "높음" and row["Excess_Profit"] < 0:
        return "고위험·저수익형 (비효율적 조작 시도)"
    elif row["Risk_Level"] == "높음" and row["Excess_Profit"] >= 0:
        return "고위험·고수익형 (공동조작 가능성)"
    elif row["Risk_Level"] == "중간" and row["Excess_Profit"] >= 0.5:
        return "중위험·고수익형 (은밀한 협조 가능성)"
    elif row["Risk_Level"] == "낮음" and row["Excess_Profit"] > 1:
        return "저위험·고수익형 (은폐형 거래 가능성)"
    else:
        return "일반형 (특이행동 없음)"

merged["해석"] = merged.apply(interpret, axis=1)

# ============================================================
# 9️⃣ 최종 출력 (한글 표)
# ============================================================
merged_sorted = merged.sort_values("Organized_Score", ascending=False).reset_index(drop=True)

merged_sorted_kor = merged_sorted.rename(columns={
    "account_id": "계정 ID",
    "Organized_Score": "조직적 점수",
    "Risk_Level": "위험 등급",
    "Total_Profit_USD": "총 수익 (USD)",
    "Excess_Profit": "초과 수익",
    "Risk_Reward_Ratio": "리스크-수익 비율",
    "해석": "거래 유형 해석"
})

core_columns_kor = ["계정 ID", "조직적 점수", "위험 등급", "총 수익 (USD)", "초과 수익", "거래 유형 해석"]

print("=" * 100)
print("조직적 거래 탐지 결과 ")
print("=" * 100)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0)
print(merged_sorted_kor[core_columns_kor].head(20))
