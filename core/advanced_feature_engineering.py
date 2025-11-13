import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 120)
print("고급 피처 엔지니어링 & 퀀트 기반 이상거래 탐지")
print("=" * 120)
print("\n[연구 배경]")
print("실무 거래소 리스크팀과 퀀트 트레이더들이 사용하는 고급 지표를 통해")
print("단순 통계를 넘어선 정교한 이상거래 탐지를 수행합니다.")
print("\n주요 참고 지표:")
print("  1. Sharpe Ratio - 위험 대비 수익률 (정상: 0.5~2.0, 이상: 3.0+)")
print("  2. Maximum Drawdown - 최대 손실폭 (정상: <30%, 이상: >50%)")
print("  3. Win Rate - 승률 (정상: 45~55%, 이상: >80% or <20%)")
print("  4. Profit Factor - 총이익/총손실 (정상: 1.2~2.0, 이상: >5.0)")
print("  5. Trade Frequency Entropy - 거래 시간 다양성 (낮으면 패턴화)")
print("  6. Position Concentration - 특정 심볼 집중도")
print("  7. Kelly Criterion Deviation - 최적 포지션 크기 이탈도")

# ================================================================================
# 1. 데이터 로드 및 전처리
# ================================================================================
print("\n" + "=" * 120)
print("[1] 데이터 로드 및 포지션 재구성")
print("=" * 120)

trade_df = pd.read_csv('data/Trade.csv')
funding_df = pd.read_csv('data/Funding.csv')
reward_df = pd.read_csv('data/Reward.csv')
ip_df = pd.read_csv('data/IP.csv')

trade_df['ts'] = pd.to_datetime(trade_df['ts'])
funding_df['ts'] = pd.to_datetime(funding_df['ts'])
reward_df['ts'] = pd.to_datetime(reward_df['ts'])

print(f"✓ Trade: {len(trade_df):,} rows")
print(f"✓ Funding: {len(funding_df):,} rows")
print(f"✓ Reward: {len(reward_df):,} rows")

# 포지션 재구성
positions = {}
for _, row in trade_df.iterrows():
    pos_id = row['position_id']
    if pos_id not in positions:
        positions[pos_id] = {
            'opens': [],
            'closes': [],
            'account_id': row['account_id']
        }

    if row['openclose'] == 'OPEN':
        positions[pos_id]['opens'].append(row)
    else:
        positions[pos_id]['closes'].append(row)

# 완전한 포지션만 추출
completed_positions = []
for pos_id, pos_data in positions.items():
    if len(pos_data['opens']) > 0 and len(pos_data['closes']) > 0:
        # 첫 오픈과 마지막 클로즈 사용
        first_open = pos_data['opens'][0]
        last_close = pos_data['closes'][-1]

        # PnL 계산
        if first_open['side'] == 'LONG':
            pnl = (last_close['price'] - first_open['price']) * first_open['qty']
        else:
            pnl = (first_open['price'] - last_close['price']) * first_open['qty']

        # 수익률 (%) 계산
        initial_value = first_open['amount']
        if initial_value > 0:
            return_pct = (pnl / initial_value) * 100
        else:
            return_pct = 0

        completed_positions.append({
            'position_id': pos_id,
            'account_id': pos_data['account_id'],
            'symbol': first_open['symbol'],
            'side': first_open['side'],
            'leverage': first_open['leverage'],
            'open_time': first_open['ts'],
            'close_time': last_close['ts'],
            'open_price': first_open['price'],
            'close_price': last_close['price'],
            'qty': first_open['qty'],
            'initial_value': initial_value,
            'pnl': pnl,
            'return_pct': return_pct,
            'holding_minutes': (last_close['ts'] - first_open['ts']).total_seconds() / 60
        })

positions_df = pd.DataFrame(completed_positions)
print(f"\n완료된 포지션: {len(positions_df):,}개")
print(f"계정 수: {positions_df['account_id'].nunique()}개")

# ================================================================================
# 2. 퀀트 기반 고급 피처 계산
# ================================================================================
print("\n" + "=" * 120)
print("[2] 퀀트 기반 고급 피처 엔지니어링")
print("=" * 120)

advanced_features = []

all_accounts = positions_df['account_id'].unique()

for account in all_accounts:
    account_positions = positions_df[positions_df['account_id'] == account].copy()
    account_positions = account_positions.sort_values('close_time')

    if len(account_positions) == 0:
        continue

    # ============================================================
    # 2.1 Sharpe Ratio (위험 대비 수익률)
    # ============================================================
    returns = account_positions['return_pct'].values
    if len(returns) > 1 and returns.std() > 0:
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(len(returns))
    else:
        sharpe_ratio = 0

    # ============================================================
    # 2.2 Maximum Drawdown (최대 낙폭)
    # ============================================================
    cumulative_returns = (1 + account_positions['return_pct'] / 100).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min() * 100  # 백분율로

    # ============================================================
    # 2.3 Win Rate (승률)
    # ============================================================
    wins = (account_positions['pnl'] > 0).sum()
    losses = (account_positions['pnl'] < 0).sum()
    total_trades = len(account_positions)
    win_rate = wins / total_trades if total_trades > 0 else 0

    # ============================================================
    # 2.4 Profit Factor (총이익/총손실 비율)
    # ============================================================
    total_profit = account_positions[account_positions['pnl'] > 0]['pnl'].sum()
    total_loss = abs(account_positions[account_positions['pnl'] < 0]['pnl'].sum())
    profit_factor = total_profit / total_loss if total_loss > 0 else (10 if total_profit > 0 else 0)

    # ============================================================
    # 2.5 Average Trade Duration (평균 보유시간)
    # ============================================================
    avg_holding = account_positions['holding_minutes'].mean()
    median_holding = account_positions['holding_minutes'].median()

    # ============================================================
    # 2.6 Trade Frequency Entropy (거래 시간 다양성)
    # ============================================================
    # 시간대별 거래 분포의 엔트로피 (낮으면 특정 시간에만 거래)
    hour_dist = account_positions['open_time'].dt.hour.value_counts(normalize=True)
    if len(hour_dist) > 0:
        entropy = -sum(hour_dist * np.log2(hour_dist + 1e-10))
        max_entropy = np.log2(24)  # 24시간 균등 분포 시 최대 엔트로피
        normalized_entropy = entropy / max_entropy
    else:
        normalized_entropy = 0

    # ============================================================
    # 2.7 Position Concentration (포지션 집중도)
    # ============================================================
    symbol_dist = account_positions['symbol'].value_counts(normalize=True)
    # Herfindahl-Hirschman Index (HHI): 높을수록 집중
    hhi = (symbol_dist ** 2).sum()

    # ============================================================
    # 2.8 Leverage Statistics (레버리지 통계)
    # ============================================================
    avg_leverage = account_positions['leverage'].mean()
    max_leverage = account_positions['leverage'].max()
    leverage_std = account_positions['leverage'].std()

    # ============================================================
    # 2.9 Kelly Criterion Deviation (켈리 기준 이탈도)
    # ============================================================
    # Kelly = (Win Rate × Avg Win) - (Loss Rate × Avg Loss)
    if wins > 0 and losses > 0:
        avg_win = account_positions[account_positions['pnl'] > 0]['pnl'].mean()
        avg_loss = abs(account_positions[account_positions['pnl'] < 0]['pnl'].mean())
        kelly_optimal = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win if avg_win > 0 else 0
        kelly_optimal = max(min(kelly_optimal, 1), 0)  # 0~1 범위

        # 실제 사용 레버리지와 비교
        actual_leverage_ratio = avg_leverage / 100
        kelly_deviation = abs(actual_leverage_ratio - kelly_optimal)
    else:
        kelly_optimal = 0
        kelly_deviation = 0

    # ============================================================
    # 2.10 Consistency Score (일관성 점수)
    # ============================================================
    # 수익률의 변동성이 낮고, 꾸준히 수익내면 높은 점수
    if len(returns) > 1:
        returns_std = returns.std()
        returns_mean = returns.mean()
        if returns_std > 0:
            consistency = abs(returns_mean) / returns_std
        else:
            consistency = 0
    else:
        consistency = 0

    # ============================================================
    # 2.11 Risk-Adjusted Return (위험조정수익)
    # ============================================================
    total_return = account_positions['pnl'].sum()
    total_capital = account_positions['initial_value'].sum()
    roi = (total_return / total_capital * 100) if total_capital > 0 else 0

    # ============================================================
    # 2.12 Trade Interval Statistics (거래 간격 통계)
    # ============================================================
    if len(account_positions) > 1:
        time_diffs = account_positions['open_time'].diff().dt.total_seconds() / 60
        avg_interval = time_diffs.mean()
        std_interval = time_diffs.std()
        cv_interval = std_interval / avg_interval if avg_interval > 0 else 0
    else:
        avg_interval = 0
        std_interval = 0
        cv_interval = 0

    advanced_features.append({
        'account_id': account,
        'total_positions': total_trades,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_holding_minutes': avg_holding,
        'median_holding_minutes': median_holding,
        'time_entropy': normalized_entropy,
        'position_concentration_hhi': hhi,
        'avg_leverage': avg_leverage,
        'max_leverage': max_leverage,
        'leverage_std': leverage_std,
        'kelly_optimal': kelly_optimal,
        'kelly_deviation': kelly_deviation,
        'consistency_score': consistency,
        'roi_pct': roi,
        'total_pnl': total_return,
        'avg_trade_interval_min': avg_interval,
        'trade_interval_cv': cv_interval
    })

features_df = pd.DataFrame(advanced_features)
features_df = features_df.sort_values('sharpe_ratio', ascending=False)

print(f"\n고급 피처 계산 완료: {len(features_df)}개 계정")

# ================================================================================
# 3. 이상 패턴 탐지 (실무 기준)
# ================================================================================
print("\n" + "=" * 120)
print("[3] 실무 기반 이상 패턴 탐지")
print("=" * 120)

print("\n[핵심 발견사항]")

# 3.1 Sharpe Ratio 이상
print(f"\n1️⃣ Sharpe Ratio 분석")
print(f"  평균: {features_df['sharpe_ratio'].mean():.4f}")
print(f"  중앙값: {features_df['sharpe_ratio'].median():.4f}")
print(f"  도메인 지식: 정상(0.5~2.0), 의심(2.0~3.0), 고위험(>3.0 또는 <-2.0)")

abnormal_sharpe = features_df[
    (features_df['sharpe_ratio'] > 3.0) | (features_df['sharpe_ratio'] < -2.0)
]
print(f"  이상 Sharpe: {len(abnormal_sharpe)}개 ({len(abnormal_sharpe)/len(features_df)*100:.1f}%)")

# 3.2 Win Rate 이상
print(f"\n2️⃣ Win Rate (승률) 분석")
print(f"  평균: {features_df['win_rate'].mean()*100:.2f}%")
print(f"  중앙값: {features_df['win_rate'].median()*100:.2f}%")
print(f"  도메인 지식: 정상(45~60%), 의심(>80% 또는 <30%)")

abnormal_winrate = features_df[
    (features_df['win_rate'] > 0.8) | (features_df['win_rate'] < 0.3)
]
print(f"  이상 승률: {len(abnormal_winrate)}개 ({len(abnormal_winrate)/len(features_df)*100:.1f}%)")

# 3.3 Profit Factor 이상
print(f"\n3️⃣ Profit Factor 분석")
pf_normal = features_df[features_df['profit_factor'] < 20]  # 극단값 제외
print(f"  평균 (정상범위): {pf_normal['profit_factor'].mean():.2f}")
print(f"  중앙값: {features_df['profit_factor'].median():.2f}")
print(f"  도메인 지식: 정상(1.2~2.5), 의심(>5.0)")

abnormal_pf = features_df[features_df['profit_factor'] > 5.0]
print(f"  이상 Profit Factor: {len(abnormal_pf)}개 ({len(abnormal_pf)/len(features_df)*100:.1f}%)")

# 3.4 Time Entropy 이상 (패턴화된 거래)
print(f"\n4️⃣ Time Entropy (거래 시간 다양성) 분석")
print(f"  평균: {features_df['time_entropy'].mean():.4f}")
print(f"  중앙값: {features_df['time_entropy'].median():.4f}")
print(f"  도메인 지식: 정상(>0.7), 의심(<0.5 = 특정 시간에만 거래)")

low_entropy = features_df[features_df['time_entropy'] < 0.5]
print(f"  낮은 엔트로피: {len(low_entropy)}개 ({len(low_entropy)/len(features_df)*100:.1f}%)")

# 3.5 Position Concentration (특정 심볼 집중)
print(f"\n5️⃣ Position Concentration (HHI) 분석")
print(f"  평균: {features_df['position_concentration_hhi'].mean():.4f}")
print(f"  중앙값: {features_df['position_concentration_hhi'].median():.4f}")
print(f"  도메인 지식: 정상(<0.5), 의심(>0.7 = 소수 심볼 집중)")

high_concentration = features_df[features_df['position_concentration_hhi'] > 0.7]
print(f"  고집중도: {len(high_concentration)}개 ({len(high_concentration)/len(features_df)*100:.1f}%)")

# 3.6 Kelly Deviation (비합리적 레버리지)
print(f"\n6️⃣ Kelly Criterion Deviation 분석")
print(f"  평균: {features_df['kelly_deviation'].mean():.4f}")
print(f"  도메인 지식: 정상(<0.3), 의심(>0.5 = 최적값 크게 벗어남)")

high_kelly_dev = features_df[features_df['kelly_deviation'] > 0.5]
print(f"  높은 이탈: {len(high_kelly_dev)}개 ({len(high_kelly_dev)/len(features_df)*100:.1f}%)")

# ================================================================================
# 4. 종합 Quant-Based Anomaly Score
# ================================================================================
print("\n" + "=" * 120)
print("[4] 퀀트 기반 종합 이상거래 점수 산출")
print("=" * 120)

quant_scores = []

for _, row in features_df.iterrows():
    # 1. Sharpe Score (극단적일수록 높음)
    if abs(row['sharpe_ratio']) > 3.0:
        sharpe_score = 1.0
    elif abs(row['sharpe_ratio']) > 2.0:
        sharpe_score = 0.7
    else:
        sharpe_score = 0.0

    # 2. Win Rate Score (극단적일수록 높음)
    if row['win_rate'] > 0.8 or row['win_rate'] < 0.3:
        winrate_score = 1.0
    elif row['win_rate'] > 0.7 or row['win_rate'] < 0.4:
        winrate_score = 0.6
    else:
        winrate_score = 0.0

    # 3. Profit Factor Score
    if row['profit_factor'] > 5.0:
        pf_score = 1.0
    elif row['profit_factor'] > 3.0:
        pf_score = 0.6
    else:
        pf_score = 0.0

    # 4. Time Entropy Score (낮을수록 높음)
    if row['time_entropy'] < 0.3:
        entropy_score = 1.0
    elif row['time_entropy'] < 0.5:
        entropy_score = 0.6
    else:
        entropy_score = 0.0

    # 5. Concentration Score
    if row['position_concentration_hhi'] > 0.8:
        concentration_score = 1.0
    elif row['position_concentration_hhi'] > 0.6:
        concentration_score = 0.5
    else:
        concentration_score = 0.0

    # 6. Kelly Deviation Score
    if row['kelly_deviation'] > 0.7:
        kelly_score = 1.0
    elif row['kelly_deviation'] > 0.5:
        kelly_score = 0.6
    else:
        kelly_score = 0.0

    # 가중치: w1=0.25, w2=0.20, w3=0.20, w4=0.15, w5=0.10, w6=0.10
    total_score = (0.25 * sharpe_score +
                   0.20 * winrate_score +
                   0.20 * pf_score +
                   0.15 * entropy_score +
                   0.10 * concentration_score +
                   0.10 * kelly_score)

    quant_scores.append({
        'account_id': row['account_id'],
        'sharpe_score': sharpe_score,
        'winrate_score': winrate_score,
        'pf_score': pf_score,
        'entropy_score': entropy_score,
        'concentration_score': concentration_score,
        'kelly_score': kelly_score,
        'quant_anomaly_score': total_score,
        **row.to_dict()
    })

quant_df = pd.DataFrame(quant_scores)
quant_df = quant_df.sort_values('quant_anomaly_score', ascending=False)

print(f"\n[종합 점수 분포]")
print(f"  평균: {quant_df['quant_anomaly_score'].mean():.4f}")
print(f"  중앙값: {quant_df['quant_anomaly_score'].median():.4f}")
print(f"  표준편차: {quant_df['quant_anomaly_score'].std():.4f}")
print(f"  95th: {quant_df['quant_anomaly_score'].quantile(0.95):.4f}")

quant_df['risk_level'] = pd.cut(
    quant_df['quant_anomaly_score'],
    bins=[-np.inf, 0.3, 0.6, np.inf],
    labels=['Low', 'Medium', 'High']
)

risk_counts = quant_df['risk_level'].value_counts()
print(f"\n[위험도 분류]")
for level in ['High', 'Medium', 'Low']:
    count = risk_counts.get(level, 0)
    print(f"  {level}: {count}개 ({count/len(quant_df)*100:.2f}%)")

print(f"\n[고위험 계정 Top 10]")
for i, row in quant_df.head(10).iterrows():
    print(f"  {row['account_id']}: Score={row['quant_anomaly_score']:.4f}")
    print(f"    Sharpe={row['sharpe_ratio']:.2f}, WinRate={row['win_rate']*100:.1f}%, "
          f"PF={row['profit_factor']:.2f}, Entropy={row['time_entropy']:.3f}")

# ================================================================================
# 5. 결과 저장
# ================================================================================
print("\n" + "=" * 120)
print("[5] 결과 저장")
print("=" * 120)

quant_df.to_csv('output/funding_analysis/quant_features_all.csv', index=False)
print(f"✓ 전체 피처: output/funding_analysis/quant_features_all.csv")

high_risk = quant_df[quant_df['risk_level'] == 'High']
high_risk.to_csv('output/funding_analysis/quant_high_risk.csv', index=False)
print(f"✓ 고위험 계정: output/funding_analysis/quant_high_risk.csv ({len(high_risk)}개)")

# ================================================================================
# 6. 시각화
# ================================================================================
print("\n" + "=" * 120)
print("[6] 시각화 생성")
print("=" * 120)

fig = plt.figure(figsize=(22, 16))
gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

# 1. Sharpe Ratio 분포
ax1 = fig.add_subplot(gs[0, 0])
sharpe_plot = features_df[features_df['sharpe_ratio'].between(-5, 10)]['sharpe_ratio']
ax1.hist(sharpe_plot, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax1.axvline(3.0, color='red', linestyle='--', linewidth=2, label='고위험 (3.0)')
ax1.axvline(-2.0, color='orange', linestyle='--', linewidth=2, label='고위험 (-2.0)')
ax1.set_xlabel('Sharpe Ratio', fontsize=11)
ax1.set_ylabel('계정 수', fontsize=11)
ax1.set_title('1. Sharpe Ratio 분포', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Win Rate 분포
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(features_df['win_rate']*100, bins=30, edgecolor='black', alpha=0.7, color='green')
ax2.axvline(30, color='red', linestyle='--', linewidth=2, label='이상 (<30%)')
ax2.axvline(80, color='red', linestyle='--', linewidth=2, label='이상 (>80%)')
ax2.set_xlabel('Win Rate (%)', fontsize=11)
ax2.set_ylabel('계정 수', fontsize=11)
ax2.set_title('2. 승률 분포', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Profit Factor 분포
ax3 = fig.add_subplot(gs[0, 2])
pf_plot = features_df[features_df['profit_factor'] < 10]['profit_factor']
ax3.hist(pf_plot, bins=50, edgecolor='black', alpha=0.7, color='orange')
ax3.axvline(5.0, color='red', linestyle='--', linewidth=2, label='이상 (>5.0)')
ax3.set_xlabel('Profit Factor', fontsize=11)
ax3.set_ylabel('계정 수', fontsize=11)
ax3.set_title('3. Profit Factor 분포', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Time Entropy
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(features_df['time_entropy'], bins=30, edgecolor='black', alpha=0.7, color='purple')
ax4.axvline(0.5, color='orange', linestyle='--', linewidth=2, label='의심 (<0.5)')
ax4.axvline(0.3, color='red', linestyle='--', linewidth=2, label='고위험 (<0.3)')
ax4.set_xlabel('Time Entropy', fontsize=11)
ax4.set_ylabel('계정 수', fontsize=11)
ax4.set_title('4. 거래 시간 다양성', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Position Concentration (HHI)
ax5 = fig.add_subplot(gs[1, 1])
ax5.hist(features_df['position_concentration_hhi'], bins=30, edgecolor='black', alpha=0.7, color='coral')
ax5.axvline(0.7, color='orange', linestyle='--', linewidth=2, label='의심 (>0.7)')
ax5.set_xlabel('HHI (Concentration)', fontsize=11)
ax5.set_ylabel('계정 수', fontsize=11)
ax5.set_title('5. 포지션 집중도', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Kelly Deviation
ax6 = fig.add_subplot(gs[1, 2])
ax6.hist(features_df['kelly_deviation'], bins=30, edgecolor='black', alpha=0.7, color='gold')
ax6.axvline(0.5, color='orange', linestyle='--', linewidth=2, label='의심 (>0.5)')
ax6.set_xlabel('Kelly Deviation', fontsize=11)
ax6.set_ylabel('계정 수', fontsize=11)
ax6.set_title('6. 켈리 기준 이탈도', fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 7. Quant Anomaly Score
ax7 = fig.add_subplot(gs[2, 0])
ax7.hist(quant_df['quant_anomaly_score'], bins=30, edgecolor='black', alpha=0.7, color='crimson')
ax7.axvline(0.3, color='orange', linestyle='--', linewidth=2, label='Medium')
ax7.axvline(0.6, color='darkred', linestyle='--', linewidth=2, label='High')
ax7.set_xlabel('Quant Anomaly Score', fontsize=11)
ax7.set_ylabel('계정 수', fontsize=11)
ax7.set_title('7. 퀀트 기반 종합 이상점수', fontsize=12, fontweight='bold')
ax7.legend()
ax7.grid(True, alpha=0.3)

# 8. 위험도 분류
ax8 = fig.add_subplot(gs[2, 1])
risk_sorted = risk_counts.reindex(['High', 'Medium', 'Low'])
colors = ['red', 'orange', 'green']
bars = ax8.bar(range(len(risk_sorted)), risk_sorted.values, color=colors, edgecolor='black')
ax8.set_xticks(range(len(risk_sorted)))
ax8.set_xticklabels(risk_sorted.index, fontsize=11)
ax8.set_ylabel('계정 수', fontsize=11)
ax8.set_title('8. 위험도 분류', fontsize=12, fontweight='bold')
ax8.grid(True, alpha=0.3, axis='y')
for bar, count in zip(bars, risk_sorted.values):
    ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{count}\n({count/len(quant_df)*100:.1f}%)',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# 9. Sharpe vs Win Rate 산점도
ax9 = fig.add_subplot(gs[2, 2])
scatter = ax9.scatter(features_df['sharpe_ratio'], features_df['win_rate']*100,
                     c=quant_df['quant_anomaly_score'], cmap='Reds',
                     s=50, alpha=0.6, edgecolors='black')
ax9.axvline(3.0, color='gray', linestyle='--', alpha=0.5)
ax9.axhline(80, color='gray', linestyle='--', alpha=0.5)
ax9.set_xlabel('Sharpe Ratio', fontsize=11)
ax9.set_ylabel('Win Rate (%)', fontsize=11)
ax9.set_title('9. Sharpe vs Win Rate', fontsize=12, fontweight='bold')
ax9.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax9, label='Anomaly Score')

# 10. 고위험 계정 Top 15
ax10 = fig.add_subplot(gs[3, :])
top_15 = quant_df.head(15)
y_pos = np.arange(len(top_15))
bars = ax10.barh(y_pos, top_15['quant_anomaly_score'], color='darkred', edgecolor='black')
ax10.set_yticks(y_pos)
ax10.set_yticklabels([f"{acc[:15]}..." for acc in top_15['account_id']], fontsize=9)
ax10.set_xlabel('Quant Anomaly Score', fontsize=11)
ax10.set_title('10. 퀀트 기반 이상거래 의심 계정 Top 15', fontsize=12, fontweight='bold')
ax10.invert_yaxis()
ax10.grid(True, alpha=0.3, axis='x')
for i, (idx, row) in enumerate(top_15.iterrows()):
    ax10.text(row['quant_anomaly_score'], i, f" {row['quant_anomaly_score']:.3f}",
              va='center', fontsize=8, fontweight='bold')

fig.suptitle('퀀트 기반 고급 피처 엔지니어링 & 이상거래 탐지',
             fontsize=18, fontweight='bold', y=0.998)

plt.savefig('output/funding_analysis/quant_analysis_visualization.png', dpi=300, bbox_inches='tight')
print(f"✓ 시각화: output/funding_analysis/quant_analysis_visualization.png")

print("\n" + "=" * 120)
print("고급 피처 분석 완료!")
print("=" * 120)
print("\n[핵심 공식]")
print("QuantAnomalyScore = 0.25×Sharpe + 0.20×WinRate + 0.20×ProfitFactor + ")
print("                    0.15×TimeEntropy + 0.10×Concentration + 0.10×Kelly")
