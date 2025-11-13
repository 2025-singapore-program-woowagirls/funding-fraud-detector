#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
펀딩피 수익 비중 재계산 (funding_profit_ratio)
- Trade.csv에서 position_id별 OPEN/CLOSE amount 차이로 PnL 계산
- Funding.csv에서 펀딩비 합계 계산
- 비율 = abs(펀딩비 합계) / (abs(실현손익) + abs(펀딩비 합계))
"""

import pandas as pd
import numpy as np
from pathlib import Path

# 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent.parent
data_dir = BASE_DIR / 'data'

print("=" * 80)
print("펀딩피 수익 비중 재계산 (Funding Profit Ratio Recalculation)")
print("=" * 80)

# 1. 데이터 로드
print("\n[1] 데이터 로드")
trade_df = pd.read_csv(data_dir / 'Trade.csv')
funding_df = pd.read_csv(data_dir / 'Funding.csv')
print(f"  ✓ Trade: {len(trade_df):,}건")
print(f"  ✓ Funding: {len(funding_df):,}건")

# 2. PnL 계산 (position_id별 OPEN/CLOSE 매칭)
print("\n[2] 실현손익(PnL) 계산")
positions = []

for (account_id, position_id), group in trade_df.groupby(['account_id', 'position_id']):
    open_trades = group[group['openclose'] == 'OPEN']
    close_trades = group[group['openclose'] == 'CLOSE']

    if len(open_trades) > 0 and len(close_trades) > 0:
        open_amount = open_trades['amount'].sum()
        close_amount = close_trades['amount'].sum()
        pnl = close_amount - open_amount

        positions.append({
            'account_id': account_id,
            'position_id': position_id,
            'open_amount': open_amount,
            'close_amount': close_amount,
            'pnl': pnl
        })

pnl_df = pd.DataFrame(positions)
print(f"  ✓ 완결된 포지션: {len(pnl_df):,}개")
print(f"  ✓ PnL 합계: ${pnl_df['pnl'].sum():,.2f}")
print(f"  ✓ 수익 포지션: {len(pnl_df[pnl_df['pnl'] > 0]):,}개 (${pnl_df[pnl_df['pnl'] > 0]['pnl'].sum():,.2f})")
print(f"  ✓ 손실 포지션: {len(pnl_df[pnl_df['pnl'] < 0]):,}개 (${pnl_df[pnl_df['pnl'] < 0]['pnl'].sum():,.2f})")

# 계정별 총 PnL
account_pnl = pnl_df.groupby('account_id')['pnl'].sum().reset_index(name='total_pnl')
print(f"  ✓ 거래 계정 수: {len(account_pnl):,}개")

# 3. 펀딩비 합계
print("\n[3] 펀딩비 합계 계산")
account_funding = funding_df.groupby('account_id')['funding_fee'].sum().reset_index(name='total_funding')
account_funding['total_funding_abs'] = account_funding['total_funding'].abs()
print(f"  ✓ 펀딩비 총합: ${account_funding['total_funding'].sum():,.2f}")
print(f"  ✓ 펀딩비 절댓값 총합: ${account_funding['total_funding_abs'].sum():,.2f}")

# 4. 펀딩피 수익 비중 계산
print("\n[4] 펀딩피 수익 비중 계산")
merged = pd.merge(account_pnl, account_funding, on='account_id', how='outer').fillna(0)
merged['total_pnl_abs'] = merged['total_pnl'].abs()

# funding_profit_ratio = abs(펀딩비) / (abs(펀딩비) + abs(실현손익))
merged['funding_profit_ratio'] = merged.apply(
    lambda row: row['total_funding_abs'] / (row['total_funding_abs'] + row['total_pnl_abs'] + 0.001)
    if (row['total_funding_abs'] + row['total_pnl_abs']) > 0 else 0,
    axis=1
)

# 통계 계산
print(f"  ✓ 계산 완료: {len(merged):,}개 계정")
print(f"\n  통계:")
print(f"    - 평균: {merged['funding_profit_ratio'].mean():.4f} ({merged['funding_profit_ratio'].mean()*100:.2f}%)")
print(f"    - 중앙값: {merged['funding_profit_ratio'].median():.4f} ({merged['funding_profit_ratio'].median()*100:.2f}%)")
print(f"    - 표준편차: {merged['funding_profit_ratio'].std():.4f}")
print(f"    - 최솟값: {merged['funding_profit_ratio'].min():.4f}")
print(f"    - 최댓값: {merged['funding_profit_ratio'].max():.4f}")

# 백분위수
percentiles = [25, 50, 75, 90, 95, 99]
print(f"\n  백분위수:")
for p in percentiles:
    val = merged['funding_profit_ratio'].quantile(p / 100)
    print(f"    - {p}th: {val:.4f} ({val*100:.2f}%)")

# 5. 그룹별 분석
print("\n[5] 그룹별 분석")

# 0.5 (50%) 기준으로 분류
high_funding_ratio = merged[merged['funding_profit_ratio'] >= 0.5]
low_funding_ratio = merged[merged['funding_profit_ratio'] < 0.5]

print(f"\n  펀딩비 비중 ≥ 50%: {len(high_funding_ratio):,}개 계정")
print(f"    - 평균 펀딩비 비중: {high_funding_ratio['funding_profit_ratio'].mean():.4f} ({high_funding_ratio['funding_profit_ratio'].mean()*100:.2f}%)")
print(f"    - 평균 펀딩비: ${high_funding_ratio['total_funding_abs'].mean():,.2f}")
print(f"    - 평균 PnL: ${high_funding_ratio['total_pnl'].mean():,.2f}")

print(f"\n  펀딩비 비중 < 50%: {len(low_funding_ratio):,}개 계정")
print(f"    - 평균 펀딩비 비중: {low_funding_ratio['funding_profit_ratio'].mean():.4f} ({low_funding_ratio['funding_profit_ratio'].mean()*100:.2f}%)")
print(f"    - 평균 펀딩비: ${low_funding_ratio['total_funding_abs'].mean():,.2f}")
print(f"    - 평균 PnL: ${low_funding_ratio['total_pnl'].mean():,.2f}")

# 변별력 계산
if len(high_funding_ratio) > 0 and len(low_funding_ratio) > 0:
    ratio_diff = high_funding_ratio['funding_profit_ratio'].mean() / low_funding_ratio['funding_profit_ratio'].mean()
    print(f"\n  변별력: {ratio_diff:.2f}배 차이")

# 6. 상세 분포
print("\n[6] 구간별 분포")
bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
labels = ['0-10%', '10-30%', '30-50%', '50-70%', '70-90%', '90-100%']
merged['ratio_bin'] = pd.cut(merged['funding_profit_ratio'], bins=bins, labels=labels, include_lowest=True)
distribution = merged['ratio_bin'].value_counts().sort_index()

for label in labels:
    count = distribution.get(label, 0)
    pct = count / len(merged) * 100
    print(f"  {label}: {count:,}개 ({pct:.1f}%)")

# 7. 결과 저장
print("\n[7] 결과 저장")
output_file = BASE_DIR / 'output' / 'final' / 'funding_profit_ratio_analysis.csv'
output_file.parent.mkdir(parents=True, exist_ok=True)
merged[['account_id', 'total_pnl', 'total_funding', 'total_funding_abs', 'funding_profit_ratio']].to_csv(
    output_file, index=False
)
print(f"  ✓ 저장 완료: {output_file}")

# 8. 결론
print("\n" + "=" * 80)
print("결론 (Conclusion)")
print("=" * 80)

# 상위 10% 분석
top_10_pct = merged['funding_profit_ratio'].quantile(0.9)
top_10 = merged[merged['funding_profit_ratio'] >= top_10_pct]

print(f"\n상위 10% (펀딩비 비중 ≥ {top_10_pct:.2f}):")
print(f"  - 계정 수: {len(top_10):,}개")
print(f"  - 평균 펀딩비 비중: {top_10['funding_profit_ratio'].mean():.4f} ({top_10['funding_profit_ratio'].mean()*100:.2f}%)")
print(f"  - 평균 펀딩비: ${top_10['total_funding_abs'].mean():,.2f}")
print(f"  - 평균 PnL: ${top_10['total_pnl'].mean():,.2f}")

# 하위 90% 분석
bottom_90 = merged[merged['funding_profit_ratio'] < top_10_pct]
print(f"\n하위 90% (펀딩비 비중 < {top_10_pct:.2f}):")
print(f"  - 계정 수: {len(bottom_90):,}개")
print(f"  - 평균 펀딩비 비중: {bottom_90['funding_profit_ratio'].mean():.4f} ({bottom_90['funding_profit_ratio'].mean()*100:.2f}%)")
print(f"  - 평균 펀딩비: ${bottom_90['total_funding_abs'].mean():,.2f}")
print(f"  - 평균 PnL: ${bottom_90['total_pnl'].mean():,.2f}")

if len(top_10) > 0 and len(bottom_90) > 0:
    diff_ratio = top_10['funding_profit_ratio'].mean() / bottom_90['funding_profit_ratio'].mean()
    print(f"\n변별력: 상위 10%는 하위 90%보다 {diff_ratio:.2f}배 높은 펀딩비 비중")

    if diff_ratio >= 1.5:
        print("  ✅ 유의미한 피처 (1.5배 이상 차이)")
    else:
        print("  ⚠️  변별력 부족 (1.5배 미만 차이)")

print("\n" + "=" * 80)
