import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# =============================================================================
# 1. 데이터 로드 및 전처리
# =============================================================================

def load_and_preprocess_data():
    """데이터 로드 및 기본 전처리"""

    funding = pd.read_csv("../../data/Funding.csv")
    trade = pd.read_csv("../../data/Trade.csv")

    print("\n 데이터 구조 확인:")
    print("\n[Funding 컬럼]")
    print(funding.columns.tolist())
    print("\n[Trade 컬럼]")
    print(trade.columns.tolist())
    print("\n[Trade 샘플 데이터]")
    print(trade.head(3))

    funding['ts'] = pd.to_datetime(funding['ts'])
    trade['ts'] = pd.to_datetime(trade['ts'])

    funding['fee_rate'] = pd.to_numeric(funding['fee_rate'], errors='coerce')
    funding['funding_fee'] = pd.to_numeric(funding['funding_fee'], errors='coerce')

    numeric_cols = ['price', 'qty', 'amount', 'leverage']
    for col in numeric_cols:
        if col in trade.columns:
            trade[col] = pd.to_numeric(trade[col], errors='coerce')

    funding['date'] = funding['ts'].dt.date
    trade['date'] = trade['ts'].dt.date

    funding['hour'] = funding['ts'].dt.hour
    trade['hour'] = trade['ts'].dt.hour

    print(f"\n✓ 펀딩 데이터: {len(funding)} rows")
    print(f"✓ 트레이딩 데이터: {len(trade)} rows")
    print(f"✓ 기간: {trade['ts'].min()} ~ {trade['ts'].max()}")
    print(f"✓ 계정 수: {trade['account_id'].nunique()}")

    return funding, trade


# =============================================================================
# 2. 거래 손익 계산 (openclose 컬럼 사용)
# =============================================================================

def calculate_pnl(trade):
    """각 거래의 손익 계산"""

    trade = trade.sort_values(['account_id', 'position_id', 'ts'])

    pnl_list = []

    for position_id in trade['position_id'].unique():
        pos_trades = trade[trade['position_id'] == position_id].copy()

        # openclose 컬럼 사용
        opens = pos_trades[pos_trades['openclose'] == 'OPEN']
        closes = pos_trades[pos_trades['openclose'] == 'CLOSE']

        if len(opens) == 0 or len(closes) == 0:
            continue

        # 평균 진입가와 청산가
        avg_open_price = (opens['price'] * opens['qty']).sum() / opens['qty'].sum()
        avg_close_price = (closes['price'] * closes['qty']).sum() / closes['qty'].sum()
        total_qty = closes['qty'].sum()

        # 손익 계산 (롱/숏 구분)
        side = opens['side'].iloc[0]
        if side == 'LONG':
            pnl = (avg_close_price - avg_open_price) * total_qty
        else:  # SHORT
            pnl = (avg_open_price - avg_close_price) * total_qty

        pnl_list.append({
            'position_id': position_id,
            'account_id': opens['account_id'].iloc[0],
            'pnl': pnl,
            'open_time': opens['ts'].min(),
            'close_time': closes['ts'].max(),
            'holding_hours': (closes['ts'].max() - opens['ts'].min()).total_seconds() / 3600,
            'side': side
        })

    pnl_df = pd.DataFrame(pnl_list)
    print(f"✓ {len(pnl_df)} 포지션의 손익 계산 완료")

    return pnl_df


# =============================================================================
# 3. 펀딩 타임 전후 거래 패턴 분석
# =============================================================================

def analyze_funding_time_pattern(trade, funding, window_minutes=60):
    """펀딩 타임(00:00, 08:00, 16:00) 전후 거래 패턴 분석"""

    funding_hours = [0, 8, 16]
    results = []

    for account in trade['account_id'].unique():
        acc_trades = trade[trade['account_id'] == account].copy()
        total_trades = len(acc_trades)

        if total_trades == 0:
            continue

        funding_time_trades = 0

        for _, row in acc_trades.iterrows():
            trade_hour = row['ts'].hour
            trade_minute = row['ts'].minute

            for f_hour in funding_hours:
                trade_minutes = trade_hour * 60 + trade_minute
                funding_minutes = f_hour * 60

                distance = min(
                    abs(trade_minutes - funding_minutes),
                    abs(trade_minutes - funding_minutes + 1440),
                    abs(trade_minutes - funding_minutes - 1440)
                )

                if distance <= window_minutes:
                    funding_time_trades += 1
                    break

        funding_ratio = funding_time_trades / total_trades

        results.append({
            'account_id': account,
            'total_trades': total_trades,
            'funding_time_trades': funding_time_trades,
            'funding_time_ratio': funding_ratio
        })

    pattern_df = pd.DataFrame(results)
    print(f"✓ {len(pattern_df)} 계정의 펀딩 타임 패턴 분석 완료")

    return pattern_df


# =============================================================================
# 4. 계정별 종합 통계
# =============================================================================

def calculate_account_stats(trade, funding, pnl_df):
    """계정별 종합 통계 계산"""

    stats_list = []

    for account in trade['account_id'].unique():
        acc_pnl = pnl_df[pnl_df['account_id'] == account]
        total_pnl = acc_pnl['pnl'].sum() if len(acc_pnl) > 0 else 0
        avg_holding_hours = acc_pnl['holding_hours'].mean() if len(acc_pnl) > 0 else 0

        acc_funding = funding[funding['account_id'] == account]
        total_funding_fee = acc_funding['funding_fee'].sum() if len(acc_funding) > 0 else 0
        funding_count = len(acc_funding)

        acc_trades = trade[trade['account_id'] == account]
        total_volume = acc_trades['amount'].sum() if 'amount' in acc_trades.columns else 0

        stats_list.append({
            'account_id': account,
            'total_pnl': total_pnl,
            'total_funding_fee': total_funding_fee,
            'funding_count': funding_count,
            'avg_holding_hours': avg_holding_hours,
            'total_volume': total_volume,
            'trade_count': len(acc_trades),
            'position_count': len(acc_pnl)
        })

    stats_df = pd.DataFrame(stats_list)
    print(f"✓ {len(stats_df)} 계정의 통계 계산 완료")

    return stats_df


# =============================================================================
# 5. 펀딩 헌터 탐지 (매우 엄격한 기준 - 상위 10개 정도만)
# =============================================================================

def detect_funding_hunters(stats_df, pattern_df,
                           funding_ratio_threshold=0.75,     # 75%로 상향
                           pnl_funding_ratio_threshold=5.0,  # 5배로 상향
                           holding_time_min=7.0,             # 7~9시간으로 좁힘
                           holding_time_max=9.0,
                           min_funding_fee=50.0):            # $50로 상향
    """
    펀딩 헌터 탐지 알고리즘 (매우 엄격한 기준)

    목표: 상위 10개 정도의 확실한 케이스만 검출
    """

    merged = stats_df.merge(pattern_df, on='account_id', how='left')

    merged['suspicion_score'] = 0
    merged['suspicion_reasons'] = ''

    for idx, row in merged.iterrows():
        score = 0
        reasons = []

        # Pattern 1: 펀딩 타임 거래 집중 (최대 60점)
        if row['funding_time_ratio'] >= funding_ratio_threshold:
            score += 35
            reasons.append(f"펀딩타임_거래집중({row['funding_time_ratio']:.1%})")

        if row['funding_time_ratio'] >= 0.85:
            score += 25  # 85% 이상이면 추가

        # Pattern 2: 펀딩피 과다 수익 (최대 80점)
        if row['total_funding_fee'] > min_funding_fee:
            if row['total_pnl'] != 0:
                pnl_funding_ratio = abs(row['total_funding_fee'] / row['total_pnl'])
                if pnl_funding_ratio >= pnl_funding_ratio_threshold:
                    score += 40
                    reasons.append(f"펀딩피과다(손익대비_{pnl_funding_ratio:.1f}배)")
                if pnl_funding_ratio >= 20.0:
                    score += 40  # 20배 이상이면 추가
            elif row['total_funding_fee'] > 0:
                score += 60
                reasons.append("거래손익없이_펀딩피수익")

        # Pattern 3: 8시간 홀딩 패턴 (30점)
        if holding_time_min <= row['avg_holding_hours'] <= holding_time_max:
            score += 30
            reasons.append(f"보유시간의심({row['avg_holding_hours']:.1f}h)")

        # Pattern 4: 펀딩피 수익이 있음 (10점)
        if row['total_funding_fee'] > min_funding_fee:
            score += 10
            reasons.append(f"펀딩수익(${row['total_funding_fee']:.2f})")

        # Pattern 5: 펀딩 횟수 과다 (20점)
        if row['funding_count'] > 30:  # 10일 이상
            score += 20
            reasons.append(f"펀딩횟수과다({row['funding_count']}회)")

        # Pattern 6: 손실을 펀딩으로 커버 (40점)
        if row['total_pnl'] < 0 and row['total_funding_fee'] > abs(row['total_pnl']) * 3:
            score += 40
            reasons.append("손실을_펀딩으로_대폭_커버")

        merged.at[idx, 'suspicion_score'] = score
        merged.at[idx, 'suspicion_reasons'] = ', '.join(reasons) if reasons else 'N/A'

    # 의심 등급 부여 (더 엄격하게)
    merged['risk_level'] = pd.cut(merged['suspicion_score'],
                                  bins=[-1, 40, 70, 100, 250],
                                  labels=['Low', 'Medium', 'High', 'Critical'])

    merged = merged.sort_values('suspicion_score', ascending=False)

    print(f"\n{'='*80}")
    print(f"✓ 펀딩 헌터 탐지 완료")
    print(f"  - Critical Risk: {len(merged[merged['risk_level'] == 'Critical'])} 계정")
    print(f"  - High Risk: {len(merged[merged['risk_level'] == 'High'])} 계정")
    print(f"  - Medium Risk: {len(merged[merged['risk_level'] == 'Medium'])} 계정")
    print(f"  - Low Risk: {len(merged[merged['risk_level'] == 'Low'])} 계정")
    print(f"{'='*80}\n")

    return merged


# =============================================================================
# 6. 결과 출력
# =============================================================================

def print_results(results, min_score=40, max_display=15):
    """의심 점수가 일정 이상인 계정 출력"""

    suspects = results[results['suspicion_score'] >= min_score]

    print(f"\n{'='*100}")
    print(f" 의심 점수 {min_score}점 이상 계정 (총 {len(suspects)}개)")
    print(f"{'='*100}\n")

    display_count = min(max_display, len(suspects))
    top_suspects = suspects.head(display_count)

    for i, (idx, row) in enumerate(top_suspects.iterrows(), 1):
        print(f"[{i}] [{row['risk_level']}] {row['account_id']}")
        print(f"    의심점수: {row['suspicion_score']:.0f}점")
        print(f"    펀딩피 수익: ${row['total_funding_fee']:.2f} ({row['funding_count']}회)")
        print(f"    거래 손익: ${row['total_pnl']:.2f}")

        if row['total_pnl'] != 0:
            ratio = abs(row['total_funding_fee'] / row['total_pnl'])
            print(f"    펀딩/손익 비율: {ratio:.2f}배")
        else:
            print(f"    펀딩/손익 비율: N/A (거래손익 0)")

        print(f"    펀딩타임 거래비율: {row['funding_time_ratio']:.1%} ({row['funding_time_trades']}/{row['total_trades']}건)")
        print(f"    평균 보유시간: {row['avg_holding_hours']:.2f}시간")
        print(f"    거래량: ${row['total_volume']:.2f} (포지션 {row['position_count']}개)")
        print(f"    의심 이유: {row['suspicion_reasons']}")
        print()

    if len(suspects) > max_display:
        print(f"... 외 {len(suspects) - max_display}개 계정 (CSV 파일 참조)\n")


# =============================================================================
# 7. 요약 통계
# =============================================================================

def print_summary(results):
    """전체 요약 통계"""

    print(f"\n{'='*100}")
    print(" 전체 요약 통계")
    print(f"{'='*100}\n")

    print(f"총 계정 수: {len(results)}")
    print(f"\n[위험도별 분포]")
    for level in ['Critical', 'High', 'Medium', 'Low']:
        count = len(results[results['risk_level'] == level])
        pct = count / len(results) * 100
        print(f"  {level:10s}: {count:4d}개 ({pct:5.1f}%)")

    print(f"\n[의심 점수 분포]")
    print(f"  평균: {results['suspicion_score'].mean():.1f}점")
    print(f"  중간값: {results['suspicion_score'].median():.1f}점")
    print(f"  최대: {results['suspicion_score'].max():.1f}점")

    print(f"\n[점수 구간별 분포]")
    for threshold in [120, 100, 80, 70, 50]:
        count = len(results[results['suspicion_score'] >= threshold])
        print(f"  {threshold}점 이상: {count}개")

    print(f"\n[펀딩 피 통계]")
    print(f"  총 펀딩피: ${results['total_funding_fee'].sum():.2f}")
    print(f"  평균: ${results['total_funding_fee'].mean():.2f}")
    print(f"  중간값: ${results['total_funding_fee'].median():.2f}")
    print(f"  최대: ${results['total_funding_fee'].max():.2f}")

    print(f"\n[거래 손익 통계]")
    print(f"  총 거래손익: ${results['total_pnl'].sum():.2f}")
    print(f"  평균: ${results['total_pnl'].mean():.2f}")
    print(f"  흑자 계정: {len(results[results['total_pnl'] > 0])}개")
    print(f"  적자 계정: {len(results[results['total_pnl'] < 0])}개")

    print(f"\n[펀딩타임 거래 패턴]")
    print(f"  평균 펀딩타임 거래비율: {results['funding_time_ratio'].mean():.1%}")
    print(f"  75% 이상: {len(results[results['funding_time_ratio'] >= 0.75])}개")
    print(f"  85% 이상: {len(results[results['funding_time_ratio'] >= 0.85])}개")
    print(f"  95% 이상: {len(results[results['funding_time_ratio'] >= 0.95])}개")

    print(f"\n[보유 시간 패턴]")
    print(f"  평균 보유시간: {results['avg_holding_hours'].mean():.2f}시간")
    print(f"  7~9시간 보유: {len(results[(results['avg_holding_hours'] >= 7) & (results['avg_holding_hours'] <= 9)])}개")

    print()


# =============================================================================
# 8. 상세 분석 리포트 (상위 계정)
# =============================================================================

def print_detailed_analysis(results, top_n=10):
    """상위 N개 계정에 대한 상세 분석"""

    print(f"\n{'='*100}")
    print(f" 상위 {top_n}개 계정 상세 분석")
    print(f"{'='*100}\n")

    top = results.head(top_n)

    for i, (idx, row) in enumerate(top.iterrows(), 1):
        print(f"\n{'─'*100}")
        print(f"순위 #{i} | {row['account_id']} | {row['risk_level']} Risk | {row['suspicion_score']:.0f}점")
        print(f"{'─'*100}")

        # 기본 정보
        print(f"\n 기본 정보")
        print(f"  • 총 거래 횟수: {row['trade_count']}건")
        print(f"  • 총 포지션: {row['position_count']}개")
        print(f"  • 총 거래량: ${row['total_volume']:,.2f}")
        print(f"  • 거래 기간: {row['funding_count']} 펀딩 주기 (약 {row['funding_count'] * 8 / 24:.1f}일)")

        # 수익 구조
        print(f"\n 수익 구조")
        print(f"  • 거래 손익: ${row['total_pnl']:,.2f}")
        print(f"  • 펀딩피 수익: ${row['total_funding_fee']:,.2f}")
        total_profit = row['total_pnl'] + row['total_funding_fee']
        print(f"  • 순수익: ${total_profit:,.2f}")

        if total_profit != 0:
            funding_contribution = (row['total_funding_fee'] / total_profit) * 100
            print(f"  • 펀딩 기여도: {funding_contribution:.1f}%")

        # 거래 패턴
        print(f"\n 거래 패턴")
        print(f"  • 펀딩타임 거래 비율: {row['funding_time_ratio']:.1%}")
        print(f"  • 펀딩타임 거래: {row['funding_time_trades']}건 / 전체 {row['total_trades']}건")
        print(f"  • 평균 포지션 보유시간: {row['avg_holding_hours']:.2f}시간")

        # 의심 이유
        print(f"\n  의심 이유")
        reasons = row['suspicion_reasons'].split(', ')
        for reason in reasons:
            print(f"  • {reason}")

        print()


# =============================================================================
# 9. 메인 실행
# =============================================================================

def main():
    """메인 실행 함수"""

    print("\n" + "="*100)
    print(" 펀딩 피 이상거래 탐지 시스템")
    print("="*100)

    # 1. 데이터 로드
    funding, trade = load_and_preprocess_data()

    # 2. 거래 손익 계산
    pnl_df = calculate_pnl(trade)

    # 3. 펀딩 타임 패턴 분석 (윈도우 60분으로 축소)
    pattern_df = analyze_funding_time_pattern(trade, funding, window_minutes=60)

    # 4. 계정별 통계
    stats_df = calculate_account_stats(trade, funding, pnl_df)

    # 5. 펀딩 헌터 탐지 (매우 엄격한 기준)
    results = detect_funding_hunters(
        stats_df,
        pattern_df,
        funding_ratio_threshold=0.75,     # 75% 이상만
        pnl_funding_ratio_threshold=5.0,  # 5배 이상만
        holding_time_min=7.0,             # 7~9시간만
        holding_time_max=9.0,
        min_funding_fee=50.0              # $50 이상만
    )

    # 6. 결과 출력 (70점 이상, 최대 15개)
    print_results(results, min_score=70, max_display=15)

    # 7. 요약 통계
    print_summary(results)

    # 8. 상세 분석 (상위 10개)
    print_detailed_analysis(results, top_n=10)

    # 9. 결과 저장
    output_path = "../../data/funding_hunter_detection_results.csv"
    results.to_csv(output_path, index=False)
    print(f"\n 전체 결과가 CSV로 저장되었습니다: {output_path}")

    # 10. 고위험군만 별도 저장 (70점 이상)
    high_risk = results[results['suspicion_score'] >= 70]
    if len(high_risk) > 0:
        high_risk_path = "../../data/funding_hunter_HIGH_RISK.csv"
        high_risk.to_csv(high_risk_path, index=False)
        print(f" 고위험 계정({len(high_risk)}개)이 별도 저장되었습니다: {high_risk_path}")

    # 11. 최종 요약
    print(f"\n{'='*100}")
    print(f" 분석 완료!")
    print(f"   • 총 분석 계정: {len(results)}개")
    print(f"   • High Risk 이상: {len(results[results['risk_level'].isin(['High', 'Critical'])])}개")
    print(f"   • 70점 이상: {len(high_risk)}개")
    print(f"{'='*100}\n")

    return results


# 실행
if __name__ == "__main__":
    results = main()