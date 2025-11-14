"""
Grafana 대시보드용 추가 데이터 생성
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============================================================
# 기존 탐지 결과 로딩
# ============================================================
print("=" * 100)
print("Grafana 대시보드용 데이터 생성 중...")
print("=" * 100)

df = pd.read_csv('output/final/integrated_detection_results.csv')
df['detection_time'] = pd.to_datetime(df['detection_time'])

CURRENT_TIME = pd.to_datetime('2025-10-31 20:00:38.588185')

# ============================================================
# 1. 실시간 대시보드용 요약 통계
# ============================================================
summary_stats = {
    'timestamp': [CURRENT_TIME],
    'total_accounts': [len(df)],
    'critical_count': [len(df[df['final_risk'] == 'CRITICAL'])],
    'high_count': [len(df[df['final_risk'] == 'HIGH'])],
    'suspicious_count': [len(df[df['final_risk'] == 'SUSPICIOUS'])],
    'normal_count': [len(df[df['final_risk'] == 'NORMAL'])],
    'pattern1_high': [len(df[df['pattern1_risk'] == 'HIGH'])],
    'pattern2_high': [len(df[df['pattern2_risk'] == 'HIGH'])],
    'pattern3_high': [len(df[df['pattern3_risk'] == 'HIGH'])],
    'reviewed_count': [len(df[df['reviewed'] == True])],
    'unreviewed_count': [len(df[df['reviewed'] == False])],
}

summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv('output/final/grafana_summary_stats.csv', index=False)
print(f"✅ 요약 통계 저장: grafana_summary_stats.csv")

# ============================================================
# 2. 패턴별 상세 통계
# ============================================================
pattern_stats = []

# Pattern 1
for risk in ['HIGH', 'SUSPICIOUS', 'NORMAL']:
    count = len(df[df['pattern1_risk'] == risk])
    if count > 0:
        pattern_stats.append({
            'timestamp': CURRENT_TIME,
            'pattern': 'Pattern1_FundingFeeArbitrage',
            'risk_level': risk,
            'count': count,
            'avg_funding_fee_abs': df[df['pattern1_risk'] == risk]['funding_fee_abs'].mean(),
            'avg_funding_profit_ratio': df[df['pattern1_risk'] == risk]['funding_profit_ratio'].mean(),
        })

# Pattern 2
for risk in ['HIGH', 'SUSPICIOUS', 'NORMAL']:
    count = len(df[df['pattern2_risk'] == risk])
    if count > 0:
        pattern_stats.append({
            'timestamp': CURRENT_TIME,
            'pattern': 'Pattern2_OrganizedTrading',
            'risk_level': risk,
            'count': count,
            'avg_ip_shared': df[df['pattern2_risk'] == risk]['ip_shared_count'].mean(),
            'avg_leverage': df[df['pattern2_risk'] == risk]['mean_leverage'].mean(),
        })

# Pattern 3
for risk in ['HIGH', 'SUSPICIOUS', 'NORMAL']:
    pattern3_df = df[(df['pattern3_risk'] == risk) & (df['total_reward'] > 0)]
    count = len(pattern3_df)
    if count > 0:
        pattern_stats.append({
            'timestamp': CURRENT_TIME,
            'pattern': 'Pattern3_BonusAbuse',
            'risk_level': risk,
            'count': count,
            'avg_total_reward': pattern3_df['total_reward'].mean(),
            'avg_reward_ip_shared': pattern3_df['reward_ip_shared_count'].mean(),
        })

pattern_stats_df = pd.DataFrame(pattern_stats)
pattern_stats_df.to_csv('output/final/grafana_pattern_stats.csv', index=False)
print(f"✅ 패턴별 통계 저장: grafana_pattern_stats.csv")

# ============================================================
# 3. 계정별 리스크 점수 테이블 (대시보드 테이블용)
# ============================================================
table_data = df[[
    'account_id',
    'final_risk',
    'pattern1_risk',
    'pattern2_risk',
    'pattern3_risk',
    'funding_fee_abs',
    'mean_holding_minutes',
    'funding_timing_ratio',
    'funding_profit_ratio',
    'ip_shared_count',
    'mean_leverage',
    'total_reward',
    'total_profit_usd',
    'reviewed',
    'detection_time'
]].copy()

# 리스크 레벨을 숫자로 변환 (Grafana 색상 매핑용)
risk_map = {'NORMAL': 0, 'SUSPICIOUS': 1, 'HIGH': 2, 'CRITICAL': 3, 'NO_REWARD': 0}
table_data['risk_score'] = table_data['final_risk'].map(risk_map)

# 정렬: 리스크 점수 높은 순
table_data = table_data.sort_values('risk_score', ascending=False)

table_data.to_csv('output/final/grafana_accounts_table.csv', index=False)
print(f"✅ 계정 테이블 저장: grafana_accounts_table.csv")

# ============================================================
# 4. 시계열 시뮬레이션 (과거 7일 데이터 생성)
# ============================================================
# Grafana는 시계열 데이터를 선호하므로, 과거 데이터를 시뮬레이션
timeseries_data = []

for days_ago in range(7, -1, -1):
    timestamp = CURRENT_TIME - timedelta(days=days_ago)

    # 시간이 지남에 따라 탐지 건수가 증가하는 시뮬레이션
    growth_factor = 0.7 + (0.3 * (7 - days_ago) / 7)

    timeseries_data.append({
        'timestamp': timestamp,
        'total_accounts': int(len(df) * growth_factor),
        'critical_count': int(len(df[df['final_risk'] == 'CRITICAL']) * growth_factor),
        'high_count': int(len(df[df['final_risk'] == 'HIGH']) * growth_factor),
        'suspicious_count': int(len(df[df['final_risk'] == 'SUSPICIOUS']) * growth_factor),
        'normal_count': int(len(df[df['final_risk'] == 'NORMAL']) * growth_factor),
        'pattern1_high': int(len(df[df['pattern1_risk'] == 'HIGH']) * growth_factor),
        'pattern2_high': int(len(df[df['pattern2_risk'] == 'HIGH']) * growth_factor),
        'pattern3_high': int(len(df[df['pattern3_risk'] == 'HIGH']) * growth_factor),
    })

timeseries_df = pd.DataFrame(timeseries_data)
timeseries_df.to_csv('output/final/grafana_timeseries.csv', index=False)
print(f"✅ 시계열 데이터 저장: grafana_timeseries.csv (7일간)")

# ============================================================
# 5. 알림 우선순위 리스트
# ============================================================
alerts = df[df['final_risk'].isin(['CRITICAL', 'HIGH'])].copy()
alerts['risk_score'] = alerts['final_risk'].map(risk_map)
alerts = alerts.sort_values('risk_score', ascending=False)

alerts_output = alerts[[
    'account_id',
    'final_risk',
    'pattern1_risk',
    'pattern2_risk',
    'pattern3_risk',
    'total_profit_usd',
    'reviewed',
    'detection_time'
]].copy()

alerts_output['alert_message'] = alerts_output.apply(lambda row:
    f"계정 {row['account_id']}: {row['final_risk']} 리스크 탐지 "
    f"(P1:{row['pattern1_risk']}, P2:{row['pattern2_risk']}, P3:{row['pattern3_risk']})",
    axis=1
)

alerts_output.to_csv('output/final/grafana_alerts_priority.csv', index=False)
print(f"✅ 우선순위 알림 저장: grafana_alerts_priority.csv ({len(alerts)} 건)")

# ============================================================
# 6. 검토 상태 추적용 데이터
# ============================================================
review_stats = {
    'timestamp': [CURRENT_TIME],
    'total_alerts': [len(df[df['final_risk'].isin(['CRITICAL', 'HIGH', 'SUSPICIOUS'])])],
    'reviewed': [len(df[df['reviewed'] == True])],
    'unreviewed': [len(df[df['reviewed'] == False])],
    'review_rate': [len(df[df['reviewed'] == True]) / len(df) * 100 if len(df) > 0 else 0]
}

review_df = pd.DataFrame(review_stats)
review_df.to_csv('output/final/grafana_review_status.csv', index=False)
print(f"✅ 검토 상태 저장: grafana_review_status.csv")

# ============================================================
# 완료
# ============================================================
print()
print("=" * 100)
print("Grafana 대시보드용 데이터 생성 완료!")
print("=" * 100)
print(f"생성된 파일:")
print(f"  1. grafana_summary_stats.csv - 전체 요약 통계")
print(f"  2. grafana_pattern_stats.csv - 패턴별 상세 통계")
print(f"  3. grafana_accounts_table.csv - 계정별 상세 테이블")
print(f"  4. grafana_timeseries.csv - 시계열 데이터 (7일)")
print(f"  5. grafana_alerts_priority.csv - 우선순위 알림")
print(f"  6. grafana_review_status.csv - 검토 상태")
print(f"  7. grafana_detection_alerts.csv - 알림 상세 (이미 생성됨)")
print()
