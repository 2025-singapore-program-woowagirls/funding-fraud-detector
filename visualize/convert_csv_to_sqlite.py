"""
CSV 파일을 SQLite 데이터베이스로 변환
Grafana SQLite 데이터 소스와 함께 사용
"""
import pandas as pd
import sqlite3
import os

print("=" * 100)
print("CSV → SQLite 변환 시작")
print("=" * 100)

# SQLite DB 파일 경로
db_path = 'output/final/detection.db'

# 기존 DB 파일 삭제 (새로 생성)
if os.path.exists(db_path):
    os.remove(db_path)
    print(f"✅ 기존 DB 삭제: {db_path}")

# SQLite 연결
conn = sqlite3.connect(db_path)

# CSV 파일들을 테이블로 저장
csv_files = {
    'summary_stats': 'grafana_summary_stats.csv',
    'pattern_stats': 'grafana_pattern_stats.csv',
    'accounts': 'grafana_accounts_table.csv',
    'timeseries': 'grafana_timeseries.csv',
    'alerts_priority': 'grafana_alerts_priority.csv',
    'review_status': 'grafana_review_status.csv',
    'detection_alerts': 'grafana_detection_alerts.csv',
    'all_results': 'integrated_detection_results.csv',
    'high_risk_only': 'high_risk_accounts_detected.csv',
}

print()
for table_name, csv_file in csv_files.items():
    csv_path = f'output/final/{csv_file}'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        print(f"✅ {table_name:20s} 테이블 생성 ({len(df):4d} rows) ← {csv_file}")
    else:
        print(f"⚠️  {table_name:20s} 스킵 (파일 없음) ← {csv_file}")

# 인덱스 생성 (쿼리 성능 향상)
print()
print("=" * 100)
print("인덱스 생성 중...")
print("=" * 100)

cursor = conn.cursor()

# 계정 ID 인덱스
cursor.execute("CREATE INDEX IF NOT EXISTS idx_accounts_id ON accounts(account_id)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_accounts_risk ON accounts(final_risk)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_all_results_id ON all_results(account_id)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_account ON alerts_priority(account_id)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_risk ON alerts_priority(final_risk)")

# 시간 인덱스
cursor.execute("CREATE INDEX IF NOT EXISTS idx_timeseries_ts ON timeseries(timestamp)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_detection_ts ON detection_alerts(timestamp)")

print("✅ 인덱스 생성 완료")

# 뷰 생성 (자주 사용하는 쿼리)
print()
print("=" * 100)
print("뷰 생성 중...")
print("=" * 100)

# 1. 고위험 계정 뷰
cursor.execute("""
CREATE VIEW IF NOT EXISTS v_high_risk_accounts AS
SELECT
    account_id,
    final_risk,
    pattern1_risk,
    pattern2_risk,
    pattern3_risk,
    funding_fee_abs,
    mean_holding_minutes,
    funding_timing_ratio,
    funding_profit_ratio,
    ip_shared_count,
    mean_leverage,
    total_reward,
    total_profit_usd,
    reviewed,
    detection_time
FROM accounts
WHERE final_risk IN ('CRITICAL', 'HIGH')
ORDER BY
    CASE final_risk
        WHEN 'CRITICAL' THEN 1
        WHEN 'HIGH' THEN 2
        ELSE 3
    END,
    total_profit_usd DESC
""")
print("✅ v_high_risk_accounts 뷰 생성")

# 2. 미확인 알림 뷰
cursor.execute("""
CREATE VIEW IF NOT EXISTS v_unreviewed_alerts AS
SELECT
    account_id,
    final_risk,
    pattern1_risk,
    pattern2_risk,
    pattern3_risk,
    total_profit_usd,
    detection_time
FROM accounts
WHERE reviewed = 0
  AND final_risk IN ('CRITICAL', 'HIGH', 'SUSPICIOUS')
ORDER BY
    CASE final_risk
        WHEN 'CRITICAL' THEN 1
        WHEN 'HIGH' THEN 2
        WHEN 'SUSPICIOUS' THEN 3
        ELSE 4
    END
""")
print("✅ v_unreviewed_alerts 뷰 생성")

# 3. 패턴별 요약 뷰
cursor.execute("""
CREATE VIEW IF NOT EXISTS v_pattern_summary AS
SELECT
    'Pattern1_FundingFeeArbitrage' as pattern,
    SUM(CASE WHEN pattern1_risk = 'HIGH' THEN 1 ELSE 0 END) as high_count,
    SUM(CASE WHEN pattern1_risk = 'SUSPICIOUS' THEN 1 ELSE 0 END) as suspicious_count,
    SUM(CASE WHEN pattern1_risk = 'NORMAL' THEN 1 ELSE 0 END) as normal_count
FROM accounts
UNION ALL
SELECT
    'Pattern2_OrganizedTrading' as pattern,
    SUM(CASE WHEN pattern2_risk = 'HIGH' THEN 1 ELSE 0 END) as high_count,
    SUM(CASE WHEN pattern2_risk = 'SUSPICIOUS' THEN 1 ELSE 0 END) as suspicious_count,
    SUM(CASE WHEN pattern2_risk = 'NORMAL' THEN 1 ELSE 0 END) as normal_count
FROM accounts
UNION ALL
SELECT
    'Pattern3_BonusAbuse' as pattern,
    SUM(CASE WHEN pattern3_risk = 'HIGH' THEN 1 ELSE 0 END) as high_count,
    SUM(CASE WHEN pattern3_risk = 'SUSPICIOUS' THEN 1 ELSE 0 END) as suspicious_count,
    SUM(CASE WHEN pattern3_risk = 'NORMAL' THEN 1 ELSE 0 END) as normal_count
FROM accounts
WHERE total_reward > 0
""")
print("✅ v_pattern_summary 뷰 생성")

conn.commit()
conn.close()

print()
print("=" * 100)
print("✅ SQLite 변환 완료!")
print("=" * 100)
print(f"DB 파일: {db_path}")
print(f"절대 경로: {os.path.abspath(db_path)}")
print()
print("Grafana에서 사용 방법:")
print("1. SQLite 데이터 소스 플러그인 설치")
print("   grafana-cli plugins install frser-sqlite-datasource")
print()
print("2. 데이터 소스 추가")
print(f"   Path: {os.path.abspath(db_path)}")
print()
print("3. 쿼리 예시:")
print("   SELECT * FROM v_high_risk_accounts")
print("   SELECT * FROM v_unreviewed_alerts")
print("   SELECT * FROM v_pattern_summary")
print("=" * 100)
