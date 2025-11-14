# Grafana 대시보드 설정 가이드

## 📋 개요

이상 거래 탐지 시스템의 결과를 Grafana 대시보드로 시각화하는 방법을 안내합니다.

**현재 시점**: 2025-10-31 20:00:38 (데이터의 최신 시점)

---

## 🎯 대시보드 구성

### 생성된 CSV 파일 목록

`output/final/` 디렉토리에 다음 파일들이 생성됩니다:

1. **grafana_summary_stats.csv** - 전체 요약 통계
2. **grafana_pattern_stats.csv** - 패턴별 상세 통계
3. **grafana_accounts_table.csv** - 계정별 상세 테이블
4. **grafana_timeseries.csv** - 시계열 데이터 (7일간)
5. **grafana_alerts_priority.csv** - 우선순위 알림
6. **grafana_review_status.csv** - 검토 상태
7. **grafana_detection_alerts.csv** - 알림 상세
8. **integrated_detection_results.csv** - 전체 탐지 결과
9. **high_risk_accounts_detected.csv** - 고위험 계정만

---

## 🚀 빠른 설정 방법 (추천)

### Option 1: Infinity Data Source 플러그인 사용 (가장 쉬움)

#### 1단계: Grafana 설치 및 실행

```bash
# macOS (Homebrew)
brew install grafana
brew services start grafana

# 또는 Docker
docker run -d -p 3000:3000 --name=grafana grafana/grafana
```

Grafana 접속: http://localhost:3000
- 기본 계정: admin / admin

#### 2단계: Infinity Data Source 플러그인 설치

1. Grafana UI에서 **Configuration** → **Plugins** 이동
2. **Infinity** 검색 후 설치
3. 또는 CLI로 설치:

```bash
grafana-cli plugins install yesoreyeram-infinity-datasource
brew services restart grafana
```

#### 3단계: 데이터 소스 추가

1. **Configuration** → **Data sources** → **Add data source**
2. **Infinity** 선택
3. 이름: `Detection CSV Data`
4. **Save & test**

#### 4단계: CSV 파일을 웹 서버로 서빙

간단한 HTTP 서버 실행:

```bash
cd output/final
python3 -m http.server 8080
```

또는 프로젝트 루트에서:

```bash
cd /Users/gimhyejin/Library/CloudStorage/OneDrive-한성대학교/문서/Projects/singapore-prestolabs/BE
python3 -m http.server 8080
```

이제 CSV 파일 접근 가능:
- http://localhost:8080/output/final/grafana_summary_stats.csv
- http://localhost:8080/output/final/grafana_timeseries.csv
- 등등...

---

## 📊 대시보드 패널 구성

### 1️⃣ 실시간 요약 통계 (Stat Panel)

**데이터 소스**: grafana_summary_stats.csv

**패널 설정**:
- Type: **Stat**
- URL: `http://localhost:8080/output/final/grafana_summary_stats.csv`
- Parser: **CSV**
- Timestamp field: `timestamp`

**메트릭**:
- Total Accounts: `total_accounts`
- Critical: `critical_count` (빨강)
- High Risk: `high_count` (주황)
- Suspicious: `suspicious_count` (노랑)
- Normal: `normal_count` (녹색)

---

### 2️⃣ 리스크 레벨 분포 (Pie Chart)

**데이터**: grafana_summary_stats.csv

**쿼리**:
```
critical_count as "CRITICAL"
high_count as "HIGH"
suspicious_count as "SUSPICIOUS"
normal_count as "NORMAL"
```

**색상 설정**:
- CRITICAL: 빨강 (#F2495C)
- HIGH: 주황 (#FF9830)
- SUSPICIOUS: 노랑 (#FADE2A)
- NORMAL: 녹색 (#73BF69)

---

### 3️⃣ 시계열 추세 (Time Series Graph)

**데이터**: grafana_timeseries.csv

**쿼리**:
```
timestamp (X축)
critical_count (빨강)
high_count (주황)
suspicious_count (노랑)
```

**설정**:
- Type: **Time series**
- X-axis: `timestamp`
- Y-axis: 건수
- Legend: 하단에 표시

---

### 4️⃣ 패턴별 탐지 건수 (Bar Chart)

**데이터**: grafana_pattern_stats.csv

**설정**:
- Type: **Bar chart**
- X-axis: `pattern`
- Y-axis: `count`
- Color by: `risk_level`

**패턴 설명 추가**:
- Pattern1_FundingFeeArbitrage: 펀딩피 차익거래
- Pattern2_OrganizedTrading: 조직적 거래
- Pattern3_BonusAbuse: 보너스 악용

---

### 5️⃣ 고위험 계정 테이블 (Table Panel)

**데이터**: grafana_alerts_priority.csv

**설정**:
- Type: **Table**
- Columns:
  - account_id
  - final_risk (색상 매핑)
  - pattern1_risk, pattern2_risk, pattern3_risk
  - total_profit_usd (통화 포맷)
  - reviewed (체크박스 아이콘)
  - alert_message

**색상 규칙**:
```
final_risk:
  CRITICAL → 빨강
  HIGH → 주황
  SUSPICIOUS → 노랑
  NORMAL → 녹색
```

**정렬**: `final_risk` 내림차순

---

### 6️⃣ 검토 진행률 (Gauge)

**데이터**: grafana_review_status.csv

**설정**:
- Type: **Gauge**
- Value: `review_rate` (퍼센트)
- Min: 0
- Max: 100
- Thresholds:
  - 0-30: 빨강
  - 30-70: 노랑
  - 70-100: 녹색

---

### 7️⃣ 알림 스트림 (Logs Panel)

**데이터**: grafana_detection_alerts.csv

**설정**:
- Type: **Logs**
- Message: `description`
- Time: `timestamp`
- Level: `risk_level` (HIGH=error, SUSPICIOUS=warning)

**필터**:
- reviewed=false (미확인만 표시)

---

## 🔧 Option 2: CSV to SQLite 변환 후 SQLite 데이터 소스 사용

### 단계 1: SQLite DB 생성

```bash
cd /Users/gimhyejin/Library/CloudStorage/OneDrive-한성대학교/문서/Projects/singapore-prestolabs/BE
```

Python으로 CSV를 SQLite로 변환:

```python
import pandas as pd
import sqlite3

# SQLite DB 생성
conn = sqlite3.connect('output/final/detection.db')

# CSV 파일들을 테이블로 저장
csv_files = {
    'summary_stats': 'grafana_summary_stats.csv',
    'pattern_stats': 'grafana_pattern_stats.csv',
    'accounts': 'grafana_accounts_table.csv',
    'timeseries': 'grafana_timeseries.csv',
    'alerts': 'grafana_alerts_priority.csv',
    'review': 'grafana_review_status.csv',
}

for table_name, csv_file in csv_files.items():
    df = pd.read_csv(f'output/final/{csv_file}')
    df.to_sql(table_name, conn, if_exists='replace', index=False)

conn.close()
print("✅ SQLite DB 생성 완료: output/final/detection.db")
```

### 단계 2: Grafana SQLite 데이터 소스 설치

```bash
grafana-cli plugins install frser-sqlite-datasource
brew services restart grafana
```

### 단계 3: 데이터 소스 설정

1. **Configuration** → **Data sources** → **Add data source**
2. **SQLite** 선택
3. Path: `/Users/gimhyejin/Library/CloudStorage/OneDrive-한성대학교/문서/Projects/singapore-prestolabs/BE/output/final/detection.db`

### 단계 4: 쿼리 예시

**요약 통계**:
```sql
SELECT * FROM summary_stats ORDER BY timestamp DESC LIMIT 1
```

**시계열**:
```sql
SELECT
  timestamp,
  critical_count,
  high_count,
  suspicious_count
FROM timeseries
ORDER BY timestamp
```

**고위험 계정**:
```sql
SELECT
  account_id,
  final_risk,
  pattern1_risk,
  pattern2_risk,
  pattern3_risk,
  total_profit_usd,
  reviewed
FROM accounts
WHERE final_risk IN ('CRITICAL', 'HIGH')
ORDER BY
  CASE final_risk
    WHEN 'CRITICAL' THEN 1
    WHEN 'HIGH' THEN 2
    ELSE 3
  END
```

---

## 🎨 대시보드 레이아웃 예시

```
┌─────────────────────────────────────────────────────────┐
│           이상 거래 탐지 대시보드                        │
│           현재 시점: 2025-10-31 20:00:38                │
└─────────────────────────────────────────────────────────┘

┌──────────┬──────────┬──────────┬──────────┬──────────┐
│  총 계정  │ CRITICAL │   HIGH   │SUSPICIOUS│  NORMAL  │
│    63    │    6     │    15    │    6     │    36    │
└──────────┴──────────┴──────────┴──────────┴──────────┘

┌─────────────────────────┬─────────────────────────────┐
│  리스크 레벨 분포 (Pie) │  패턴별 탐지 건수 (Bar)     │
│                         │                             │
│    [파이 차트]          │    [막대 그래프]            │
└─────────────────────────┴─────────────────────────────┘

┌───────────────────────────────────────────────────────┐
│        시계열 추세 (7일간)                             │
│                                                        │
│        [시계열 그래프]                                 │
└───────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────┐
│        고위험 계정 테이블                              │
│                                                        │
│  account_id | risk | P1 | P2 | P3 | reviewed         │
│  A_1f97...  | CRIT | H  | H  | -  | ☐                │
│  A_26ff...  | HIGH | N  | S  | H  | ☐                │
│  ...                                                   │
└───────────────────────────────────────────────────────┘

┌─────────────────────┬─────────────────────────────────┐
│  검토 진행률 (Gauge)│  최근 알림 (Logs)               │
│                     │                                 │
│    [게이지]         │    [로그 스트림]                │
└─────────────────────┴─────────────────────────────────┘
```

---

## 🔄 데이터 업데이트 방법

### 실시간 업데이트 스크립트

```bash
#!/bin/bash
# update_dashboard.sh

cd /Users/gimhyejin/Library/CloudStorage/OneDrive-한성대학교/문서/Projects/singapore-prestolabs/BE

# 탐지 실행
source .venv/bin/activate
python detection/integrated_detection.py
python detection/create_grafana_dashboard_data.py

# Grafana 자동 새로고침 (대시보드 설정에서 auto-refresh 활성화)
echo "✅ 데이터 업데이트 완료"
```

### Cron 설정 (매시간 자동 업데이트)

```bash
crontab -e
```

추가:
```
0 * * * * /path/to/update_dashboard.sh
```

---

## 📌 체크박스: 검토 상태 업데이트

### 수동 업데이트 방법

CSV 파일을 직접 수정하거나, Python 스크립트로 업데이트:

```python
import pandas as pd

# 계정 검토 완료 처리
df = pd.read_csv('output/final/integrated_detection_results.csv')
df.loc[df['account_id'] == 'A_1f97e16953', 'reviewed'] = True
df.to_csv('output/final/integrated_detection_results.csv', index=False)

# Grafana용 데이터 재생성
# python detection/create_grafana_dashboard_data.py
```

### 웹 인터페이스 (향후 개선)

Django/Flask로 간단한 웹 인터페이스를 만들어 체크박스 클릭으로 업데이트 가능:

```python
# 예시: Flask API
@app.route('/api/review/<account_id>', methods=['POST'])
def mark_reviewed(account_id):
    df = pd.read_csv('output/final/integrated_detection_results.csv')
    df.loc[df['account_id'] == account_id, 'reviewed'] = True
    df.to_csv('output/final/integrated_detection_results.csv', index=False)
    return {'success': True}
```

---

## 🎯 대시보드 활용 시나리오

### 1️⃣ 일일 모니터링

1. 대시보드 접속
2. **CRITICAL + HIGH** 알림 확인
3. 테이블에서 계정 상세 확인
4. 수동 검토 후 `reviewed` 체크
5. 검토 진행률 트래킹

### 2️⃣ 패턴 분석

1. 패턴별 통계 확인
2. 시계열 추세로 증가/감소 파악
3. 특정 패턴 집중 탐지

### 3️⃣ 보고서 생성

1. 대시보드 스냅샷 저장
2. CSV 파일 다운로드
3. 엑셀/BI 도구로 추가 분석

---

## ⚙️ 고급 설정

### Variables (드롭다운 필터)

대시보드에 변수 추가:

1. **Settings** → **Variables** → **Add variable**

**예시**:
- Name: `risk_level`
- Type: Custom
- Values: `CRITICAL,HIGH,SUSPICIOUS,NORMAL`

쿼리에서 사용:
```
WHERE final_risk = '$risk_level'
```

### Alerts 설정

1. 패널 선택 → **Alert** 탭
2. 조건 설정:
   ```
   WHEN critical_count > 10
   FOR 5m
   ```
3. Notification channel 설정 (Slack, Email 등)

---

## 🐛 트러블슈팅

### 문제 1: CSV 파일 로딩 안됨

**해결**:
- HTTP 서버가 실행 중인지 확인
- CORS 이슈 → Grafana 설정에서 허용

### 문제 2: 시간대 문제

**해결**:
- Grafana 설정에서 timezone을 Asia/Seoul로 변경
- CSV의 timestamp가 ISO 8601 형식인지 확인

### 문제 3: 플러그인 설치 오류

**해결**:
```bash
# 플러그인 수동 설치
cd /usr/local/var/lib/grafana/plugins
git clone https://github.com/yesoreyeram/grafana-infinity-datasource
brew services restart grafana
```

---

## 📚 추가 자료

- [Grafana 공식 문서](https://grafana.com/docs/)
- [Infinity Data Source](https://github.com/yesoreyeram/grafana-infinity-datasource)
- [CSV 데이터 시각화](https://grafana.com/docs/grafana/latest/datasources/csv/)

---

## ✅ 완료 체크리스트

- [ ] Grafana 설치 및 실행
- [ ] Infinity 플러그인 설치
- [ ] HTTP 서버 실행 (CSV 서빙)
- [ ] 데이터 소스 추가
- [ ] 대시보드 생성
- [ ] 패널 7개 추가 (요약, 파이, 시계열, 막대, 테이블, 게이지, 로그)
- [ ] 색상 및 포맷 설정
- [ ] Auto-refresh 활성화
- [ ] 테스트 및 검증

---

**작성일**: 2025-11-14
**데이터 시점**: 2025-10-31 20:00:38
