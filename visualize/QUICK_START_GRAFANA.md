# 🚀 Grafana 대시보드 빠른 시작 가이드

## 단계별 실행 방법 (5분 완성)

### 1️⃣ 탐지 실행 및 데이터 생성 (1분)

```bash
# 프로젝트 루트에서 실행
./run_detection_and_generate_dashboard_data.sh
```

이 스크립트는 자동으로:
- ✅ 통합 이상 거래 탐지 실행
- ✅ Grafana용 CSV 파일 7개 생성
- ✅ SQLite 데이터베이스 생성

### 2️⃣ CSV 파일 서빙 (30초)

새 터미널 창에서:

```bash
cd output/final
python3 -m http.server 8080
```

웹 브라우저에서 확인: http://localhost:8080

### 3️⃣ Grafana 설치 및 실행 (2분)

```bash
# macOS
brew install grafana
brew services start grafana

# 또는 Docker
docker run -d -p 3000:3000 grafana/grafana
```

브라우저에서 접속: http://localhost:3000
- 기본 계정: admin / admin

### 4️⃣ Infinity 플러그인 설치 (1분)

```bash
grafana-cli plugins install yesoreyeram-infinity-datasource
brew services restart grafana
```

### 5️⃣ 데이터 소스 추가 (30초)

1. Grafana에서 **⚙️ Configuration** → **Data sources**
2. **Add data source** 클릭
3. **Infinity** 검색 후 선택
4. 이름: `Detection CSV Data`
5. **Save & test**

### 6️⃣ 첫 번째 대시보드 만들기 (1분)

#### 방법 A: 간단한 테이블 (추천)

1. **➕** → **Dashboard** → **Add new panel**
2. 데이터 소스: `Detection CSV Data`
3. Type: **CSV**
4. URL: `http://localhost:8080/grafana_accounts_table.csv`
5. Parsing: **CSV**
6. Visualization: **Table**
7. **Apply** 클릭

완성! 계정별 리스크 테이블이 표시됩니다.

#### 방법 B: 요약 통계

1. 새 패널 추가
2. URL: `http://localhost:8080/grafana_summary_stats.csv`
3. Visualization: **Stat**
4. Value: `critical_count`
5. Title: "CRITICAL 계정 수"
6. Color: 빨강

---

## 📊 생성된 데이터 파일

| 파일명 | 용도 | 추천 패널 타입 |
|--------|------|----------------|
| `grafana_summary_stats.csv` | 전체 요약 통계 | Stat, Gauge |
| `grafana_timeseries.csv` | 시계열 데이터 (7일) | Time series |
| `grafana_accounts_table.csv` | 계정 상세 테이블 | Table |
| `grafana_pattern_stats.csv` | 패턴별 통계 | Bar chart |
| `grafana_alerts_priority.csv` | 우선순위 알림 | Table, Logs |
| `grafana_review_status.csv` | 검토 상태 | Gauge |
| `grafana_detection_alerts.csv` | 알림 상세 | Logs |

---

## 🎨 대시보드 패널 빠른 생성 템플릿

### 패널 1: 리스크 레벨 분포 (Pie Chart)

```
Data Source: Detection CSV Data
Type: CSV
URL: http://localhost:8080/grafana_summary_stats.csv
Visualization: Pie chart

Values:
- critical_count (빨강)
- high_count (주황)
- suspicious_count (노랑)
- normal_count (녹색)
```

### 패널 2: 시계열 추세 (Time Series)

```
URL: http://localhost:8080/grafana_timeseries.csv
Visualization: Time series
X-axis: timestamp
Y-axis: critical_count, high_count, suspicious_count
```

### 패널 3: 고위험 계정 테이블

```
URL: http://localhost:8080/grafana_alerts_priority.csv
Visualization: Table
Columns: account_id, final_risk, pattern1_risk, pattern2_risk, pattern3_risk, total_profit_usd, reviewed
```

---

## 🔄 데이터 업데이트

```bash
# 탐지 재실행 및 데이터 업데이트
./run_detection_and_generate_dashboard_data.sh
```

Grafana 대시보드에서 **Auto-refresh** 설정:
- 우측 상단 ⏱️ 아이콘 클릭
- **5s, 10s, 30s, 1m, 5m** 중 선택

---

## 🆚 CSV vs SQLite 비교

| 방법 | 장점 | 단점 |
|------|------|------|
| **CSV + HTTP 서버** | ✅ 설치 간단<br>✅ 즉시 사용 가능 | ❌ 쿼리 기능 제한<br>❌ 대용량 데이터 느림 |
| **SQLite DB** | ✅ SQL 쿼리 가능<br>✅ 빠른 성능<br>✅ 뷰/인덱스 지원 | ❌ 플러그인 추가 설치 필요 |

**추천**: 소규모 데이터(< 1000건) → CSV, 대규모 → SQLite

---

## 📌 현재 데이터 현황

```
총 계정 수: 63
├─ CRITICAL: 6 계정
├─ HIGH: 15 계정
├─ SUSPICIOUS: 6 계정
└─ NORMAL: 36 계정

패턴별 고위험:
├─ Pattern 1 (펀딩피 차익거래): 6 계정
├─ Pattern 2 (조직적 거래): 9 계정
└─ Pattern 3 (보너스 악용): 12 계정

현재 시점: 2025-10-31 20:00:38
```

---

## 🐛 트러블슈팅

### Q: CSV 파일이 로딩되지 않습니다

**A**: HTTP 서버가 실행 중인지 확인
```bash
# 터미널에서 확인
curl http://localhost:8080/grafana_summary_stats.csv
```

### Q: "No data" 메시지가 표시됩니다

**A**:
1. URL이 올바른지 확인
2. Parser를 **CSV**로 설정했는지 확인
3. 브라우저에서 직접 URL 접속 테스트

### Q: 시간대가 맞지 않습니다

**A**: Grafana 설정 변경
1. **⚙️ Configuration** → **Preferences**
2. Timezone: **Browser Time** 또는 **Asia/Seoul**

---

## 📚 상세 가이드

- **상세 설정**: [GRAFANA_SETUP_GUIDE.md](./GRAFANA_SETUP_GUIDE.md)
- **피처 분석**: [FEATURE_ANALYSIS_REPORT.md](./FEATURE_ANALYSIS_REPORT.md)
- **모델 설계**: [MODEL_DESIGN.md](./MODEL_DESIGN.md)

---

## ✅ 완료!

이제 대시보드에서 실시간으로 이상 거래를 모니터링할 수 있습니다!

🎯 다음 단계:
1. [ ] 알림 규칙 설정 (CRITICAL 계정 > 10개 시 알림)
2. [ ] Slack/Email 연동
3. [ ] 자동 업데이트 스케줄링 (Cron)
4. [ ] 커스텀 대시보드 디자인

---

**작성일**: 2025-11-14
