# 🚀 Grafana 빠른 참조 가이드

> **한 눈에 보는 핵심 명령어와 URL**

---

## ⚡ 빠른 실행

```bash
# 1. 데이터 생성
./run_detection_and_generate_dashboard_data.sh

# 2. CSV 서버 시작 (새 터미널)
cd output/final && python3 -m http.server 8080

# 3. Grafana 접속
open http://localhost:3000
```

---

## 🔗 주요 URL

| 서비스 | URL | 계정 |
|--------|-----|------|
| Grafana 대시보드 | http://localhost:3000 | admin / admin |
| CSV 파일 서버 | http://localhost:8080 | - |
| 요약 통계 CSV | http://localhost:8080/grafana_summary_stats.csv | - |
| 계정 테이블 CSV | http://localhost:8080/grafana_accounts_table.csv | - |
| 시계열 CSV | http://localhost:8080/grafana_timeseries.csv | - |

---

## 📊 패널 템플릿

### 1. 숫자 카드 (Stat)

```
Data Source: Detection CSV Data
Type: URL
URL: http://localhost:8080/grafana_summary_stats.csv
Parser: CSV
Visualization: Stat
Field: critical_count
```

### 2. 테이블 (Table)

```
URL: http://localhost:8080/grafana_accounts_table.csv
Visualization: Table
```

### 3. 시계열 (Time Series)

```
URL: http://localhost:8080/grafana_timeseries.csv
Visualization: Time series
Transform: Convert field type (timestamp → Time)
```

### 4. 파이 차트 (Pie Chart)

```
URL: http://localhost:8080/grafana_summary_stats.csv
Visualization: Pie chart
Filter: critical_count, high_count, suspicious_count, normal_count
```

---

## 🎨 색상 코드

| 리스크 레벨 | 색상 코드 | 이름 |
|-------------|-----------|------|
| CRITICAL | `#F2495C` | 빨강 |
| HIGH | `#FF9830` | 주황 |
| SUSPICIOUS | `#FADE2A` | 노랑 |
| NORMAL | `#73BF69` | 녹색 |

---

## 🔧 자주 쓰는 명령어

### Grafana 제어

```bash
# 시작
brew services start grafana

# 중지
brew services stop grafana

# 재시작
brew services restart grafana

# 상태 확인
brew services list | grep grafana
```

### 플러그인 관리

```bash
# Infinity 설치
grafana-cli plugins install yesoreyeram-infinity-datasource

# 플러그인 목록
grafana-cli plugins ls

# 플러그인 업데이트
grafana-cli plugins update yesoreyeram-infinity-datasource
```

### 데이터 업데이트

```bash
# 전체 재실행
./run_detection_and_generate_dashboard_data.sh

# 탐지만 재실행
source .venv/bin/activate && python detection/integrated_detection.py

# Grafana 데이터만 재생성
python visualize/create_grafana_dashboard_data.py

# SQLite만 재생성
python visualize/convert_csv_to_sqlite.py
```

---

## 🐛 트러블슈팅 체크리스트

### "No data" 에러

```bash
# 1. CSV 서버 확인
curl http://localhost:8080/grafana_summary_stats.csv

# 2. 파일 존재 확인
ls -l output/final/grafana_*.csv

# 3. 서버 재시작
cd output/final && python3 -m http.server 8080
```

### Infinity 플러그인 없음

```bash
grafana-cli plugins install yesoreyeram-infinity-datasource
brew services restart grafana
# 1분 대기 후 새로고침
```

### 포트 충돌

```bash
# 8080 포트 사용 중인 프로세스 확인
lsof -i :8080

# 프로세스 종료
kill -9 <PID>

# 다른 포트 사용
python3 -m http.server 9000
# URL을 http://localhost:9000/... 로 변경
```

---

## 📱 단축키

### Grafana 대시보드

| 단축키 | 기능 |
|--------|------|
| `d` + `s` | 대시보드 저장 |
| `d` + `d` | 대시보드 설정 |
| `e` | 패널 편집 |
| `v` | 패널 보기 모드 |
| `r` | 새로고침 |
| `t` + `z` | 시간 범위 줌 아웃 |

### 패널 편집 모드

| 단축키 | 기능 |
|--------|------|
| `Ctrl` + `S` | 저장 |
| `Esc` | 편집 취소 |

---

## 📋 CSV 파일 매핑

| CSV 파일 | 행 수 | 주요 컬럼 | 용도 |
|----------|-------|-----------|------|
| `grafana_summary_stats.csv` | 1 | total_accounts, critical_count, high_count | 요약 통계 |
| `grafana_timeseries.csv` | 8 | timestamp, critical_count, high_count | 시계열 그래프 |
| `grafana_accounts_table.csv` | 63 | account_id, final_risk, reviewed | 전체 계정 테이블 |
| `grafana_pattern_stats.csv` | 9 | pattern, risk_level, count | 패턴별 통계 |
| `grafana_alerts_priority.csv` | 21 | account_id, final_risk, alert_message | 고위험 알림 |
| `grafana_review_status.csv` | 1 | total_alerts, reviewed, unreviewed | 검토 진행률 |
| `grafana_detection_alerts.csv` | 64 | timestamp, pattern, risk_level, description | 알림 로그 |

---

## 🎯 1분 체크리스트

시작 전:
- [ ] `./run_detection_and_generate_dashboard_data.sh` 실행
- [ ] `cd output/final && python3 -m http.server 8080` 실행
- [ ] http://localhost:3000 접속 가능
- [ ] Infinity 플러그인 설치됨

대시보드 생성:
- [ ] 데이터 소스 "Detection CSV Data" 추가
- [ ] 패널 1개 이상 생성
- [ ] 대시보드 저장 완료

---

## 📞 도움말

- 🎓 처음이라면: [GRAFANA_BEGINNER_GUIDE.md](GRAFANA_BEGINNER_GUIDE.md)
- ⚡ 빠르게 시작: [QUICK_START_GRAFANA.md](QUICK_START_GRAFANA.md)
- 📖 상세 설명: [GRAFANA_SETUP_GUIDE.md](GRAFANA_SETUP_GUIDE.md)
- 📁 전체 구조: [README.md](README.md)

---

**인쇄해서 책상에 붙여두기!** 📌
