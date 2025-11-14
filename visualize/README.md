# 📊 Visualize - 데이터 시각화 디렉토리

이 디렉토리는 탐지 결과를 Grafana 대시보드로 시각화하는 모든 파일을 포함합니다.

---

## 📁 디렉토리 구조

```
visualize/
├── README.md                           # 이 파일
├── GRAFANA_BEGINNER_GUIDE.md          # 🎓 초보자용 완벽 가이드 (추천!)
├── QUICK_START_GRAFANA.md             # ⚡ 5분 빠른 시작
├── GRAFANA_SETUP_GUIDE.md             # 📖 상세 설정 가이드
├── create_grafana_dashboard_data.py   # Grafana용 CSV 생성 스크립트
└── convert_csv_to_sqlite.py           # SQLite DB 변환 스크립트
```

---

## 🚀 빠른 시작

### 1. 데이터 생성

프로젝트 루트에서:

```bash
./run_detection_and_generate_dashboard_data.sh
```

이 스크립트가 자동으로:
- ✅ 탐지 실행
- ✅ CSV 파일 7개 생성 (`output/final/grafana_*.csv`)
- ✅ SQLite DB 생성 (`output/final/detection.db`)

### 2. CSV 서빙

```bash
cd output/final
python3 -m http.server 8080
```

### 3. Grafana 대시보드 만들기

📖 **[GRAFANA_BEGINNER_GUIDE.md](GRAFANA_BEGINNER_GUIDE.md) 참조**

---

## 📄 가이드 문서 선택

### 🎓 처음 사용자 → [GRAFANA_BEGINNER_GUIDE.md](GRAFANA_BEGINNER_GUIDE.md)
- Grafana를 한 번도 안 써봤다면 이거!
- 설치부터 대시보드 완성까지 30분 완벽 가이드
- UI 설명, 클릭할 버튼, 입력할 값 모두 포함
- **추천!** ⭐⭐⭐⭐⭐

### ⚡ 경험자 → [QUICK_START_GRAFANA.md](QUICK_START_GRAFANA.md)
- Grafana를 써본 적 있다면 이거!
- 5분 빠른 시작
- 핵심만 간단히

### 📖 고급 사용자 → [GRAFANA_SETUP_GUIDE.md](GRAFANA_SETUP_GUIDE.md)
- SQLite 연동, 고급 쿼리, 알림 설정 등
- 모든 옵션 상세 설명

---

## 🔧 스크립트 사용법

### create_grafana_dashboard_data.py

Grafana 대시보드용 CSV 파일 7개 생성:

```bash
cd /Users/gimhyejin/Library/CloudStorage/OneDrive-한성대학교/문서/Projects/singapore-prestolabs/BE
source .venv/bin/activate
python visualize/create_grafana_dashboard_data.py
```

**생성 파일**:
- `grafana_summary_stats.csv` - 요약 통계
- `grafana_timeseries.csv` - 시계열 데이터 (7일)
- `grafana_accounts_table.csv` - 계정 테이블
- `grafana_pattern_stats.csv` - 패턴별 통계
- `grafana_alerts_priority.csv` - 알림 우선순위
- `grafana_review_status.csv` - 검토 상태
- `grafana_detection_alerts.csv` - 알림 상세

### convert_csv_to_sqlite.py

CSV → SQLite 데이터베이스 변환:

```bash
python visualize/convert_csv_to_sqlite.py
```

**생성 파일**:
- `output/final/detection.db`

**포함 내용**:
- 9개 테이블 (CSV 파일들)
- 7개 인덱스 (성능 최적화)
- 3개 뷰 (자주 쓰는 쿼리)

---

## 📊 생성된 대시보드 데이터

### CSV 파일 상세

| 파일명 | 행 수 | 용도 | Grafana 패널 타입 |
|--------|-------|------|-------------------|
| `grafana_summary_stats.csv` | 1 | 전체 요약 | Stat, Gauge |
| `grafana_timeseries.csv` | 8 | 시계열 (7일) | Time series |
| `grafana_accounts_table.csv` | 63 | 계정 상세 | Table |
| `grafana_pattern_stats.csv` | 9 | 패턴별 통계 | Bar chart |
| `grafana_alerts_priority.csv` | 21 | 고위험 알림 | Table |
| `grafana_review_status.csv` | 1 | 검토 상태 | Gauge |
| `grafana_detection_alerts.csv` | 64 | 알림 상세 | Logs, Table |

### 데이터 구조 예시

**grafana_summary_stats.csv**:
```csv
timestamp,total_accounts,critical_count,high_count,suspicious_count,normal_count
2025-10-31 20:00:38,63,6,15,6,36
```

**grafana_accounts_table.csv**:
```csv
account_id,final_risk,pattern1_risk,pattern2_risk,pattern3_risk,total_profit_usd,reviewed
A_1f97e16953,CRITICAL,HIGH,HIGH,NO_REWARD,66743.18,False
...
```

---

## 🔄 데이터 업데이트 주기

### 수동 업데이트
```bash
./run_detection_and_generate_dashboard_data.sh
```

### 자동 업데이트 (Cron)

```bash
# crontab 편집
crontab -e

# 매시간 실행
0 * * * * cd /path/to/BE && ./run_detection_and_generate_dashboard_data.sh
```

---

## 🎯 대시보드 구성 추천

### 필수 패널 (5개)

1. **요약 통계** (Stat)
   - 총 계정 수
   - CRITICAL, HIGH, SUSPICIOUS, NORMAL 각각

2. **리스크 분포** (Pie Chart)
   - 리스크 레벨별 비율

3. **시계열 추세** (Time Series)
   - 7일간 리스크 변화

4. **고위험 계정 테이블** (Table)
   - 조치 필요 계정 목록

5. **검토 진행률** (Gauge)
   - 확인/미확인 비율

### 추가 패널 (선택)

6. **패턴별 통계** (Bar Chart)
7. **알림 로그** (Logs)
8. **수익 분포** (Histogram)

---

## 🐛 문제 해결

### "No data" 에러

**원인**: CSV 파일을 못 읽음

**해결**:
```bash
# HTTP 서버 실행 확인
curl http://localhost:8080/grafana_summary_stats.csv

# 안 되면 재시작
cd output/final
python3 -m http.server 8080
```

### Infinity 플러그인 없음

```bash
grafana-cli plugins install yesoreyeram-infinity-datasource
brew services restart grafana
```

### 시간축이 이상함

Transform → Convert field type → timestamp를 Time으로

---

## 📚 참고 자료

- [Grafana 공식 문서](https://grafana.com/docs/)
- [Infinity 플러그인](https://grafana.com/grafana/plugins/yesoreyeram-infinity-datasource/)
- [CSV 데이터 소스 가이드](https://grafana.com/docs/grafana/latest/datasources/csv/)

---

## ✅ 체크리스트

설정:
- [ ] Grafana 설치
- [ ] Infinity 플러그인 설치
- [ ] CSV 서버 실행
- [ ] 데이터 소스 추가

대시보드:
- [ ] 요약 통계 패널
- [ ] 리스크 분포 패널
- [ ] 시계열 패널
- [ ] 테이블 패널
- [ ] 검토 진행률 패널
- [ ] 대시보드 저장

---

**문의**: 가이드 문서 참조 또는 이슈 생성

**작성일**: 2025-11-14
