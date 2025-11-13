# 암호화폐 이상 거래 패턴 탐지 프로젝트

## 프로젝트 개요

무기한 선물 거래소의 거래 데이터를 분석하여 3가지 이상 거래 패턴을 탐지하는 시스템입니다.

### 탐지 대상 패턴

1. **펀딩피 차익거래 (Funding Fee Arbitrage)** - 펀딩 시각 전후 단기 포지션 집중
2. **조직적 거래 (Organized Trading)** - 다계정 동시 거래
3. **보너스 악용 (Bonus Abuse)** - 다계정 보너스 반복 수령

---

## 프로젝트 구조

```
BE/
├── data/                          # 원본 데이터
│   ├── Funding.csv               # 펀딩피 데이터
│   ├── Trade.csv                 # 거래 데이터
│   ├── Reward.csv                # 보너스 데이터
│   ├── IP.csv                    # IP 데이터
│   └── Spec.csv                  # 거래 사양 데이터
│
├── core/                          # 분석 코드
│   ├── 1_funding_arbitrage/      # 펀딩피 차익거래 탐지
│   ├── 2_organized_trading/      # 조직적 거래 탐지
│   ├── 3_bonus_abuse/            # 보너스 악용 탐지
│   └── 4_integrated/             # 통합 시스템
│
├── output/                        # 분석 결과
│   ├── 1_funding_arbitrage/      # 펀딩피 분석 결과
│   ├── 2_organized_trading/      # 조직적 거래 분석 결과
│   ├── 3_bonus_abuse/            # 보너스 악용 분석 결과
│   └── 4_integrated/             # 통합 분석 결과
│
├── CLAUDE.md                      # 프로젝트 상세 문서
└── README.md                      # 이 파일
```

각 폴더의 상세 내용은 해당 폴더의 README.md를 참조하세요.

---

## 환경 설정

### 1. 가상환경 생성 및 활성화

```bash
cd <프로젝트_루트>
python3 -m venv venv
source venv/bin/activate
```

### 2. 패키지 설치

```bash
python -m pip install -U pip
pip install -r requirements-eda.txt
```

필요한 패키지:
- pandas, numpy
- matplotlib, seaborn
- scipy

---

## 실행 방법

### 1. 펀딩피 차익거래 분석

```bash
cd core/1_funding_arbitrage
python analyze_funding_arbitrage.py
```

### 2. 조직적 거래 분석

```bash
cd core/2_organized_trading
python analyze_organized_trading.py
```

### 3. 보너스 악용 분석

```bash
cd core/3_bonus_abuse
python analyze_bonus_abuse.py
```

### 4. 통합 분석 (전체)

```bash
cd core/4_integrated
python final_integrated_system.py
```

### 5. 상세 시각화 생성

```bash
cd core/1_funding_arbitrage
python visualize_features_detail.py
```

---

## 주요 출력물

### 통합 분석 결과 (`output/4_integrated/`)

- **FEATURES_DEFINITION.md** - 전체 피처 정의 및 해석 가이드
- **final_integrated_risk_scores.csv** - 모든 계정의 최종 리스크 점수
- **critical_high_risk_accounts.csv** - 최고위험 계정 목록
- **multi_pattern_accounts.csv** - 복수 패턴 탐지 계정
- **pattern{1,2,3}_detailed_distribution.png** - 패턴별 상세 분포 시각화

### 개별 패턴 결과

각 패턴별 폴더(`output/1_funding_arbitrage/`, `output/2_organized_trading/`, `output/3_bonus_abuse/`)에 다음이 저장됩니다:
- 리스크 점수 CSV
- 고위험 계정 목록
- 분석 시각화 PNG
- 통계 요약

---

## 최종 리스크 스코어 공식

```
FinalRiskScore = 0.30 * FundingScore
               + 0.35 * OrganizedScore
               + 0.20 * BonusScore
               + 0.15 * QuantScore
```

- **High Risk**: Score ≥ 0.4
- **Medium Risk**: 0.2 ≤ Score < 0.4
- **Low Risk**: Score < 0.2

---

## 참고 문서

- [CLAUDE.md](CLAUDE.md) - 프로젝트 전체 설계 및 방법론
- [core/1_funding_arbitrage/README.md](core/1_funding_arbitrage/README.md) - 펀딩피 탐지 상세
- [core/2_organized_trading/README.md](core/2_organized_trading/README.md) - 조직적 거래 탐지 상세
- [core/3_bonus_abuse/README.md](core/3_bonus_abuse/README.md) - 보너스 악용 탐지 상세
- [core/4_integrated/README.md](core/4_integrated/README.md) - 통합 시스템 상세
- [output/4_integrated/FEATURES_DEFINITION.md](output/4_integrated/FEATURES_DEFINITION.md) - 피처 정의

---

## 기술 스택

- Python 3.9+
- pandas, numpy - 데이터 처리
- matplotlib, seaborn - 시각화
- scipy - 통계 분석

---

## 문의 및 기여

이슈나 개선사항은 GitHub Issues를 통해 제출해주세요.
