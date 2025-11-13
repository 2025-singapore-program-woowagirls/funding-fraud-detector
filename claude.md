# 이상 거래 패턴 탐지 프로젝트 (Abusive Trading Pattern Detection)

## 프로젝트 개요

### 프로젝트명

암호화폐 무기한 선물 거래소의 이상 거래 패턴 탐지 및 알고리즘 설계

### 목적

* 실제 거래 데이터를 기반으로 정상 거래와 이상 거래(True/False)를 구분하는 탐지 모델 구축
* 거래소 리스크 관리 및 시장 건전성 강화를 위한 데이터 기반 의사결정 지원
* 펀딩비(Funding Fee), 보너스(Reward), 조직적 거래(Collusive Trading) 세 가지 핵심 이상 패턴 정의 및 모델링

### 접근 방향

본 프로젝트는 단순 규칙 기반 탐지에서 출발하되,

1. **데이터 피처(feature)별 통계적 분포 및 정상/이상 사례 분석**,
2. **각 상황별 도메인 지식(Funding/Reward/Trading Behavior)을 반영한 가중치 설계**,
3. **수식 기반 탐지 알고리즘 수립 및 성능 평가(Test/Validation)**
   의 3단계 과정을 거친다.

---

## 데이터 분석 설계

### 1️⃣ 데이터 탐색 및 피처 분석 (EDA)

각 데이터셋의 주요 피처를 시각화 및 통계적으로 검토하여 **정상 거래와 이상 거래의 차이점**을 정의한다.

#### 주요 단계

* 각 CSV(Trade, Funding, Reward, IP, Spec) 로드 및 결합
* 결측치, 이상치, 중복 제거 및 시간순 정렬
* 거래 단위(계정·포지션)별 통계량 계산
* 펀딩비, 보너스 수령, 거래빈도, 레버리지, 포지션 유지시간 등의 분포 확인
* **정규분포·신뢰구간·상위/하위 퍼센타일 기준**으로 이상치 기준 제시

#### 분석 포인트 예시

| Feature             | 정상 거래 분포 예시 | 이상 거래 시 특징   |
| ------------------- | ----------- | ------------ |
| `leverage`          | 평균 10~30배   | 50배 이상 고빈도   |
| `holding_time`      | 1~3시간       | 5분 미만 청산 반복  |
| `funding_fee_ratio` | ±0.001 내외   | 펀딩비 중심 수익 구조 |
| `reward_amount`     | 소액·1회성      | 다계정 반복 수령    |
| `ip_shared_count`   | 1           | 2개 이상 중복 계정  |

→ 통계적으로 확인 가능한 패턴 외에도, 펀딩비 제도/보너스 구조 등 **도메인 지식에 따라 가중치 직접 설정** 가능.

---

## 이상 거래 패턴 정의 및 알고리즘 설계

### ① Funding Fee Arbitrage (펀딩비 차익거래형)

**특징:** 펀딩 시각 전후 단기 포지션 집중, 수익 대부분이 Funding Fee에서 발생
**가중치 요소:** `funding_profit_ratio`, `holding_time`, `fee_rate_direction`, `PnL_ratio`

```
FundingScore = w1 * funding_profit_ratio + w2 * (1 / holding_time) + w3 * PnL_ratio
```

→ w1~w3은 도메인 지식 기반(펀딩 구조 이해)에 따라 초기 설정 후, 테스트 데이터로 조정.

---

### ② Organized Trading (조직적 거래형)

**특징:** 다계정이 동일 시간대·심볼·가격에서 거래
**가중치 요소:** `shared_ip_ratio`, `time_overlap`, `price_similarity`, `avg_leverage`

```
OrganizedScore = α1 * shared_ip_ratio + α2 * time_overlap + α3 * price_similarity + α4 * (avg_leverage / 100)
```

→ 동일 시각·동일 가격대 거래 빈도(네트워크 분석 기반)로 패턴 검출.

---

### ③ Bonus Abuse (보너스 악용형)

**특징:** 동일 IP에서 다수 계정이 유사 시각에 Reward 수령 후 비활성
**가중치 요소:** `duplicate_reward`, `inactive_ratio`, `reward_to_trade_ratio`

```
BonusScore = β1 * duplicate_reward + β2 * inactive_ratio + β3 * reward_to_trade_ratio
```

---

## 종합 위험 점수 (Composite Risk Score)

모든 패턴을 정규화(0~1) 후 가중 평균:

```
RiskScore = 0.4 * FundingScore + 0.35 * OrganizedScore + 0.25 * BonusScore
```

**참고:** 초기 가중치는 도메인 지식 기반으로 설정하고,
이후 실험(Test) 단계에서 정확도(Accuracy), 정밀도(Precision), 재현율(Recall)에 따라 조정.

---

## 모델 실험 및 성능 검증

### Step 1. 기준 설정

* 각 패턴별 이상(True)/정상(False) 레이블 예시 수집
* 단순 Rule-Base 기준으로 1차 라벨링 → 이후 개선 시 Ground Truth로 사용

### Step 2. 알고리즘 테스트

* 각 수식(FundingScore, OrganizedScore, BonusScore)의 결과값 비교
* Accuracy / Precision / Recall 계산
* 상관계수 및 ROC Curve 분석으로 임계값 결정

### Step 3. 알고리즘 조정

* **정규분포 기반 기준:** μ ± 2σ 바깥 구간 이상치로 간주
* **신뢰구간 기반 기준:** 95% CI 밖의 값 이상거래로 탐지
* **직관적 조정:** 도메인 지식 기반으로 수동 가중치 조정

### Step 4. 통합 모델

* 각 Score 결과를 Ensemble로 결합하여 최종 RiskScore 생성
* 임계값(Threshold)을 기준으로 High / Medium / Low 등급 분류

---

## 예시 실험 계획 (펀딩비 케이스)

| 실험 항목             | 방법                               | 기대결과                  |
| ----------------- | -------------------------------- | --------------------- |
| 펀딩 시각 ±30분 거래 집중도 | 시계열 밀도그래프                        | 이상거래는 펀딩 직전 피크        |
| 포지션 유지시간 분포       | KDE Plot                         | 이상거래는 짧은 Tail         |
| Funding Fee 수익 비중 | Boxplot                          | 정상: 0~10%, 이상: 70% 이상 |
| True/False 모델 검증  | Logistic Regression or Threshold | 탐지 정확도 ≥ 85% 목표       |

---

## 수행 일정 (개선 후)

| 단계      | 주요 목표                | 기간   |
| ------- | -------------------- | ---- |
| Phase 1 | 피처 분석 및 정상/이상 데이터 탐색 | 1~2일 |
| Phase 2 | 패턴별 수식 정의 및 가중치 설계   | 2~3일 |
| Phase 3 | 알고리즘 테스트 및 성능 검증     | 2~3일 |
| Phase 4 | 결과 분석 및 보고서 작성       | 1~2일 |

---

## 기대 효과 및 향후 확장

* 단순 탐지에서 벗어나 **정량적 기준 + 도메인 기반 해석형 탐지 시스템** 구축
* 이상 거래 발생 원인(동기) 해석 가능
* 추후 머신러닝 기반 자동 탐지로 확장 (Isolation Forest, One-Class SVM 등)

---

*이 문서는 프로젝트 진행 상황에 따라 지속적으로 업데이트됩니다.*
