# 📈 Crypto Trading Bot  
**Binance API와 OpenAI 기반 AI 트레이딩 봇**  
🔍 **백테스팅 및 자동 매매 기능 포함**  

## 📌 프로젝트 개요  
이 프로젝트는 **Binance API**를 활용하여 실시간 **암호화폐 거래(트레이딩) 자동화**를 목표로 합니다.  
트레이딩 봇은 **AI 모델(OpenAI API)**을 활용해 매매 신호를 생성하며,  
**기술적 지표(RSI, 볼린저 밴드, MACD 등)**을 기반으로 매매 결정을 내립니다.  

---

## ✅ 주요 기능  

📊 **실시간 시장 데이터 수집** (Binance API)  
🤖 **AI 기반 매매 신호 생성** (OpenAI API)  
📉 **기술적 분석 지표 추가** (RSI, 볼린저 밴드, 이동평균선 등)  
🏦 **자동 매매 기능** (BUY/SELL 시뮬레이션)  
🧪 **백테스팅 기능 포함** (과거 데이터를 활용한 전략 검증)  
📺 **Streamlit 대시보드 제공** (실시간 트레이딩 모니터링)  
🛠 **리스크 관리 및 동적 손절/익절 설정**  

---

## 🛠 사용 기술  

- **Python 3.8+**  
- **Binance API** (실시간 가격 데이터 및 주문 실행)  
- **OpenAI API** (AI 기반 트레이딩 신호 분석)  
- **Streamlit** (웹 대시보드 제공)  
- **Pandas, NumPy** (데이터 처리 및 분석)  
- **Dotenv** (API 키 보안 관리)  

---

## 📊 트레이딩 전략 (AI 기반)  

AI는 다음과 같은 **기술적 분석 지표**를 활용하여 매매 신호를 생성합니다.  
🔹 **RSI** (Relative Strength Index) – 과매수/과매도 영역 분석  
🔹 **볼린저 밴드** – 가격 변동성 분석 및 돌파 신호 감지  
🔹 **이동평균선 (MA)** – 골든크로스/데드크로스 활용  
🔹 **MACD** – 상승/하락 모멘텀 분석  
🔹 **ATR** – 시장 변동성 및 리스크 조정  

---

## 🚀 백테스팅 및 성과 분석  

트레이딩 전략을 검증하기 위해 **백테스팅 기능**을 제공합니다.  

🔹 **초기 자본**: 1000 USDT  
🔹 **거래 기록 저장** (`trade_log1.csv`)  
🔹 **손익 분석** (최대 손실, 최대 수익, 승률 등 포함)  
🔹 **동적 손절/익절 설정**  

**실행 결과 예시:**  
**Final Balance: 950.32 USDT  
Total Profit: -49.68 USDT  
Win Rate: 42.5%  
Max Drawdown: 85.30 USDT**
⚠️ 현재 손실이 발생하는 전략으로 **개선이 필요합니다.**  

---

## 📉 실패 원인 분석  

이 트레이딩 봇은 OpenAI API를 활용한 매매 신호와 기술적 지표를 결합하여 자동 트레이딩을 수행했습니다.  
그러나 백테스팅 결과 **손실이 발생**했으며, 주요 원인은 다음과 같습니다.  

### ❌ 실패 원인  
1️⃣ **AI 신호의 신뢰도 문제**  
   - AI가 생성한 신호가 시장 상황을 제대로 반영하지 못하는 경우가 많았음  
   - 일부 신호는 과거 데이터에 과적합(Overfitting)된 것으로 보임  

2️⃣ **리스크 관리 부족**  
   - 손절/익절 전략이 일괄적이어서 시장 변동성을 충분히 반영하지 못함  
   - 특히 급등락이 심한 구간에서 큰 손실 발생  

3️⃣ **거래 비용(슬리피지, 수수료) 고려 부족**  
   - Binance의 수수료 및 슬리피지(체결 가격 차이)가 누적되면서 손익에 영향  

4️⃣ **데이터 불균형**  
   - 학습 및 테스트 데이터가 일부 특정 시장 상황에 치우쳐 있음  
   - 횡보장(박스권)에서는 수익을 내지 못하고, 트렌드가 강한 시장에서만 수익 발생  

---

## 🔄 개선 방향 (업데이트 예정)  

현재 결과를 바탕으로 **다음과 같은 개선을 계획**하고 있습니다.  

✅ **AI 신호 개선**  
   - 단순 OpenAI API 호출이 아니라, 추가적인 머신러닝 모델을 활용한 신호 필터링 적용  
   - 온체인 데이터(예: Nansen, Dune Analytics) 활용  

✅ **리스크 관리 최적화**  
   - ATR 기반 동적 손절/익절 설정 (시장 변동성에 따라 자동 조절)  
   - 거래량 기반 트레이딩 전략 적용 (거래량 증가 시 트렌드 신뢰도 상승)  

✅ **거래 비용 최적화**  
   - 슬리피지를 고려한 주문 전략 도입 (Limit Order 활용)  
   - 거래 수수료 최적화 (Binance BNB 할인 적용)  

---

## 🏗️ 프로젝트 기여도  
- **데이터 수집**: Binance API를 활용한 실시간 가격 데이터 저장 및 전처리  
- **AI 트레이딩 신호 연구**: OpenAI API를 활용한 매매 신호 생성 및 백테스팅  
- **기술적 지표 적용**: RSI, MACD, 볼린저 밴드 등 다양한 전략 테스트  
- **리스크 관리 전략 구현**: 손절/익절 규칙 및 포트폴리오 관리  
- **백테스팅 결과 분석**: 매매 기록을 CSV로 저장하고 성과 분석  
