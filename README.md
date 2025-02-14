Crypto Trading Bot
📈 Binance API와 OpenAI 기반 AI 트레이딩 봇
🔍 백테스팅 및 자동 매매 기능 포함

📌 프로젝트 개요
이 프로젝트는 Binance API를 활용하여 실시간 암호화폐 거래(트레이딩) 자동화를 목표로 합니다.
트레이딩 봇은 **AI 모델(OpenAI API)**을 활용해 매매 신호를 생성하며, **기술적 지표(RSI, 볼린저 밴드, MACD 등)**를 기반으로 매매 결정을 내립니다.

✅ 주요 기능

📊 실시간 시장 데이터 수집 (Binance API)
🤖 AI 기반 매매 신호 생성 (OpenAI API)
📉 기술적 분석 지표 추가 (RSI, 볼린저 밴드, 이동평균선 등)
🏦 자동 매매 기능 (BUY/SELL 시뮬레이션)
🧪 백테스팅 기능 포함 (과거 데이터를 활용한 전략 검증)
📺 Streamlit 대시보드 제공 (실시간 트레이딩 모니터링)
🛠 리스크 관리 및 동적 손절/익절 설정
🛠 사용 기술

- Python 3.8+
- Binance API (실시간 가격 데이터 및 주문 실행)
- OpenAI API (AI 기반 트레이딩 신호 분석)
- Streamlit (웹 대시보드 제공)
- Pandas, NumPy (데이터 처리 및 분석)
- Dotenv (API 키 보안 관리)

📊 트레이딩 전략 (AI 기반)
AI는 다음과 같은 기술적 분석 지표를 활용하여 매매 신호를 생성합니다.
🔹 RSI (Relative Strength Index) – 과매수/과매도 영역 분석
🔹 볼린저 밴드 – 가격 변동성 분석 및 돌파 신호 감지
🔹 이동평균선 (MA) – 골든크로스/데드크로스 활용
🔹 MACD – 상승/하락 모멘텀 분석
🔹 ATR – 시장 변동성 및 리스크 조정

🚀 백테스팅 및 성과 분석
트레이딩 전략을 검증하기 위해 백테스팅 기능을 제공합니다.
🔹 초기 자본: 1000 USDT
🔹 거래 기록 저장 (trade_log1.csv)
🔹 손익 분석 (최대 손실, 최대 수익, 승률 등 포함)
🔹 동적 손절/익절 설정

실행 결과 예시:
**Final Balance: 950.32 USDT  
Total Profit: -49.68 USDT  
Win Rate: 42.5%  
Max Drawdown: 85.30 USDT**
⚠️ 현재 손실이 발생하는 전략으로 개선이 필요합니다.
