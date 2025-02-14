import pandas as pd
from binance.client import Client
import time
import numpy as np
from dotenv import load_dotenv
import os
from openai import OpenAI
import streamlit as st

# Streamlit 환경 설정
st.title("Crypto Trading Bot Dashboard")
st.sidebar.header("Bot Settings")

# 전역 변수 초기화
current_position = 0  # 보유 중인 BTC 수량
entry_price = 0  # 매수 진입 가격
current_balance = 0  # 테스트 모드 잔고

# 환경 선택
environment = st.sidebar.selectbox("Choose Environment", ["test", "live"]).lower()

# 환경 변수 파일 로드
if environment == "test":
    load_dotenv("TestKey.env")  # 테스트용 키 파일 로드
    initial_test_balance = 1000  # 테스트 환경 초기 잔고 (1000 USDT)
    current_balance = initial_test_balance
elif environment == "live":
    load_dotenv("Key_File.env")  # 실제 환경 키 파일 로드
else:
    st.error("Invalid environment. Please choose 'test' or 'live'.")
    st.stop()

# API 키 가져오기
openai_api_key = os.getenv("OPENAI_API_KEY")
binance_api_key = os.getenv("BINANCE_API_KEY")
binance_api_secret = os.getenv("BINANCE_API_SECRET")

if not openai_api_key or not binance_api_key or not binance_api_secret:
    st.error("Missing API keys. Please check your environment files.")
    st.stop()

# OpenAI 및 Binance 클라이언트 초기화
client = OpenAI(api_key=openai_api_key)
binance_client = Client(binance_api_key, binance_api_secret, testnet=(environment == "test"))

# 매매 기록 초기화
trade_log = []
last_signal = None
last_trade_time = 0  # 마지막 거래 시간
cooldown_period = 30  # 최소 거래 간격 (초)

# 리스크 관리 변수 설정
MAX_LOSS = 0.02  # 최대 손실 2%
MAX_PROFIT = 0.05  # 최대 수익 5%
DEFAULT_RISK_RATIO = 0.02  # 기본 리스크 비율 2%


# Streamlit 대시보드 설정
SYMBOL = st.sidebar.text_input("Trading Pair", value="BTCUSDT")
INTERVAL = st.sidebar.selectbox("Interval", ["1m", "5m", "15m", "30m", "1h"], index=0)
log_display = st.empty()  # 거래 로그 디스플레이 초기화

# 거래 기록 표시
def display_trade_log():
    """거래 로그를 테이블로 표시"""
    if len(trade_log) > 0:
        df_trade_log = pd.DataFrame(trade_log)  # trade_log를 데이터프레임으로 변환
        log_display.table(df_trade_log)  # 테이블 업데이트
    else:
        log_display.write("No trades have been executed yet.")  # 초기 메시지


def backtest(data, strategy):
    """
    Backtesting logic to evaluate strategy performance.
    """
    trade_log = []  # 거래 기록
    balance = 1000  # 초기 잔고
    position = 0  # 초기 포지션
    entry_price = 0  # 초기 매수가격
    initial_balance = balance  # 초기 잔고 저장
    max_balance = balance  # 최대 잔고 추적
    max_drawdown = 0  # 최대 드로우다운

    for i in range(len(data)):
        current_price = data["close"].iloc[i]
        signal, confidence = strategy(data.iloc[:i+1])

        # 실패 신호 학습: confidence가 낮은 경우 실패로 기록
        if signal == "SELL" and confidence < 50:
            log_failure(data.iloc[:i+1], "Low confidence for SELL")
            continue

        # BUY 로직
        if signal == "BUY" and position == 0 and confidence >= 80:
            quantity = balance * 0.8 / current_price  # 잔고의 80%로 매수
            position += quantity
            entry_price = current_price
            balance -= quantity * current_price
            trade_log.append({"Action": "BUY", "Price": current_price, "Balance": balance, "Position": position})

        # SELL 로직
        elif signal == "SELL" and position > 0:
            quantity = position  # 전체 포지션 매도
            balance += quantity * current_price
            profit = (current_price - entry_price) * quantity
            trade_log.append({"Action": "SELL", "Price": current_price, "Balance": balance, "Profit": profit, "Position": 0})
            position = 0  # 포지션 초기화
            entry_price = 0  # 매수가 초기화

        # 최대 잔고 업데이트
        max_balance = max(max_balance, balance + (position * current_price))

        # 드로우다운 계산
        drawdown = max_balance - (balance + (position * current_price))
        max_drawdown = max(max_drawdown, drawdown)

    # 최종 잔고 계산
    final_balance = balance + (position * data["close"].iloc[-1])
    trade_log.append({"Action": "FINAL", "Balance": final_balance, "Position": position})

    # 통계 계산
    total_profit = final_balance - initial_balance
    win_trades = [trade for trade in trade_log if trade.get("Profit", 0) > 0]
    loss_trades = [trade for trade in trade_log if trade.get("Profit", 0) <= 0]
    win_rate = len(win_trades) / len([t for t in trade_log if "Profit" in t]) * 100 if len(trade_log) > 0 else 0
    avg_win = np.mean([t["Profit"] for t in win_trades]) if win_trades else 0
    avg_loss = np.mean([t["Profit"] for t in loss_trades]) if loss_trades else 0
    risk_reward_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else "N/A"

    # 결과 출력
    st.write(f"Final Balance: {final_balance:.2f} USDT")
    st.write(f"Total Profit: {total_profit:.2f} USDT")
    st.write(f"Win Rate: {win_rate:.2f}%")
    st.write(f"Risk-Reward Ratio: {risk_reward_ratio}")
    st.write(f"Max Drawdown: {max_drawdown:.2f} USDT")

    # 거래 로그 반환
    trade_log_df = pd.DataFrame(trade_log)
    return trade_log_df

def initial_market_analysis(data):
    """AI가 초기 매수 타이밍을 설정"""
    signal = generate_signal(data, current_position)

    # AI 신호가 "BUY"일 때 초기 매수 진행
    if signal == "BUY":
        price = simulate_order(SYMBOL, "BUY", 0.001)
        if price:
            st.write(f"Initial BUY executed at {price}")
            trade_log.append({"Signal": "BUY", "Price": price, "Time": time.strftime("%H:%M:%S")})
        return True
    else:
        st.write("AI recommends HOLD. Waiting for better conditions.")
        return False

MIN_PRICE_CHANGE = 0.005  # 최소 가격 변화율 (0.1%)



def evaluate_and_decide(data, current_price):
    global entry_price, current_position, last_signal, current_balance

    # Generate a signal using AI or technical indicators
    signal, confidence = generate_signal(data, current_position, last_signal)
    st.write(f"Generated Signal: {signal}, Confidence: {confidence}%")

    # Check cooldown before trading
    if not check_cooldown_and_trade(signal, data, trade_log):
        st.write("Skipping trade due to active cooldown.")
        return  # 쿨다운이 아직 진행 중이라면 거래 중단

    # Check risk management rules if a position exists
    if current_position > 0 and entry_price > 0:
        take_profit, stop_loss = dynamic_risk_management(data)
        if take_profit is None or stop_loss is None:
            st.write("Risk management skipped due to insufficient data.")
        else:
            risk_signal = check_risk_management(entry_price, current_price, take_profit, stop_loss)
            if risk_signal == "SELL":
                st.write("Risk management triggered SELL signal.")
                signal = "SELL"

    # Process trading signals
    if confidence >= 80:
        if signal == "BUY":
            st.write(f"Signal is BUY with high confidence ({confidence}%). Proceeding to buy.")
            process_buy(current_price, confidence)
        elif signal == "SELL":
            if current_position > 0:
                st.write(f"Signal is SELL with high confidence ({confidence}%). Proceeding to sell.")
                process_sell(current_price, confidence)
            else:
                st.write("SELL signal received but no position to sell.")
        else:
            st.write("Signal is HOLD. No trade executed.")
    else:
        st.write(f"Confidence too low ({confidence}%). Skipping trade.")


def process_buy(current_price, confidence):
    """
    Execute a buy order and update the portfolio based on confidence level.
    """
    global current_balance, current_position, entry_price

    # Calculate dynamic quantity based on confidence
    buy_percentage = confidence / 100  # Confidence determines percentage to use
    if buy_percentage > 0.5:  # Limit max buy percentage to 50% of balance
        buy_percentage = 0.5

    quantity = current_balance * buy_percentage / current_price
    cost = current_price * quantity
    total_fee = cost * FEE_RATE

    # 최소 거래 금액 확인 (예: 10 USDT 이상)
    if cost < 10:
        st.write("Order cost is below minimum trade amount. Skipping buy.")
        return

    if cost + total_fee > current_balance:
        st.write("Insufficient balance. Skipping buy.")
        return

    # Execute buy order
    price = simulate_order(SYMBOL, "BUY", quantity)
    if price:
        # Update average entry price
        if current_position > 0:
            entry_price = (entry_price * current_position + cost) / (current_position + quantity)
        else:
            entry_price = price

        # Update position and balance
        current_position += quantity
        current_balance -= (cost + total_fee)

        st.write(f"BUY executed at {price} for {quantity:.4f} BTC")
        log_trade("BUY", price, quantity)


def process_sell(current_price, confidence):
    global current_balance, current_position, entry_price

    # Calculate dynamic sell quantity based on confidence
    sell_percentage = confidence / 100  # Confidence determines percentage to sell
    if sell_percentage > 0.5:  # Limit max sell percentage to 50% of position
        sell_percentage = 0.5

    quantity = current_position * sell_percentage

    # 최소 거래 금액 확인
    if quantity * current_price < 10:
        st.write("Order value is below minimum trade amount. Skipping sell.")
        return

    # Simulate order and calculate profit
    price = simulate_order(SYMBOL, "SELL", quantity)
    if price:
        # 매수 비용 및 매도 수익 계산
        cost_basis = entry_price * quantity + (entry_price * quantity * FEE_RATE)  # 매수 비용 + 매수 수수료
        revenue = price * quantity - (price * quantity * FEE_RATE)  # 매도 수익 - 매도 수수료
        profit = revenue - cost_basis  # 순수익 계산

        # Update portfolio
        current_balance += revenue
        current_position -= quantity

        # Reset entry price if all positions are sold
        if current_position == 0:
            entry_price = 0

        st.write(f"SELL executed at {price}. Profit: {profit:.2f} USDT")
        log_trade("SELL", price, quantity, profit)


def reset_position():
    """
    Reset the current position and entry price after a sell.
    """
    global current_position, entry_price
    current_position = 0
    entry_price = 0

def log_trade(action, price, quantity, profit=None):
    """
    Log trade details to the trade log.
    """
    trade_log.append({
        "Signal": action,
        "Price": price,
        "Quantity": quantity,
        "Profit": profit if profit is not None else 0.0,  # 기본값을 0.0으로 설정
        "Time": time.strftime("%H:%M:%S")
    })
    display_trade_log()

def dynamic_risk_management(data, risk_ratio=DEFAULT_RISK_RATIO):
    """
    동적으로 익절 및 손절 기준을 설정.
    """
    if entry_price == 0:
        return None, None  # 포지션이 없는 경우 기준 없음

    try:
        atr = data["ATR"].iloc[-1]
        if pd.isna(atr):
            raise ValueError("ATR calculation failed or insufficient data.")
    except Exception as e:
        st.write(f"ATR calculation failed or insufficient data: {e}")
        return None, None

    # ATR 기반으로 동적 손절 및 익절 계산
    adjusted_risk_ratio = min(max(atr * 100, 1), 5)
    take_profit = entry_price * (1 + adjusted_risk_ratio / 100)
    stop_loss = entry_price * (1 - risk_ratio)
    return take_profit, stop_loss

def check_risk_management(entry_price, current_price, take_profit, stop_loss):
    """
    익절/손절 기준 확인.
    """
    if current_price >= take_profit:
        return "SELL"  # 익절 기준 도달
    elif current_price <= stop_loss:
        return "SELL"  # 손절 기준 도달
    return None

def calculate_rsi(prices, period=14):
    """RSI 계산"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)  # ZeroDivisionError 방지
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(prices, window=20):
    """볼린저 밴드 계산"""
    ma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper_band = ma + (std * 2)
    lower_band = ma - (std * 2)
    return upper_band, lower_band

# 데이터 가져오기
def fetch_data(symbol="BTCUSDT", interval="1m", limit=50):
    """Binance에서 데이터를 가져옵니다."""
    klines = binance_client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        "time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    df["close"] = df["close"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["time"] = pd.to_datetime(df["time"], unit='ms')
    df = df.loc[:, ["time", "high", "low", "close"]]
    return df


def calculate_atr(data, period=14):
    """ATR (Average True Range) 계산"""
    data["H-L"] = data["high"] - data["low"]
    data["H-C"] = abs(data["high"] - data["close"].shift(1))
    data["L-C"] = abs(data["low"] - data["close"].shift(1))
    data["TR"] = data[["H-L", "H-C", "L-C"]].max(axis=1)
    data["ATR"] = data["TR"].rolling(window=period).mean()
    return data.drop(columns=["H-L", "H-C", "L-C", "TR"])

def add_technical_indicators(data):
    """
    Add technical indicators such as moving averages, RSI, Bollinger Bands, and ATR.
    """
    data["MA_10"] = data["close"].rolling(window=10).mean()
    data["MA_50"] = data["close"].rolling(window=50).mean()
    data["RSI"] = calculate_rsi(data["close"])
    data["Bollinger_Upper"], data["Bollinger_Lower"] = calculate_bollinger_bands(data["close"])
    data = calculate_atr(data)  # ATR 계산 추가
    data.dropna(inplace=True)  # NaN 값 제거
    return data


# 쿨다운 및 거래 로직
last_trade_time = 0  # 마지막 거래 시간
cooldown_period = 30  # 최소 거래 간격 (초)

# 쿨다운 시간 계산
def calculate_dynamic_cooldown(data, signal):
    """시장 변동성과 신호 품질을 바탕으로 유동적인 쿨다운 시간 계산"""
    if len(data) < 10:  # 최소 데이터 길이 확인
        return 30  # 기본 쿨다운 시간 반환

    rsi = data["RSI"].iloc[-1]
    price_change = (data["close"].iloc[-1] - data["close"].iloc[-5]) / data["close"].iloc[-5]
    base_cooldown = 30

    if rsi > 70 or rsi < 30:
        cooldown_adjustment = -10
    else:
        cooldown_adjustment = 10

    if abs(price_change) > 0.01:
        cooldown_adjustment -= 5

    if signal in ["BUY", "SELL"]:
        cooldown_adjustment -= 5

    return max(base_cooldown + cooldown_adjustment, 5)

def log_failure(data, reason):
    """
    Log failed trade conditions for analysis and debugging.
    """
    failed_trade = {
        "Time": time.strftime("%H:%M:%S"),
        "Close": data["close"].iloc[-1],
        "Reason": reason
    }
    st.write(f"Failed Trade Logged: {failed_trade}")
    # You can save these logs to a file or database for further analysis

def adaptive_cooldown(data, last_trades):
    """
    Adjust cooldown period based on recent losses.
    """
    # 최근 거래 기록에서 손실 건수 계산
    recent_losses = sum(
        1 for trade in last_trades[-10:] if trade.get("Profit") is not None and trade["Profit"] < 0
    )
    st.write(f"Recent losses: {recent_losses}")  # 디버깅용 로그
    return max(10, 30 + (recent_losses * 5))  # 손실 비율에 따라 쿨다운 증가



# 수정된 쿨다운 확인 및 거래 로직
def initial_market_analysis(data):
    """
    AI가 초기 매수/매도 타이밍을 설정합니다.
    첫 번째 거래를 강제 실행하여 쿨다운과 신호 처리가 정상 작동하도록 보장합니다.
    """
    global last_trade_time, current_balance, current_position, entry_price

    # AI 신호 생성
    signal, confidence = generate_signal(data, current_position)
    st.write(f"Initial Signal: {signal}, Confidence: {confidence}%")

    # 강제 실행: 첫 거래에서 confidence 조건 완화
    if signal == "BUY" and current_position == 0:
        process_buy(data["close"].iloc[-1], confidence=confidence)
        last_trade_time = time.time()  # 쿨다운 타이머 갱신
        return True
    elif signal == "SELL" and current_position > 0:
        st.write("Initial SELL triggered regardless of confidence.")
        process_sell(data["close"].iloc[-1])
        last_trade_time = time.time()  # 쿨다운 타이머 갱신
        return True
    else:
        st.write("AI recommends HOLD. Waiting for better conditions.")
        last_trade_time = time.time()  # 강제 초기화
        return False

# 쿨다운 로직 보완
def check_cooldown_and_trade(signal, data, trade_log):
    """
    쿨다운 타이머를 확인하고, 거래 실행 가능 여부를 판단합니다.
    """
    global last_trade_time, cooldown_period
    current_time = time.time()

    # 첫 거래 강제 실행을 보장하기 위해 초기화 처리
    if last_trade_time == 0:
        st.write("Initializing cooldown timer.")
        last_trade_time = current_time - cooldown_period  # 초기화

    # 적응형 쿨다운 계산
    cooldown_period = adaptive_cooldown(data, trade_log)

    # 쿨다운 상태 확인
    if current_time - last_trade_time < cooldown_period:
        remaining_time = cooldown_period - (current_time - last_trade_time)
        st.write(f"Cooldown active. Next trade allowed in {remaining_time:.2f} seconds.")
        return False  # 거래 스킵

    # 쿨다운 해제
    last_trade_time = current_time
    return True
    



# AI 신호 생성
def generate_signal(data, position, last_signal=None):
    price_data = data.tail(50).to_dict("records")
    prompt = f"""
Here is the recent Bitcoin market data:
{price_data}

Current position: {position} BTC.

Analyze the following Bitcoin market data and decide the next action using advanced indicators:
1. **Close Prices**: Evaluate short-term and long-term trends.
2. **RSI (Relative Strength Index)**:
   - Overbought condition: RSI > 70
   - Oversold condition: RSI < 30
3. **Moving Averages (MA)**:
   - Bullish crossover: MA(10) crosses above MA(50)
   - Bearish crossover: MA(10) crosses below MA(50)
4. **Bollinger Bands**:
   - Expansion: High volatility; look for breakout opportunities.
   - Contraction: Low volatility; anticipate potential trend reversals.
5. **MACD (Moving Average Convergence Divergence)**:
   - Bullish signal: MACD line crosses above the Signal line.
   - Bearish signal: MACD line crosses below the Signal line.
6. **ATR (Average True Range)**:
   - Estimate market volatility and adjust confidence levels.
7. **Fibonacci Retracement Levels**:
   - Identify potential support and resistance levels.
8. **Ichimoku Cloud**:
   - Bullish signal: Price above the cloud.
   - Bearish signal: Price below the cloud.

**Decision Criteria**:
1. **BUY**: High probability of price increase.
   - RSI < 30 (oversold condition)
   - Bullish crossover in MA(10/50) or MACD
   - Bollinger Bands contraction with upward breakout
   - Price near Fibonacci support level or above the Ichimoku Cloud.
2. **SELL**: High probability of price decrease.
   - RSI > 70 (overbought condition)
   - Bearish crossover in MA(10/50) or MACD
   - Bollinger Bands expansion with downward breakout
   - Price near Fibonacci resistance level or below the Ichimoku Cloud.
3. **HOLD**: No clear trend or conflicting signals.

Your response must follow this format:
SIGNAL, CONFIDENCE%

**Example response**:
BUY, 85%

Respond strictly in this format.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a trading assistant."},
                      {"role": "user", "content": prompt}],
            max_tokens=10
        )
        response_text = response.choices[0].message.content.strip()
        st.write(f"Raw AI Response: {response_text}")

        if "," in response_text:
            signal, confidence = response_text.split(",")
            signal = signal.strip().upper()
            confidence = int(confidence.strip().replace("%", ""))
            if signal not in ["BUY", "SELL", "HOLD"]:
                raise ValueError("Invalid signal received.")
            return signal, confidence
        else:
            raise ValueError("Response format is incorrect.")
    except Exception as e:
        st.error(f"Error generating signal: {e}")
        return "HOLD", 0


FEE_RATE = 0.001  # 거래 수수료 (0.1%)

def simulate_order(symbol, side, quantity):
    global current_balance, current_position
    price = float(binance_client.get_symbol_ticker(symbol=symbol)["price"])
    cost = price * quantity
    fee = cost * FEE_RATE

    if side == "BUY":
        if cost + fee > current_balance:
            st.error("Insufficient balance in test mode.")
            return None
        current_balance -= (cost + fee)  # 매수 비용 차감
        current_position += quantity  # 포지션 증가
        return price
    elif side == "SELL":
        if quantity > current_position:
            st.error("Insufficient position to sell in test mode.")
            return None
        revenue = cost - fee  # 매도 수익 = 매도 금액 - 수수료
        current_balance += revenue  # 잔고 증가
        current_position -= quantity  # 포지션 감소
        return price


    
    @st.cache_data
    def fetch_and_prepare_data(symbol, interval, limit):
        """
        데이터를 가져오고 기술적 지표를 추가한 뒤 캐싱.
        """
        raw_data = fetch_data(symbol, interval, limit)
        prepared_data = add_technical_indicators(raw_data)
        return prepared_data

    # Main loop optimization
    while True:
        data = fetch_and_prepare_data(SYMBOL, INTERVAL, 50)
        current_price = data["close"].iloc[-1]
        evaluate_and_decide(data, current_price)
        time.sleep(10)  # 거래 간격 조정


# 메인 실행
try:
    initial_balance = current_balance
    st.sidebar.write(f"Initial Balance: {initial_balance:.2f} USDT")

    # 초기 시장 분석
    data = fetch_data(SYMBOL, INTERVAL)
    while not initial_market_analysis(data):
        data = fetch_data(SYMBOL, INTERVAL)
        time.sleep(5)

    # 지속적인 트레이딩 루프
    while True:
        data = fetch_data(SYMBOL, INTERVAL)
        current_price = data["close"].iloc[-1]

        # 동적 리스크 관리
        take_profit, stop_loss = dynamic_risk_management(data)

        # 매매 판단
        evaluate_and_decide(data, current_price)

        # 최소 대기 시간
        time.sleep(5)
except KeyboardInterrupt:
    st.warning("Execution interrupted by user.")
finally:
    # 최종 기록 저장 및 수익 계산
    if len(trade_log) > 0:
        df_trade_log = pd.DataFrame(trade_log)
        df_trade_log.to_csv("trade_log1.csv", index=False)
        st.write("Trade log saved to 'trade_log.csv'")
    else:
        st.write("No trades executed. Nothing to save.")

    profit_loss = current_balance - initial_balance
    st.write(f"Final Balance: {current_balance:.2f} USDT")
    st.write(f"Profit/Loss: {profit_loss:.2f} USDT")
