import yfinance as yf
import pandas as pd
import numpy as np

def run_backtest():
    try:
        # 1. Fetch Data
        ticker = 'MSFT'
        df = yf.download(ticker, period='1y', interval='1d', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 2. Pre-calculate Standard Indicators (The God-Mode Boilerplate)
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_Mid'] = df['SMA_20']
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)
        
        # RSI 14
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))

        df['Signal'] = 0

        # ==========================================
        # --- AI GENERATED SIGNAL LOGIC START ---
        df['Signal'] = np.where(((df['RSI_14'].shift() < df['RSI_14']) & (df['SMA_20'] > df['SMA_50']) & (df['Close'] > df['BB_Upper']) & 
                              ((df['Volume'] < df['Volume'].rolling(window=3).mean()) | (df['RSI_14'] > 70))), 1, 0)
        # --- AI GENERATED SIGNAL LOGIC END ---
        # ==========================================

        # 3. Portfolio Simulation
        df['Returns'] = df['Close'].pct_change()
        df['Strategy_Returns'] = df['Signal'].shift(1) * df['Returns']
        df['Strategy_Returns'].fillna(0, inplace=True)

        initial_capital = 10000
        df['Portfolio_Value'] = initial_capital * (1 + df['Strategy_Returns']).cumprod()

        final_value = df['Portfolio_Value'].iloc[-1]
        print(f"Final Portfolio Value: ${final_value:.2f}")

    except Exception as e:
        import traceback
        print(f"Backtest failed: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    run_backtest()
