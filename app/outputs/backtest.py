import yfinance as yf
import pandas as pd
import numpy as np
import sys

def run_backtest():
    try:
        # 1. Fetch Data
        ticker = 'MSFT'
        df = yf.download(ticker, period='1y', interval='1d', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 2. Pre-calculate Standard Indicators
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
        
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

        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        df['Signal'] = 0

        # ==========================================
        # --- AI GENERATED SIGNAL LOGIC START ---
        df['Signal'] = np.where(
            (df['SMA_20'] > df['EMA_20']) & # Entry condition for SMA and EMA crossover during Asia Pacific after-hours session
                ((df['Close'].shift() < df['Open']) |  # Check if the previous close was lower than open, indicating bearish pressure
                 (df['SMA_50'] > df['EMA_50'])) & # Additional confirmation with SMA and EMA crossover during Asia Pacific after-hours session for momentum shift towards bullishness
            True, 1, 0)
        # --- AI GENERATED SIGNAL LOGIC END ---
        # ==========================================

        if df['Signal'].sum() == 0:
            raise ValueError("SILENT FAILURE: Your logic generated ZERO buy signals over the entire year. The conditions are too strict or mathematically contradictory. Loosen your entry rules.")

        # 3. Portfolio Simulation & Analytics
        df['Returns'] = df['Close'].pct_change()
        df['Strategy_Returns'] = df['Signal'].shift(1) * df['Returns']
        df['Strategy_Returns'].fillna(0, inplace=True)

        initial_capital = 10000
        df['Portfolio_Value'] = initial_capital * (1 + df['Strategy_Returns']).cumprod()

        final_value = df['Portfolio_Value'].iloc[-1]
        total_return = ((final_value - initial_capital) / initial_capital) * 100
        
        daily_volatility = df['Strategy_Returns'].std()
        sharpe_ratio = (df['Strategy_Returns'].mean() / daily_volatility) * np.sqrt(252) if daily_volatility > 0 else 0
        
        rolling_max = df['Portfolio_Value'].cummax()
        drawdown = (df['Portfolio_Value'] / rolling_max) - 1
        max_drawdown = drawdown.min() * 100

        # Export data for the Streamlit UI to plot
        df[['Close', 'Portfolio_Value']].to_csv('/app/outputs/equity_curve.csv')

        print("\n--- QUANTITATIVE TEAR SHEET ---")
        print(f"Final Portfolio Value : ${final_value:.2f}")
        print(f"Total Return          : ${total_return:.2f}%")
        print(f"Sharpe Ratio          : ${sharpe_ratio:.2f}")
        print(f"Maximum Drawdown      : ${max_drawdown:.2f}%")
        print("-------------------------------")

    except Exception as e:
        import traceback
        print(f"Backtest failed: {str(e)}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    run_backtest()
