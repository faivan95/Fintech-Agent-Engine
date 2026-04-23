import textwrap
from langchain_core.prompts import ChatPromptTemplate
from llm import llm_coder
from state import AgentState

def write_code(state: AgentState) -> dict:
    current_concept = state.get("current_concept", "")
    execution_error = state.get("execution_error", "")
    code_revisions = state.get("code_revisions", 0)
    
    print(f"--- [AGENT: CODER] Writing/Refactoring Python Code (Attempt {code_revisions + 1}/5) ---")
    
    system_instruction = """You are a Quantitative Developer.
    Write ONLY the pandas boolean logic to generate the 'Signal' column.
    
    The DataFrame 'df' ALREADY has the following columns calculated for you:
    - 'Close', 'Open', 'High', 'Low', 'Volume'
    - 'SMA_20', 'SMA_50', 'SMA_200'
    - 'EMA_20', 'EMA_50', 'EMA_200'
    - 'RSI_14'
    - 'BB_Upper', 'BB_Mid', 'BB_Lower'
    - 'MACD', 'MACD_Signal', 'MACD_Hist'
    
    CRITICAL PANDAS SYNTAX RULES:
    1. DO NOT INVENT COLUMNS OR VARIABLES. Use ONLY the exact columns listed above. Do not use undeclared variables like 'volume_threshold'.
    2. STRICT BINARY OUTPUT: Your np.where condition MUST return 1 for a signal and 0 for no signal. NEVER use strings like 'Buy' or 'Sell'.
    3. You MUST use np.where() for your logic. You must start your code directly with: df['Signal'] = np.where(...)
    4. If combining multiple conditions with & or |, you MUST wrap EVERY individual condition in its own parentheses.
    5. ABSOLUTELY NO COMMENTS. Do not use the '#' symbol. Write pure, silent code.
    6. NO FUNCTIONS OR LAMBDAS. Do not use 'def' or 'lambda'. Do not assign your logic to a custom variable.
    
    Output ONLY valid Python code. No explanations. No imports."""

    if execution_error:
        # CRITICAL FIX: Escape curly braces in the error trace so LangChain doesn't crash!
        safe_error = execution_error.replace("{", "{{").replace("}", "}}")
        
        system_instruction += f"""
        
        CRITICAL FIX REQUIRED: Your previous code crashed. 
        If the error below is a 'KeyError', YOU MISSPELLED A COLUMN NAME. 
        Check your work and use ONLY the columns explicitly listed in your instructions.
        
        Traceback:
        {safe_error}
        """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("human", "Strategy:\n{concept}")
    ])
    
    chain = prompt | llm_coder
    response = chain.invoke({"concept": current_concept})
    
    raw_ai_logic = response.content.replace("```python", "").replace("```", "").strip()
    indented_ai_logic = textwrap.indent(raw_ai_logic, '        ')
    
    full_code = f"""import yfinance as yf
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
{indented_ai_logic}
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

        print("\\n--- QUANTITATIVE TEAR SHEET ---")
        print(f"Final Portfolio Value : ${{final_value:.2f}}")
        print(f"Total Return          : ${{total_return:.2f}}%")
        print(f"Sharpe Ratio          : ${{sharpe_ratio:.2f}}")
        print(f"Maximum Drawdown      : ${{max_drawdown:.2f}}%")
        print("-------------------------------")

    except Exception as e:
        import traceback
        print(f"Backtest failed: {{str(e)}}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    run_backtest()
"""
    
    return {
        "backtest_code": full_code,
        "code_revisions": code_revisions + 1
    }