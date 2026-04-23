import textwrap
from langchain_core.prompts import ChatPromptTemplate
from llm import llm_coder
from state import AgentState

def write_code(state: AgentState) -> dict:
    print("--- [AGENT: CODER] Writing/Refactoring Python Code ---")
    
    current_concept = state.get("current_concept", "")
    execution_error = state.get("execution_error", "")
    code_revisions = state.get("code_revisions", 0)
    
    system_instruction = """You are a Quantitative Developer.
    Write ONLY the pandas boolean logic to generate the 'Signal' column.
    
    The DataFrame 'df' ALREADY has the following columns calculated for you:
    - 'Close', 'Open', 'High', 'Low', 'Volume'
    - 'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50'
    - 'RSI_14'
    - 'BB_Upper', 'BB_Mid', 'BB_Lower'
    
    Set df['Signal'] to 1 for Buy, and 0 for Flat/Sell based on the strategy rules.
    Use numpy (np.where) or pandas loc.
    
    CRITICAL: Output ONLY valid Python code. No explanations. No imports. No boilerplate. Do not define functions."""

    if execution_error:
        system_instruction += f"\n\nFIX THIS ERROR in your logic:\n{execution_error}"

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
{indented_ai_logic}
        # --- AI GENERATED SIGNAL LOGIC END ---
        # ==========================================

        # 3. Portfolio Simulation
        df['Returns'] = df['Close'].pct_change()
        df['Strategy_Returns'] = df['Signal'].shift(1) * df['Returns']
        df['Strategy_Returns'].fillna(0, inplace=True)

        initial_capital = 10000
        df['Portfolio_Value'] = initial_capital * (1 + df['Strategy_Returns']).cumprod()

        final_value = df['Portfolio_Value'].iloc[-1]
        print(f"Final Portfolio Value: ${{final_value:.2f}}")

    except Exception as e:
        import traceback
        print(f"Backtest failed: {{str(e)}}")
        traceback.print_exc()

if __name__ == "__main__":
    run_backtest()
"""
    
    return {
        "backtest_code": full_code,
        "code_revisions": code_revisions + 1
    }