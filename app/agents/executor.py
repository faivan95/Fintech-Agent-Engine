import subprocess
import os
from state import AgentState

def execute_backtest(state: AgentState) -> dict:
    """Runs the generated Python code and captures the output or errors."""
    print("--- [AGENT: EXECUTOR] Running Backtest ---")
    
    # 1. Save the current code to the file (just like we did in invoke_agent.py)
    output_dir = "/app/outputs"
    os.makedirs(output_dir, exist_ok=True)
    py_file_path = os.path.join(output_dir, "backtest.py")
    
    with open(py_file_path, "w", encoding="utf-8") as f:
        f.write(state.get("backtest_code", ""))
        
    # 2. Attempt to run the file
    try:
        # We run the file and capture what it prints out (stdout) and its errors (stderr)
        result = subprocess.run(
            ["python", py_file_path],
            capture_output=True,
            text=True,
            timeout=30 # Don't let it run forever
        )
        
        # 3. Check if it crashed
        if result.returncode != 0:
            print(f"--- [EXECUTOR: ERROR CAUGHT] ---")
            return {"execution_error": result.stderr, "simulation_results": ""}
            
        print(f"--- [EXECUTOR: SUCCESS] ---")
        return {"execution_error": "", "simulation_results": result.stdout}
        
    except subprocess.TimeoutExpired:
        print(f"--- [EXECUTOR: TIMEOUT] ---")
        return {"execution_error": "Execution timed out after 30 seconds.", "simulation_results": ""}