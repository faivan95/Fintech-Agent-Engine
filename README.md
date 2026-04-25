# 🤖 FinTech Agent Engine: Autonomous Quant Backtesting Pipeline

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-orange?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?style=for-the-badge&logo=docker)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-State_Memory-blue?style=for-the-badge&logo=postgresql)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?style=for-the-badge&logo=streamlit)
![LangSmith](https://img.shields.io/badge/LangSmith-Observability-gray?style=for-the-badge&logo=langchain)

An autonomous, containerized multi-agent AI system that ideates, codes, self-heals, and executes quantitative trading strategies via Python and Pandas. Powered by local LLMs and LangGraph state orchestration.

---

### 🎥 Live Execution Demo

https://github.com/user-attachments/assets/272b8e8d-1c04-462f-9f52-8996c0e84959

### Usage Workflow
Input: Type a strategy concept into the Streamlit sidebar (e.g., "Write a momentum strategy using RSI and MACD crossovers").

Ideation: The Generator Agent formulates the mathematical entry/exit rules.

HITL Review: The graph pauses. The UI displays the proposed strategy for human approval.

Code & Self-Heal: The Coder Agent writes the Pandas logic. If it fails, it enters a recursive 5-attempt self-healing loop.

Execution: The yfinance engine backtests the code and renders the interactive equity curve.

---

## 📊 Analytics & Outputs (The Tear Sheet)
Upon successful compilation, the engine outputs a Quantitative Tear Sheet with the following metrics:

Final Portfolio Value: The absolute dollar value of a $10,000 initial capital investment after a 1-year backtest.

Total Return (%): The net percentage gain or loss.

Sharpe Ratio: The risk-adjusted return. A Sharpe ratio > 1.0 indicates strong returns relative to the daily volatility the strategy endured.

Maximum Drawdown (%): The largest single drop from peak to trough in the portfolio's value. Evaluates the strategy's worst-case risk scenario.

---

## 🔎 Observability & Agent Tracing (LangSmith)
To ensure system reliability and track LLM token efficiency, this pipeline is deeply integrated with **LangSmith**. 

> **<img width="378" height="535" alt="Trace Run" src="https://github.com/user-attachments/assets/cbf8f98f-e67a-498f-9f36-be73d9a2192f" />**

### Key Telemetry Metrics Tracked
By inspecting the LangSmith traces, we monitor the health of the autonomous loops:

* **Validator Revisions (`revision_count`):** Tracks how many times the Risk Manager agent rejected a strategy for violating indicator constraints. 
* **Self-Healing Index (`code_revisions`):** Tracks the number of times the Executor caught a Python exception (`SyntaxError` / `ValueError`) and routed the traceback back to the Coder. *Example: In a recent run backtesting an Nvidia strategy, the system autonomously caught an overly-strict entry logic constraint, rewrote its own Pandas code twice, and successfully compiled on Attempt 3.*
* **Execution Latency & Token Usage:** Monitors the inference speed and context-window saturation of the local LLM to prevent context overpower and memory leaks.

> **<img width="375" height="467" alt="Trace Tree" src="https://github.com/user-attachments/assets/6bdb8b4c-5931-42c0-81b1-babf773d0bb7" />**

---

## 🚧 Engineering Hurdles & System Constraints
Building an autonomous coding AI requires strict guardrails to prevent infinite loops and syntax hallucinations. Here is how the system forces deterministic behavior from local LLMs:

### 1. Mitigating "Context Overpowering" (The God-Mode Sandbox)
When tasked with writing code, LLMs frequently hallucinate unsupported libraries or indicators (e.g., trying to import arbitrary moving averages). 
* **The Solution:** I engineered a strict Pandas execution sandbox ("God-Mode"). The LLM is prohibited from defining functions or imports. It is mathematically forced to output pure, binary `np.where()` logic using a predefined set of pre-calculated vectors (SMA, EMA, RSI, MACD, Bollinger Bands).

### 2. The "Zero-Trade" Logic Guard (Silent Failures)
Often, an LLM will write perfectly valid Python syntax, but the logic is so restrictive (e.g., *RSI < 20 AND Price > 200 SMA*) that the strategy never executes a single trade over a 1-year backtest, resulting in a silent 0% return.
* **The Solution:** The execution engine includes a deterministic logic guard. If `df['Signal'].sum() == 0`, the system intentionally crashes with a `ValueError`. This traceback is fed back to the Coder Agent, forcing a recursive self-healing loop to loosen its algorithmic constraints until trades are actually executed.

---

### Project Vision & Progression
The initial scope of this project explored using a Retrieval-Augmented Generation (RAG) microservice (via ChromaDB) to feed financial literature to the LLM. However, during early prototyping, it became clear that the primary bottleneck was not *knowledge retrieval*, but *execution stability and deterministic state memory*.

**The Pivot:** The architecture was refactored to prioritize **State Checkpointing**. By migrating to `langgraph-checkpoint-postgres`, I engineered a persistent Human-In-The-Loop (HITL) pipeline. This allows human operators to pause the multi-agent graph, review the AI's proposed mathematical models, and approve or reject them without losing the system's operational memory.

---

### Prerequisites
* Docker & Docker Compose
* NVIDIA GPU (Optional, but recommended for local LLM inference)
