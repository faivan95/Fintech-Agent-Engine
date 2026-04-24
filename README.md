# 🤖 FinTech Agent Engine: Autonomous Quant Backtesting Pipeline

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-orange?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?style=for-the-badge&logo=docker)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-State_Memory-blue?style=for-the-badge&logo=postgresql)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?style=for-the-badge&logo=streamlit)

An autonomous, containerized multi-agent AI system that ideates, codes, self-heals, and executes quantitative trading strategies via Python and Pandas. Powered by local LLMs and LangGraph state orchestration.

---

### 🎥 Live Execution Demo

https://github.com/user-attachments/assets/272b8e8d-1c04-462f-9f52-8996c0e84959

### Project Vision & Progression
The initial scope of this project explored using a Retrieval-Augmented Generation (RAG) microservice (via ChromaDB) to feed financial literature to the LLM. However, during early prototyping, it became clear that the primary bottleneck was not *knowledge retrieval*, but *execution stability and deterministic state memory*.

**The Pivot:** The architecture was refactored to prioritize **State Checkpointing**. By migrating to `langgraph-checkpoint-postgres`, I engineered a persistent Human-In-The-Loop (HITL) pipeline. This allows human operators to pause the multi-agent graph, review the AI's proposed mathematical models, and approve or reject them without losing the system's operational memory.

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

### Prerequisites
* Docker & Docker Compose
* NVIDIA GPU (Optional, but recommended for local LLM inference)
