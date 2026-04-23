import streamlit as st
import pandas as pd
import plotly.express as px
import uuid
import os
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from graph import workflow

# --- UI CONFIGURATION ---
st.set_page_config(page_title="AI Quant Engine", page_icon="📈", layout="wide")
st.title("🤖 Autonomous Quantitative Trading AI")

# --- INITIALIZE SESSION STATE ---
# This is how Streamlit remembers what is happening between button clicks!
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "app_state" not in st.session_state:
    st.session_state.app_state = "input" # States: 'input', 'running_gen', 'review', 'running_code', 'finished'

# --- DATABASE SETUP ---
@st.cache_resource
def get_db_pool():
    """Caches the database connection so Streamlit doesn't open a new one on every click."""
    db_user = os.getenv("POSTGRES_USER", "postgres")
    db_pass = os.getenv("POSTGRES_PASSWORD", "postgres")
    db_host = os.getenv("DB_HOST", "postgres-db")
    db_name = os.getenv("POSTGRES_DB", "postgres")
    DB_URI = f"postgresql://{db_user}:{db_pass}@{db_host}:5432/{db_name}"
    
    return ConnectionPool(conninfo=DB_URI, kwargs={"autocommit": True})

# Fetch the cached pool
pool = get_db_pool()

checkpointer = PostgresSaver(pool)
checkpointer.setup()

# WE ADD THE BREAKPOINT BACK IN!
app_graph = workflow.compile(checkpointer=checkpointer, interrupt_after=["validator"])

# --- SIDEBAR ---
with st.sidebar:
    st.header("Strategy Configuration")
    user_prompt = st.text_area("Trading Strategy Prompt:", "Write a momentum strategy for MSFT using moving average crossovers and RSI.", height=150)
    
    if st.button("🚀 Start AI Engine", disabled=(st.session_state.app_state != "input")):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.app_state = "running_gen"
        st.rerun() # Force UI refresh

# --- PHASE 1: GENERATE STRATEGY ---
if st.session_state.app_state == "running_gen":
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    initial_state = {"user_query": user_prompt, "code_revisions": 0}
    
    with st.spinner("Agents are ideating..."):
        app_graph.invoke(initial_state, config)
        
    st.session_state.app_state = "review"
    st.rerun()

# --- PHASE 2: HUMAN IN THE LOOP (REVIEW) ---
elif st.session_state.app_state == "review":
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    snapshot = app_graph.get_state(config)
    
    if snapshot.next and "coder" in snapshot.next:
        st.warning("⚠️ HUMAN-IN-THE-LOOP: Review Required")
        st.subheader("Proposed Strategy")
        
        # Display the AI's concept
        concept = snapshot.values.get("current_concept", "No concept found.")
        st.info(concept)
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("✅ Approve & Write Code", width="stretch"):
                st.session_state.app_state = "running_code"
                st.rerun()
        with col2:
            if st.button("❌ Reject & Reset", width="stretch"):
                st.session_state.app_state = "input"
                st.session_state.thread_id = None
                st.rerun()

# --- PHASE 3: CODER LOOP ---
elif st.session_state.app_state == "running_code":
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    with st.spinner("Coder & Executor are running... this may take a minute as it self-heals."):
        app_graph.invoke(None, config) # Passing None tells LangGraph to RESUME
    st.session_state.app_state = "finished"
    st.rerun()

# --- PHASE 4: RESULTS DISPLAY ---
elif st.session_state.app_state == "finished":
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    snapshot = app_graph.get_state(config)
    final_state = snapshot.values

    st.success("✅ Execution Pipeline Finished!")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("💻 Execution Logs")
        revisions = final_state.get("code_revisions", 0)
        error = final_state.get("execution_error", "")
        
        # If it failed 5 times, show exactly why
        if revisions >= 5 and error:
            st.error(f"Failed to compile after {revisions} attempts.\n\n**Final Traceback:**\n```python\n{error}\n```")
        else:
            st.text(final_state.get("simulation_results", "No Tear Sheet Output."))

    with col2:
        st.subheader("📈 Portfolio Equity Curve")
        if revisions >= 5 and error:
             st.warning("Chart cannot be displayed due to execution failure.")
        elif os.path.exists('/app/outputs/equity_curve.csv'):
            chart_df = pd.read_csv('/app/outputs/equity_curve.csv', index_col='Date', parse_dates=True)
            fig = px.line(chart_df, y='Portfolio_Value', title="Strategy Performance over 1 Year")
            fig.update_layout(yaxis_title="Portfolio Value ($)", xaxis_title="Date")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Chart data not found. The backtest may have failed without saving CSV.")
            
    if st.button("🔄 Start New Run"):
        st.session_state.app_state = "input"
        st.session_state.thread_id = None
        st.rerun()