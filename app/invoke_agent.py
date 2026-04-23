import os
import sys
import uuid
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from graph import workflow # <-- Importing the blueprint, not the compiled graph

if __name__ == "__main__":
    print("\n=== FinTech Agent Engine: Execution ===")
    
    input_file_path = "/app/user_query.md"
    if not os.path.exists(input_file_path):
        print(f"[ERROR] Input file not found at {input_file_path}")
        sys.exit(1)
        
    with open(input_file_path, "r", encoding="utf-8") as f:
        user_input = f.read().strip()
        
    initial_state = {
        "user_query": user_input,
        "current_concept": "",
        "simulation_results": "",
        "market_context": "",
        "critique_feedback": "",
        "revision_count": 0,
        "approved": False,
        "backtest_code": "",
        "execution_error": "",
        "code_revisions": 0
    }

    print("[SYSTEM] Sending request to the Multi-Agent Graph. Please wait...")

    # 1. Ensure the DB connection uses your specific host: "postgres-db"
    # Using defaults for local Docker, but dynamically fetching if env vars exist
    db_user = os.getenv("POSTGRES_USER", "postgres")
    db_pass = os.getenv("POSTGRES_PASSWORD", "postgres")
    db_host = os.getenv("DB_HOST", "postgres-db")
    db_name = os.getenv("POSTGRES_DB", "postgres")
    
    DB_URI = f"postgresql://{db_user}:{db_pass}@{db_host}:5432/{db_name}"
    
    # 2. Open the Database Connection safely using a Context Manager
    with ConnectionPool(conninfo=DB_URI, kwargs={"autocommit": True}) as pool:
        checkpointer = PostgresSaver(pool)
        
        # This automatically creates LangGraph's internal memory tables in your DB
        checkpointer.setup() 
        
        # 3. Compile the graph WITH the checkpointer and breakpoint
        app_graph = workflow.compile(
            checkpointer=checkpointer,
            interrupt_after=["validator"] # The HITL Breakpoint
        )

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        # Run the graph until it hits a breakpoint or finishes
        current_state = app_graph.invoke(initial_state, config)
        
        # Check if the graph paused (HITL)
        snapshot = app_graph.get_state(config)
        
        if snapshot.next and "coder" in snapshot.next:
            print("\n=== HUMAN-IN-THE-LOOP: REVIEW REQUIRED ===")
            print("The Validator has approved the following strategy and saved the state to PostgreSQL:\n")
            print(current_state["current_concept"])
            print("\n==========================================")
            
            user_approval = input("\n[SYSTEM] Do you approve this concept for coding? (y/n): ")
            
            if user_approval.lower() == 'y':
                print("\n[SYSTEM] Human Approved. Retrieving state from PostgreSQL and resuming to CODER...")
                final_state = app_graph.invoke(None, config)
            else:
                print("\n[SYSTEM] Human rejected the strategy. Execution aborted.")
                sys.exit(0)
        else:
            final_state = current_state

        # --- OUTPUT PRINTING ---
        simulation_results = final_state.get("simulation_results", "")
        execution_error = final_state.get("execution_error", "")
        
        print("\n================ BACKTEST RESULTS ================\n")
        if simulation_results:
            print(simulation_results)
        elif execution_error:
            revisions = final_state.get("code_revisions", 0)
            print(f"[FAILED TO EXECUTE] The AI could not fix the code after {revisions} attempts.")
            print(f"Final Error:\n{execution_error}")
        else:
            print("No simulation was run.")
        print("\n==================================================")