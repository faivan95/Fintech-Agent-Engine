from langchain_core.prompts import ChatPromptTemplate
from llm import llm_creative
from state import AgentState

def validate_strategy(state: AgentState) -> dict:
    """
    Node function for the Validator Agent.
    Critiques the strategy and decides whether to approve or send it back.
    """
    print("--- [AGENT: VALIDATOR] Reviewing Strategy for Risk ---")
    
    current_concept = state.get("current_concept", "")
    # CRITICAL: Fetch the current revision count from memory
    current_revisions = state.get("revision_count", 0) 
    
    # NEW: System prompt aligned with the Coder's capabilities
    system_instruction = """You are the Chief Risk Officer at a quantitative hedge fund. 
    Review the provided trading strategy.
    
    CRITICAL CONSTRAINTS:
    1. Our execution engine ONLY supports 1/0 position signals using SMA, EMA, RSI, BB, and MACD.
    2. Our engine currently DOES NOT support dynamic stop-loss or take-profit orders. Do NOT reject a strategy for lacking a stop-loss.
    3. If the strategy uses the allowed indicators logically and safely, you MUST start your response with: "DECISION: APPROVED".
    4. If the strategy relies on unsupported indicators (like VWAP, Stochastic, ATR) or is completely illogical, you MUST start your response with: "DECISION: REJECTED".
    
    Following your decision, provide a 2-3 sentence critique."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("human", "Strategy to review:\n{concept}")
    ])
    
    chain = prompt | llm_creative
    response = chain.invoke({"concept": current_concept})
    review_text = response.content
    
    # Parse the LLM's text to determine the boolean approval state
    approved = "DECISION: APPROVED" in review_text.upper()
    
    # Print the decision to the terminal for our visibility
    decision_str = "APPROVED" if approved else "REJECTED"
    print(f"--- [VALIDATOR DECISION: {decision_str}] ---")
    print(f"Critique: {review_text}\n")
    
    # CRITICAL FIX: Return the updated state variables with the incremented counter
    if approved:
        return {
            "critique_feedback": review_text,
            "approved": True,
            "revision_count": current_revisions # Keep count the same if approved
        }
    else:
        return {
            "critique_feedback": review_text,
            "approved": False,
            "revision_count": current_revisions + 1 # Increment to eventually break the loop!
        }