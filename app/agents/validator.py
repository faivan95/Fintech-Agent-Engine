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
    
    # The System Prompt forces the AI to output exactly "DECISION: APPROVED" or "DECISION: REJECTED"
    system_instruction = """You are the Chief Risk Officer at a quantitative hedge fund. 
    Review the provided trading strategy. You are looking for strict risk management, stop-loss mechanisms, and market viability.
    
    If the strategy is well-rounded and safe, you MUST start your response with: "DECISION: APPROVED".
    If the strategy is too risky, lacks stop-losses, or is logically flawed, you MUST start your response with: "DECISION: REJECTED".
    
    Following your decision, provide a 2-3 sentence critique explaining why."""

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
    
    # Return the updated state variables
    return {
        "critique_feedback": review_text,
        "approved": approved
    }