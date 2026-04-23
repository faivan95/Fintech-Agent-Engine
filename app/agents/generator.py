from langchain_core.prompts import ChatPromptTemplate
from llm import llm_creative
from state import AgentState

def generate_strategy(state: AgentState) -> dict:
    """
    Node function for the Generator Agent.
    It takes the current state, reads the user query (and any critiques), 
    and generates an algorithmic trading strategy concept.
    """
    print("--- [AGENT: GENERATOR] Ideating Strategy ---")
    
    user_query = state.get("user_query")
    if not user_query:
        print("Error: No user query provided to Generator.")
        return {"current_concept": "Error: Missing Input"}
    
    critique_feedback = state.get("critique_feedback", "")
    
    # System Prompts define the persona and constraints of the AI
    system_instruction = """You are an elite Quantitative Analyst at a Munich-based FinTech hedge fund. 
    Your objective is to design logical, algorithmic trading strategies based on user requests.
    Structure your response clearly with:
    1. Strategy Name
    2. Core Mechanism (e.g., Mean Reversion, Momentum)
    3. Indicators Used (e.g., RSI, MACD, Moving Averages)
    4. Proposed Entry and Exit Rules.
    Do NOT write code. Only define the mathematical concept."""

    # If the Critic agent previously rejected the concept, we append the feedback so the model learns and revises.
    if critique_feedback:
        system_instruction += f"\n\nCRITICAL FEEDBACK FROM RISK MANAGEMENT: You must revise your previous strategy based on this feedback: {critique_feedback}"

    # Construct the final prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("human", "{query}")
    ])
    
    # Chain the prompt to our local Phi-3 model
    chain = prompt | llm_creative
    
    # Execute the LLM call
    response = chain.invoke({"query": user_query})
    
    # Return the updated state. LangGraph will merge this dict into the global AgentState.
    # We also increment the revision count to track how many times it has looped.
    current_revisions = state.get("revision_count", 0)
    
    return {
        "current_concept": response.content,
        "revision_count": current_revisions + 1
    }