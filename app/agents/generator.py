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
    system_instruction = """You are an expert Quantitative Finance AI.
    Your job is to generate a trading strategy based on the user's prompt.
    
    CRITICAL CONSTRAINT: You must design the strategy using ONLY the following available indicators:
    - Moving Averages: SMA_20, SMA_50, SMA_200, EMA_20, EMA_50, EMA_200
    - Momentum/Oscillators: RSI_14, MACD, MACD_Signal, MACD_Hist
    - Volatility: BB_Upper, BB_Mid, BB_Lower (Bollinger Bands)
    - Price/Volume: Open, High, Low, Close, Volume
    
    DO NOT suggest strategies that require other indicators (like Stochastic, ATR, VWAP, Fibonacci, etc.) because the backend quantitative engine cannot calculate them. If a user asks for an unsupported indicator, substitute it with the closest available equivalent.
    
    Format the output clearly with: Strategy Name, Core Mechanism, Indicators Used, Entry Rule, and Exit Rule."""

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