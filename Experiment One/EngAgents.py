import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, trace
from openai.types.responses import ResponseTextDeltaEvent
from typing import Dict, List
import os

# Load environment variables (e.g., API keys)
load_dotenv(override=True)

# --- Agent Definitions for Asset Analysis ---

instructions1 = "You are a Diagram Analyst agent. Your job is to receive a query, a list of asset numbers, and a **pre-computed analysis** of a diagram. \
You must first state the query's goal. Then, you will clearly present the provided analysis for the requested asset numbers."

instructions2 = "You are a Legend Interpreter agent. You receive a report from a Diagram Analyst and a **pre-computed interpretation** of a legend file. \
Your job is to clearly state the meaning of any symbols mentioned in the report, based on the interpretation provided."

instructions3 = "You are a Chief Analyst agent. You review the full chain of information, \
including the original query, the diagram analysis, and the legend interpretation. \
Your job is to synthesize all this information into a final, clear, and conclusive answer \
to the user's original query (e.g., 'are there new lanterns?')."

# --- Agent Initialization ---

diagram_analyst_agent = Agent(
    name="Diagram Analyst Agent",
    instructions=instructions1,
    model="gpt-4o-mini"
)

legend_interpreter_agent = Agent(
    name="Legend Interpreter Agent",
    instructions=instructions2,
    model="gpt-4o-mini"
)

chief_analyst_agent = Agent(
    name="Chief Analyst Agent",
    instructions=instructions3,
    model="gpt-4o-mini"
)

# --- Main Execution Workflow ---

async def main():

    # --- Parameters ---
    asset_numbers: List[str] = ["2004859", "2004858"]
    asset_type_query: str = "new lanterns"
    diagram_file: str = "diagram_v1.pdf"  # Parameterized file path (e.g., PDF)
    legend_file: str = "legend_v1.pdf"    # Parameterized file path (e.g., PDF)
    
    # ---
    # This is the "simulated" analysis.
    # In a real-world scenario, you would run a PDF/image analysis tool
    # and feed the results in here.
    # ---
    simulated_diagram_analysis = f"""
    Analysis of diagram '{diagram_file}':
    - Asset 2004859 is located next to a symbol: 'a circle with spikes'.
    - Asset 2004858 is located next to a symbol: 'a circle with spikes (labeled F1)'.
    """
    
    simulated_legend_interpretation = f"""
    Interpretation of legend '{legend_file}':
    - The symbol 'a circle with spikes' represents 'NEW LANTERN'.
    """
    # --- End of Simulated Data ---


    # This variable will store the entire conversation chain
    full_response = ""
    
    # 1. Create the initial query from the parameters
    initial_query = f"Check asset numbers {asset_numbers} on diagram '{diagram_file}' and identify if there are '{asset_type_query}'."

    print("--- Agent 1: Diagram Analyst ---")
    print(f"Initial Query: {initial_query}\n")

    # 2. Create the first prompt, including the simulated analysis
    prompt1 = f"Query: {initial_query}\n\nHere is the pre-computed analysis of '{diagram_file}':\n{simulated_diagram_analysis}\n\nPlease present this analysis clearly."
    
    result1 = Runner.run_streamed(diagram_analyst_agent, input=prompt1)
    async for event in result1.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            text_chunk = event.data.delta
            print(text_chunk, end="", flush=True)
            full_response += text_chunk

    print("\n\n--- Agent 2: Legend Interpreter ---")
    
    # 3. Create the second prompt, passing the analyst's report and the legend interpretation
    prompt2 = f"Here is the analyst's report:\n{full_response}\n\nHere is the pre-computed interpretation of '{legend_file}':\n{simulated_legend_interpretation}\n\nPlease state the interpretation for the symbols found."
    
    result2 = Runner.run_streamed(legend_interpreter_agent, input=prompt2)
    async for event in result2.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            text_chunk = event.data.delta
            print(text_chunk, end="", flush=True)
            full_response += text_chunk

    print("\n\n--- Agent 3: Chief Analyst (Final Conclusion) ---")
    
    # 4. Create the final prompt for the chief analyst to synthesize everything
    prompt3 = f"Based on all the following information, provide a final, conclusive answer to the initial query ({initial_query}):\n\n{full_response}"

    result3 = Runner.run_streamed(chief_analyst_agent, input=prompt3)
    async for event in result3.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            text_chunk = event.data.delta
            print(text_chunk, end="", flush=True)
            full_response += text_chunk
    
    print("\n\n--- Analysis Complete ---")

if __name__ == "__main__":
    asyncio.run(main())

