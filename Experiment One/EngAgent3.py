import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, trace
from openai.types.responses import ResponseTextDeltaEvent
from typing import Dict, List
import os

# --- NEW IMPORTS ---
import base64
import io
from pdf2image import convert_from_path  # To read PDFs as images
from openai import AsyncOpenAI          # To make direct vision calls
# --- END NEW IMPORTS ---

# Load environment variables (e.g., API keys)
load_dotenv(override=True)

# --- Agent Definitions for Asset Analysis ---
# (These are unchanged, as they process the *results* of the vision analysis)

instructions1 = "You are a Diagram Analyst agent. Your job is to receive a query, a list of asset numbers, and a **pre-computed analysis** of a diagram. \
You must first state the query's goal. Then, you will clearly present the provided analysis for the requested asset numbers."

instructions2 = "You are a Legend Interpreter agent. You receive a report from a Diagram Analyst and a **pre-computed interpretation** of a legend file. \
Your job is to clearly state the meaning of any symbols mentioned in the report, based on the interpretation provided."

instructions3 = "You are a Chief Analyst agent. You review the full chain of information, \
including the original query, the diagram analysis, and the legend interpretation. \
Your job is to synthesize all this information into a final, clear, and conclusive answer \
to the user's original query (e.g., 'are there new lanterns?')."

# --- Agent Initialization ---
# (Unchanged)

diagram_analyst_agent = Agent(
    name="Diagram Analyst Agent",
    instructions=instructions1,
    model="gpt-5"
)

legend_interpreter_agent = Agent(
    name="Legend Interpreter Agent",
    instructions=instructions2,
    model="gpt-5"
)

chief_analyst_agent = Agent(
    name="Chief Analyst Agent",
    instructions=instructions3,
    model="gpt-5"
)

# --- NEW HELPER FUNCTIONS for PDF/Vision Analysis ---

def pdf_to_base64_images(pdf_path: str) -> List[str]:
    """Converts each page of a PDF into a list of base64 encoded image strings."""
    print(pdf_path)
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return []
    
    try:
        images = convert_from_path(pdf_path)
        base64_list = []
        for image in images:
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            base64_list.append(img_str)
        print(f"Successfully converted {pdf_path} to {len(base64_list)} image(s).")
        return base64_list
    except Exception as e:
        print(f"Error converting PDF {pdf_path}: {e}")
        print("Please ensure 'poppler' is installed and in your system's PATH.")
        return []

async def run_vision_analysis(client: AsyncOpenAI, user_prompt: str, base64_images: List[str], model: str = "gpt-5") -> str:
    """
    Runs a multimodal vision analysis on a list of images and returns the text result.
    """
    if not base64_images:
        return f"Error: No images provided for analysis. (Source prompt: {user_prompt})"

    # Build the content list for the API
    content: List[Dict] = [{"type": "text", "text": user_prompt}]
    for img_str in base64_images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_str}"}
        })
    
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": content}
            ],
            max_tokens=1024
        )
        result = response.choices[0].message.content
        return result if result else "No analysis returned from vision model."
    except Exception as e:
        print(f"Error during vision analysis: {e}")
        return f"Error analyzing images: {e}"

# --- Main Execution Workflow (Updated) ---

async def main():

    # --- Parameters ---
    asset_numbers: List[str] = ["2004859", "2004858"]
    asset_type_query: str = "new column"
    
    # These files MUST exist in the same directory (or provide full paths)
    diagram_file: str = "diagram_v1.pdf"  
    legend_file: str = "legend_v1.pdf"   
    
    # --- NEW: Initialize OpenAI Client for Vision ---
    client = AsyncOpenAI()
    vision_model = "gpt-4o" # Use the same model for consistency

    # ---
    # This is the "REAL" analysis step.
    # We replace the simulated strings with actual vision model calls.
    # ---
    
    print("--- Step 0: Analyzing PDF files with Vision Model ---")

    # 1. Convert PDFs to images
    diagram_images = pdf_to_base64_images(diagram_file)
    legend_images = pdf_to_base64_images(legend_file)

    # 2. Define vision prompts
    # This prompt asks the vision model to *perform* the analysis
    vision_prompt_diagram = f"""
    You are a PDF diagram analyst. Analyze the attached diagram image(s).
    Specifically, find the asset numbers {asset_numbers} and describe any symbols located next to them.
    Be precise and descriptive. For example: 'Asset X is next to a [symbol description]'.
    """
    
    # This prompt asks the vision model to *interpret* the legend
    # In a more advanced setup, you would first find the symbol, then ask about it.
    # For this example, we'll ask for the symbol we expect to find.
    vision_prompt_legend = f"""
    You are a PDF legend analyst. Analyze the attached legend image(s).
    I need to know the meaning of a specific symbol.
    Please find the symbol that looks like 'a circle with spikes' (it might be labeled) and state what it represents.
    """

    # 3. Run vision analysis
    print(f"Running vision analysis on {diagram_file}...")
    real_diagram_analysis = await run_vision_analysis(
        client, vision_prompt_diagram, diagram_images, model=vision_model
    )
    
    print(f"Running vision analysis on {legend_file}...")
    real_legend_interpretation = await run_vision_analysis(
        client, vision_prompt_legend, legend_images, model=vision_model
    )
    
    print("\n--- Vision Analysis Complete ---")
    print(f"Diagram Analysis Result:\n{real_diagram_analysis}")
    print(f"Legend Interpretation Result:\n{real_legend_interpretation}")
    # --- End of Real Analysis ---


    # This variable will store the entire conversation chain
    full_response = ""
    
    # 1. Create the initial query from the parameters
    initial_query = f"Check asset numbers {asset_numbers} on diagram '{diagram_file}' and identify if there are '{asset_type_query}'."

    print("\n\n--- Agent 1: Diagram Analyst ---")
    print(f"Initial Query: {initial_query}\n")

    # 2. Create the first prompt, feeding in the REAL analysis
    prompt1 = f"Query: {initial_query}\n\nHere is the pre-computed analysis of '{diagram_file}':\n{real_diagram_analysis}\n\nPlease present this analysis clearly."
    
    result1 = Runner.run_streamed(diagram_analyst_agent, input=prompt1)
    async for event in result1.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            text_chunk = event.data.delta
            print(text_chunk, end="", flush=True)
            full_response += text_chunk

    print("\n\n--- Agent 2: Legend Interpreter ---")
    
    # 3. Create the second prompt, feeding in the REAL interpretation
    prompt2 = f"Here is the analyst's report:\n{full_response}\n\nHere is the pre-computed interpretation of '{legend_file}':\n{real_legend_interpretation}\n\nPlease state the interpretation for the symbols found."
    
    result2 = Runner.run_streamed(legend_interpreter_agent, input=prompt2)
    async for event in result2.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            text_chunk = event.data.delta
            print(text_chunk, end="", flush=True)
            full_response += text_chunk

    print("\n\n--- Agent 3: Chief Analyst (Final Conclusion) ---")
    
    # 4. Create the final prompt for the chief analyst to synthesize everything
    # (This step is unchanged)
    prompt3 = f"Based on all the following information, provide a final, conclusive answer to the initial query ({initial_query}):\n\n{full_response}"

    result3 = Runner.run_streamed(chief_analyst_agent, input=prompt3)
    async for event in result3.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            text_chunk = event.data.delta
            print(text_chunk, end="", flush=True)
            full_response += text_chunk
    
    print("\n\n--- Analysis Complete ---")

if __name__ == "__main__":

    print("Start")
    # You must have diagram_v1.pdf and legend_v1.pdf in this directory
    # or the script will fail.
    if not os.path.exists("diagram_v1.pdf") or not os.path.exists("legend_v1.pdf"):
        print("="*50)
        print("ERROR: Missing PDF files.")
        print("Please create 'diagram_v1.pdf' and 'legend_v1.pdf' and place them")
        print("in the same directory as this script to run the analysis.")
        print("="*50)
    else:
        asyncio.run(main())