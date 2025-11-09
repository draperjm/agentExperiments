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

instructions1 = " \
    You are a Emgineering Diagram Analyst agent. Your job is to receive a query, \
    a list of asset numbers, and a **pre-computed analysis** of a diagram. \
    You must first state the query's goal. Then, you will clearly present the provided analysis \
    for the requested asset numbers."

# --- Agent Initialization ---
# (Unchanged)

diagram_analyst_agent = Agent(
    name="Engineer Diagram Analyst Agent",
    instructions=instructions1,
    model="gpt-4o"
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


async def main():

    # --- Parameters ---
    asset_numbers: List[str] = ["2139485", "2004859", "2004858", "2004865", "2004866", "2004867"]
    asset_type_query: str = "new column"
    
    # These files MUST exist in the same directory (or provide full paths)
    diagram_file: str = "diagram_v2.pdf" 

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

    # 2. Define vision prompts
    # This prompt asks the vision model to *perform* the analysis
    vision_prompt_diagram = f"""
    You are a Engineer diagram analyst. Analyze the attached diagram image(s).
    Specifically, find the asset numbers {asset_numbers} and describe their location on the diagram.
    Be precise and descriptive. For example: 'Asset X is located on [street] or next to [onject]'.
    Also highlight any {asset_numbers} not found on the diagram.
    """

    # This prompt asks the vision model to *perform* the analysis
    vision_prompt_diagram_assets = f"""
    You are a Engineer diagram analyst. Analyze the attached diagram image(s).
    Specifically, find the asset numbers {asset_numbers} on the diagram.
    For the {asset_numbers} found on the diagram, describe the icon they are next to.
    Be precise and descriptive. For example: 'Asset X is located next to [symbol]'.
    """

    # 3. Run vision analysis
    print(f"Running vision analysis on {diagram_file}...")
    real_diagram_analysis = await run_vision_analysis(
        client, vision_prompt_diagram, diagram_images, model=vision_model
    )

    print(f"Running vision analysis on {diagram_file}...")
    real_diagram_analysis_assets = await run_vision_analysis(
        client, vision_prompt_diagram_assets, diagram_images, model=vision_model
    )

    print("\n--- Vision Analysis Complete ---")
    print(f"Diagram Analysis Result:\n{real_diagram_analysis}")
    print(f"Diagram Analysis Result:\n{real_diagram_analysis_assets}")
    # --- End of Real Analysis ---

    
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