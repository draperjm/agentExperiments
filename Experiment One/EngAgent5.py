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
# (These are unchanged)

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


async def run_vision_analysis(client: AsyncOpenAI, user_prompt: str, base64_images: List[str], model: str = "gpt-4o") -> str:
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

# --- MODIFIED main() FUNCTION ---

async def main():

    # --- Parameters ---
    asset_numbers: List[str] = ["2139485", "2004859", "2004858", "2004865", "2004866", "2004867"]
    
    # This file MUST exist in the same directory (or provide full paths)
    diagram_file: str = "diagram_v1.pdf" 

    # --- NEW: Initialize OpenAI Client for Vision ---
    client = AsyncOpenAI()
    vision_model = "gpt-4o" # Use the same model for consistency

    # ---
    # This is the "REAL" analysis step.
    # ---
    
    print("--- Step 0: Analyzing PDF files with Vision Model ---")

    # 1. Convert PDFs to images (Done once)
    diagram_images = pdf_to_base64_images(diagram_file)
    
    if not diagram_images:
        print(f"Stopping analysis as PDF {diagram_file} could not be processed.")
        return

    # --- NEW: Loop for first analysis ---
    print("\n--- Running Loop for Individual Asset Location Analysis ---")
    
    # This is the new array (list) to store results for *found* assets
    found_asset_analyses: List[Dict] = [] 

    for asset_number in asset_numbers:
        print(f"\n[LOOP] Analyzing asset: {asset_number}...")

        # 2. Define vision prompt for a SINGLE asset
        # We ask the model to explicitly state "NOT_FOUND" if it can't find it.
        vision_prompt_diagram_single = f"""
        You are an Engineer diagram analyst. Analyze the attached diagram image(s).
        
        Your task is to find the asset number **{asset_number}** and describe its location.
        
        1.  Carefully search the diagram for the number **{asset_number}**.
        2.  If you **find** it, describe its location precisely. 
            (e.g., 'Asset {asset_number} is located on [street] next to [object]').
        3.  If you **cannot find** the asset number **{asset_number}** on the diagram, 
            you must respond with *only* the single word: NOT_FOUND
        """

        # 3. Run vision analysis for the single asset
        single_analysis_result = await run_vision_analysis(
            client, vision_prompt_diagram_single, diagram_images, model=vision_model
        )

        # 4. Check result and add to the new array if found
        if "NOT_FOUND" not in single_analysis_result.upper():
            print(f"-> STATUS: FOUND. Result: {single_analysis_result}")
            found_asset_analyses.append({
                "asset_number": asset_number,
                "location_analysis": single_analysis_result
            })
        else:
            print(f"-> STATUS: NOT_FOUND.")
    
    print("\n--- Individual Asset Analysis Complete ---")
    print(f"Found {len(found_asset_analyses)} assets. Results:")
    # Pretty print the collected results
    import json
    print(json.dumps(found_asset_analyses, indent=2))
    # --- End of new loop ---


    # --- UNCHANGED: Second Analysis (all assets at once) ---
    # This prompt asks the vision model to *perform* the analysis
    vision_prompt_diagram_assets = f"""
    You are a Engineer diagram analyst. Analyze the attached diagram image(s).
    Specifically, find the asset numbers {found_asset_analyses} on the diagram.
    For the {found_asset_analyses} found on the diagram, describe the icon they are next to.
    Be precise and descriptive. For example: 'Asset X is located next to [symbol]'.
    """
    
    print(f"\n--- Running Second Analysis (All Assets / Icon Description) ---")
    real_diagram_analysis_assets = await run_vision_analysis(
        client, vision_prompt_diagram_assets, diagram_images, model=vision_model
    )

    print("\n--- Vision Analysis Complete ---")
    print(f"Icon Analysis Result:\n{real_diagram_analysis_assets}")
    # --- End of Real Analysis ---

    
# --- UPDATED Main Execution Block ---
if __name__ == "__main__":

    print("Start")
    
    # **Updated this check to match the file used in main()**
    if not os.path.exists("diagram_v2.pdf"):
        print("="*50)
        print("ERROR: Missing PDF file.")
        print("Please create 'diagram_v2.pdf' and place it")
        print("in the same directory as this script to run the analysis.")
        print("="*50)
    else:
        asyncio.run(main())