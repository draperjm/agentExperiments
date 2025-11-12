import asyncio
import base64
import io
import os
import json  # Import json for parsing
from dotenv import load_dotenv
from agents import Agent, Runner, trace  # Assuming these are in a local file
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent
from pdf2image import convert_from_path  # To read PDFs as images
from typing import Dict, List

# Load environment variables
load_dotenv(override=True)

# --- Parameters ---
# **MOVED TO GLOBAL SCOPE** so it can be accessed by main() and the startup check
diagram_file: str = "diagram_v1.pdf"

# --- Agent Definitions ---
# (Unchanged)
instructions1 = " \
    You are a Emgineering Diagram Analyst agent. Your job is to receive a query, \
    a list of asset numbers, and a **pre-computed analysis** of a diagram. \
    You must first state the query's goal. Then, you will clearly present the provided analysis \
    for the requested asset numbers."

diagram_analyst_agent = Agent(
    name="Engineer Diagram Analyst Agent",
    instructions=instructions1,
    model="gpt-4o"
)

# --- HELPER FUNCTIONS ---

def pdf_to_base64_images(pdf_path: str) -> List[str]:
    """Converts each page of a PDF into a list of base64 encoded image strings."""
    print(f"Converting PDF: {pdf_path}")
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return []

    try:
        # **IMPROVEMENT 1: Increase DPI for better accuracy**
        images = convert_from_path(
            pdf_path,
            dpi=300,  # Increased from default (200) to 300
            fmt="png",
            thread_count=4
        )
        
        base64_list = []
        for image in images:
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            base64_list.append(img_str)
            
        print(f"Successfully converted {pdf_path} to {len(base64_list)} image(s) at 300 DPI.")
        return base64_list
    except Exception as e:
        print(f"Error converting PDF {pdf_path}: {e}")
        print("Please ensure 'poppler' is installed and in your system's PATH.")
        return []


async def run_vision_analysis(client: AsyncOpenAI, user_prompt: str, base64_images: List[str], model: str = "gpt-4o", use_json: bool = False) -> str:
    """
    Runs a multimodal vision analysis on a list of images and returns the text result.
    
    **IMPROVEMENT 2: Added 'use_json' flag and 'detail: high'**
    """
    if not base64_images:
        return f"Error: No images provided for analysis. (Source prompt: {user_prompt})"

    # Build the content list for the API
    content: List[Dict] = [{"type": "text", "text": user_prompt}]
    for img_str in base64_images:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_str}",
                "detail": "high"  # Ask model to use high-res image
            }
        })
    
    # Build request arguments
    request_args = {
        "model": model,
        "messages": [
            {"role": "user", "content": content}
        ],
        "max_tokens": 1024
    }

    # Enable JSON mode if requested
    if use_json:
        request_args["response_format"] = {"type": "json_object"}

    try:
        response = await client.chat.completions.create(**request_args)
        result = response.choices[0].message.content
        return result if result else "No analysis returned from vision model."
    except Exception as e:
        print(f"Error during vision analysis: {e}")
        return f"Error analyzing images: {e}"

# --- MODIFIED main() FUNCTION ---

async def main():

    # --- Parameters ---"
    asset_numbers: List[str] = ["2139485", "2004859", "2004858", "2004865", "2004866", "2004867", "2004855", "2004856", "2004854", "2004853", "9999999"]
    
    # Note: 'diagram_file' is now defined globally
    
    # --- Initialize OpenAI Client for Vision ---
    client = AsyncOpenAI()
    vision_model = "gpt-4o"

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
    print("\n--- Running Loop for Individual Asset Location Analysis (with JSON) ---")
    
    found_asset_analyses: List[Dict] = [] 

    for asset_number in asset_numbers:
        print(f"\n[LOOP] Analyzing asset: {asset_number}...")

        # **IMPROVEMENT 2: Updated prompt to request JSON**
        vision_prompt_diagram_single = f"""
        You are an Engineer diagram analyst. Analyze the attached diagram image(s).
        Your task is to find the asset number **{asset_number}**.
        Conduct a second review to ensure the status of "found" is accurate.

        Respond in a valid JSON format.
        The JSON object must have two keys:
        1. "found": A boolean (true/false) indicating if you found the asset.
        2. "location_description": A string describing the location. If not found, this must be null.

        Example (if found):
        {{
          "found": true,
          "location_description": "Asset {asset_number} is on Street X, next to a transformer."
        }}

        Example (if not found):
        {{
          "found": false,
          "location_description": null
        }}
        """

        # 3. Run vision analysis for the single asset, requesting JSON
        single_analysis_result_str = await run_vision_analysis(
            client, vision_prompt_diagram_single, diagram_images, model=vision_model,
            use_json=True  # <-- Tell the runner to enable JSON mode
        )

        # 4. Check result by parsing JSON (much more reliable)
        try:
            analysis_data = json.loads(single_analysis_result_str)
            
            if analysis_data.get("found") == True:
                location = analysis_data.get("location_description", "Found but no description.")
                print(f"-> STATUS: FOUND. Result: {location}")
                found_asset_analyses.append({
                    "asset_number": asset_number,
                    "location_analysis": location
                })
            else:
                print(f"-> STATUS: NOT_FOUND (as per JSON).")

        except json.JSONDecodeError:
            print(f"-> STATUS: ERROR. Failed to decode JSON from model: {single_analysis_result_str}")
    
    print("\n--- Individual Asset Analysis Complete ---")
    print(f"Found {len(found_asset_analyses)} assets. Results:")
    print(json.dumps(found_asset_analyses, indent=2))
    # --- End of new loop ---


    # --- UNCHANGED: Second Analysis (all assets at once) ---

    # **IMPROVEMENT 3: Clean up asset list for the prompt**
    found_numbers = [asset["asset_number"] for asset in found_asset_analyses]

    if not found_numbers:
        print("\nNo assets were found. Skipping second analysis.")
        return

    print(f"\n--- Running Second Analysis on {len(found_numbers)} found assets ---")

    vision_prompt_diagram_assets = f"""
    You are a Engineer diagram analyst. Analyze the attached diagram image(s).
    Your task is to analyze the following asset numbers: **{found_numbers}**.

    For each asset number in that list, describe the icon or symbol it is next to on the diagram.
    Be precise and descriptive. For example: 'Asset X is located next to [symbol]'.
    """
    
    real_diagram_analysis_assets = await run_vision_analysis(
        client, vision_prompt_diagram_assets, diagram_images, model=vision_model
    )

    print("\n--- Vision Analysis Complete ---")
    print(f"Icon Analysis Result:\n{real_diagram_analysis_assets}")
    # --- End of Real Analysis ---

    
# --- UPDATED Main Execution Block ---
if __name__ == "__main__":

    print("Start")
    
    # **IMPROVEMENT 4: Fixed bug in file check**
    # This now correctly uses the global 'diagram_file' variable
    if not os.path.exists(diagram_file):
        print("="*50)
        print("ERROR: Missing PDF file.")
        print(f"Please create '{diagram_file}' and place it")
        print("in the same directory as this script to run the analysis.")
        print("="*50)
    else:
        asyncio.run(main())