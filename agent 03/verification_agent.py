import os
import json
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from dotenv import load_dotenv
from datetime import datetime
import google.generativeai as genai

# Load environment variables
load_dotenv()

app = FastAPI(title="Asset Verification Agent")

# --- Logging Setup ---
logger = logging.getLogger("Verification_Agent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# --- AI Client Setup ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY not found. Agent may fail on visual tasks.")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

# --- Logic ---
def verify_with_gemini(file_bytes: bytes, file_type: str, asset_list: List[Dict], legend: List[Dict]) -> Dict:
    """
    Sends the drawing + data to Gemini 1.5 Pro/Flash for visual cross-referencing.
    """
    model_name = "gemini-2.5-flash" 
    model = genai.GenerativeModel(model_name)
    
    asset_str = json.dumps(asset_list, indent=2)
    legend_str = json.dumps(legend, indent=2)
    
    system_prompt = (
        "You are a Senior QA Engineer validating P&ID (Piping and Instrumentation Diagrams).\n"
        "Your Task: Cross-reference an Asset List against the provided Drawing, using a Legend for visual verification.\n\n"
        "INPUT DATA:\n"
        f"1. ASSET LIST (Expected items): {asset_str}\n"
        f"2. LEGEND (Visual Definitions): {legend_str}\n\n"
        "INSTRUCTIONS:\n"
        "1. Search the Drawing for every 'asset_number' in the Asset List.\n"
        "2. If found, look at the symbol visually associated with that number.\n"
        "3. Check if that symbol matches the description in the Legend for that 'asset_type'.\n"
        "4. Return a JSON Report."
    )
    
    output_req = (
        "\nOUTPUT FORMAT:\n"
        "Return a single JSON Object with a key 'verification_report' containing a list of objects:\n"
        "{\n"
        "  'verification_report': [\n"
        "    {\n"
        "      'asset_number': 'P-101',\n"
        "      'status': 'VERIFIED' | 'MISSING_ON_DRAWING' | 'ICON_MISMATCH',\n"
        "      'observed_symbol': 'Description of what you actually see',\n"
        "      'confidence': 'High' | 'Medium' | 'Low'\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "Do not use Markdown formatting."
    )

    mime_type = "application/pdf" if file_type == "pdf" else "image/png"
    if file_type not in ['pdf', 'png', 'jpg', 'jpeg']:
        mime_type = "application/pdf"

    prompt_parts = [
        system_prompt + output_req,
        {"mime_type": mime_type, "data": file_bytes}
    ]

    logger.info(f"Sending request to {model_name}...")
    
    try:
        response = model.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        logger.error(f"Gemini API Error: {e}")
        raise HTTPException(status_code=500, detail=f"AI Processing Failed: {e}")

# --- Endpoints ---

@app.post("/verify_assets")
async def verify_assets(
    # Make drawing Optional so we can ask for it if it's missing
    drawing: Optional[UploadFile] = File(None),
    asset_list_json: str = Form(..., description="JSON string of the asset list"),
    legend_json: str = Form(..., description="JSON string of the legend")
):
    """
    Main Verification Endpoint.
    1. Checks if Drawing is present. 
    2. If NOT, returns a 'request_file' action to the Frontend.
    3. If YES, proceeds with AI verification.
    """
    print(f"\n[VERIFIER] 📥 Request Received. Checking for file...")

    # --- 1. Interaction Logic: Request File from User ---
    if not drawing:
        print("[VERIFIER] ⚠️ No file provided. Requesting upload from User...")
        return {
            "status": "interaction_required",
            "action": "request_file_upload",
            "message": "Please upload the P&ID Drawing (PDF or Image) to proceed with verification.",
            "required_fields": ["drawing"]
        }

    # --- 2. Processing Logic (File is present) ---
    print(f"[VERIFIER] 📄 Processing {drawing.filename}...")
    
    # Parse Inputs
    try:
        assets = json.loads(asset_list_json)
        legend = json.loads(legend_json)
        if isinstance(assets, dict) and "assets" in assets:
            assets = assets["assets"]
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON input: {e}")

    # Read File
    content = await drawing.read()
    file_ext = drawing.filename.split('.')[-1].lower()

    # AI Verification
    raw_response = verify_with_gemini(content, file_ext, assets, legend)
    
    # Clean Output
    clean_json = raw_response.replace("```json", "").replace("```", "").strip()
    
    try:
        parsed_report = json.loads(clean_json)
    except json.JSONDecodeError:
        parsed_report = {"raw_output": clean_json, "status": "parse_error"}

    # Save Output
    output_dir = "OUTPUT"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(output_dir, f"Verification_Report_{timestamp}.json")
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(parsed_report, f, indent=2)
        
    print(f"[VERIFIER] ✅ Report generated: {output_filename}")
    
    return {
        "status": "success", 
        "report": parsed_report,
        "output_file": output_filename
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8086)