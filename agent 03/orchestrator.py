import time
import uuid
import requests
import os
import json
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Orchestrator Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- State ---
# Modified to store "results" so we can pass data between steps
executions: Dict[str, dict] = {}

# --- Models ---
class StepValidation(BaseModel):
    criteria: Optional[str] = None
    critical_fail: Optional[bool] = False

class StepResources(BaseModel):
    agent_id: Optional[str] = None
    tool_id: Optional[str] = None

class PlanStep(BaseModel):
    step_number: int
    description: str
    name: Optional[str] = None
    assigned_resource_id: Optional[str] = None
    required_resources: Optional[StepResources] = None
    validation: Optional[StepValidation] = None

class ExecutionRequest(BaseModel):
    plan_overview: str
    steps: List[PlanStep]

@app.post("/execute")
async def initialize_plan(request: ExecutionRequest):
    plan_id = str(uuid.uuid4())
    executions[plan_id] = {
        "plan": request,
        "current_step_index": 0,
        "total_steps": len(request.steps),
        "is_complete": False,
        "results": {} # Store output context here (e.g., assets, legend)
    }
    return {"status": "ready", "plan_id": plan_id}

@app.post("/run_next/{plan_id}")
async def run_next_step(plan_id: str, file: UploadFile = File(None)):
    if plan_id not in executions:
        raise HTTPException(status_code=404, detail="Plan ID not found")
    
    state = executions[plan_id]
    if state["is_complete"]:
        return {"status": "completed", "message": "Plan is already finished."}
    
    idx = state["current_step_index"]
    step: PlanStep = state["plan"].steps[idx]
    
    # Identify Resource
    resource = step.assigned_resource_id
    if step.required_resources and step.required_resources.agent_id:
        resource = step.required_resources.agent_id
    
    resource_key = (resource or "").lower()
    log_message = f"[STEP {step.step_number}] Action: {step.name}\n"
    
    # --- ROUTING LOGIC ---

    # A. ASSET OPS AGENT (Port 8084)
    if "asset" in resource_key and "ops" in resource_key:
        if not file:
            return {"status": "error", "log": log_message + "   ❌ ERROR: File required!", "step_index": idx}
        
        log_message += f"   📤 Sending to Asset Ops (Port 8084)...\n"
        try:
            files = {'file': (file.filename, file.file, file.content_type)}
            resp = requests.post("http://asset-ops:8084/extract_assets", files=files, timeout=30)
            
            if resp.status_code == 200:
                data = resp.json()
                count = len(data.get('assets', []))
                log_message += f"   ✅ Extracted {count} assets.\n"
                
                # SAVE CONTEXT for Step 3
                state["results"]["asset_list"] = data
            else:
                log_message += f"   ⚠️ Agent Error: {resp.text}\n"
        except Exception as e:
            log_message += f"   ❌ Network Error: {e}\n"

    # B. CONTENT REVIEWER AGENT (Port 8085)
    elif "content" in resource_key and "reviewer" in resource_key:
        if not file:
            return {"status": "error", "log": log_message + "   ❌ ERROR: File required!", "step_index": idx}
        
        log_message += f"   🧠 Sending to Content Reviewer (Port 8085)...\n"
        
        form_data = {
            "instruction": step.description,
            "validation_criteria": step.validation.criteria if step.validation else ""
        }
        
        try:
            await file.seek(0)
            files = {'file': (file.filename, file.file, file.content_type)}
            
            resp = requests.post(
                "http://content-reviewer:8085/review_content", 
                files=files, 
                data=form_data, 
                timeout=60
            )
            
            if resp.status_code == 200:
                res_json = resp.json()
                log_message += f"   ✅ Success (Model: {res_json.get('used_model')})\n"
                log_message += f"   📝 Result: {str(res_json.get('result'))[:100]}...\n"
                
                # SAVE CONTEXT for Step 3
                state["results"]["legend"] = res_json.get('result')
            else:
                log_message += f"   ⚠️ Validation Failed or Error: {resp.text}\n"
        except Exception as e:
            log_message += f"   ❌ Network Error: {e}\n"

    # C. VERIFICATION AGENT (Port 8086) - [NEW]
    elif "verification" in resource_key:
        log_message += f"   👁️ Routing to Verification Agent (Port 8086)...\n"
        
        # 1. Retrieve Context from previous steps
        # We need the output from Step 1 (Asset List) and Step 2 (Legend)
        # Note: In a real system, we might look up by step index, but keys work here.
        asset_ctx = state["results"].get("asset_list", {})
        legend_ctx = state["results"].get("legend", {})

        payload_data = {
            "asset_list_json": json.dumps(asset_ctx),
            "legend_json": json.dumps(legend_ctx)
        }

        try:
            url = "http://agent-verification:8086/verify_assets"
            
            # 2. Call Logic: Handle File Presence
            if file:
                # User has uploaded the P&ID drawing
                await file.seek(0)
                files = {'drawing': (file.filename, file.file, file.content_type)}
                resp = requests.post(url, data=payload_data, files=files, timeout=60)
            else:
                # First pass: Call without file to see if agent requests it
                resp = requests.post(url, data=payload_data, timeout=10)

            # 3. Process Response
            if resp.status_code == 200:
                res_json = resp.json()
                
                # CHECK: Does the agent need user interaction?
                if res_json.get("status") == "interaction_required":
                    log_message += f"   🛑 Paused: {res_json.get('message')}\n"
                    
                    # Return specific status to Frontend to trigger File Upload
                    # We DO NOT increment step index here.
                    return {
                        "status": "paused",
                        "action": "request_file_upload",
                        "message": res_json.get("message"),
                        "log": log_message,
                        "step_index": idx # Stay on current step
                    }
                
                # Success
                log_message += f"   ✅ Verification Complete. Report Generated.\n"
                state["results"]["verification_report"] = res_json
            else:
                log_message += f"   ⚠️ Agent Error: {resp.text}\n"
                
        except Exception as e:
            log_message += f"   ❌ Network Error: {e}\n"

    # D. STANDARD SIMULATION
    else:
        time.sleep(1)
        log_message += "   ✅ Step Simulated (No external call).\n"

    # Advance State (Only if we didn't return 'paused' above)
    state["current_step_index"] += 1
    remaining = state["total_steps"] - state["current_step_index"]
    
    if remaining == 0:
        state["is_complete"] = True
        log_message += "\n🏁 PLAN COMPLETED."

    return {
        "status": "completed" if state["is_complete"] else "waiting",
        "step_index": idx + 1,
        "log": log_message,
        "remaining_steps": remaining
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("orchestrator:app", host="0.0.0.0", port=8001, reload=True)