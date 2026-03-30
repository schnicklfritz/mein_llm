"""
mcp_server.py — ComfyUI MCP Server
Exposes tools for the LLM to interact with ComfyUI and supporting services.

Tools (planned):
  - search_workflows    : RAG query against ChromaDB workflow corpus
  - run_workflow        : POST workflow JSON to ComfyUI /prompt
  - list_models         : List available models from ComfyUI /object_info
  - download_model      : aria2c wrapper for model downloads
  - generate_prompt     : Use DeepSeek API for complex prompt engineering

Actor routing:
  - Fast/simple tasks   → local comfy-llm (Ollama)
  - Complex reasoning   → DeepSeek API (cheap, capable)
"""

import os
import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="ComfyUI MCP Server", version="0.1.0")

# ── Clients ───────────────────────────────────────────────────────────────────
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "comfy-llm")
COMFYUI_URL = os.getenv("COMFYUI_URL", "")  # Set at runtime via env

deepseek = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", ""),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)

# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


# ── Tool: generate_prompt ─────────────────────────────────────────────────────
class PromptRequest(BaseModel):
    description: str
    model_type: str = "flux"   # flux | wan22 | chroma | hunyuan
    use_deepseek: bool = False  # True = route to DeepSeek for complex prompts

@app.post("/tools/generate_prompt")
async def generate_prompt(req: PromptRequest):
    system = (
        f"You are an expert ComfyUI prompt engineer specializing in {req.model_type} models. "
        "Output only the prompt text, no explanations."
    )
    user = f"Write a detailed generation prompt for: {req.description}"

    if req.use_deepseek:
        resp = deepseek.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return {"prompt": resp.choices[0].message.content, "actor": "deepseek"}
    else:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{OLLAMA_BASE}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "stream": False,
                },
                timeout=120,
            )
        data = resp.json()
        return {"prompt": data["message"]["content"], "actor": "ollama"}


# ── Tool: run_workflow ────────────────────────────────────────────────────────
class WorkflowRequest(BaseModel):
    workflow: dict

@app.post("/tools/run_workflow")
async def run_workflow(req: WorkflowRequest):
    if not COMFYUI_URL:
        return {"error": "COMFYUI_URL not set"}
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{COMFYUI_URL}/prompt",
            json={"prompt": req.workflow},
            timeout=30,
        )
    return resp.json()


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("MCP_PORT", 3000)),
    )
