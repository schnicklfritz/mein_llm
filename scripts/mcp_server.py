import os
import subprocess
import httpx
import chromadb
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from typing import Optional

app = FastAPI(title="ComfyUI MCP Server", version="0.2.0")

# ── Clients ───────────────────────────────────────────────────────────────────
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "comfy-llm")
COMFYUI_URL = os.getenv("COMFYUI_URL", "")

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "comfyui_workflows")

deepseek = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", ""),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)

def get_chroma_collection():
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    return client.get_or_create_collection(name=CHROMA_COLLECTION)


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "version": "0.2.0"}


# ── Tool: search_workflows ────────────────────────────────────────────────────
class WorkflowSearchRequest(BaseModel):
    query: str
    n_results: int = 5

@app.post("/tools/search_workflows")
async def search_workflows(req: WorkflowSearchRequest):
    """
    Semantic search against ChromaDB workflow corpus.
    Returns filepath + summary for each match.
    LLM should load the full JSON from filepath when needed.
    """
    try:
        collection = get_chroma_collection()
        results = collection.query(
            query_texts=[req.query],
            n_results=req.n_results,
            include=["documents", "metadatas", "distances"]
        )
        matches = []
        for i in range(len(results["ids"][0])):
            matches.append({
                "id": results["ids"][0][i],
                "summary": results["documents"][0][i],
                "filepath": results["metadatas"][0][i].get("path", ""),
                "filename": results["metadatas"][0][i].get("filename", ""),
                "source": results["metadatas"][0][i].get("source", ""),
                "score": round(1 - results["distances"][0][i], 4),
            })
        return {"query": req.query, "results": matches}
    except Exception as e:
        return {"error": str(e)}


# ── Tool: list_models ─────────────────────────────────────────────────────────
@app.get("/tools/list_models")
async def list_models():
    """
    List available models from ComfyUI /object_info endpoint.
    Returns checkpoints, loras, vaes, and other model types.
    """
    if not COMFYUI_URL:
        return {"error": "COMFYUI_URL not set"}
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{COMFYUI_URL}/object_info",
                timeout=30,
            )
        data = resp.json()
        # Extract model lists from CheckpointLoaderSimple and related nodes
        models = {}
        for node_name, node_info in data.items():
            if not isinstance(node_info, dict):
                continue
            inputs = node_info.get("input", {}).get("required", {})
            for param, param_info in inputs.items():
                if isinstance(param_info, list) and len(param_info) > 0:
                    if isinstance(param_info[0], list) and param in (
                        "ckpt_name", "lora_name", "vae_name",
                        "unet_name", "clip_name", "control_net_name",
                        "upscale_model_name", "model_name"
                    ):
                        if param not in models:
                            models[param] = param_info[0]
        return {"models": models}
    except Exception as e:
        return {"error": str(e)}


# ── Tool: download_model ──────────────────────────────────────────────────────
class DownloadRequest(BaseModel):
    url: str
    destination: str  # e.g. /workspace/models/checkpoints/
    filename: Optional[str] = None  # override filename, else inferred from URL
    hf_token: Optional[str] = None  # for private HuggingFace models

@app.post("/tools/download_model")
async def download_model(req: DownloadRequest):
    """
    Download a model file using aria2c.
    Runs async in background — returns immediately with status.
    """
    os.makedirs(req.destination, exist_ok=True)
    cmd = ["aria2c", "-x16", "-s16", "-d", req.destination]
    if req.filename:
        cmd += ["-o", req.filename]
    if req.hf_token:
        cmd += [f"--header=Authorization: Bearer {req.hf_token}"]
    cmd.append(req.url)
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return {
            "status": "started",
            "pid": proc.pid,
            "destination": req.destination,
            "url": req.url,
        }
    except Exception as e:
        return {"error": str(e)}


# ── Tool: generate_prompt ─────────────────────────────────────────────────────
class PromptRequest(BaseModel):
    description: str
    model_type: str = "flux"   # flux | wan22 | chroma | hunyuan
    use_deepseek: bool = False

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
