import os
import json
import glob
import chromadb
from chromadb.utils import embedding_functions

WORKFLOWS_DIR = "/workspace/rag/workflows"
COMFY_ORG_DIR = "/workspace/rag/comfy-org-templates"
BATCH_SIZE = 50

# ── ChromaDB client ───────────────────────────────────────────────────────────
client = chromadb.HttpClient(host="localhost", port=8000)

ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434",
    model_name="nomic-embed-text"
)

collection = client.get_or_create_collection(
    name="comfyui_workflows",
    embedding_function=ef
)

# ── Load Comfy-Org AI descriptions ────────────────────────────────────────────
# The workflow_templates repo has .md or frontmatter descriptions per workflow
# Structure: src/content/workflows/<name>.md or similar
def load_comfy_org_descriptions():
    descriptions = {}
    if not os.path.isdir(COMFY_ORG_DIR):
        print(f"[WARN] Comfy-Org templates dir not found: {COMFY_ORG_DIR}")
        return descriptions

    # Search for markdown files with descriptions
    for md_path in glob.glob(f"{COMFY_ORG_DIR}/**/*.md", recursive=True):
        try:
            with open(md_path) as f:
                content = f.read()
            # Extract name from filename
            name = os.path.splitext(os.path.basename(md_path))[0]
            # Use full markdown content as description (frontmatter + body)
            if len(content.strip()) > 20:
                descriptions[name.lower()] = content.strip()
        except Exception:
            pass

    # Also check for JSON metadata files with description fields
    for json_path in glob.glob(f"{COMFY_ORG_DIR}/**/*.json", recursive=True):
        try:
            with open(json_path) as f:
                data = json.load(f)
            name = os.path.splitext(os.path.basename(json_path))[0]
            # Look for description field in various locations
            desc = (
                data.get("description") or
                data.get("meta", {}).get("description") or
                data.get("extra", {}).get("description")
            )
            if desc and isinstance(desc, str) and len(desc) > 20:
                descriptions[name.lower()] = desc
        except Exception:
            pass

    print(f"[INFO] Loaded {len(descriptions)} Comfy-Org descriptions")
    return descriptions


# ── Workflow validation ───────────────────────────────────────────────────────
def is_comfyui_workflow(data):
    if not isinstance(data, dict):
        return False
    keys = list(data.keys())
    if any(k.isdigit() for k in keys):
        return True
    if "nodes" in data and "links" in data:
        return True
    return False


# ── Fallback summary from node types ─────────────────────────────────────────
def workflow_to_summary(filepath, data):
    name = os.path.splitext(os.path.basename(filepath))[0]
    node_types = []

    if "nodes" in data:
        for node in data["nodes"]:
            t = node.get("type", "")
            if t:
                node_types.append(t)
    else:
        for node in data.values():
            if isinstance(node, dict):
                t = node.get("class_type", "")
                if t:
                    node_types.append(t)

    unique_nodes = list(dict.fromkeys(node_types))  # preserve order, dedupe
    node_str = ", ".join(unique_nodes[:30]) if unique_nodes else "unknown"

    return (
        f"Workflow: {name}. "
        f"Contains {len(node_types)} nodes including: {node_str}. "
        f"ComfyUI workflow for AI image or video generation."
    )


# ── Main seeding loop ─────────────────────────────────────────────────────────
def main():
    descriptions = load_comfy_org_descriptions()

    json_files = []
    for root, dirs, files in os.walk(WORKFLOWS_DIR):
        for f in files:
            if f.endswith(".json"):
                fp = os.path.join(root, f)
                json_files.append(fp)

    print(f"[INFO] Found {len(json_files)} JSON files")

    success, failed, skipped, used_description = 0, 0, 0, 0
    batch_docs, batch_ids, batch_metas = [], [], []

    for i, filepath in enumerate(json_files):
        try:
            with open(filepath) as f:
                data = json.load(f)

            if not is_comfyui_workflow(data):
                skipped += 1
                continue

            name = os.path.splitext(os.path.basename(filepath))[0]
            name_lower = name.lower()

            # Use Comfy-Org description if available, else fallback summary
            if name_lower in descriptions:
                text = f"Workflow: {name}.\n{descriptions[name_lower]}"
                used_description += 1
            else:
                text = workflow_to_summary(filepath, data)

            doc_id = f"workflow_{i}"
            meta = {
                "filename": os.path.basename(filepath),
                "path": filepath,
                "source": filepath.split("/")[-2],
                "has_description": name_lower in descriptions,
            }

            batch_docs.append(text)
            batch_ids.append(doc_id)
            batch_metas.append(meta)

            if len(batch_docs) >= BATCH_SIZE:
                collection.upsert(
                    documents=batch_docs,
                    ids=batch_ids,
                    metadatas=batch_metas
                )
                success += len(batch_docs)
                batch_docs, batch_ids, batch_metas = [], [], []
                print(f"[INFO] Progress: {i+1}/{len(json_files)} | "
                      f"Success: {success} | Skipped: {skipped} | "
                      f"With description: {used_description}")

        except Exception as e:
            failed += 1
            print(f"[ERROR] {filepath}: {e}")

    # Final batch
    if batch_docs:
        collection.upsert(
            documents=batch_docs,
            ids=batch_ids,
            metadatas=batch_metas
        )
        success += len(batch_docs)

    print(f"\n[DONE] Success: {success} | Failed: {failed} | "
          f"Skipped (non-workflow): {skipped}")
    print(f"[DONE] Used AI descriptions: {used_description} / {success}")
    print(f"[DONE] Collection count: {collection.count()}")


if __name__ == "__main__":
    main()
