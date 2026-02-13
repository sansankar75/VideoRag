import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --------------------------
# CONFIG
# --------------------------
JSON_FILE = "newData.json"
OUTPUT_DB = "video_rag_faiss_index"

# Local embedding model — NO API KEY required
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --------------------------
# Load JSON file
# --------------------------
with open(JSON_FILE, "r") as f:
    data = json.load(f)

texts = []
metadatas = []

# --------------------------
# Convert each object → embedding text
# --------------------------
for frame in data:
    frame_id = frame.get("frame")

    for obj in frame.get("objects", []):
        obj_id = obj.get("id")
        label = obj.get("label")
        bbox = obj.get("bbox")

        # This text is what will be embedded
        text = f"frame={frame_id}; object_id={obj_id}; label={label}; bbox={bbox}"

        texts.append(text)
        metadatas.append({
            "frame_id": frame_id,
            "object_id": obj_id,
            "label": label,
            "bbox": bbox
        })

# --------------------------
# Build FAISS DB locally
# --------------------------
db = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

db.save_local(OUTPUT_DB)

print(f"[OK] Stored {len(texts)} embeddings → {OUTPUT_DB}")
