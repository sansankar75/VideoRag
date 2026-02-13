from langchain.tools import tool
import json, math

JSON_FILE = "cleaned_detection.json"        # vector db used to find the data in db and json file is used to collect data from db
EMBED_DB_FILE = "embedded_objects.json"

def load_raw_data():
    with open(JSON_FILE) as f:
        return json.load(f)

def load_vector_db():
    try:
        with open(EMBED_DB_FILE) as f:
            return json.load(f)
    except:
        return []

def save_vector_db(db):
    with open(EMBED_DB_FILE, "w") as f:
        json.dump(db, f, indent=2)

def cosine(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(y*y for y in b))
    if norm_a==0 or norm_b==0:
        return 0
    return dot/(norm_a*norm_b)

# ---------------- Tools ---------------- #

@tool
def get_frame(frame_id: int):
    """Return full metadata for a frame by frame_id"""
    for frame in load_raw_data():
        if frame.get("frame") == frame_id:
            return frame
    return {"error": "frame not found"}

@tool
def get_objects(frame_id: int):
    """Return all objects in a specific frame"""
    for frame in load_raw_data():
        if frame.get("frame") == frame_id:
            return frame.get("objects", [])
    return []

@tool
def get_object_by_unique_id(frame_id: int, object_id: int):
    """Return a single object by its UNIQUE combination of frame_id and object_id"""
    for frame in load_raw_data():
        if frame.get("frame") == frame_id:
            for obj in frame.get("objects", []):
                if obj.get("id") == object_id:
                    return {
                        "frame": frame_id,
                        "object": obj
                    }
    return {"error": "object not found"}

@tool
def search_objects_by_label(label: str):
    """Return all objects with the given label across all frames"""
    results = []
    for frame in load_raw_data():
        for obj in frame.get("objects", []):
            if obj.get("label") == label:
                results.append({
                    "frame": frame.get("frame"), 
                    "object_id": obj.get("id"),
                    "object": obj
                })
    return results

@tool
def get_object_bbox(frame_id: int, object_id: int):
    """Get bounding box coordinates for a specific object"""
    obj_data = get_object_by_unique_id.invoke({"frame_id": frame_id, "object_id": object_id})
    if "error" in obj_data:
        return obj_data
    return obj_data["object"].get("bbox", {})

@tool
def get_object_confidence(frame_id: int, object_id: int):
    """Get confidence score for a specific object detection"""
    obj_data = get_object_by_unique_id.invoke({"frame_id": frame_id, "object_id": object_id})
    if "error" in obj_data:
        return obj_data
    return {"confidence": obj_data["object"].get("confidence")}

@tool
def get_object_state(frame_id: int, object_id: int):
    """Get the state (static/moving) of a specific object"""
    obj_data = get_object_by_unique_id.invoke({"frame_id": frame_id, "object_id": object_id})
    if "error" in obj_data:
        return obj_data
    return {
        "state": obj_data["object"].get("state"),
        "velocity": obj_data["object"].get("velocity")
    }

@tool
def filter_objects_by_confidence(min_confidence: float):
    """Return all objects with confidence >= min_confidence"""
    results = []
    for frame in load_raw_data():
        for obj in frame.get("objects", []):
            if obj.get("confidence", 0) >= min_confidence:
                results.append({
                    "frame": frame.get("frame"),
                    "object_id": obj.get("id"),
                    "label": obj.get("label"),
                    "confidence": obj.get("confidence")
                })
    return results

@tool
def filter_objects_by_state(state: str):
    """Return all objects with specific state (static/moving)"""
    results = []
    for frame in load_raw_data():
        for obj in frame.get("objects", []):
            if obj.get("state") == state:
                results.append({
                    "frame": frame.get("frame"),
                    "object_id": obj.get("id"),
                    "label": obj.get("label"),
                    "state": obj.get("state"),
                    "velocity": obj.get("velocity")
                })
    return results

@tool
def filter_objects_by_area(min_area: int = 0, max_area: int = 999999):
    """Return all objects within a specific area range"""
    results = []
    for frame in load_raw_data():
        for obj in frame.get("objects", []):
            area = obj.get("area", 0)
            if min_area <= area <= max_area:
                results.append({
                    "frame": frame.get("frame"),
                    "object_id": obj.get("id"),
                    "label": obj.get("label"),
                    "area": area
                })
    return results

@tool
def get_all_labels():
    """Get a list of all unique object labels in the dataset"""
    labels = set()
    for frame in load_raw_data():
        for obj in frame.get("objects", []):
            labels.add(obj.get("label"))
    return {"labels": sorted(list(labels))}

@tool
def count_objects_by_label():
    """Count how many times each label appears across all frames"""
    label_counts = {}
    for frame in load_raw_data():
        for obj in frame.get("objects", []):
            label = obj.get("label")
            label_counts[label] = label_counts.get(label, 0) + 1
    return label_counts

@tool
def get_frame_range():
    """Get the range of frames in the dataset (min and max frame numbers)"""
    frames = [frame.get("frame") for frame in load_raw_data()]
    if not frames:
        return {"error": "no frames found"}
    return {
        "min_frame": min(frames),
        "max_frame": max(frames),
        "total_frames": len(frames)
    }

@tool
def get_objects_in_frame_range(start_frame: int, end_frame: int):
    """Get all objects within a specific frame range"""
    results = []
    for frame in load_raw_data():
        frame_num = frame.get("frame")
        if start_frame <= frame_num <= end_frame:
            for obj in frame.get("objects", []):
                results.append({
                    "frame": frame_num,
                    "object_id": obj.get("id"),
                    "label": obj.get("label")
                })
    return results

@tool
def search_objects_by_bbox_size(min_width: int = 0, min_height: int = 0):
    """Find objects with bounding box dimensions larger than specified"""
    results = []
    for frame in load_raw_data():
        for obj in frame.get("objects", []):
            bbox = obj.get("bbox", {})
            if bbox.get("w", 0) >= min_width and bbox.get("h", 0) >= min_height:
                results.append({
                    "frame": frame.get("frame"),
                    "object_id": obj.get("id"),
                    "label": obj.get("label"),
                    "bbox": bbox
                })
    return results

@tool
def get_static_objects():
    """Get all static objects (velocity = 0)"""
    return filter_objects_by_state.invoke({"state": "static"})

@tool
def get_moving_objects():
    """Get all moving objects (velocity > 0)"""
    results = []
    for frame in load_raw_data():
        for obj in frame.get("objects", []):
            if obj.get("velocity", 0) > 0:
                results.append({
                    "frame": frame.get("frame"),
                    "object_id": obj.get("id"),
                    "label": obj.get("label"),
                    "velocity": obj.get("velocity")
                })
    return results

# ---------------- Vector DB Tools ---------------- #

@tool
def store_embedding(frame_id: int, object_id: int, embedding: list):
    """Store an object's embedding in the vector DB using unique frame_id + object_id"""
    db = load_vector_db()
    unique_id = f"{frame_id}_{object_id}"
    db.append({
        "unique_id": unique_id,
        "frame_id": frame_id,
        "object_id": object_id,
        "embedding": embedding
    })
    save_vector_db(db)
    return {"status": "saved", "unique_id": unique_id}

@tool
def search_embedding(query_vector: list, top_k: int = 5):
    """Search the vector DB for top-K similar embeddings"""
    db = load_vector_db()
    if not db:
        return {"error": "vector database is empty"}
    
    scored = []
    for item in db:
        score = cosine(query_vector, item.get("embedding", []))
        scored.append({
            "unique_id": item.get("unique_id"),
            "frame_id": item.get("frame_id"),
            "object_id": item.get("object_id"),
            "score": score
        })
    scored = sorted(scored, key=lambda x: x["score"], reverse=True)
    return scored[:top_k]

@tool
def search_vector_db(query: str, top_k: int = 3):
    """Search the vector database for relevant objects based on text query"""
    from langchain.embeddings import HuggingFaceEmbeddings
    
    db = load_vector_db()
    if not db:
        return {"error": "vector database is empty"}
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    query_vector = embeddings.embed_query(query)
    
    scored = []
    for item in db:
        score = cosine(query_vector, item.get("embedding", []))
        scored.append({
            "unique_id": item.get("unique_id"),
            "frame_id": item.get("frame_id"),
            "object_id": item.get("object_id"),
            "score": score
        })
    
    scored = sorted(scored, key=lambda x: x["score"], reverse=True)
    top_results = scored[:top_k]
    
    # Enrich with actual object data from JSON
    enriched_results = []
    for result in top_results:
        obj_data = get_object_by_unique_id.invoke({
            "frame_id": result["frame_id"],
            "object_id": result["object_id"]
        })
        enriched_results.append({
            "unique_id": result["unique_id"],
            "frame_id": result["frame_id"],
            "object_id": result["object_id"],
            "similarity_score": result["score"],
            "object_data": obj_data
        })
    
    return enriched_results