import json
from collections import defaultdict
import math

# ---------------- CONFIG ----------------
JSON_INPUT = "detection.json"
JSON_OUTPUT = "cleaned_detection.json"
RELATION_DISTANCE = 100  # pixels threshold for "near/far"
# ----------------------------------------

# Load JSON
with open(JSON_INPUT, "r") as f:
    data = json.load(f)

# ---------------- Helpers ----------------
def compute_area(bbox):
    return bbox["w"] * bbox["h"]

def compute_velocity(prev_bbox, curr_bbox, fps=30):
    # Simple Euclidean distance / frame difference
    dx = (curr_bbox["x"] + curr_bbox["w"]/2) - (prev_bbox["x"] + prev_bbox["w"]/2)
    dy = (curr_bbox["y"] + curr_bbox["h"]/2) - (prev_bbox["y"] + prev_bbox["h"]/2)
    distance = math.sqrt(dx**2 + dy**2)
    return distance * fps  # approximate pixels per second

def compute_state(velocity):
    if velocity < 10:
        return "static"
    elif velocity < 200:
        return "slow"
    else:
        return "moving"

def compute_relations(objects):
    relations = []
    for i, obj1 in enumerate(objects):
        for j, obj2 in enumerate(objects):
            if i >= j:
                continue
            dx = (obj1["bbox"]["x"] + obj1["bbox"]["w"]/2) - (obj2["bbox"]["x"] + obj2["bbox"]["w"]/2)
            dy = (obj1["bbox"]["y"] + obj1["bbox"]["h"]/2) - (obj2["bbox"]["y"] + obj2["bbox"]["h"]/2)
            dist = math.sqrt(dx**2 + dy**2)
            if dist < RELATION_DISTANCE:
                relations.append({
                    "type": "near",
                    "obj1": obj1["id"],
                    "obj2": obj2["id"],
                    "distance": round(dist, 2)
                })
    return relations

# ---------------- Process ----------------
prev_objects = {}  # store previous frame bbox per object id
cleaned_data = []

for frame in data:
    frame_objects = []
    seen_ids = set()  # to remove duplicate IDs in same frame

    for obj in frame.get("objects", []):
        obj_id = obj["id"]
        if obj_id in seen_ids:
            continue  # skip duplicate
        seen_ids.add(obj_id)

        # Compute area if missing
        if "area" not in obj:
            obj["area"] = compute_area(obj["bbox"])

        # Compute velocity
        if obj_id in prev_objects:
            obj["velocity"] = compute_velocity(prev_objects[obj_id]["bbox"], obj["bbox"])
        else:
            obj["velocity"] = 0

        # Compute state
        obj["state"] = compute_state(obj["velocity"])

        # Save current bbox for next frame velocity
        prev_objects[obj_id] = obj

        frame_objects.append(obj)

    # Compute relations
    frame_relations = compute_relations(frame_objects)

    cleaned_data.append({
        "frame": frame["frame"],
        "objects": frame_objects,
        "relations": frame_relations
    })

# ---------------- Summary ----------------
object_ids = defaultdict(set)
object_detections = defaultdict(int)

for frame in cleaned_data:
    for obj in frame["objects"]:
        label = obj["label"]
        obj_id = obj["id"]
        object_ids[label].add(obj_id)
        object_detections[label] += 1

print("\n===== OBJECT COUNT SUMMARY =====\n")
for label in object_ids:
    print(f"Object: {label}")
    print(f"  Unique count     : {len(object_ids[label])}")
    print(f"  Total detections : {object_detections[label]}")
    print()

# ---------------- Save cleaned JSON ----------------
with open(JSON_OUTPUT, "w") as f:
    json.dump(cleaned_data, f, indent=4)

print(f"\n✅ Cleaned JSON saved at: {JSON_OUTPUT}")
