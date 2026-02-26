import cv2
import json
from ultralytics import YOLO
from pathlib import Path

# ---------------- CONFIG ----------------
VIDEO_PATH = "VideoData/sample_video.mp4"
MODEL_PATH = "yolov8n.pt"
FRAME_SKIP = 5
CONF_THRESHOLD = 0.5
JSON_OUTPUT = "detection.json"

RESIZE_WIDTH = 960
RESIZE_HEIGHT = 540
# ----------------------------------------

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
video_fps = cap.get(cv2.CAP_PROP_FPS) or 30

frame_count = 0
all_frames_data = []

def extract_properties(obj_data, prev_obj=None, fps=30):
    bbox = obj_data["bbox"]
    obj_data["area"] = bbox["w"] * bbox["h"]
    if prev_obj:
        dx = bbox["x"] - prev_obj["bbox"]["x"]
        dy = bbox["y"] - prev_obj["bbox"]["y"]
        dist = (dx**2 + dy**2)**0.5
        obj_data["velocity"] = dist * fps
    else:
        obj_data["velocity"] = 0
    obj_data["state"] = "unknown"
    return obj_data

def compute_relations(all_objs, iou_threshold=0.1):
    relations = []
    for i, obj1 in enumerate(all_objs):
        for j, obj2 in enumerate(all_objs):
            if i >= j:
                continue
            xA = max(obj1["bbox"]["x"], obj2["bbox"]["x"])
            yA = max(obj1["bbox"]["y"], obj2["bbox"]["y"])
            xB = min(obj1["bbox"]["x"]+obj1["bbox"]["w"], obj2["bbox"]["x"]+obj2["bbox"]["w"])
            yB = min(obj1["bbox"]["y"]+obj1["bbox"]["h"], obj2["bbox"]["y"]+obj2["bbox"]["h"])
            inter_area = max(0, xB - xA) * max(0, yB - yA)
            area1 = obj1["bbox"]["w"] * obj1["bbox"]["h"]
            area2 = obj2["bbox"]["w"] * obj2["bbox"]["h"]
            iou = inter_area / float(area1 + area2 - inter_area + 1e-6)
            relation = "far" if iou < iou_threshold else "near"
            relations.append({
                "obj1_id": obj1["id"],
                "obj2_id": obj2["id"],
                "iou": round(iou,3),
                "relation": relation
            })
    return relations

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=CONF_THRESHOLD,
        verbose=False
    )

    frame_objects = []
    for r in results:
        if r.boxes.id is None:
            continue
        for box, track_id, cls_id, conf in zip(r.boxes.xyxy, r.boxes.id, r.boxes.cls, r.boxes.conf):
            label = model.names[int(cls_id)]
            x1, y1, x2, y2 = map(int, box)
            obj_data = {
                "id": int(track_id),
                "label": label,
                "confidence": round(float(conf),3),
                "bbox": {"x":x1, "y":y1, "w":x2-x1, "h":y2-y1}
            }
            # property extraction
            prev_obj = None
            for prev_frame in reversed(all_frames_data):
                for p_obj in prev_frame["objects"]:
                    if p_obj["id"] == obj_data["id"]:
                        prev_obj = p_obj
                        break
                if prev_obj: break
            obj_data = extract_properties(obj_data, prev_obj, fps=video_fps)
            frame_objects.append(obj_data)

    frame_relations = compute_relations(frame_objects)

    all_frames_data.append({
        "frame": frame_count,
        "objects": frame_objects,
        "relations": frame_relations
    })

cap.release()

# Write JSON
output_path = Path(JSON_OUTPUT).resolve()
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path,"w") as f:
    json.dump(all_frames_data, f, indent=4)

print(f"✅ Detection + Tracking JSON saved at: {output_path}")
