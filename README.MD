# 📌 **PROJECT DOCUMENTATION — VideoRAG (Video-Region Attention Retrieval System)**

**Version:** 2.0
**Type:** Technical Design Document
**Purpose:** Define the full engineering background, architecture, module dependencies, and code structure required to build an agentic LLM system capable of querying, reasoning over, and extracting structured information from videos using frame-region embeddings and tool-augmented retrieval.

---

# 1. **Project Overview**

VideoRAG is a **multi-stage agentic AI system** that transforms raw video into a structured knowledge base of:

* frames
* regions
* objects
* textual summaries
* embeddings

These are stored in an embedded vector database.
The LLM interacts with this database through  **tools/functions** , enabling retrieval-augmented reasoning over visual data.

**Key Capabilities:**

* Query video content using natural language
* Retrieve relevant frames via vector similarity
* Extract region-level features (objects, bbox, embeddings)
* Produce high-precision answers using multi-tool agentic execution

---

# 2. **Project Objectives**

* Enable **semantic video querying** (“find all frames with chairs”).
* Support **region attention reasoning** (“object next to window”).
* Implement an **LLM-Tool-DB execution chain** that simulates real agent intelligence.
* Provide a clean research-grade **pipeline** for future expansion.

---

# 3. **Core System Architecture**

```
VideoRAG System
|
├── Ingestion Layer
│     - Video Loader
│     - Frame Generator
│     - Region Detector
│
├── Intelligence Layer
│     - Object Detection (ROI extraction)
│     - Region Feature Encoder (CLIP/VIT)
│     - Frame Text Summarizer (caption)
│
├── Embedding & Storage Layer
│     - Frame Embeddings Store
│     - Region Embeddings Store
│     - Frame Metadata Store
│     - Vector Index (FAISS/Chroma)
│
├── Agent Tool Layer
│     - search_frame_embeddings
│     - get_frame_metadata
│     - get_region_features
│
└── Reasoning Layer (LLM)
      - Query Understanding
      - Tool Selection
      - Evidence Fusion
      - Final Response
```

---

# 4. **Justification of Architecture**

### 4.1 Frame-level embeddings

* Necessary for fast ranking of video segments.
* Avoids sending raw video to the LLM.
* Outperforms global video embeddings for fine-grained tasks.

### 4.2 Region-level features

* Required for counting, detection, and spatial reasoning.
* Enables “what is left of X”, “how many Y”, “track object Z”.

### 4.3 Vector database

* JSON or raw files cannot perform ANN search.
* Required for any real RAG retrieval.

### 4.4 Tool-driven agentic design

* LLM cannot directly access embeddings.
* Tools provide deterministic and auditable retrieval.

---

# 5. **Dependencies (Strict List)**

## 5.1 Core AI Libraries

| Dependency                           | Purpose                    |
| ------------------------------------ | -------------------------- |
| **PyTorch**                    | Frame encoding & detectors |
| **OpenCV**                     | Frame extraction           |
| **Transformers (HuggingFace)** | CLIP / ViT embeddings      |
| **Ultralytics YOLO**           | Object detection           |
| **LangChain**                  | Tool execution layer       |
| **FAISS / ChromaDB**           | Vector storage             |

---

## 5.2 Utility Libraries

| Library        | Purpose                     |
| -------------- | --------------------------- |
| numpy          | Tensor manipulation         |
| pillow         | Image processing            |
| python-dotenv  | Config                      |
| orjson / ujson | High-speed metadata storage |
| tqdm           | Processing progress         |

---

# 6. **Project Directory Structure (Gold Standard)**

```
videorag/
│
├── data/
│   ├── videos/
│   │   └── sample.mp4
│   ├── frames/
│   │   ├── vid001/
│   │   │   ├── frame_0001.jpg
│   │   │   ├── frame_0002.jpg
│   │   │   └── ...
│   └── metadata/
│       └── vid001.json
│
├── embeddings/
│   ├── frame_embeddings/
│   └── region_embeddings/
│
├── index/
│   ├── frame_index.faiss
│   └── region_index.faiss
│
├── src/
│   ├── ingestion/
│   │   ├── video_loader.py
│   │   ├── frame_extractor.py
│   │   └── metadata_builder.py
│   │
│   ├── vision/
│   │   ├── object_detector.py
│   │   ├── region_encoder.py
│   │   ├── frame_encoder.py
│   │   └── summarizer.py
│   │
│   ├── storage/
│   │   ├── vector_store.py
│   │   ├── metadata_store.py
│   │   └── index_builder.py
│   │
│   ├── tools/
│   │   ├── search_frame_embeddings.py
│   │   ├── get_frame_metadata.py
│   │   └── get_region_features.py
│   │
│   ├── agent/
│   │   ├── tool_router.py
│   │   ├── reasoning_engine.py
│   │   └── query_planner.py
│   │
│   └── main.py
│
├── requirements.txt
└── README.md
```

---

# 7. **Code-Level Architecture (Module Responsibilities)**

## 7.1 **ingestion/video_loader.py**

**Purpose:**
Load raw videos → generate frame timestamps.

**Inputs:** video file
**Outputs:** frame images

---

## 7.2 **ingestion/frame_extractor.py**

**Purpose:**
Convert video to a sequence of frames at fixed intervals.

---

## 7.3 **vision/object_detector.py**

**Purpose:**
Run region detection on each frame (YOLO/DETR).
Return: object classes + bounding boxes + scores.

---

## 7.4 **vision/region_encoder.py**

**Purpose:**
Encode cropped region into a 512/768-dim dense vector (CLIP ViT).

---

## 7.5 **vision/frame_encoder.py**

**Purpose:**
Encode entire frame for semantic retrieval.

---

## 7.6 **vision/summarizer.py**

**Purpose:**
Produce caption describing frame (“room with 3 chairs…”).

---

## 7.7 **storage/vector_store.py**

**Purpose:**
Insert/retrieve vectors from FAISS/Chroma.

---

## 7.8 **storage/metadata_store.py**

**Purpose:**
Store raw metadata:

* bboxes
* class labels
* region features references

---

## 7.9 **tools/search_frame_embeddings.py**

Agent tool for ANN retrieval using  **frame embeddings** .

---

## 7.10 **tools/get_frame_metadata.py**

Agent tool returning entire frame metadata.

---

## 7.11 **tools/get_region_features.py**

Agent tool returning specific object’s embedding vector.

---

## 7.12 **agent/query_planner.py**

**Purpose:**
Break user query into a multi-step tool execution plan.

---

## 7.13 **agent/reasoning_engine.py**

**Purpose:**
Combine tool outputs → generate final answer.

---

# 8. **Execution Flow (End-to-End)**

```



1. User Query → LLM  
2. LLM decides → search_frame_embeddings  
3. Retrieve candidate frames  
4. LLM fetches → get_frame_metadata  
5. If object-specific → get_region_features  
6. LLM fuses all evidence  
7. Produce final natural-language answer
```





# *Flow of file*


main.py   ==> detection.json ==> json_processing.py ==>cleaned_detection.json =>data_cleaning.py ==> newData.json ==>final_cleaning.py ==> (again newData.json) ==> store_vector_db.py (index.faiss and index.pkl) ==>rag_agent.py (AI)
