"""
Optimized LLM Agent - Fast, Human-like Responses
This version pre-indexes the data so responses are instant, not iterating through frames
"""

import requests
import json
import re
import sys
import os
from typing import List, Dict, Any
from collections import Counter, defaultdict

# Fix imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
tools_dir = os.path.join(parent_dir, 'tools')
sys.path.insert(0, tools_dir)

from llm_tools import (
    # Basic Retrieval
    get_frame,
    get_objects,
    get_object_by_unique_id,
    
    # Property Extraction
    get_object_bbox,
    get_object_confidence,
    get_object_state,
    
    # Search & Filter
    search_objects_by_label,
    filter_objects_by_confidence,
    filter_objects_by_state,
    filter_objects_by_area,
    search_objects_by_bbox_size,
    
    # Statistics & Analysis
    get_all_labels,
    count_objects_by_label,
    get_frame_range,
    
    # Temporal Queries
    get_objects_in_frame_range,
    get_static_objects,
    get_moving_objects,
    
    # Vector Search (RAG)
    store_embedding,
    search_embedding,
    search_vector_db
)

# ============ Configuration ============
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:7b"

# ============ Data Index Cache ============
class VideoDataIndex:
    """Pre-computed index for instant queries - no iterating through frames!"""
    
    def __init__(self):
        self.index = None
        self.build_index()
    
    def build_index(self):
        """Build the index once at startup"""
        print("🔄 Building data index for fast queries...")
        
        # Load all data once
        try:
            with open("newData.json") as f:
                data = json.load(f)
        except:
            print("⚠️  Warning: Could not load video data. Using empty index.")
            data = []
        
        # Pre-compute everything
        self.index = {
            "total_frames": len(data),
            "frame_ids": [frame.get("frame") for frame in data],
            "total_objects": 0,
            "objects_by_label": defaultdict(list),
            "objects_by_frame": {},
            "label_counts": Counter(),
            "frames_with_objects": set(),
            "summary": ""
        }
        
        # Index all objects
        for frame in data:
            frame_id = frame.get("frame")
            objects = frame.get("objects", [])
            
            self.index["total_objects"] += len(objects)
            self.index["objects_by_frame"][frame_id] = len(objects)
            
            if objects:
                self.index["frames_with_objects"].add(frame_id)
            
            for obj in objects:
                label = obj.get("label")
                self.index["label_counts"][label] += 1
                self.index["objects_by_label"][label].append({
                    "frame": frame_id,
                    "object_id": obj.get("id")
                })
        
        # Create human-readable summary
        top_labels = self.index["label_counts"].most_common(5)
        label_summary = ", ".join([f"{count} {label}s" for label, count in top_labels])
        
        self.index["summary"] = (
            f"Video with {self.index['total_frames']} frames containing "
            f"{self.index['total_objects']} objects. "
            f"Main objects: {label_summary}."
        )
        
        print(f"✅ Index built! {self.index['summary']}")
    
    def get_summary(self):
        """Instant video summary"""
        return self.index["summary"]
    
    def count_label(self, label: str):
        """Instant count of objects by label"""
        return self.index["label_counts"].get(label, 0)
    
    def get_frames_with_label(self, label: str):
        """Instant list of frames containing a label"""
        return [item["frame"] for item in self.index["objects_by_label"].get(label, [])]
    
    def get_all_labels(self):
        """All unique object types"""
        return list(self.index["label_counts"].keys())
    
    def get_busiest_frames(self, top_n=5):
        """Frames with most objects"""
        sorted_frames = sorted(
            self.index["objects_by_frame"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_frames[:top_n]

# Global index instance
VIDEO_INDEX = VideoDataIndex()

# ============ Smart Tools (Use Index First) ============
SMART_TOOLS = {
    # ============ BASIC RETRIEVAL ============
    "get_frame": {
        "description": "Get complete metadata for a specific frame",
        "parameters": ["frame_id (int)"],
        "example": '{"frame_id": 110}'
    },
    "get_objects": {
        "description": "Get all objects detected in a specific frame",
        "parameters": ["frame_id (int)"],
        "example": '{"frame_id": 110}'
    },
    "get_object_by_unique_id": {
        "description": "Get a single object using unique frame_id + object_id combination",
        "parameters": ["frame_id (int)", "object_id (int)"],
        "example": '{"frame_id": 110, "object_id": 3}'
    },
    
    # ============ PROPERTY EXTRACTION ============
    "get_object_bbox": {
        "description": "Get bounding box coordinates (x, y, w, h) for a specific object",
        "parameters": ["frame_id (int)", "object_id (int)"],
        "example": '{"frame_id": 110, "object_id": 3}'
    },
    "get_object_confidence": {
        "description": "Get detection confidence score for a specific object",
        "parameters": ["frame_id (int)", "object_id (int)"],
        "example": '{"frame_id": 110, "object_id": 3}'
    },
    "get_object_state": {
        "description": "Get state (static/moving) and velocity of a specific object",
        "parameters": ["frame_id (int)", "object_id (int)"],
        "example": '{"frame_id": 110, "object_id": 3}'
    },
    
    # ============ SEARCH & FILTER ============
    "search_objects_by_label": {
        "description": "Find all objects of a specific type across all frames",
        "parameters": ["label (str)"],
        "example": '{"label": "tv"}'
    },
    "filter_objects_by_confidence": {
        "description": "Find all objects with confidence score >= threshold",
        "parameters": ["min_confidence (float)"],
        "example": '{"min_confidence": 0.8}'
    },
    "filter_objects_by_state": {
        "description": "Find all objects with specific state (static or moving)",
        "parameters": ["state (str)"],
        "example": '{"state": "static"}'
    },
    "filter_objects_by_area": {
        "description": "Find objects within a specific area range in pixels",
        "parameters": ["min_area (int, default=0)", "max_area (int, default=999999)"],
        "example": '{"min_area": 50000, "max_area": 100000}'
    },
    "search_objects_by_bbox_size": {
        "description": "Find objects with bounding box larger than specified dimensions",
        "parameters": ["min_width (int, default=0)", "min_height (int, default=0)"],
        "example": '{"min_width": 200, "min_height": 200}'
    },
    
    # ============ STATISTICS & ANALYSIS ============
    "get_all_labels": {
        "description": "Get a list of all unique object types in the dataset",
        "parameters": [],
        "example": "{}"
    },
    "count_objects_by_label": {
        "description": "Count how many times each object type appears across all frames",
        "parameters": [],
        "example": "{}"
    },
    "get_frame_range": {
        "description": "Get min/max frame numbers and total frame count in dataset",
        "parameters": [],
        "example": "{}"
    },
    
    # ============ TEMPORAL QUERIES ============
    "get_objects_in_frame_range": {
        "description": "Get all objects within a specific frame range",
        "parameters": ["start_frame (int)", "end_frame (int)"],
        "example": '{"start_frame": 100, "end_frame": 120}'
    },
    "get_static_objects": {
        "description": "Get all objects that are not moving (velocity = 0)",
        "parameters": [],
        "example": "{}"
    },
    "get_moving_objects": {
        "description": "Get all objects that are moving (velocity > 0)",
        "parameters": [],
        "example": "{}"
    },
    
    # ============ VECTOR SEARCH (RAG) ============
    "search_vector_db": {
        "description": "Semantic search - find objects similar to text query using AI embeddings",
        "parameters": ["query (str)", "top_k (int, default=3)"],
        "example": '{"query": "person wearing red shirt", "top_k": 5}'
    },
    "search_embedding": {
        "description": "Search using raw embedding vector for similarity matching",
        "parameters": ["query_vector (list)", "top_k (int, default=5)"],
        "example": '{"query_vector": [0.1, 0.2, ...], "top_k": 5}'
    },
    "store_embedding": {
        "description": "Store an object's embedding in vector database for semantic search",
        "parameters": ["frame_id (int)", "object_id (int)", "embedding (list)"],
        "example": '{"frame_id": 110, "object_id": 3, "embedding": [...]}'
    }
}

# ============ Smart Tool Executor ============
def execute_smart_tool(tool_name: str, arguments: Dict[str, Any] = None) -> Any:
    """Execute tools using the pre-built index (instant results!)"""
    
    if tool_name == "video_summary":
        return {
            "summary": VIDEO_INDEX.get_summary(),
            "total_frames": VIDEO_INDEX.index["total_frames"],
            "total_objects": VIDEO_INDEX.index["total_objects"],
            "object_types": len(VIDEO_INDEX.index["label_counts"]),
            "top_objects": dict(VIDEO_INDEX.index["label_counts"].most_common(5))
        }
    
    elif tool_name == "count_objects":
        label = arguments.get("label")
        count = VIDEO_INDEX.count_label(label)
        frames = VIDEO_INDEX.get_frames_with_label(label)
        return {
            "label": label,
            "count": count,
            "appears_in_frames": len(frames),
            "frame_list": frames[:10]  # First 10 frames only
        }
    
    elif tool_name == "list_object_types":
        labels = VIDEO_INDEX.get_all_labels()
        counts = {label: VIDEO_INDEX.count_label(label) for label in labels}
        return {
            "total_types": len(labels),
            "object_types": sorted(counts.items(), key=lambda x: x[1], reverse=True)
        }
    
    elif tool_name == "find_busiest_frames":
        top_n = arguments.get("top_n", 5) if arguments else 5
        busiest = VIDEO_INDEX.get_busiest_frames(top_n)
        return {
            "busiest_frames": [
                {"frame": frame_id, "object_count": count}
                for frame_id, count in busiest
            ]
        }
    
    elif tool_name == "frames_with_object":
        label = arguments.get("label")
        frames = VIDEO_INDEX.get_frames_with_label(label)
        return {
            "label": label,
            "total_frames": len(frames),
            "frames": frames
        }
    
    elif tool_name == "get_frame_details":
        frame_id = arguments.get("frame_id")
        # Only call actual tool for specific frame requests
        result = get_objects.invoke({"frame_id": frame_id})
        return {
            "frame_id": frame_id,
            "objects": result
        }
    
    else:
        return {"error": f"Unknown tool: {tool_name}"}

# ============ Enhanced System Prompt ============
def create_smart_system_prompt():
    """System prompt that encourages fast, human-like responses"""
    
    return f"""You are a helpful AI assistant analyzing a video. 

VIDEO OVERVIEW:
VIDEO_INDEX.get_summary()

IMPORTANT INSTRUCTIONS:
1. **Be conversational and human-like** - Don't sound robotic
2. **Use the summary tools FIRST** - They're instant! Only use detailed frame tools if user asks about a specific frame
3. **Don't iterate through all frames** - Use the index/summary tools instead
4. **Give quick, helpful answers** - Don't make the user wait
5. **Give a replay after understand the data**- Dont make stright answer analysis first
6. **Count all other operation by proper method by understand the below Data structure in db**- dont skip

Below Sample DataStructure for reference
frame: 120,
        objects: [

        id: 3,
        "label": "tv",
        "confidence": 0.803,
    ]

AVAILABLE TOOLS:

**Fast Summary Tools (USE THESE FIRST!):**
- video_summary: Get instant overview of entire video
- count_objects: Instantly count objects by type, e.g. {{"label": "person"}}
- list_object_types: List all object types in video
- find_busiest_frames: Find frames with most activity
- frames_with_object: List frames containing an object type

**Detailed Tools (only for specific frame queries):**
- get_frame_details: Get objects in ONE specific frame, e.g. {{"frame_id": 5}}

RESPONSE FORMAT:
<tool>tool_name</tool>
<args>{{"param": "value"}}</args>

EXAMPLES:
User: "What's in the video?"
You: <tool>video_summary</tool><args>{{}}</args>

User: "How many people are there?"
You: <tool>count_objects</tool><args>{{"label": "person"}}</args>

User: "What's in frame 5?"
You: <tool>get_frame_details</tool><args>{{"frame_id": 5}}</args>

Remember: Be fast, friendly, and human! Use summary tools for general questions.
"""

# ============ Tool Parser ============
def parse_tool_calls(response: str) -> List[Dict[str, Any]]:
    """Parse tool calls from response"""
    tool_calls = []
    pattern = r'<tool>(.*?)</tool>\s*<args>(.*?)</args>'
    matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
    
    for tool_name, args_str in matches:
        try:
            args_str = args_str.strip()
            args = json.loads(args_str) if args_str and args_str != '{}' else {}
            tool_calls.append({
                "tool": tool_name.strip(),
                "arguments": args
            })
        except json.JSONDecodeError as e:
            print(f"⚠️  Failed to parse: {args_str}")
    
    return tool_calls

# ============ Ollama Chat ============
def ollama_chat(messages: List[Dict], stream: bool = False):
    """Send messages to Ollama"""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "stream": stream,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 256  # Limit tokens for faster response
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()["message"]["content"]
        else:
            raise Exception(f"Ollama error: {response.text}")
    
    except requests.exceptions.ConnectionError:
        raise Exception("Cannot connect to Ollama. Run: ollama serve")
    except requests.exceptions.Timeout:
        raise Exception("Request timed out")

# ============ Smart Agent ============
class SmartVideoAgent:
    """Fast, human-like agent with pre-indexed data"""
    
    def __init__(self):
        self.conversation = []
        self.system_prompt = create_smart_system_prompt()
    
    def chat(self, user_message: str, verbose: bool = True):
        """Chat with instant responses"""
        
        # Add user message
        self.conversation.append({
            "role": "user",
            "content": user_message
        })
        
        # Build messages
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation)
        
        # Get response
        if verbose:
            print(f"🤔 Thinking...")
        
        response = ollama_chat(messages)
        
        # Parse tools
        tool_calls = parse_tool_calls(response)
        
        if not tool_calls:
            # Direct response
            self.conversation.append({
                "role": "assistant",
                "content": response
            })
            return response
        
        # Execute tools (should be instant with index!)
        if verbose:
            print(f"🔧 Using {len(tool_calls)} tool(s)...")
        
        tool_results = {}
        for call in tool_calls:
            tool_name = call["tool"]
            arguments = call["arguments"]
            
            if verbose:
                print(f"  ├─ {tool_name}({json.dumps(arguments)})")
            
            result = execute_smart_tool(tool_name, arguments)
            tool_results[tool_name] = result
        
        # Add to conversation
        self.conversation.append({
            "role": "assistant",
            "content": response
        })
        
        tool_context = f"""Tool results:
{json.dumps(tool_results, indent=2)}

Now give a natural, conversational answer based on these results. Be friendly and human-like!"""
        
        self.conversation.append({
            "role": "user",
            "content": tool_context
        })
        
        # Get final response
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation)
        
        final_response = ollama_chat(messages)
        
        self.conversation.append({
            "role": "assistant",
            "content": final_response
        })
        
        return final_response
    
    def reset(self):
        """Clear conversation"""
        self.conversation = []
        print("🔄 Conversation reset!")

# ============ Main ============
def main():
    print("\n" + "="*70)
    print("🚀 Fast Video Analysis Agent (Human-like Responses)")
    print("="*70)
    
    # Check Ollama
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            print("\n❌ Ollama not running! Start with: ollama serve")
            return
        
        models = [m["name"] for m in response.json().get("models", [])]
        if MODEL_NAME not in models:
            print(f"\n❌ Model {MODEL_NAME} not found!")
            print(f"Available: {', '.join(models)}")
            print(f"\nInstall with: ollama pull {MODEL_NAME}")
            return
        
        print(f"✅ Connected to Ollama with {MODEL_NAME}")
    
    except:
        print("\n❌ Cannot connect to Ollama! Run: ollama serve")
        return
    
    agent = SmartVideoAgent()
    
    print("\n📝 Try these fast queries:")
    print("  • 'What's in this video?'")
    print("  • 'How many people are there?'")
    print("  • 'What types of objects are detected?'")
    print("  • 'Which frames are busiest?'")
    print("  • 'Show me frames with cars'")
    
    print("\n💡 Commands: 'quit', 'reset', 'stats'")
    print("="*70 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!\n")
                break
            
            if user_input.lower() == 'reset':
                agent.reset()
                continue
            
            if user_input.lower() == 'stats':
                print(f"\n📊 Video Stats:")
                print(f"  Frames: {VIDEO_INDEX.index['total_frames']}")
                print(f"  Objects: {VIDEO_INDEX.index['total_objects']}")
                print(f"  Types: {len(VIDEO_INDEX.index['label_counts'])}")
                print()
                continue
            
            # Get response (should be fast!)
            response = agent.chat(user_input, verbose=True)
            
            print(f"\n🤖 {response}\n")
            print("-"*70 + "\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

if __name__ == "__main__":
    main()