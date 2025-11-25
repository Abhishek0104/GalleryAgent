import os
from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field, model_validator
from typing import List, Dict, Any, Optional, Literal
import uuid
from datetime import datetime
import re
import logging

# --- NEW IMPORTS FOR GEMINI ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY") 
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- DYNAMIC SYSTEM PROMPT ---
def get_system_prompt():
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    return (
        f"You are a helpful gallery assistant. Today is {current_date}. "
        "You have access to client-side tools to manage the user's photo gallery. "
        
        "--- DATA HANDLING RULES (CRITICAL) ---"
        "1. **CONTEXT URIs:** If the user message contains a section '[Developer Note: User context URIs: ...]', "
        "   these are the specific photos the user wants you to act on."
        "2. **TOOL ARGS:** When calling tools like `create_collage`, `delete_photos`, or `move_photos_to_album`, "
        "   you MUST provide the `photo_uris` argument."
        
        "--- SEARCH QUERY RULES (CRITICAL) ---"
        "1. **PEOPLE vs OBJECTS:** You must distinguish between searching for *content* and *people*."
        "   - If the user asks for a PERSON (e.g., 'Photos of Alice', 'Mom', 'Dad', 'John'), you MUST put their name "
        "     into the `people` list argument. Example: `search_photos(people=['Alice'])`."
        "   - If the user asks for an OBJECT or SCENE (e.g., 'Cow', 'Beach', 'Car'), put it in the `query` argument. "
        "     Example: `search_photos(query='Cow')`."
        "   - You can use both. Example: 'Alice at the beach' -> `search_photos(query='beach', people=['Alice'])`."
        "2. **'ME' / SELF:** If the user says 'photos of me', put 'Me' in the `people` list: `people=['Me']`."

        "--- EFFICIENT WORKFLOW (CRITICAL) ---"
        "To perform actions (Delete, Move, Collage) on photos, you must follow this 2-step process:"
        "1. **SEARCH FIRST:** Call `search_photos(...)` to find the images. "
        "   - The device will store the results in its local memory (cache)."
        "   - It will return a count (e.g., 'Found 5 photos')."
        "2. **ACT ON CACHE:** Immediately call the action tool (e.g., `delete_photos`) "
        "   with `use_cache=True`. "
        "   - DO NOT ask for URIs."
        "   - DO NOT try to pass a list of URIs."
        "   - ONLY pass `use_cache=True`."
        
        "Example 1: 'Delete cats from 2024'"
        "Step 1: `search_photos(query='cat', start_date='2024-01-01')`"
        "Step 2: `delete_photos(use_cache=True)`"
        
        "Example 2: 'Move vacation photos to Paris album'"
        "Step 1: `search_photos(query='vacation', location='Paris')`"
        "Step 2: `move_photos_to_album(album_name='Paris', use_cache=True)`"
        
        "Example 3: 'Create a collage of my dog'"
        "Step 1: `search_photos(query='dog')`"
        "Step 2: `create_collage(title='Dog Collage', use_cache=True)`"
        
        "--- AVAILABLE TOOLS & CAPABILITIES ---"
        "1. `search_photos`: Finds images by content, date, location, or people."
        "2. `delete_photos`: Removes images (can use cache from search)."
        "3. `move_photos_to_album`: Organizes images into folders (can use cache)."
        "4. `create_collage`: Stitches 2-9 images into one (can use cache)."
        "5. `apply_filter`: Applies visual effects like 'grayscale' or 'sepia' (can use cache)."
        "6. `scan_for_cleanup`: Finds duplicates. Returns 'uris_to_delete'. To act, call `delete_photos(use_cache=True)`."
        "7. `get_photo_metadata`: Reads details like camera model and location."

        "--- RESPONSE FORMAT ---"
        "If you want to suggest quick replies, add them at the end of your message in this format: "
        "||Suggestion 1||Suggestion 2||"
        "Example: 'I found 5 photos. Should I create a collage? ||Create Collage||Delete Them||'"
    )

# --- 1. DEFINE TOOLS ---

@tool
def search_photos(query: str = "", start_date: str = None, end_date: str = None, location: str = None, people: List[str] = None):
    """
    Searches for photos. Results are CACHED on the device.
    Args:
        query: Content description (e.g. "cat").
        start_date: Format YYYY-MM-DD.
        end_date: Format YYYY-MM-DD.
        location: City name.
        people: List of names.
    Returns:
        JSON object with count. The actual photos are stored in the device cache.
    """
    return "Client tool"

@tool
def delete_photos(use_cache: bool = False, photo_uris: List[str] = None):
    """
    Deletes photos.
    Args:
        use_cache: Set to True to delete the photos found in the LAST search/scan.
        photo_uris: (Optional) Specific list of URIs if NOT using cache.
    """
    return "Client tool"

@tool
def move_photos_to_album(album_name: str, use_cache: bool = False, photo_uris: List[str] = None):
    """
    Moves photos to an album.
    Args:
        album_name: Destination folder.
        use_cache: Set to True to move results from the last search.
    """
    return "Client tool"

@tool
def create_collage(title: str = "My Collage", use_cache: bool = False, photo_uris: List[str] = None):
    """
    Creates a collage.
    Args:
        title: File title.
        use_cache: Set to True to use results from the last search (Max 9).
    """
    return "Client tool"

@tool
def apply_filter(filter_name: Literal["grayscale", "sepia"] = "grayscale", use_cache: bool = False, photo_uris: List[str] = None):
    """
    Applies filter.
    Args:
        use_cache: Set to True to use results from the last search.
    """
    return "Client tool"

@tool
def scan_for_cleanup():
    """
    Scans for duplicates. 
    IMPORTANT: If duplicates are found, they are automatically CACHED. 
    To delete them, simply call `delete_photos(use_cache=True)`.
    """
    return "Client tool"

@tool
def get_photo_metadata(use_cache: bool = False, photo_uris: List[str] = None):
    """Reads EXIF data."""
    return "Client tool"

tools = [search_photos, delete_photos, move_photos_to_album, create_collage, apply_filter, scan_for_cleanup, get_photo_metadata]

# ... (The rest of the file with FastAPI, AgentResponse, etc. remains exactly the same) ...
# ... (I will output the full file to be safe) ...

# --- 2. SETUP MODEL ---
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    google_api_key=api_key
).bind_tools(tools)

# --- 3. GRAPH DEFINITION ---
async def chatbot(state: MessagesState):
    return {"messages": [await llm.ainvoke(state["messages"])]}

graph_builder = StateGraph(MessagesState)
graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile(checkpointer=MemorySaver())

# --- 4. FASTAPI SERVER ---
app = FastAPI()

class ToolResult(BaseModel):
    tool_call_id: str
    content: str

class AgentRequest(BaseModel):
    sessionId: Optional[str] = None
    userInput: Optional[str] = None
    toolResult: Optional[dict] = None
    selectedUris: Optional[List[str]] = None
    base64Images: Optional[List[str]] = None

    @model_validator(mode='before')
    def check_input(cls, values):
        return values

class ToolCall(BaseModel):
    id: str
    name: str
    args: Dict[str, Any]

class Suggestion(BaseModel):
    label: str
    prompt: str

class AgentResponse(BaseModel):
    sessionId: str
    status: Literal["complete", "requires_action"]
    agentMessage: Optional[str] = None
    nextActions: Optional[List[ToolCall]] = None
    suggestedActions: Optional[List[Suggestion]] = None

def extract_suggestions(text: str):
    pattern = r"\|\|(.*?)\|\|"
    matches = re.findall(pattern, text)
    clean_text = re.sub(pattern, "", text).strip()
    suggestions = []
    for match in matches:
        suggestions.append({"label": match, "prompt": match})
    return clean_text, suggestions

def parse_message_content(content):
    if isinstance(content, list):
        return " ".join([c["text"] for c in content if c["type"] == "text"])
    return str(content)

@app.post("/agent/invoke", response_model=AgentResponse)
async def invoke_agent(request: AgentRequest):
    session_id = request.sessionId or str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    
    graph_input = None

    if request.userInput:
        user_text = request.userInput
        if request.selectedUris:
            user_text += f"\n\n[Developer Note: User context URIs: {request.selectedUris}]"
        
        sys_msg = SystemMessage(content=get_system_prompt())
        
        if request.base64Images:
            content_parts = [{"type": "text", "text": user_text}]
            for b64 in request.base64Images:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                })
            graph_input = {"messages": [sys_msg, HumanMessage(content=content_parts)]}
        else:
            graph_input = {"messages": [sys_msg, HumanMessage(content=user_text)]}
            
    elif request.toolResult:
        graph_input = {"messages": [ToolMessage(
            tool_call_id=request.toolResult["tool_call_id"], 
            content=str(request.toolResult["content"])
        )]}
    else:
        return {"error": "Invalid request"}, 400

    async for chunk in graph.astream(graph_input, config): 
        pass
    
    latest_state = await graph.aget_state(config)
    last_msg = latest_state.values["messages"][-1]

    if last_msg.tool_calls:
        actions = [ToolCall(id=tc["id"], name=tc["name"], args=tc["args"]) for tc in last_msg.tool_calls]
        response_obj = AgentResponse(sessionId=session_id, status="requires_action", nextActions=actions)
        logging.info(f"Outgoing response (Tool Call): {response_obj.model_dump_json(indent=2)}") 
        return response_obj
    else:
        raw_text = parse_message_content(last_msg.content)
        clean_text, suggestions_list = extract_suggestions(raw_text)
        suggestions_model = [Suggestion(**s) for s in suggestions_list] if suggestions_list else None

        response_obj = AgentResponse(
            sessionId=session_id, 
            status="complete", 
            agentMessage=clean_text,
            suggestedActions=suggestions_model
        )
        logging.info(f"Outgoing response (Final Answer): {response_obj.model_dump_json(indent=2)}")
        return response_obj

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

