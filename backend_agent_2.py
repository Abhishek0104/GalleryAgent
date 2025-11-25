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
        "You have access to tools to manage the user's photo gallery. "
        
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

        "--- TOOL CHAINING RULES ---"
        "1. **SEARCH FIRST:** If a user asks to 'delete cat photos', you cannot just guess."
        "   First, call `search_photos(query='cat')`."
        "2. **READ OUTPUT:** The search tool will return a JSON object like: "
        "   `{'count': 5, 'uris': ['file://a', 'file://b']}`."
        "3. **PASS URIS:** You MUST extract those specific strings from the `uris` list and pass them "
        "   into the next tool. "
        "   Example: Call `delete_photos(photo_uris=['file://a', 'file://b'])`."
        "   DO NOT make up URIs. DO NOT use placeholders like 'uris_from_search'."
        
        "--- RESPONSE FORMAT ---"
        "If you want to suggest quick replies, add them at the end of your message in this format: "
        "||Suggestion 1||Suggestion 2||"
        "Example: 'I found 5 photos. Should I create a collage? ||Create Collage||Delete Them||'"
    )

# --- 1. DEFINE TOOLS (Client-Side Placeholders) ---

@tool
def search_photos(query: str = "", start_date: str = None, end_date: str = None, location: str = None, people: List[str] = None):
    """
    Searches for photos on the device.
    Args:
        query: Content description (e.g. "cat", "receipt", "sunset"). Use this for objects/scenes.
        start_date: Format YYYY-MM-DD.
        end_date: Format YYYY-MM-DD.
        location: City or country name (e.g. "Paris").
        people: List of PERSON NAMES (e.g. ["Alice", "Bob", "Me"]). Use this for humans.
    Returns:
        JSON object: {"count": int, "uris": List[str]}.
        IMPORTANT: The 'uris' list contains the identifiers you NEED for other tools.
    """
    return "This tool runs on the client."

@tool
def delete_photos(photo_uris: List[str]):
    """
    Deletes the specified photos.
    Args:
        photo_uris: The EXACT list of strings returned by a previous search or scan.
                    Do not invent these.
    """
    return "This tool runs on the client."

@tool
def move_photos_to_album(photo_uris: List[str], album_name: str):
    """
    Moves photos to an album (folder). Creates the album if it doesn't exist.
    Args:
        photo_uris: The list of photo URIs to move.
        album_name: The destination folder name.
    """
    return "This tool runs on the client."

@tool
def create_collage(photo_uris: List[str], title: str = "My Collage"):
    """
    Creates a single collage image from the provided photos.
    Args:
        photo_uris: List of 2-9 photo URIs.
        title: Title for the file.
    """
    return "This tool runs on the client."

@tool
def apply_filter(photo_uris: List[str], filter_name: Literal["grayscale", "sepia"] = "grayscale"):
    """
    Applies a visual filter to photos and saves copies.
    """
    return "This tool runs on the client."

@tool
def scan_for_cleanup():
    """
    Scans the gallery for duplicate or bad photos.
    Returns:
        JSON object: {"found_sets": int, "uris_to_delete": List[str]}.
        You can pass 'uris_to_delete' to the delete_photos tool if the user agrees.
    """
    return "This tool runs on the client."

@tool
def get_photo_metadata(photo_uris: List[str]):
    """
    Reads EXIF data (date, camera, location) from photos.
    """
    return "This tool runs on the client."

tools = [search_photos, delete_photos, move_photos_to_album, create_collage, apply_filter, scan_for_cleanup, get_photo_metadata]

# --- 2. SETUP MODEL ---
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    google_api_key=api_key
).bind_tools(tools)

# --- 3. GRAPH DEFINITION ---
def chatbot(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}

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
    toolResult: Optional[dict] = None # Using dict to match client JSON
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
    """Parses ||Label|| from text and returns clean text + suggestions list"""
    pattern = r"\|\|(.*?)\|\|"
    matches = re.findall(pattern, text)
    clean_text = re.sub(pattern, "", text).strip()
    
    suggestions = []
    for match in matches:
        # Simple heuristic: label is the text, prompt is the same
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
    
    # SYSTEM PROMPT INJECTION
    # We inject it at the start if the history is empty, 
    # or we trust the model's memory for subsequent turns.
    # Ideally, we prepend it to the graph state.
    
    graph_input = None

    if request.userInput:
        user_text = request.userInput
        # 1. Handle "Context Selection" (User selected photos in UI)
        if request.selectedUris:
            user_text += f"\n\n[Developer Note: User context URIs: {request.selectedUris}]"
        
        # 2. Add System Prompt to every interaction to ensure rules stick
        # (Or add it once. Adding it as a SystemMessage is best).
        sys_msg = SystemMessage(content=get_system_prompt())
        
        # 3. Vision capabilities (Base64)
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
        # Tool Result comes back. We don't need to re-inject system prompt 
        # because it's in the thread history (MemorySaver).
        # We assume the last message was a AIMessage with tool_calls.
        
        # PARSE the JSON content from client
        # The client sends a JSON string. We should just pass it as string to the LLM.
        graph_input = {"messages": [ToolMessage(
            tool_call_id=request.toolResult["tool_call_id"], 
            content=str(request.toolResult["content"])
        )]}
    else:
        return {"error": "Invalid request"}, 400

    # RUN GRAPH
    # Stream allows us to handle the "loop" internally if we wanted, 
    # but here we just step once.
    for chunk in graph.stream(graph_input, config): 
        pass
    
    # GET LATEST STATE
    latest_state = graph.get_state(config)
    last_msg = latest_state.values["messages"][-1]

    # DETERMINE RESPONSE
    if last_msg.tool_calls:
        # 1. Agent wants to run a tool
        actions = [ToolCall(id=tc["id"], name=tc["name"], args=tc["args"]) for tc in last_msg.tool_calls]
        response_obj = AgentResponse(sessionId=session_id, status="requires_action", nextActions=actions)
        logging.info(f"Outgoing response (Tool Call): {response_obj.model_dump_json(indent=2)}") 
        return response_obj
    else:
        # 2. Agent has a final answer
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
