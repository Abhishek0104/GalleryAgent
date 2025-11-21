import os
from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
import uuid
from datetime import datetime
import re
import logging # Import logging module

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
        "2. **TOOL ARGS:** When calling tools like `create_collage`, `delete_photos`, or `apply_filter`, "
        "   you MUST copy those exact URIs into the `photo_uris` argument."
        "3. **DO NOT ASK:** Never ask 'Could you provide the URIs?' if they are already present in the Developer Note. Just proceed."
        
        "--- CONTEXT AWARENESS ---"
        "If the user refines a search (e.g., 'show only the ones from 2023'), you MUST maintain the original search query from the conversation history. "
        "Example: User said 'find cats', then 'from 2023'. You call search_photos(query='cats', start_date='2023-01-01'...). "
        "Do NOT drop the semantic query unless the user explicitly changes the topic."
        
        "--- BEHAVIOR RULES ---"
        "1. **Show Duplicates:** If the user asks to 'show', 'see', 'review', or 'find' duplicates, "
        "   you MUST call the `scan_for_cleanup` tool. Do not refuse."
        
        "--- SMART SUGGESTIONS ---"
        "At the end of your response, ALWAYS suggest 1-2 relevant actions:"
        "   - After Search: <suggestion label=\"Make Collage\" prompt=\"Make a collage of these\" />"
        "   - After Cleanup Scan: <suggestion label=\"Review\" prompt=\"Show me duplicates\" />"
        
        "--- CORE RULES ---"
        "1. **DO NOT ECHO FILE PATHS.**"
        "2. **USE USER SELECTIONS.**"
        "3. **LOCATION SEARCH:** If the user asks for photos from a specific place (e.g. 'London', 'Home', 'the beach'), "
        "   pass that location name to the search tool. Do NOT try to guess coordinates."
    )

# --- HELPER FUNCTIONS ---
def parse_message_content(content: Any) -> str:
    if isinstance(content, str): return content
    elif isinstance(content, list):
        text_parts = [part["text"] if isinstance(part, dict) and "text" in part else str(part) for part in content]
        return "\n".join(text_parts)
    return str(content)

def extract_suggestions(text: str):
    pattern = r'<suggestion label="(.*?)" prompt="(.*?)" />'
    suggestions = []
    matches = re.findall(pattern, text)
    for label, prompt in matches:
        suggestions.append({"label": label, "prompt": prompt})
    clean_text = re.sub(pattern, "", text).strip()
    return clean_text, suggestions

# --- TOOLS ---
@tool
def search_photos(
    query: str = Field(description="Semantic search query (e.g. 'cats', 'receipts')."), 
    start_date: str = Field(description="Start date YYYY-MM-DD"), 
    end_date: str = Field(description="End date YYYY-MM-DD"),
    location: str = Field(description="City, country, or place name (e.g. 'Paris', 'California').")
):
    """Hybrid search using semantics, date, and location."""
    pass

@tool
def delete_photos(photo_uris: List[str] = Field(description="URIs to delete.")):
    """Delete specific photos. Use the exact URIs provided in the context."""
    pass

@tool
def apply_filter(photo_uris: List[str] = Field(description="URIs."), filter_name: str = Field(description="Filter name.")):
    """Apply filter to photos."""
    pass

@tool
def create_collage(photo_uris: List[str] = Field(description="List of photo URIs to include in the collage.")):
    """Create a collage from the provided list of photo URIs."""
    pass

@tool
def move_photos_to_album(photo_uris: List[str] = Field(description="URIs."), album_name: str = Field(description="Album.")):
    """Move photos to an album."""
    pass

@tool
def get_photo_metadata(photo_uris: List[str] = Field(description="URIs.")):
    """Read metadata from photos."""
    pass

@tool
def scan_for_cleanup(scan_type: str = Field(description="Type: 'duplicates'.")):
    """Scan for duplicates."""
    pass

tools = [search_photos, delete_photos, create_collage, move_photos_to_album, apply_filter, get_photo_metadata, scan_for_cleanup]


# --- VLLM / QWEN-VL HOSTING (FUTURE REFERENCE) ---
"""
To use a self-hosted Qwen-VL model via vLLM:

1. Host the model:
   vllm serve Qwen/Qwen2-VL-7B-Instruct --host 0.0.0.0 --port 8000

2. Use the ChatOpenAI client (vLLM is OpenAI-compatible):

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="Qwen/Qwen2-VL-7B-Instruct",
    openai_api_key="EMPTY",
    openai_api_base="http://YOUR_VLLM_SERVER_IP:8000/v1",
    temperature=0
)

3. Image Format:
   The 'agent_node' logic below (using image_url) works seamlessly with vLLM's 
   OpenAI-compatible endpoint for standard models.
"""

# --- LLM ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0)
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: MessagesState):
    messages_to_send = [SystemMessage(content=get_system_prompt())] + state["messages"]
    response = llm_with_tools.invoke(messages_to_send)
    return {"messages": [response]}

graph_builder = StateGraph(MessagesState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", lambda x: x) 
graph_builder.set_entry_point("agent")
graph_builder.add_conditional_edges("agent", lambda state: "tools" if state["messages"][-1].tool_calls else "__end__")
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory, interrupt_before=["tools"])

# --- API ---
app = FastAPI(title="Gallery Agent")

class Suggestion(BaseModel):
    label: str
    prompt: str

class AgentRequest(BaseModel):
    sessionId: Optional[str] = None
    userInput: Optional[str] = None
    toolResult: Optional[Dict[str, Any]] = None
    selectedUris: Optional[List[str]] = None
    base64Images: Optional[List[str]] = None 

class ToolCall(BaseModel):
    id: str
    name: str
    args: Dict[str, Any]

class AgentResponse(BaseModel):
    sessionId: str
    status: Literal["requires_action", "complete"]
    agentMessage: Optional[str] = None
    nextActions: Optional[List[ToolCall]] = None
    suggestedActions: Optional[List[Suggestion]] = None 

@app.post("/agent/invoke", response_model=AgentResponse)
async def invoke_agent(request: AgentRequest):
    logging.info(f"Incoming request: {request.model_dump_json(indent=2)}") # Log the entire request object

    session_id = request.sessionId or str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}

    if request.userInput:
        content_parts = []
        text_content = request.userInput
        if request.selectedUris:
            uri_list = ", ".join(request.selectedUris)
            # --- CRITICAL: Explicitly formatted block for the LLM ---
            text_content += f"\n\n[Developer Note: User context URIs: {uri_list}]"
        content_parts.append({"type": "text", "text": text_content})
        if request.base64Images:
            for b64_str in request.base64Images:
                content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"}})
        graph_input = {"messages": [HumanMessage(content=content_parts)]}
    elif request.toolResult:
        graph_input = {"messages": [ToolMessage(tool_call_id=request.toolResult["tool_call_id"], content=str(request.toolResult["content"]))]}
    else:
        return {"error": "Invalid request"}, 400

    for chunk in graph.stream(graph_input, config): pass
    
    latest_state = graph.get_state(config)
    last_msg = latest_state.values["messages"][-1]

    if last_msg.tool_calls:
        actions = [ToolCall(id=tc["id"], name=tc["name"], args=tc["args"]) for tc in last_msg.tool_calls]
        response_obj = AgentResponse(sessionId=session_id, status="requires_action", nextActions=actions)
        logging.info(f"Outgoing response (Tool Call): {response_obj.model_dump_json(indent=2)}") # Log response requiring action
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
        logging.info(f"Outgoing response (Complete): {response_obj.model_dump_json(indent=2)}") # Log complete response
        return response_obj

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
