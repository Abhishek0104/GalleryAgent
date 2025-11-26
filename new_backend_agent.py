import os
from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
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
        "1. **MANUAL SELECTION:** If the user has manually selected photos (indicated by 'User context URIs' in the input), "
        "   and asks to act on 'them', 'these', or 'the selection', you MUST set `use_current_selection=True` in the tool call. "
        "   Do NOT repeat the list of URIs in the `photo_uris` argument."
        
        "2. **LARGE SEARCH RESULTS:** If a search yields many results and the user wants to act on ALL of them (e.g., 'move them all'), "
        "   set `act_on_last_search_results=True`. Do NOT list the URIs."
        
        "3. **SPECIFIC SUBSETS:** Only populate `photo_uris` explicitly if the user asks for a *subset* of the context "
        "   (e.g., 'only the first one', 'just the cat photos from that group')."
        
        "--- CONTEXT AWARENESS ---"
        "If the user refines a search (e.g., 'show only the ones from 2023'), you MUST maintain the original search query from the conversation history. "
        "Example: User said 'find cats', then 'from 2023'. You call search_photos(query='cats', start_date='2023-01-01'...). "
        "Do NOT drop the semantic query unless the user explicitly changes the topic."
        
        "--- BEHAVIOR RULES ---"
        "1. **People Search (CRITICAL):** If the user mentions a Proper Noun or Name (e.g., 'Modi', 'Alice', 'Mom', 'Dad'), "
        "   you MUST put that name in the `people` list argument of `search_photos`. "
        "   Do NOT put names in the `query` string if they are clearly people."
        "   Example: 'Show Modi' -> search_photos(people=['Modi']) (CORRECT) vs search_photos(query='Modi') (WRONG)."
        
        "2. **Show Duplicates:** If the user asks to 'show', 'see', 'review', or 'find' duplicates, "
        "   you MUST call the `scan_for_cleanup` tool. Do not refuse."
        
        "3. **Describe Images:** If the user asks 'What is in these photos?', 'Describe them', 'What food is this?', or generally asks a question about the visual content of photos, "
        "   you MUST use the `describe_images` tool. Pass the user's question as the `question` argument."
        
        "--- SMART SUGGESTIONS ---"
        "At the end of your response, ALWAYS suggest 1-2 relevant actions:"
        "   - After Search: <suggestion label=\"Describe\" prompt=\"What is in these photos?\" />"
        "   - After Selection: <suggestion label=\"Collage\" prompt=\"Make a collage\" />"
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
    query: str = Field(default="", description="Semantic search query (e.g. 'cats', 'receipts'). Leave empty if searching only by person/date."), 
    start_date: Optional[str] = Field(default=None, description="Start date YYYY-MM-DD"), 
    end_date: Optional[str] = Field(default=None, description="End date YYYY-MM-DD"),
    location: Optional[str] = Field(default=None, description="City, country, or place name."),
    people: List[str] = Field(default=[], description="List of person names to filter by. ALWAYS use this for names (e.g. 'Modi', 'Me').")
):
    """Hybrid search using semantics, date, location, and people. Populate ONLY the fields relevant to the user's request."""
    pass

@tool
def delete_photos(
    photo_uris: List[str] = Field(default=[], description="Specific URIs to delete."),
    act_on_last_search_results: bool = Field(default=False, description="If True, deletes ALL photos found in the last search."),
    use_current_selection: bool = Field(default=False, description="If True, deletes the photos currently selected by the user.")
):
    """Delete photos based on selection, search history, or specific URIs."""
    pass

@tool
def apply_filter(
    photo_uris: List[str] = Field(default=[], description="URIs."), 
    filter_name: str = Field(description="Filter name."),
    act_on_last_search_results: bool = Field(default=False, description="If True, applies filter to last search results."),
    use_current_selection: bool = Field(default=False, description="If True, applies filter to current user selection.")
):
    """Apply filter to photos."""
    pass

@tool
def create_collage(
    photo_uris: List[str] = Field(default=[], description="List of photo URIs to include in the collage."),
    use_current_selection: bool = Field(default=False, description="If True, uses the currently selected photos.")
):
    """Create a collage."""
    pass

@tool
def move_photos_to_album(
    album_name: str = Field(description="Target Album Name."),
    photo_uris: List[str] = Field(default=[], description="Specific URIs to move."),
    act_on_last_search_results: bool = Field(default=False, description="If True, moves last search results."),
    use_current_selection: bool = Field(default=False, description="If True, moves currently selected photos.")
):
    """Move photos to an album."""
    pass

@tool
def get_photo_metadata(
    photo_uris: List[str] = Field(default=[], description="URIs."),
    use_current_selection: bool = Field(default=False, description="If True, uses currently selected photos.")
):
    """Read metadata from photos."""
    pass

@tool
def scan_for_cleanup(scan_type: str = Field(description="Type: 'duplicates'.")):
    """Scan for duplicates."""
    pass

# --- NEW TOOL: DESCRIBE IMAGES ---
@tool
def describe_images(
    photo_uris: List[str] = Field(default=[], description="List of photo URIs to describe."),
    question: str = Field(default="Describe these images", description="The specific question to answer about the images (e.g. 'What food is this?')."),
    act_on_last_search_results: bool = Field(default=False, description="If True, describes the results of the last search."),
    use_current_selection: bool = Field(default=False, description="If True, describes the currently selected photos.")
):
    """Analyze and describe the visual content of photos using an AI model. Use this for 'What is this?', 'Describe', or visual questions."""
    pass

tools = [search_photos, delete_photos, create_collage, move_photos_to_album, apply_filter, get_photo_metadata, scan_for_cleanup, describe_images]

# --- LLM ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0)
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
    logging.info(f"Incoming request: {request.model_dump_json(indent=2)}")

    session_id = request.sessionId or str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}

    if request.userInput:
        content_parts = []
        text_content = request.userInput
        if request.selectedUris:
            # --- OPTIMIZATION: We don't need to send ALL URIs to the LLM anymore ---
            # Just send a count and maybe the first few for context
            count = len(request.selectedUris)
            preview = ", ".join(request.selectedUris[:5])
            if count > 5:
                preview += f", ... and {count-5} more"
            text_content += f"\n\n[Developer Note: User context URIs: {preview}. Count: {count}]"
        content_parts.append({"type": "text", "text": text_content})
        # --- CRITICAL CHANGE: If base64 images are provided in the request, 
        #     it means the client is responding to a `describe_images` tool call.
        #     We pass them to the LLM so it can "see" them.
        if request.base64Images:
            for b64_str in request.base64Images:
                content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"}})
        graph_input = {"messages": [HumanMessage(content=content_parts)]}
    
    elif request.toolResult:
        # Handle tool result normally
        graph_input = {"messages": [ToolMessage(tool_call_id=request.toolResult["tool_call_id"], content=str(request.toolResult["content"]))]}
        
        # --- HANDLING "DESCRIBE IMAGES" LOOP ---
        # If the tool result contains Base64 images (which happens when the client 
        # executes 'describe_images' and sends back visual data), we need to send 
        # those images to the LLM as a HumanMessage immediately following the ToolMessage.
        # NOTE: In standard LangGraph/LangChain, tools return text. 
        # If we want the LLM to "see" the result of `describe_images`, the client usually 
        # sends the images in the NEXT turn. 
        # However, we can pass them here if the client sends them in `base64Images` field 
        # ALONG with `toolResult`.
        
        if request.base64Images:
             # Append a HumanMessage with the visual data so the LLM can answer the question
             # "Here are the images you asked for..."
             img_content = [{"type": "text", "text": "Here are the images for analysis:"}]
             for b64_str in request.base64Images:
                img_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"}})
             
             graph_input["messages"].append(HumanMessage(content=img_content))

    else:
        return {"error": "Invalid request"}, 400

    for chunk in graph.stream(graph_input, config): pass
    
    latest_state = graph.get_state(config)
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
        logging.info(f"Outgoing response (Complete): {response_obj.model_dump_json(indent=2)}")
        return response_obj

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
