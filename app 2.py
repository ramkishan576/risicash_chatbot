from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
import requests
from dotenv import load_dotenv
import os
from datetime import datetime
from langgraph.prebuilt import ToolNode
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
import time
from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_ollama.chat_models import ChatOllama
import asyncio
from typing import Any, Dict, List
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition
from langchain_core.prompts import ChatPromptTemplate
#from datetime import datetime
import logging
#import redis
from typing import TypedDict, Annotated, Any
from langchain_core.runnables import RunnableConfig, Runnable
from langgraph.graph.message import AnyMessage, add_messages
import uvicorn

# Initialize logging
# logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# # Load environment variables from .env file
load_dotenv()
x_api_key = os.getenv('X-API-KEY')
# os.environ["TAVILY_API_KEY"] = os.getenv('Tavily_Search_API')
gemini_api_key = os.getenv('GEMINI_API_KEY')
BASE_URL = os.getenv('BASE_URL')

# Load LLM
generation_config = {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 40,
            "response_mime_type":  "application/json"} #"text/plain"

#llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.4, streaming=True, api_key=gpt_api_key) #, callbacks=[MyCustomSyncHandler(), MyCustomAsyncHandler()])
llm = ChatGoogleGenerativeAI(model=os.getenv('GEMINI_TXT_MODEL'), api_key = gemini_api_key ,generation_config=generation_config ,verbose=False)


@tool
def fetch_tour_information(Destination:str) -> str:
    """Fetch all corresponding tour information."""
    if not Destination:
        raise ValueError("No Destination provided.")
    
    Category = "religious"
    
    if not Category:
        raise ValueError("No Category provided.")
    
    # headers = {'x-api-key': x_api_key, 'Content-Type': 'application/json'}

    try:
        response = requests.get(f'{BASE_URL}/ai/get-mainPackage-and-package-by-detail?destinationName={Destination}&categoryName={Category}')
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException as e:
        return f"Request failed: {str(e)}"


## Handle Tools Error
def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls

    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

def create_tool_node_with_fallback(tools: list) -> dict:
    # Create a ToolNode with async error handling
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


tools = [
    fetch_tour_information
        ]

prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        **Objective:**
        You are a virtual assistant named LUNA for tour assistanse and provide any tour relevent ans by using `fetch_tour_information` only.
        Provide Greeting message `Hello ! This side LUNA , Your Virtual Tour Assistant . How can i help You today`.
        Ask Desired location for tour for easy process.
        If any other question out of knowledge provide simple `We do\'t server this here. Please contact to Administration`

        **Knowledge Sources:**      
        
        - `fetch_tour_information` : for for tour related package , budget , timing and other relevent details.
        """
    ),
    ("placeholder", "{messages}"),
]).partial() 


class State(TypedDict):
    messages: Annotated[list[Any], add_messages]

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig) -> dict:
        while True:
            try:
                state = {**state} 
                # print(f"Current State : {state}")
                result = self.runnable.invoke(state)
                # print(f'Current Result : {result}')
                # If the LLM happens to return an empty response, re-prompt for a response
                if not result.tool_calls and (
                    not result.content
                    or (isinstance(result.content, list) and not result.content[0].get("text"))
                ):
                    messages = state["messages"] + [("user", "Respond with a real output.")]
                    state = {**state, "messages": messages}
                else:
                    break
            except Exception as e:
                # Check if error is a rate limit error (429)
                if "429" in str(e):
                    print("Rate limit exceeded .")
                    error_message = 'We are facing some technical issue. Please direct contact to clinic.'
                    state["messages"].append({"role": "assistant", "content": error_message})
                    return {"messages": state["messages"]}

        return {"messages": result}
    

runnable_chain =  prompt_template | llm.bind_tools(tools) #
builder = StateGraph(State)
# Define nodes: these do the work
builder.add_node("assistant", Assistant(runnable_chain))
builder.add_node("tools", create_tool_node_with_fallback(tools))
# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")

# The checkpointer lets the graph persist its state
# this is a complete memory for the entire graph.
memory = MemorySaver()
runnable_graph = builder.compile(checkpointer=memory)


async def chatbot_session(input_text, config):
    #result = runnable_graph.batch([{"messages": ("user", input_text)}], config, stream_mode="values")
    print(config)
    result = await runnable_graph.ainvoke({"messages": [("user", input_text)]}, config, stream_mode="values")
    #await asyncio.sleep(0.001)
    return result['messages'][-1].content #result[-1]['messages'][-1].content      


# Initialize FastAPI
app = FastAPI()

class ChatRequest(BaseModel):
    human: str

@app.get("/")
def home():
    return {"message": "Welcome to the Chatbot AI by LAKHAN SINGH!"}
    

@app.post("/api/chatbot/tour_ai")
async def chatbot(request: Request, chat_request: ChatRequest, x_api_key: str = Header(None)):
    if x_api_key != os.getenv('X-API-KEY'):
        raise HTTPException(status_code=403, detail="Unauthorized")

    # Query parameters
    session_id = request.query_params.get('session_id')

    if not session_id:
        raise HTTPException(status_code=400, detail="Missing required parameters")

    user_input = chat_request.human
    if not user_input:
        raise HTTPException(status_code=400, detail="Please provide input text")

    config = {
        "configurable": {
            "thread_id": session_id,
        }
    }

    print(config , user_input)

    try:
        chatbot_response = await chatbot_session(user_input, config)
        return {"model_response": chatbot_response}
    
    except Exception as e:
        logging.error(f"Error during chatbot session: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during chatbot session")
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
