import os
import re
import json
import logging
from typing import Dict, Any

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.messages import SystemMessage
from dotenv import load_dotenv

load_dotenv()

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("LUNA_Travel_Assistant")


# Environment Variables and Constants
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME")
API_BASE_URL = os.getenv("API_BASE_URL")
DEFAULT_TEMPERATURE = 0.7
REQUEST_TIMEOUT = 10  # seconds

if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY environment variable is not set.")
    raise ValueError("Environment variable 'GEMINI_API_KEY' is required.")


# FastAPI App Initialization
app = FastAPI(
    title="LUNA Travel Assistant API",
    description="API for chatbot to assist in finding Rishikesh travel packages",
    version="1.0.0",
)


# Pydantic Request Model
class ChatRequest(BaseModel):
    message: str


# Utility Functions
def clean_json_string(json_str: str) -> str:
    """
    Clean control characters and BOM from JSON response string.
    """
    cleaned = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", json_str)
    cleaned = cleaned.lstrip("\ufeff")
    return cleaned


# LangChain Tool Definition
@tool
def fetch_rishikesh_packages(query: str) -> Dict[str, Any]:
    """
    Fetch travel packages for Rishikesh based on the user's query.
    
    The query can include filters like destination and category.
    Example query: "destination:Rishikesh category:adventure"
    """
    try:
        # Expect query like: "destination:Rishikesh category:adventure"
        destinationName = re.search(r"destination:(\w+)", query)
        categoryName = re.search(r"category:(\w+)", query)
        destinationName = destinationName.group(1) if destinationName else "Rishikesh"
        categoryName = categoryName.group(1) if categoryName else ""
        
        url = f"{API_BASE_URL}/ai/get-mainPackage-and-package-by-detail"
        params = {"destinationName": destinationName, "categoryName": categoryName}
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        cleaned_json = clean_json_string(response.text)
        data = json.loads(cleaned_json)
        return data
    except Exception as e:
        return {"error": str(e)}



# Initialize LLM with LangChain
llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL_NAME,
    temperature=DEFAULT_TEMPERATURE,
    google_api_key=GEMINI_API_KEY,
)

# Define system prompt with clear instructions
SYSTEM_PROMPT = SystemMessage(
    content="""
**Objective:**
You are LUNA, a smart and friendly virtual travel assistant specialized in helping users discover travel packages specifically for Rishikesh.

**Responsibilities:**
- Understand user queries related to travel packages in Rishikesh.
- Identify the destination and category of packages the user is interested in.
- Use the tool `fetch_rishikesh_packages` to fetch accurate and up-to-date package information based on user input.
- Provide concise, helpful, and clear answers based on the data retrieved.
- If the user’s request is vague or missing key details like destination or category, ask polite and clarifying questions to gather more information before proceeding.
- Handle categories like "religious", "adventure", "yoga", "wellness", etc., and inform the user if a category is not available.
- Always maintain a friendly and professional tone.

**Guidelines:**
- The destination is usually "Rishikesh", but be ready to handle or correct if the user inputs a different destination.
- Categories refer to the type of travel packages such as religious tours, adventure activities, yoga retreats, wellness packages, etc.
- If the API returns an error or no packages found, inform the user politely and suggest trying a different category or destination.
- Do not provide irrelevant or speculative information — stick to facts retrieved from the API.
- Keep responses short and to the point unless the user asks for more details.
"""
)

# Initialize LangChain Agent
agent = initialize_agent(
    tools=[fetch_rishikesh_packages],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    agent_kwargs={"system_message": SYSTEM_PROMPT},
)


def chatbot_interaction(user_message: str) -> str:
    """
    Process the user input through the LangChain agent and return the response.

    Args:
        user_message (str): The user input message.

    Returns:
        str: The chatbot's response.
    """
    try:
        logger.info(f"User message received: {user_message}")
        response = agent.run(user_message)
        logger.info("Response generated successfully.")
        return response

    except Exception as exc:
        logger.error(f"Agent execution error: {exc}")
        return "Sorry, something went wrong while processing your request. Please try again."


# FastAPI Exception Handlers
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "message": "Invalid input."},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error. Please try again later."},
    )


# FastAPI Chat Endpoint
@app.post("/chat", summary="Chat with SWAMI Travel Assistant")
async def chat_endpoint(request: ChatRequest):
    """
    POST endpoint to send a message to the chatbot and receive a response.

    Args:
        request (ChatRequest): The request body containing the user's message.

    Returns:
        dict: A dictionary containing the chatbot's response.
    """
    response_text = chatbot_interaction(request.message)
    return {"response": response_text}


# Run app with Uvicorn if executed as main program
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
