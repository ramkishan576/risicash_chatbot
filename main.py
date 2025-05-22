# import os
# import re
# import json
# import logging
# from typing import Dict, Any
# from datetime import datetime
# import requests
# from fastapi import FastAPI, HTTPException, Request
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel, ValidationError
# from langchain_core.runnables import RunnableConfig, Runnable
# from langchain.tools import tool
# from langchain.agents import initialize_agent, AgentType
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.schema.messages import SystemMessage
# from dotenv import load_dotenv
 
# load_dotenv()
 
# # Configure Logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
# )
# logger = logging.getLogger("LUNA_Travel_Assistant")
 
# # Environment Variables and Constants
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME")
# API_BASE_URL = os.getenv("API_BASE_URL")
# DEFAULT_TEMPERATURE = 0.7
 
# if not GEMINI_API_KEY:
#     logger.error("GEMINI_API_KEY environment variable is not set.")
#     raise ValueError("Environment variable 'GEMINI_API_KEY' is required.")
 
# if not API_BASE_URL:
#     logger.error("API_BASE_URL environment variable is not set.")
#     raise ValueError("Environment variable 'API_BASE_URL' is required.")
 
# # FastAPI App Initialization
# app = FastAPI(
#     title="LUNA Travel Assistant API",
#     description="API for chatbot to assist in finding travel packages",
#     version="1.0.0",
# )
 
# # Pydantic Request Model
# class ChatRequest(BaseModel):
#     message: str
 
# # Utility Functions
# def clean_json_string(json_str: str) -> str:
#     """
#     Clean control characters and BOM from JSON response string.
#     """
#     cleaned = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", json_str)
#     cleaned = cleaned.lstrip("\ufeff")
#     return cleaned
 
# def extract_query_param(query: str, key: str) -> str:
#     """
#     Extract parameter value from query string using regex.
#     Handles cases where parameter might be followed by another parameter or end of string.
#     """
#     match = re.search(rf"{key}:([^\n\r:]+?)(?:\s+\w+:|$)", query)
#     return match.group(1).strip() if match else ""
 
# # LangChain Tool Definition
# @tool
# def fetch_travel_packages(config: RunnableConfig) -> str:
#     """Fetch travel packages based on destination and category."""
#     destination = config.get('configurable', {}).get('destinationName', None)
#     category = config.get('configurable', {}).get('categoryName', None)

#     if not destination:
#         raise ValueError("No destinationName configured.")

#     headers = {'Content-Type': 'application/json'}
#     params = {"destinationName": destination}
#     if category:
#         params["categoryName"] = category

#     try:
#         s_time = time.time()
#         response = requests.get(
#             f"{API_BASE_URL}/ai/get-mainPackage-and-package-by-detail",
#             headers=headers,
#             params=params
#         )

#         if response.status_code == 200:
#             print(f"Total time taken to fetch travel packages: {time.time() - s_time} seconds\n")
#             return response.json()
#         else:
#             return None

#     except requests.exceptions.RequestException as e:
#         return f"Request failed: {str(e)}"

 
# # Initialize LLM with LangChain
# llm = ChatGoogleGenerativeAI(
#     model=GEMINI_MODEL_NAME,
#     temperature=DEFAULT_TEMPERATURE,
#     google_api_key=GEMINI_API_KEY,
# )
 
# # Define system prompt with clear instructions
# SYSTEM_PROMPT = SystemMessage(
#     content="""
# **Objective:**
# You are LUNA, a smart and friendly virtual travel assistant specialized in helping users discover travel packages.
 
# **Responsibilities:**
# - Understand user queries related to travel packages.
# - Identify the {destination} and {category} of packages the user is interested in from their query: {description}.
# - Use the tool `fetch_travel_packages` to fetch accurate and up-to-date package information based on user input.
# - Provide concise, helpful, and clear answers based on the data retrieved.
# - If the user's request is vague or missing key details like {destination} or {category}, ask polite and clarifying questions to gather more information before proceeding.
# - Handle any destination and any category dynamically.
# - If a category is not available, inform the user politely.
# - Always maintain a friendly and professional tone.
 
# **Guidelines:**
# - The {destination} refers to places like "Rishikesh", "Manali", "Goa", etc.
# - The {category} refers to the type of travel packages such as {"religious tours"}, {"adventure activities"}, {"yoga retreats"}, {"wellness packages"}, etc.
# - If the API returns an error or no packages are found, inform the user politely and suggest trying a different {category} or {destination}.
# - Keep responses short and to the point unless the user asks for more details.
# """
# )
 
# # Initialize LangChain Agent
# agent = initialize_agent(
#     tools=[fetch_travel_packages],
#     llm=llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=False,
#     agent_kwargs={"system_message": SYSTEM_PROMPT},
# )
 
# def chatbot_interaction(user_message: str) -> str:
#     """
#     Process the user input through the LangChain agent and return the response.
 
#     Args:
#         user_message (str): The user input message.
 
#     Returns:
#         str: The chatbot's response.
#     """
#     try:
#         logger.info(f"User message received: {user_message}")
#         response = agent.run(user_message)
#         logger.info("Response generated successfully.")
#         return response
 
#     except Exception as exc:
#         logger.error(f"Agent execution error: {exc}")
#         return "Sorry, something went wrong while processing your request. Please try again."
 
# # FastAPI Exception Handlers
# @app.exception_handler(ValidationError)
# async def validation_exception_handler(request: Request, exc: ValidationError):
#     logger.warning(f"Validation error: {exc}")
#     return JSONResponse(
#         status_code=422,
#         content={"detail": exc.errors(), "message": "Invalid input."},
#     )
 
# @app.exception_handler(Exception)
# async def general_exception_handler(request: Request, exc: Exception):
#     logger.error(f"Unhandled exception: {exc}")
#     return JSONResponse(
#         status_code=500,
#         content={"message": "Internal server error. Please try again later."},
#     )
 
# # FastAPI Chat Endpoint
# @app.post("/chat", summary="Chat with LUNA Travel Assistant")
# async def chat_endpoint(request: ChatRequest):
#     """
#     POST endpoint to send a message to the chatbot and receive a response.
 
#     Args:
#         request (ChatRequest): The request body containing the user's message.
 
#     Returns:
#         dict: A dictionary containing the chatbot's response.
#     """
#     response_text = chatbot_interaction(request.message)
#     return {"response": response_text}
 
# # Run app with Uvicorn if executed as main program
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
 
 
 
 
 
 
 
 
# import os
# import json
# import time
# import logging
# import requests
# from typing import Optional
# from fastapi import FastAPI, HTTPException, Request
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel, ValidationError
# from langchain.tools import tool
# from dotenv import load_dotenv

# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.agents import create_tool_calling_agent, AgentExecutor

# # Load environment variables
# load_dotenv()

# # Configure Logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
# )
# logger = logging.getLogger("LUNA_Travel_Assistant")

# # Environment Variables
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-pro")
# API_BASE_URL = os.getenv("API_BASE_URL")
# DEFAULT_TEMPERATURE = 0.7

# if not GEMINI_API_KEY:
#     raise ValueError("Environment variable 'GEMINI_API_KEY' is required.")
# if not API_BASE_URL:
#     raise ValueError("Environment variable 'API_BASE_URL' is required.")

# # Initialize FastAPI app
# app = FastAPI(
#     title="LUNA Travel Assistant API",
#     description="Chatbot API to assist in finding travel packages",
#     version="1.0.0",
# )

# # Pydantic model for request body
# class ChatRequest(BaseModel):
#     message: str

# # LangChain Tool to fetch travel packages
# @tool
# def fetch_travel_packages(destination: str, category: Optional[str] = None) -> str:
#     """Fetch travel packages based on destination and optional category."""
#     if not destination:
#         raise ValueError("Destination is required.")

#     headers = {'Content-Type': 'application/json'}
#     params = {"destinationName": destination}
#     if category:
#         params["categoryName"] = category

#     try:
#         s_time = time.time()
#         response = requests.get(
#             f"{API_BASE_URL}/ai/get-mainPackage-and-package-by-detail",
#             headers=headers,
#             params=params
#         )
#         if response.status_code == 200:
#             print(f"Fetched in {time.time() - s_time:.2f}s")
#             return json.dumps(response.json())
#         else:
#             return "Sorry, couldn't fetch travel packages. Try again."
#     except requests.exceptions.RequestException as e:
#         return f"Request failed: {str(e)}"

# # Initialize Gemini LLM
# llm = ChatGoogleGenerativeAI(
#     model=GEMINI_MODEL_NAME,
#     temperature=DEFAULT_TEMPERATURE,
#     google_api_key=GEMINI_API_KEY,
# )

# # Prompt Template with required 'agent_scratchpad'
# prompt = ChatPromptTemplate.from_messages([
#     ("system", """
# You are LUNA, a smart and friendly virtual travel assistant specialized in helping users discover travel packages.

# Responsibilities:
# - Understand user queries related to travel packages.
# - Identify the destination and category of packages the user is interested in.
# - Use the tool 'fetch_travel_packages' to fetch up-to-date package info.
# - Ask clarifying questions if details are missing.
# - If no results, suggest alternatives.
# - Keep answers concise, friendly, and helpful.
#     """),
#     ("human", "{input}"),
#     MessagesPlaceholder(variable_name="agent_scratchpad"),
# ])

# # Create agent and executor
# tools = [fetch_travel_packages]
# agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# # Function to handle conversation
# def chatbot_interaction(user_message: str) -> str:
#     try:
#         logger.info(f"User: {user_message}")
#         response = agent_executor.invoke({"input": user_message})
#         logger.info("Bot response sent.")
#         return response.get("output", "Sorry, I couldn't generate a response.")
#     except Exception as exc:
#         logger.error(f"Error: {exc}")
#         return "Sorry, something went wrong. Please try again."

# # Chat API endpoint
# @app.post("/chat", summary="Chat with LUNA Travel Assistant")
# async def chat_endpoint(request: ChatRequest):
#     response_text = chatbot_interaction(request.message)
#     return {"response": response_text}

# # Exception Handlers
# @app.exception_handler(ValidationError)
# async def validation_exception_handler(request: Request, exc: ValidationError):
#     return JSONResponse(
#         status_code=422,
#         content={"detail": exc.errors(), "message": "Invalid input."},
#     )

# @app.exception_handler(Exception)
# async def general_exception_handler(request: Request, exc: Exception):
#     return JSONResponse(
#         status_code=500,
#         content={"message": "Internal server error. Try again later."},
#     )

# # Uvicorn runner
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)






# import os
# import re
# import json
# import logging
# import time
# from typing import Dict, Any
# from datetime import datetime
# import requests
# from fastapi import FastAPI, HTTPException, Request
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel, ValidationError
# from langchain_core.runnables import RunnableConfig, Runnable
# from langchain.tools import tool
# from langchain.agents import initialize_agent, AgentType
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.schema.messages import SystemMessage
# from dotenv import load_dotenv
 
# load_dotenv()
 
# # Configure Logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
# )
# logger = logging.getLogger("LUNA_Travel_Assistant")
 
# # Environment Variables and Constants
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME")
# API_BASE_URL = os.getenv("API_BASE_URL")
# DEFAULT_TEMPERATURE = 0.7
 
# if not GEMINI_API_KEY:
#     logger.error("GEMINI_API_KEY environment variable is not set.")
#     raise ValueError("Environment variable 'GEMINI_API_KEY' is required.")
 
# if not API_BASE_URL:
#     logger.error("API_BASE_URL environment variable is not set.")
#     raise ValueError("Environment variable 'API_BASE_URL' is required.")
 
# # FastAPI App Initialization
# app = FastAPI(
#     title="LUNA Travel Assistant API",
#     description="API for chatbot to assist in finding travel packages",
#     version="1.0.0",
# )
 
# # Pydantic Request Model
# class ChatRequest(BaseModel):
#     message: str
 
# # Utility Functions
# def clean_json_string(json_str: str) -> str:
#     """
#     Clean control characters and BOM from JSON response string.
#     """
#     cleaned = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", json_str)
#     cleaned = cleaned.lstrip("\ufeff")
#     return cleaned
 
# def extract_query_param(query: str, key: str) -> str:
#     """
#     Extract parameter value from query string using regex.
#     Handles cases where parameter might be followed by another parameter or end of string.
#     """
#     match = re.search(rf"{key}:([^\n\r:]+?)(?:\s+\w+:|$)", query)
#     return match.group(1).strip() if match else ""
 
# # LangChain Tool Definition
# @tool(description="Fetch travel packages for a given destination and category.")
# def fetch_travel_packages(destination: str, category: str = None) -> str:
#     if not destination:
#         raise ValueError("Destination is required.")
    
#     headers = {'Content-Type': 'application/json'}
#     params = {"destinationName": destination}
#     if category:
#         params["categoryName"] = category

#     try:
#         logger.info(f"Fetching packages for destination='{destination}' category='{category}'")
#         start_time = time.time()
#         response = requests.get(
#             f"{API_BASE_URL}/ai/get-mainPackage-and-package-by-detail",
#             headers=headers,
#             params=params
#         )
#         logger.info(f"API Response code: {response.status_code}")
#         json_data = response.json()
#         logger.info(f"API response data: {json.dumps(json_data)[:500]}")
#         logger.info(f"Packages fetched in {time.time() - start_time:.2f}s")

#         if response.status_code == 200:
#             # Check for API-specific error in payload
#             if json_data.get("code") == 0:
#                 # API returned error message
#                 if "Invalid category name" in json_data.get("message", ""):
#                     return (
#                         f"Sorry, the category '{category}' is not recognized. "
#                         "Please provide a valid category such as 'adventure', 'wellness', 'religious tours', etc."
#                     )
#                 else:
#                     return f"API Error: {json_data.get('message')}"
#             # Normal successful response
#             return json.dumps(json_data)
#         else:
#             logger.warning(f"API call failed with status {response.status_code}: {response.text}")
#             return "Sorry, couldn't fetch travel packages. Try again."
#     except requests.RequestException as e:
#         logger.error(f"Request error: {e}")
#         return f"Request failed: {str(e)}"
 
# # Initialize LLM with LangChain
# llm = ChatGoogleGenerativeAI(
#     model=GEMINI_MODEL_NAME,
#     temperature=DEFAULT_TEMPERATURE,
#     google_api_key=GEMINI_API_KEY,
# )
 
# # Define system prompt with clear instructions
# SYSTEM_PROMPT = SystemMessage(
#     content="""
# **Objective:**
# You are LUNA, a smart and friendly virtual travel assistant specialized in helping users discover travel packages.
 
# **Responsibilities:**
# - Understand user queries related to travel packages.
# - Identify the {destination} and {category} of packages the user is interested in from their query: {description}.
# - Use the tool `fetch_travel_packages` to fetch accurate and up-to-date package information based on user input.
# - Provide concise, helpful, and clear answers based on the data retrieved.
# - If the user's request is vague or missing key details like {destination} or {category}, ask polite and clarifying questions to gather more information before proceeding.
# - Handle any destination and any category dynamically.
# - If a category is not available, inform the user politely.
# - Always maintain a friendly and professional tone.
 
# **Guidelines:**
# - The {destination} refers to places like "Rishikesh", "Manali", "Goa", etc.
# - The {category} refers to the type of travel packages such as {"religious tours"}, {"adventure activities"}, {"yoga retreats"}, {"wellness packages"}, etc.
# - If the API returns an error or no packages are found, inform the user politely and suggest trying a different {category} or {destination}.
# - Keep responses short and to the point unless the user asks for more details.
# """
# )
 
# # Initialize LangChain Agent
# agent = initialize_agent(
#     tools=[fetch_travel_packages],
#     llm=llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=False,
#     agent_kwargs={"system_message": SYSTEM_PROMPT},
# )
 
# def chatbot_interaction(user_message: str) -> str:
#     """
#     Process the user input through the LangChain agent and return the response.
 
#     Args:
#         user_message (str): The user input message.
 
#     Returns:
#         str: The chatbot's response.
#     """
#     try:
#         logger.info(f"User message received: {user_message}")
#         response = agent.run(user_message)
#         logger.info("Response generated successfully.")
#         return response
 
#     except Exception as exc:
#         logger.error(f"Agent execution error: {exc}")
#         return "Sorry, something went wrong while processing your request. Please try again."
 
# # FastAPI Exception Handlers
# @app.exception_handler(ValidationError)
# async def validation_exception_handler(request: Request, exc: ValidationError):
#     logger.warning(f"Validation error: {exc}")
#     return JSONResponse(
#         status_code=422,
#         content={"detail": exc.errors(), "message": "Invalid input."},
#     )
 
# @app.exception_handler(Exception)
# async def general_exception_handler(request: Request, exc: Exception):
#     logger.error(f"Unhandled exception: {exc}")
#     return JSONResponse(
#         status_code=500,
#         content={"message": "Internal server error. Please try again later."},
#     )
 
# # FastAPI Chat Endpoint
# @app.post("/chat", summary="Chat with LUNA Travel Assistant")
# async def chat_endpoint(request: ChatRequest):
#     """
#     POST endpoint to send a message to the chatbot and receive a response.
 
#     Args:
#         request (ChatRequest): The request body containing the user's message.
 
#     Returns:
#         dict: A dictionary containing the chatbot's response.
#     """
#     response_text = chatbot_interaction(request.message)
#     return {"response": response_text}
 
# # Run app with Uvicorn if executed as main program
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)




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
 
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY environment variable is not set.")
    raise ValueError("Environment variable 'GEMINI_API_KEY' is required.")
 
if not API_BASE_URL:
    logger.error("API_BASE_URL environment variable is not set.")
    raise ValueError("Environment variable 'API_BASE_URL' is required.")
 
# FastAPI App Initialization
app = FastAPI(
    title="LUNA Travel Assistant API",
    description="API for chatbot to assist in finding travel packages",
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
 
def extract_query_param(query: str, key: str) -> str:
    """
    Extract parameter value from query string using regex.
    Handles cases where parameter might be followed by another parameter or end of string.
    """
    match = re.search(rf"{key}:([^\n\r:]+?)(?:\s+\w+:|$)", query)
    return match.group(1).strip() if match else ""
 
# LangChain Tool Definition
@tool
def fetch_travel_packages(query: str) -> Dict[str, Any]:
    """
    Fetch travel packages for any destination based on the user's query.
   
    The query can include filters like destination and category.
    Example query: "destination:Manali category:adventure"
    """
    try:
        destinationName = extract_query_param(query, "destination")
        categoryName = extract_query_param(query, "category")
       
        # Log the API request details
        logger.info(f"Making API request to: {API_BASE_URL}")
        logger.info(f"Parameters: destination={destinationName}, category={categoryName}")
       
        url = f"{API_BASE_URL}/ai/get-mainPackage-and-package-by-detail"
        params = {"destinationName": destinationName, "categoryName": categoryName}
       
        # Add headers if needed
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
       
        response = requests.get(
            url,
            params=params,
            headers=headers
        )
       
        # Log response status
        logger.info(f"API Response Status: {response.status_code}")
       
        response.raise_for_status()
        cleaned_json = clean_json_string(response.text)
        data = json.loads(cleaned_json)
       
        # Log successful data retrieval
        logger.info(f"Successfully retrieved data for destination: {destinationName}, category: {categoryName}")
        return data
       
    except requests.exceptions.RequestException as e:
        logger.error(f"API Request Error: {str(e)}")
        return {"error": f"API request failed: {str(e)}"}
    except json.JSONDecodeError as e:
        logger.error(f"JSON Decode Error: {str(e)}")
        return {"error": f"Invalid JSON response: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected Error: {str(e)}")
        return {"error": f"An unexpected error occurred: {str(e)}"}
 
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
You are LUNA, a smart and friendly virtual travel assistant specialized in helping users discover travel packages.
 
**Responsibilities:**
- Understand user queries related to travel packages.
- Identify the {destination} and {category} of packages the user is interested in from their query: {description}.
- Use the tool `fetch_travel_packages` to fetch accurate and up-to-date package information based on user input.
- Provide concise, helpful, and clear answers based on the data retrieved.
- If the user's request is vague or missing key details like {destination} or {category}, ask polite and clarifying questions to gather more information before proceeding.
- Handle any destination and any category dynamically.
- If a category is not available, inform the user politely.
- Always maintain a friendly and professional tone.
 
**Guidelines:**
- The {destination} refers to places like "Rishikesh", "Manali", "Goa", etc.
- The {category} refers to the type of travel packages such as {"religious tours"}, {"adventure activities"}, {"yoga retreats"}, {"wellness packages"}, etc.
- If the API returns an error or no packages are found, inform the user politely and suggest trying a different {category} or {destination}.
- Keep responses short and to the point unless the user asks for more details.
"""
)
 
# Initialize LangChain Agent
agent = initialize_agent(
    tools=[fetch_travel_packages],
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
@app.post("/chat", summary="Chat with LUNA Travel Assistant")
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