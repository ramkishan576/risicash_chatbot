# 🧭 LUNA - Virtual Tour Assistant

**LUNA** is an intelligent virtual assistant built with **FastAPI** and **LangGraph**, designed to assist users in discovering **religious tour packages** based on their preferred destination. Powered by **Google Generative AI (Gemini)** and integrated with real-time backend APIs, LUNA delivers a seamless and informative user experience for tour planning.

---

## ✨ Key Features

- 💬 **Conversational Tour Assistance**  
  Responds to natural language queries regarding religious tours.

- 🔗 **Real-time Package Retrieval**  
  Uses a custom `fetch_tour_information` tool to retrieve tour details from a backend API.

- 🧠 **AI-powered Intelligence**  
  Utilizes **Google Gemini** to generate context-aware, dynamic responses.

- 🔒 **Secure API Access**  
  Authenticated via `X-API-KEY` header to ensure secure access to endpoints.

- 🔁 **Session Persistence with LangGraph**  
  Maintains conversation flow using LangGraph's stateful session handling.

- ⚠️ **Robust Error Handling**  
  Built-in fallback mechanisms to handle tool invocation errors gracefully.

---

## 🛠️ Prerequisites

Make sure you have the following installed:

- Python 3.9+
- pip

Then, install all required dependencies:

```bash
pip install -r requirements.txt
