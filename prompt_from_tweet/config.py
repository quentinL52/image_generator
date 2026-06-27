from langchain_openai import ChatOpenAI
from typing import Dict, List, Any, Tuple, Optional, Type
from crewai import LLM
import os
from dotenv import load_dotenv 

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model_openai = "gpt-4o"  

def crew_openai():
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        api_key=OPENAI_API_KEY
    )
    return llm