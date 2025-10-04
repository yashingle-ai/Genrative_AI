from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

# Load .env
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",   
    google_api_key=gemini_api_key
)

prompt = PromptTemplate.from_template("{question}")
parser = StrOutputParser()

chain = prompt | model | parser

@traceable
def ask_question(q: str):
    return chain.invoke({"question": q})

result = ask_question("What is the capital of Peru?")
print(result)
