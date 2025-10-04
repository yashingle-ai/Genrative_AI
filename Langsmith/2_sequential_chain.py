
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Get Gemini API key
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Define prompts
prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a 5-point summary from the following text:\n{text}",
    input_variables=["text"]
)

# Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",   # or "gemini-2.5-pro"
    google_api_key=gemini_api_key
)
parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic': 'selfless decisions'})

print(result)
