import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    groq_api_key="Your_Key",
    model_name="llama-3.1-8b-instant"
)

response = llm.invoke("what is AI?")
print(response.content)
