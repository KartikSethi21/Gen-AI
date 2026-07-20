from langchain_openai import OpenAI
from dotenv import load_dotenv
import os
from pathlib import Path

# load_dotenv()

load_dotenv(Path(__file__).parent.parent / ".env")
# print(os.getenv("OPENAI_API_KEY"))

llm = OpenAI(model='gpt-3.5-turbo-instruct')

result = llm.invoke("What is the capital of india")

print(result)


