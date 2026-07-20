from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env")

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)
documents = [
    'Delhi is the capital of India',
    "Kolkata is the capital of West Bengal",
    "Paris is the captal of France"
]

result = embedding.embed_documents(documents)

print(result)
print(str(result))