from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

# Failed

# llm = HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation"
# )
chat = ChatHuggingFace(llm=llm)

response = chat.invoke("What is the capital of India?")
print(response.content)