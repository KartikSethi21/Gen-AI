from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from transformers import pipeline
import torch

model_id = "Qwen/Qwen2.5-0.5B-Instruct"

# Check GPU availability
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    raise RuntimeError(
        "CUDA is unavailable. Install a CUDA-enabled PyTorch build."
    )

# Load model on NVIDIA GPU
pipe = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=model_id,
    device=0,
    torch_dtype=torch.float16,
    max_new_tokens=150,
    do_sample=True,
    temperature=0.7,
    return_full_text=False
)

model = HuggingFacePipeline(pipeline=pipe)

prompt = PromptTemplate(
    template="Write a summary of the following text:\n\n{poem}",
    input_variables=["poem"]
)

parser = StrOutputParser()

loader = TextLoader("cricket.txt", encoding="utf-8")
docs = loader.load()

print("Document count:", len(docs))
print("Metadata:", docs[0].metadata)

chain = prompt | model | parser

result = chain.invoke({"poem": docs[0].page_content})
print(result)



# pip install -U transformers torch accelerate langchain-huggingface langchain-core langchain-community