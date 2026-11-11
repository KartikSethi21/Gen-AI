from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from transformers import pipeline
import torch
import os

os.environ["USER_AGENT"] = "Mozilla/5.0 (compatible; LangChainLearning/1.0)"

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
    template="Answer the following questions \n {question} from the following text - \n {text}",
    input_variables=["question","text"]
)

parser = StrOutputParser()



url = "https://www.myntra.com/shirts?f=Gender%3Amen%2Cmen%20women&rf=Discount%20Range%3A10.0_100.0_10.0%20TO%20100.0"


loader = WebBaseLoader(url)
docs = loader.load()
chain = prompt | model | parser

print(docs[0].page_content)


res = chain.invoke({'question':'List all the products','text':docs[0].page_content})

print(res)