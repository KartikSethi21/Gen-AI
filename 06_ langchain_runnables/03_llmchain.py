
import os

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Initialize the Hugging Face language model
llm_endpoint = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=100,
    temperature=0.3
)

# 2. Wrap the endpoint as a chat model
llm = ChatHuggingFace(llm=llm_endpoint)

# 3. Define the prompt template
prompt = PromptTemplate(
    template="Suggest a catchy blog title about {topic}",
    input_variables=["topic"]
)

# 4. Create a chain using LangChain Expression Language (LCEL)
chain = prompt | llm | StrOutputParser()

# 5. Provide the topic
topic = "Machine Learning"

# 6. Run the chain
output = chain.invoke({"topic": topic})

# 7. Print the generated title
print("Generated Blog Title:", output)
