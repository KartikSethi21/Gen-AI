from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=150,
    temperature=0.7
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

prompt2 = PromptTemplate(
    template="Explain the following joke:\n{text}",
    input_variables=["text"]
)

chain = RunnableSequence(
    prompt1,
    model,
    parser,
    RunnablePassthrough.assign(text=RunnablePassthrough()),
    prompt2,
    model,
    parser
)

result = chain.invoke({"topic": "AI"})
print(result)




# joke_chain = prompt1 | model | parser

# # Second stage: map the joke string to the "text" variable
# explanation_chain = (
#     {"text": joke_chain}
#     | prompt2
#     | model
#     | parser
# )

# result = explanation_chain.invoke({"topic": "AI"})




# chain = prompt1 | model | parser | prompt2 | model | parser

# result = chain.invoke({"topic": "Unemployment in India"})

# print(result)
# chain.get_graph().print_ascii()
