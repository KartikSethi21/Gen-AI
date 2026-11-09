
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableSequence,
    RunnableLambda,
    RunnablePassthrough,
    RunnableParallel
)
from transformers import pipeline

# Count words
def word_count(text):
    return len(text.split())

# Prompt
prompt = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

# Load Hugging Face model locally
pipe = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    return_full_text=False
)

model = HuggingFacePipeline(pipeline=pipe)

# Output parser
parser = StrOutputParser()

# Chain 1: Generate joke
joke_gen_chain = RunnableSequence(
    prompt,
    model,
    parser
)

# Chain 2: Process output in parallel
parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "word_count": RunnableLambda(word_count)
})

# Final chain
final_chain = RunnableSequence(
    joke_gen_chain,
    parallel_chain
)

# Run
result = final_chain.invoke({"topic": "AI"})

final_result = "{}\nWord count - {}".format(
    result["joke"],
    result["word_count"]
)

print(final_result)
