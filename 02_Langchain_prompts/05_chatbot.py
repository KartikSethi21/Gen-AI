from transformers.utils import logging
logging.set_verbosity_error()

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation",
    device=0,
    pipeline_kwargs=dict(
        max_new_tokens = 100,
        return_full_text = False,
        temperature=1.2
    )
)

model = ChatHuggingFace(llm = llm)

chat_history = []

while True:
    user_unput = input("You: ")
    chat_history.append(user_unput)
    if user_unput == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(result.content)
    print("AI: ",result.content)

print(chat_history)