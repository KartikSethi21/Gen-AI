from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import streamlit as st


llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens = 100,
        return_full_text = False,
        temperature=1.2
    )
)

model = ChatHuggingFace(llm = llm)
st.header('Research Tool')

user_input = st.text_input("Enter your prompt")

if st.button('Summarize'):
    # st.text('Some random text')
    # If button clicked show this
    result = model.invoke(user_input)
    st.write(result.content)


