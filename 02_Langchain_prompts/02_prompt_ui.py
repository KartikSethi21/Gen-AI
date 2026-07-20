from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

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

paper_input = st.selectbox("Select Research Paper Name",["Attention Is All You Need",
                                                         "BERT: Pre-training of Deep Bidirectional Transformers",
                                                         "GPT-3: Language Models are Few-Shot Learners",
                                                         "Diffusion Models Beat GANs o Image Synthesis"])

style_input = st.selectbox("Select Explaination Style",["Beginner-Friendly",
                                                        "Technical",
                                                        "Code-oriented",
                                                        "Mathematical"])
length_input = st.selectbox("Select Explanation Length",["Short (1-2 paragraphs)",
                                                        "Medium (3-5 paragraphs)",
                                                        "Long (detailed explanation)"
                                                        ])

template = load_prompt('template.json')

# fill the placeholders
prompt = template.invoke({
    'paper_input': paper_input,
    'style_input':style_input,
    'length_input':length_input
})

if st.button('Summarize'):
    # If button clicked show this
    result = model.invoke(prompt)
    print(result.content)
    st.write(result.content)

