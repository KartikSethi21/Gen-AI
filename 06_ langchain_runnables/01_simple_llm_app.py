from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

llm  =  ChatOpenAI(model_name='gpt-3.5-turbo',temprature=0.3)

prompt = PromptTemplate(
    template="Suggest a catchy blog title about {topic}",
    input_variables=["topic"]
)


topic = input("Enter a topic")

formatted_prompt = prompt.format(topic=topic)


blog_title = llm.predict(formatted_prompt)

print("Generated Blog Title:",blog_title)