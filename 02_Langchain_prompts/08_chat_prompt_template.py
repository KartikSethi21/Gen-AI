from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

chat_template = ChatPromptTemplate([
    ('system',"You are a helpful {domain} expert"),
    ('human', "Explain in simple terms, what is {topic}")
    # SystemMessage(content="You are a helpful {domain} expert"),
    # HumanMessage(content="Explain in simple terms, what is {topic}")
])

# same output
chat_template_2 = ChatPromptTemplate.from_messages([
    ('system',"You are a helpful {domain} expert"),
    ('human', "Explain in simple terms, what is {topic}")
])


prompt = chat_template.invoke({'domain':'cricket','topic':'dusra'})
prompt = chat_template_2.invoke({'domain':'cricket','topic':'dusra'})


print(prompt)