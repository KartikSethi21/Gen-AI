from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pathlib import Path


load_dotenv(Path(__file__).parent.parent / ".env")

model = ChatOpenAI(model="gpt-4", temperature=0, max_completion_tokens=10)

# temperature => creat
# max_completion_tokens => words t
# tokens or words r paid so we can limit them => they r not only words

results = model.invoke("What is the capital of india")

print(results)
print(results.content)


results = model.invoke("Suggest me 5 indian male names")

print(results)
print(results.content)



