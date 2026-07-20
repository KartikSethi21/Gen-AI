from langchain_anthropic import ChatAnthropic
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

model = ChatAnthropic(model="claude-3-5-sonnet-20241022")

result = model.invoke("What is Captital of india")

print(result)