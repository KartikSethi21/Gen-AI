from langchain_text_splitters import RecursiveCharacterTextSplitter


text = """
My name is kartik.
I am 25 years old.

I live in Delhi.
How are you.
"""


splitter = RecursiveCharacterTextSplitter(
    chunk_size=10,
    chunk_overlap=0
)

result = splitter.split_text(text)

print(len(result))
print(result)