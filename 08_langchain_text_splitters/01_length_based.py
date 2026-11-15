from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
text = """
LangChain is a framework for developing applications
powered by Large Language Models.
It provides tools for building RAG applications.
"""

splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    separator='',
)

result = splitter.split_text(text)

print(result)

print("\n")
loader = PyPDFLoader(r"C:\Users\kartik\OneDrive\Desktop\GenAI\07_langchain_doc_loaders\dl-curriculum.pdf")

docs = loader.load()

result = splitter.split_documents(docs)
# print(result)
print(result[0].page_content,"\n")
print(result[1].page_content)

