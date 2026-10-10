
import os

from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace
)
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA


# STEP 1: Load the document
loader = TextLoader("docs.txt", encoding="utf-8")
documents = loader.load()

print("Documents loaded:", len(documents))


# STEP 2: Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

print("Chunks created:", len(docs))


# STEP 3: Create embeddings using a Hugging Face model
# This embedding model runs locally after downloading.
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# STEP 4: Create the FAISS vector database
vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=embedding_model
)

print("FAISS vector database created successfully!")


# STEP 5: Create the retriever
# It retrieves the 3 most relevant document chunks.
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# STEP 6: Configure the Hugging Face language model
if not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
    raise ValueError(
        "Set the HUGGINGFACEHUB_API_TOKEN environment variable."
    )

llm_endpoint = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=300,
    temperature=0.3
)

# Wrap the endpoint as a chat model.
model = ChatHuggingFace(llm=llm_endpoint)


# STEP 7: Create the prompt
# RetrievalQA supplies the retrieved context and user's question.
prompt = PromptTemplate(
    template="""
You are a helpful assistant answering questions about a document.

Use only the context below to answer the question.
If the answer is not present in the context, say:
"I could not find the answer in the provided document."

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)


# STEP 8: Create the RetrievalQA chain
# The chain automatically retrieves documents, combines their text,
# inserts the context into the prompt, and calls the language model.
qa_chain = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)


# STEP 9: Ask a question
query = input("Ask a question about the document: ")

# RetrievalQA expects the input key "query".
response = qa_chain.invoke({"query": query})


# STEP 10: Display the answer
print("\nAnswer:")
print(response["result"])


# STEP 11: Display the retrieved source chunks
print("\nSource documents:")

for i, doc in enumerate(response["source_documents"], start=1):
    print(f"\nSource {i}:")
    print(doc.page_content)
