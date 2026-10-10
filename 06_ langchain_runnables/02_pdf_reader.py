
import os

# Hugging Face integration for embeddings and hosted language models
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace
)

# Tools for loading documents, splitting text, and storing vectors
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Used to create a structured prompt for the language model
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# STEP 1: LOAD THE DOCUMENT
# --------------------------------------------------

# Read the contents of docs.txt.
# TextLoader returns a list of Document objects containing text and metadata.
loader = TextLoader("docs.txt", encoding="utf-8")
documents = loader.load()

print("Documents loaded:", len(documents))


# --------------------------------------------------
# STEP 2: SPLIT THE DOCUMENT INTO CHUNKS
# --------------------------------------------------

# Large documents may exceed the context limit of an LLM.
# Splitting makes retrieval more focused and efficient.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # Maximum approximate characters per chunk
    chunk_overlap=50      # Preserve context between adjacent chunks
)

docs = text_splitter.split_documents(documents)

print("Chunks created:", len(docs))


# --------------------------------------------------
# STEP 3: CREATE EMBEDDINGS
# --------------------------------------------------

# Embeddings convert text into numerical vectors.
# This model runs locally after the initial download.
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# STEP 4: CREATE THE FAISS VECTOR DATABASE
# --------------------------------------------------

# Convert every chunk into an embedding and store it in FAISS.
# FAISS allows us to search for chunks semantically similar to a query.
vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=embedding_model
)

print("FAISS vector database created successfully!")


# --------------------------------------------------
# STEP 5: CREATE A RETRIEVER
# --------------------------------------------------

# The retriever finds relevant document chunks for a question.
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}  # Retrieve the 3 most relevant chunks
)


# --------------------------------------------------
# STEP 6: CONFIGURE THE HUGGING FACE LLM
# --------------------------------------------------

# Set your Hugging Face access token in the environment variable
# HUGGINGFACEHUB_API_TOKEN before running this script.
#
# This example uses a hosted inference endpoint.
# Model availability and provider access can vary.

if not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
    raise ValueError(
        "Please set the HUGGINGFACEHUB_API_TOKEN environment variable."
    )

llm_endpoint = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=300,
    temperature=0.3
)

# Wrap the endpoint as a chat model so it can process chat messages.
model = ChatHuggingFace(llm=llm_endpoint)


# --------------------------------------------------
# STEP 7: DEFINE THE PROMPT
# --------------------------------------------------

# Context contains retrieved document chunks.
# Question contains the user's actual query.
# The instructions encourage answers grounded in the document.

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant answering questions about a document.

Use only the context provided below to answer the question.
If the answer is not present in the context, say:
"I could not find the answer in the provided document."

Context:
{context}

Question:
{question}

Answer:
""")


# --------------------------------------------------
# STEP 8: ASK A QUESTION
# --------------------------------------------------

query = input("Ask a question about the document: ")

# Retrieve relevant chunks instead of passing the entire document.
retrieved_docs = retriever.invoke(query)

print("\nRetrieved chunks:", len(retrieved_docs))


# --------------------------------------------------
# STEP 9: PREPARE THE RETRIEVED CONTEXT
# --------------------------------------------------

# Each retrieved item is a Document object.
# Access its text using page_content.
retrieved_text = "\n\n".join(
    doc.page_content for doc in retrieved_docs
)


# --------------------------------------------------
# STEP 10: GENERATE THE ANSWER
# --------------------------------------------------

# Fill the prompt with the retrieved context and user's question.
messages = prompt.invoke({
    "context": retrieved_text,
    "question": query
})

# Invoke the chat model to generate the response.
response = model.invoke(messages)

# Chat models return an AIMessage; its answer is in .content.
print("\nAnswer:")
print(response.content)