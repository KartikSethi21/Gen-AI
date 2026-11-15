
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings


text = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.

Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.
"""


# Step 1: Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)


# Step 2: Initialize Semantic Chunker
splitter = SemanticChunker(
    embeddings=embeddings,
    buffer_size=0,
    breakpoint_threshold_type="standard_deviation",#"percentile",
    breakpoint_threshold_amount=1,
    # number_of_chunks=3
)


# Step 3: Create semantic chunks
result = splitter.create_documents([text])


# Step 4: Print results
print("Total chunks:", len(result))
print(result)
# for i, doc in enumerate(result):
#     print(f"\n--- Chunk {i+1} ---")
#     print(doc.page_content)
