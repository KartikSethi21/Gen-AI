from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader



loader = DirectoryLoader(
    path="books",
    glob='*.pdf',  # => pdf upload,  "**/*.txt"=> text , "data/*.csv" => csv file in data folder , "**/*" => all files, ** => recursive search through subfolder
    loader_cls=PyPDFLoader
)

docs = loader.load()

print(len(docs))

print(docs[0].page_content)
print(docs[0].metadata)
print(docs[325].page_content)
print(docs[325].metadata)