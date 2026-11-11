from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader



loader = DirectoryLoader(
    path="books",
    glob='*.pdf',  # => pdf upload,  "**/*.txt"=> text , "data/*.csv" => csv file in data folder , "**/*" => all files, ** => recursive search through subfolder
    loader_cls=PyPDFLoader
)

docs = loader.lazy_load()


for document in docs:
    print(document.metadata)