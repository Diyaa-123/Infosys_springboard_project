from langchain_community.document_loaders import TextLoader
loader= TextLoader('cricket.txt')
docs = loader.load()
print(docs)