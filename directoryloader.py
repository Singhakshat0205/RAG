from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader= DirectoryLoader(
    path='books',# name of the folder
    glob='*.pdf',
    loader_cls=PyPDFLoader
)


docs= loader.load()
#when we use load, it uses eager loading and loads the entire object at once

docs= loader.lazy_load()

print(len(docs))



for document in docs:
    print(document.metadata)
