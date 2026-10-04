from langchain_community.document_loaders import PyPDFLoader

loader= PyPDFLoader('dl-curriculum.pdf')

docs= loader.load()

print(docs)

print(len(docs)) ## length will be equal to the number of pages in the pdf 

print(docs[0].page_content) # page content of the first document object
print(docs[0].metadata)
## this gives the metadata of the first document object 


for document in docs:
    print(document.metadata)