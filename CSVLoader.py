from langchain_community.document_loaders import CSVLoader, PyPDFLoader

loader= CSVLoader('RAG\Social_Network_Ads.csv')


docs = loader.load()

print(docs[1])


