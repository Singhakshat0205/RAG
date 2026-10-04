from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os


load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
     huggingfacehub_api_token=os.getenv(
        "HUGGINGFACEHUB_ACCESS_TOKEN")

)

model= ChatHuggingFace(llm=llm)

prompt= PromptTemplate(
    template= 'Write the summary for the following poem \n {poem}',
    input_variables=['poem']

)

parser = StrOutputParser()

loader= TextLoader('RAG\cricket.txt',encoding='utf-8')


docs= loader.load()

print(docs[0])


chain = prompt | model | parser


result= chain.invoke({'poem':docs[0].page_content})


print(result)

