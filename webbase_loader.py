from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
import os
load_dotenv()




url="https://www.flipkart.com/apple-macbook-air-m4-24-gb-512-gb-ssd-macos-sequoia-mc6l4hn-a/p/itm6ff26e315ebd8?pid=COMH9ZWQMDBWPXHR&lid=LSTCOMH9ZWQMDBWPXHRMLWYDO&marketplace=FLIPKART&store=6bo%2Fb5g&srno=b_1_1&otracker=browse&fm=organic&iid=1e05f2e8-c000-4874-b01e-2212f39b39eb.COMH9ZWQMDBWPXHR.SEARCH&ppt=None&ppn=None&ssid=hmoyihi1q80000001791122852582&ov_redirect=true"
#we can also provide a list of urls and it will give us the text from each web page 

loader= WebBaseLoader(url)


docs= loader.load()

print(len(docs))
print(docs[0].page_content)

llm= HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
     huggingfacehub_api_token=os.getenv(
        "HUGGINGFACEHUB_ACCESS_TOKEN")

)


model= ChatHuggingFace(llm=llm)

prompt= PromptTemplate(
    template='Answer the following question \n {question} from the following text- \n {text}',
    input_variables=['question', 'text']
)

parser = StrOutputParser()


chain= prompt| model| parser

result= chain.invoke({
    'question':'what is the maximum brightness of this product',
    'text':docs[0].page_content
    })

print(result)


question=[
    'what is the price of the product',
    'what is the length and width of this product',
    'what is the hardware specification of the product'
]
for qus in question:
    print(chain.invoke({'question':qus, 
                        'text':docs[0].page_content}))
