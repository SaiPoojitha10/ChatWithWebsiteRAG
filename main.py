from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import requests
from bs4 import BeautifulSoup

# Set OpenAI API Key
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Function to scrape website content
def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "lxml")
    paragraphs = soup.find_all("p")
    content = " ".join([para.get_text() for para in paragraphs])
    return content

# URLs to scrape
urls = [
    "https://www.uchicago.edu/",
    "https://www.washington.edu/",
    "https://www.stanford.edu/",
    "https://und.edu/",
]

# Scrape and chunk data
scraped_data = {url: scrape_website(url) for url in urls}
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = [
    Document(page_content=chunk, metadata={"source": url})
    for url, content in scraped_data.items()
    for chunk in text_splitter.split_text(content)
]

# Create Embeddings and VectorStore
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vector_db = Chroma.from_documents(documents, embedding_model)

# Define LLM and Prompt
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
prompt = PromptTemplate(
    template="Given the following context:\n\n{context}\n\nAnswer the following question:\n{question}",
    input_variables=["context", "question"],
)

# Define Query Function
def query_chain(question):
    # Retrieve relevant documents
    retriever = vector_db.as_retriever()
    relevant_docs = retriever.invoke(question)
    
    # Combine context
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Generate response
    response = llm.invoke(prompt.format(context=context, question=question))
    return response.content

# Example Query
query = "What are the academic programs offered at Stanford University?"
response = query_chain(query)
print(response)