
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class StorePdfData:
    def __init__(self,pdf_files,api_key) -> None:
        self.pdf_files=pdf_files
        self.api_key=api_key
    

    def extract_pdf_data(self):
        text=""
        for pdf in self.pdf_files:
            reader=PdfReader(pdf)
            for page in reader.pages:
                text += page.extract_text()
        return text
    
    def get_text_chunks(self,text):
        splitter=RecursiveCharacterTextSplitter(chunk_size=256,chunk_overlap=0)
        chunks=splitter.split_text(text=text)
        return chunks
    
    def store_data_in_vector_store(self,chunks):
        embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=self.api_key)
        vector_stores=FAISS.from_texts(chunks,embeddings)
        vector_stores.save_local('faiss_index')



    
            

