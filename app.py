import streamlit as st
from components.store_pdf import StorePdfData
from components.chat_with_gemini import ChatWithGemini
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings



load_dotenv()

def main():

   
    api_key=os.getenv("GOOGLE_API_KEY")
    #
    st.set_page_config(
        page_title="Chat with PDFs",
        page_icon=":paperclip:",  # Emoji for document icon
        layout="wide",  # Use wider layout for better presentation
    )
    st.header("Chat with PDFs")
    st.subheader("Upload multiple pdfs and ask your questions")

    uploaded_files=st.file_uploader(label="Choose pdfs",type="pdf",accept_multiple_files=True)
    
    #when submit button clicked , then storing the pdf data in FAISS vector store
    if st.button("Submit"):
        store_pdf=StorePdfData(pdf_files=uploaded_files,api_key=api_key)
        text = store_pdf.extract_pdf_data()
        chunks = store_pdf.get_text_chunks(text)
        store_pdf.store_data_in_vector_store(chunks=chunks)
        st.write("Files Uploaded succesfully!")

    st.subheader("Pleaes ask your question here")
    question=st.text_input("Ask your question")

    #when ask button clicked, taking the apropiate docs from FAISS vector and passing to gemini with the question
    if st.button("Ask"):
         chat=ChatWithGemini(api_key=api_key)
         embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=api_key)
         new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)     
         docs=chat.question_search(question=question,new_db=new_db)
         response=chat.question_search_in_gemini(docs=docs,question=question)
         st.write(f"Answer to your Question : ",response["output_text"])



if __name__ == "__main__":
    main()

