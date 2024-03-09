from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, HarmBlockThreshold, HarmCategory
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI



class ChatWithGemini:
    def __init__(self,api_key) -> None:
        self.api_key=api_key
        self.faiss_index="faiss_index"
    
    def question_search(self,question,new_db):

        # embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=self.api_key)
        # new_db = FAISS.load_local("faiss_index", embeddings)
        similar_docs=new_db.similarity_search(question)
        return similar_docs
    
    def question_search_in_gemini(self,docs,question):
        prompt=PromptTemplate(input_variables=["context","question"],
                              template="""Please read the below context \n
                              {context} \n
                              and answer this question {question} .\n
                              answer the question only if the required information available in above context. if it is not available then you can easily say ,
                                required information is not available in your PDF.
                                """)
        # prompt_temp=prompt.format(context=docs,question=question)
        llm=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3,google_api_key=self.api_key,safety_settings={HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE})
        chain=load_qa_chain(llm,chain_type="stuff",prompt=prompt)
        # answer=chain.run(prompt_temp,)
        response=chain({"input_documents":docs,"question":question},return_only_outputs=True)
        return response
        # prompt.format()
        
        