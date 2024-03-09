# Chat-with-PDFs

This project builds a user-friendly web application using Streamlit that empowers users to upload PDFs and pose questions directly related to the document's content. The application seamlessly integrates Google's cutting-edge Gemini Pro large language model (LLM) to dynamically extract information from the uploaded PDF and generate insightful answers to user queries. LangChain, a powerful natural language processing (NLP) library, is employed to enhance the overall accuracy and robustness of the system.


step 1 (optional)
create an enviornment with python greater than 3.9

bash : conda create -n env_name python==3.10
bash : conda activate env_name

step 2
Install all the packages required

bash : pip install -r requirements.txt

step 3
create .env file and add google api key like this
GOOGLE_API_KEY="AI*******uSo"

step 4
run app.py

bash : streamlit run app.py