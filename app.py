import streamlit as st
import langchain_google_genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import HarmBlockThreshold, HarmCategory
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

api_key = "AIzaSyD9_BDASRH0MqOuv2dYJUvoIm_x3lZEKLU"

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 100,
)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    vector_store = FAISS.from_texts(text_chunks,
                               embedding_model)
    vector_store.save_local("faiss_index")


def get_conversation_chain():
    prompt_template="""
Answer the question as detailed as possible from the provided context ,make sure to provide all the details, if the answer is not in 
provide context just say " answer is not available in the context", don't provide the worng answer 
context : \n{context}?\n
Question :\n{question}\n

answer:
"""
    model = ChatGoogleGenerativeAI(model = "gemini-2.0-flash",
                               temperature = 0.3,
                               google_api_key = api_key,
                                   safety_settings={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
                              )
    prompt = PromptTemplate(
    template = prompt_template,
    input_variables = ["context","question"]
)
    chain = load_qa_chain(model,
                      chain_type = "stuff",
                      prompt=prompt)
    return chain


def user_input(user_question):
    embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')  
    new_db = FAISS.load_local("faiss_index",
                          embedding_model,
                          allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversation_chain()
    
    response = chain.invoke(
    {"input_documents":docs,"question":user_question},
    return_only_outputs=True
    )
    
    print(response)
    st.write("Reply:" , response["output_text"])


def main():
    st.set_page_config(page_title="Chat with PDF 📄🤖",
    page_icon=":robot_face:",
    layout="wide")
    st.header("Chat with Your PDF 📚💬")

    user_question = st.text_input("Ask a Question from the PDF 📘")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu 🧭")
        pdf_docs = st.file_uploader("Upload your PDF 📄 here and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process 🚀"):
            with st.spinner("Processing... ⏳"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done ✅")
                
if __name__ == "__main__":
    main()


        

    
