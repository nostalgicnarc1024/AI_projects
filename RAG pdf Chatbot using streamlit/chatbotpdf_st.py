#chatbot to chat with pdf in streamlit
import openai
import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
api_key = os.getenv("MY_KEY")

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", openai_api_key=api_key)
embedding = OpenAIEmbeddings(openai_api_key=api_key)

st.set_page_config(
    page_title="GPT-4o Chat",
    layout="centered"
)
st.title("OpenAI Chatbot")
st.write("Chat with the OpenAI model!")

pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")

if pdf_docs is not None:
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    vector_store = FAISS.from_texts(chunks, embedding=embedding)
    retriever = vector_store.as_retriever()
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    question = st.text_input("Input your question for the uploaded document")
    result = chain.invoke(question)
    st.write(result)