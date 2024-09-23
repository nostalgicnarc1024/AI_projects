#simple chatbot in streamlit with memory
import os
from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
api_key = os.getenv("MY_KEY")
model = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", openai_api_key=api_key)
# client = OpenAI()
# def llm_response(prompt):
#     response = client.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}])
#     return response.choices[0].message['content']

st.set_page_config(
    page_title="GPT-4o Chat",
    layout="centered"
)
st.title("OpenAI Chatbot")
st.write("Chat with the OpenAI model!")

# Initialize session state for conversation history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# input field for user's message
user_prompt = st.chat_input("You: ")

if user_prompt:
    # add user's message to chat and display it
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # send user's message to GPT-4o and get a response
    # all the previous responses will be sent to the api in the messages
    response = model.invoke(st.session_state.chat_history)

    assistant_response = response.content
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    # display GPT-4o's response
    with st.chat_message("assistant"):
        st.markdown(assistant_response)