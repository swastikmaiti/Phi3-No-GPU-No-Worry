import streamlit as st

import random
import time


from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


def doc_processing():
    import pre_processing
    check_point = 'mixedbread-ai/mxbai-embed-large-v1'
    embedding_model = SentenceTransformer(check_point)
    pre_processing.model = embedding_model

    first_section = "Abstract"
    ignore_after = "References"
    reader = PdfReader(st.session_state.doc)
    pre_processing.create_embedding(pre_processing.parese_doc(reader,first_section,ignore_after))



file = st.file_uploader("Upload a PDF file", type="pdf",on_change=doc_processing,key="doc")



# Streamed response emulator
def response_generator():
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


st.title("Simple chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator())
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})



    
