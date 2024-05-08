import streamlit as st

import random
import time



from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import onnxruntime_genai as og
import os


base_path = os.getcwd()

import pre_processing
check_point = 'mixedbread-ai/mxbai-embed-large-v1'
embedding_model = SentenceTransformer(check_point)
pre_processing.model = embedding_model


@st.cache_resource
def doc_processing():
    first_section = "Abstract"
    ignore_after = "References"
    reader = PdfReader(st.session_state.doc)
    context_list = pre_processing.parese_doc(reader,first_section,ignore_after)
    pre_processing.create_embedding(context_list)
    pre_processing.CONTEXT=context_list
    


index: faiss.IndexFlatL2 = faiss.read_index(os.path.join(base_path, 'doc.index'))
model_path = os.path.join(base_path, 'cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4')
model = og.Model(model_path)
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()

@st.cache_resource
def response_generator(text):
    context_list =pre_processing.CONTEXT
    chat_template = '<|user|>\n{input} <|end|>\n<|assistant|>'
    search_options ={}
    search_options['temperature'] = 1

    query_embedding = embedding_model.encode(text).reshape(1, -1)
    top_k = 1
    _scores, binary_ids = index.search(query_embedding, top_k)
    binary_ids = binary_ids[0]
    _scores = _scores[0]
    temp_list = []
    for idx in binary_ids:
            temp_list.append(context_list[idx])
    context = '. '.join(temp_list)
    
    text += " With respect to context: "+context
    

    prompt = f'{chat_template.format(input=text)}'
    print(prompt)

    input_tokens = tokenizer.encode(prompt)

    params = og.GeneratorParams(model)
    params.try_use_cuda_graph_with_max_batch_size(1)
    params.set_search_options(**search_options)
    params.input_ids = input_tokens
    generator = og.Generator(model, params)
    
    out = ""
    while not generator.is_done():
        generator.compute_logits()
        generator.generate_next_token()
        new_token = generator.get_next_tokens()[0]
        #print(tokenizer_stream.decode(new_token), end='', flush=True)
        yield tokenizer_stream.decode(new_token)

file = st.file_uploader("Upload a PDF file", type="pdf",on_change=doc_processing,key="doc")
print(file)



# Streamed response emulator
# def response_generator():
#     response = random.choice(
#         [
#             "Hello there! How can I assist you today?",
#             "Hi, human! Is there anything I can help you with?",
#             "Do you need help?",
#         ]
#     )
#     for word in response.split():
#         yield word + " "
#         time.sleep(0.05)


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
        response = st.write_stream(response_generator(prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})



    
