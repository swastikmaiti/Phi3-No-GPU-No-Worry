import gradio as gr
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


context_list = None
index = None


model_path = os.path.join(base_path, 'cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4')
model = og.Model(model_path)
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()

def doc_processing(uploaded_pdf):
    first_section = "Abstract"
    ignore_after = "References"
    reader = PdfReader(uploaded_pdf)
    context_list = pre_processing.parese_doc(reader,first_section,ignore_after)
    pre_processing.create_embedding(context_list)
    pre_processing.CONTEXT = context_list

def response_generator(history):
    context_list = pre_processing.CONTEXT
    index: faiss.IndexFlatL2 = faiss.read_index(os.path.join(base_path, 'doc.index'))
    text = history[-1][0]
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
    print("prompt: ",prompt)

    input_tokens = tokenizer.encode(prompt)

    params = og.GeneratorParams(model)
    params.try_use_cuda_graph_with_max_batch_size(1)
    params.set_search_options(**search_options)
    params.input_ids = input_tokens
    generator = og.Generator(model, params)
    
    history[-1][1] = ""
    del context_list
    del context
    del index
    while not generator.is_done():
        generator.compute_logits()
        generator.generate_next_token()
        new_token = generator.get_next_tokens()[0]
        p_word = tokenizer_stream.decode(new_token)
        history[-1][1] += p_word
        yield history

with gr.Blocks() as demo:
    with gr.Row():
        upload_button = gr.UploadButton("📁 Upload PDF", file_types=[".pdf"])

    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
        history[-1][1] = ""
        for character in bot_message:
            #history[-1][1] += character
            time.sleep(0.05)
            yield history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        response_generator, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

    upload_button.upload(doc_processing,upload_button,None,queue=False,show_progress=True,trigger_mode="once")
    
        
    
demo.queue()
demo.launch(share=True,state_session_capacity=1)
