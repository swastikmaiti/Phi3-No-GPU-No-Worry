import gradio as gr
import random
import time
import gc

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





model_path = os.path.join(base_path, 'cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4')
model = og.Model(model_path)
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()

params = og.GeneratorParams(model)
params.try_use_cuda_graph_with_max_batch_size(1)

with gr.Blocks() as demo:

    context_list = None
    index = None

    def doc_processing(uploaded_pdf):
        first_section = "Abstract"
        ignore_after = "References"
        reader = PdfReader(uploaded_pdf)
        context_list = pre_processing.parese_doc(reader,first_section,ignore_after)
        pre_processing.create_embedding(context_list)
        pre_processing.CONTEXT = context_list

    def response_generator(text):
        context_list = pre_processing.CONTEXT
        index: faiss.IndexFlatL2 = faiss.read_index(os.path.join(base_path, 'doc.index'))
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


        params.set_search_options(**search_options)
        params.input_ids = input_tokens
        generator = og.Generator(model, params)
        
        del context_list
        del context
        del index
        output = ""
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()
            new_token = generator.get_next_tokens()[0]
            p_word = tokenizer_stream.decode(new_token)
            output+=p_word
            #yield output
        del generator
        return output
        




    with gr.Row():
        upload_button = gr.UploadButton("📁 Upload PDF", file_types=[".pdf"])
    upload_button.upload(doc_processing,upload_button,None,queue=False,show_progress=True,trigger_mode="once")

    input_box = gr.Textbox(autoscroll=True)
    output_box = gr.Textbox(autoscroll=True,max_lines=30)
    gr.Interface(fn=response_generator, inputs=input_box, outputs=output_box,delete_cache=(10,10))
    
        
    
demo.queue()
demo.launch(share=True)
