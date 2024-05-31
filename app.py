import gradio as gr

from pypdf import PdfReader
import onnxruntime_genai as og
import os

import pre_processing
from pre_processing import embedding_model


base_path = os.getcwd()

model_path = os.path.join(base_path, 'cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4')
model = og.Model(model_path)
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()

# params = og.GeneratorParams(model)
# params.try_graph_capture_with_max_batch_size(1)

def doc_processing(uploaded_pdf,var):
    first_section = "abstract"
    ignore_after = "references"
    reader = PdfReader(uploaded_pdf)
    context_list = pre_processing.parese_doc(reader,first_section,ignore_after)
    index = pre_processing.create_embedding(context_list)


    return {input_box: gr.Textbox(value="Ask a question", visible=True),
            state_var:[context_list,index]}

def response_generator(text,var1):
    context_list,index = var1
    chat_template = '<|user|>\nYou are an Research Assistant. You will provide short and precise answer.<|end|>\n<|assistant|>\nYes I will keep the answer short and precise.<|end|>\n<|user|>\n{input} <|end|>\n<|assistant|>'
    search_options ={}
    search_options['temperature'] = 1
    search_options['max_length'] = 2000

    query_embedding = embedding_model.encode(text).reshape(1, -1)
    top_k = 1
    _scores, binary_ids = index.search(query_embedding, top_k)
    binary_ids = binary_ids[0]
    _scores = _scores[0]
    temp_list = []
    for idx in binary_ids:
            temp_list.append(context_list[idx])
    context = '. '.join(temp_list)
    
    text += " with respect to context: "+context
    

    prompt = f'{chat_template.format(input=text)}'
    input_tokens = tokenizer.encode(prompt)
    params = og.GeneratorParams(model)
    params.try_graph_capture_with_max_batch_size(1)
    params.set_search_options(**search_options)
    params.input_ids = input_tokens
    generator = og.Generator(model, params)
    
    output = ""
    while not generator.is_done():
        generator.compute_logits()
        generator.generate_next_token()
        new_token = generator.get_next_tokens()[0]
        p_word = tokenizer_stream.decode(new_token)
        output+=p_word
        yield {output_box:output}  
    del generator

def submit():
    return {input_box: gr.Textbox(visible=True)}

with gr.Blocks() as demo:
    
    gr.Markdown(
    """
    # Phi3 3.8B

    ## RAG - Topic based pdf Q/A

    - ***LLM:*** Phi3 Mini
    - ***Embedding:*** nomic-embed-text-v1

    """)

    state_var = gr.State([])

    with gr.Row():
        upload_button = gr.UploadButton("📁 Upload PDF", file_types=[".pdf"])
    error_box = gr.Textbox(label="Error", visible=False)

    input_box = gr.Textbox(autoscroll=True,visible=False,label='User')
    output_box = gr.Textbox(autoscroll=True,max_lines=30,value="Output",label='Assistant')
    gr.Interface(fn=response_generator, inputs=[input_box,state_var], outputs=[output_box,state_var],delete_cache=(20,10))

    upload_button.upload(doc_processing,inputs=[upload_button,state_var],outputs=[input_box,state_var],queue=False,show_progress=True,trigger_mode="once")
    upload_button.upload(submit,None,input_box)
    
demo.queue()
demo.launch()
