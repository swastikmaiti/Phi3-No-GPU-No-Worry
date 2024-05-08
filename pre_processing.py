import faiss
import numpy as np

model = None
CONTEXT = None

def parese_doc(doc,first_section,ignore_after):
    documents_1 = ''

    reader = doc
    for page in reader.pages:
        documents_1 += page.extract_text()
    
    cleaned_string = documents_1.replace('\n', ' ')

    start_index = cleaned_string.find(first_section)
    end_index = cleaned_string.rindex(ignore_after)
    cleaned_string = cleaned_string[start_index:end_index]

    sentence_list = cleaned_string.split('. ')
    context_list = []
    group_size = 20
    overlap = 5
    i = 0 
    while True:
        group = sentence_list[i:i+group_size]
        text = '. '.join(group)
        context_list.append(text)
        i+=group_size-overlap
        if i>=len(sentence_list):
            break
    return context_list

def get_embeddings(doc):
    model_input = doc
    out =  model.encode(model_input)
    return out

def create_embedding(context_list):
    embedding_dimension = model.get_sentence_embedding_dimension()
    embeddings = list(map(get_embeddings,context_list))
    embeddings_array = np.array(embeddings)
    
    index = faiss.IndexFlatL2(embedding_dimension)
    index.add(embeddings_array)
    faiss.write_index(index, 'doc.index')
    