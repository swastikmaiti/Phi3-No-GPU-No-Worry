import faiss
import numpy as np

model = None

def parese_doc(doc):
    documents_1 = ''

    reader = doc
    for page in reader.pages:
        documents_1 += page.extract_text()
    
    cleaned_string = documents_1.replace('\n', ' ')
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
    