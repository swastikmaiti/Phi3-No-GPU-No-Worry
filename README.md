# RAG-System with Microsoft Phi3 Mini

In this work we harness the power of Microsoft `Phi3 Mini 3.8` on `ONXX` CPU runtime. We build a PDF Q/A system with `nomic-embed-text-v1` as embedding moel 
`faiss` as Vector DB.

#

<img src="https://github.com/swastikmaiti/digital_research_guide/blob/a458495c7620b0af1a86104e774d0f9f03b459e9/phi3-mini-onxx.png">

# File Structures
- ***pre_processing.py:*** Contains code for parsing PDF file, creating Embedding and Vector DB.
- ***application.ipynb:*** This notebook for creating a pdf Q/A pipeline.
- ***app.py:*** Code for Gradio Application. The app is hosted on [`HF Space`](https://huggingface.co/spaces/SwastikM/RA)

# Frameworks
- **LLM:** Phi3 Mini
- **Embedding Model:** nomic-embed-text-v1
- **Vector DB:** faiss
- **Application:** Gradio

# How to RUN
- Install libraries with `make install`
- Prepare Phi3 Mini with `ONXX CPU Runtime in Linux` with `make phi3_dependency`
- Run run the app execute `python app.py`

# Acknowledgement
- `Microsoft` for the open source Phi3 Mini Quantized along with ONXX Runtime support.
- `Hugging Face` for the all the educational and open source resources.
