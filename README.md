# digital_research_guide

This project implements RAG with the cost-effective Phi-3-mini (3.8B) from `Microsoft`.

The project implements a question answer system with respect to a user provided research PDF.

# Gradio App
[🚀 Research Assistant! 🎓](https://huggingface.co/spaces/SwastikM/RA)

# Model Description
 The model used is microsoft/Phi-3-mini-4k-instruct-onnx. The model is small yet powerful. According to `Microsoft` the model provides uses for applications which require:
- Memory/compute constrained environments
- Latency bound scenarios
- Strong reasoning (especially code, math and logic)

The ONXX model used here is uses ONXX Runtime for Inference. The model run efficiently on CPU and inference speed is descent.

# Quantization
The model uses int4 quantization via RTN.

# App
- Gradio😍🚀

#NB
Cloning this repository won't work. The work has been created on Linux Cloud Enveionment using Github Codespace. The model files are missing in this repositry due to large size.
The model can be easily setup following the instruction from hosted model from `Microsoft` on [HuggingFace](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct-onnx)
