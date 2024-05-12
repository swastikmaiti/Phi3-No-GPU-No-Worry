install: 
		pip install --upgrade pip &&\
		pip install -r requirements.txt

phi3_dependency:
				apt-get install git-lfs
				git lfs install
				pip install huggingface-hub[cli]
				huggingface-cli download microsoft/Phi-3-mini-4k-instruct-onnx --include cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/* --local-dir .
				pip install numpy
				pip install --pre onnxruntime-genai
