install: 
		pip install --upgrade pip &&\
		pip install -r requirements.txt

phi3_dependency:
				pip uninstall -y transformers &&\
				pip install git+https://github.com/huggingface/transformers

all: install phi3_dependency