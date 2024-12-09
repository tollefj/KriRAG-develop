default: install run
install:
	pip install -r requirements.txt
	python3 src/install.py
run:
	cd src && streamlit run ui.py

# install-gpu:
# run docker with
# --gpus all
# after installing nvidia-container-toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

server:
	chmod +x docker/download.sh
	chmod +x docker/serve.sh
	./docker/download.sh
	./docker/serve.sh

frontend:
	docker build -t krirag -f krirag.dockerfile .
	docker run -p 8501:8501 krirag


NEWEST_UPDATE:
	mkdir llms
	echo "Move your .gguf model to the llm folder: `mv path/to/your/model.gguf llms/`"
