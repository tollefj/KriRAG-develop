# setup

## docker containers

The script below sets up two containers: the API and UI service.

```bash
chmod +x run-krirag.sh
./run-krirag.sh
```

### server-only docker container

```bash
./docker/build.server.sh
./docker/run.server.sh
```

### frontend-only docker container

```bash
./docker/build.ui.sh 
./docker/run.ui.sh
```

## local installation

### llama.cpp server

```bash
git clone git@github.com:ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
# this assumes enough VRAM to offload all layers (rarely >~ 40) and 4096 context length
./llama.cpp/llama-server -m <PATH/TO/MODEL.gguf> -ngl 100 -c 4096 --port 8052
```

### frontend

```bash
cd src
# requirements.cpu.txt if CUDA is unavailable
# the SBERT model will use the openvino backend to offload the GPU memory.
python -m pip install -r requirements.txt
python install.py
streamlit run ui.py
```
