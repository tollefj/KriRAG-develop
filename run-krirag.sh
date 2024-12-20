#!/bin/bash
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "Error: .env file not found. Please create it with the required variables."
    exit 1
fi

if [ ! -d ~/LLM_STORE ] || [ -z "$(ls -A ~/LLM_STORE/*.gguf 2>/dev/null)" ]; then
    echo "Error: ~/LLM_STORE does not exist or contains no .gguf LLM files."
else
    echo "Found .gguf files in ~/LLM_STORE:"
    ls -1 ~/LLM_STORE/*.gguf
fi

MODEL_NAME=${1:-"gemma-2-27b-it-Q5_K_M.gguf"}  # gemma is default. first user arg is the model name otherwise.

if [ ! -f ~/LLM_STORE/$MODEL_NAME ]; then
    echo "Error: $MODEL_NAME not found in ~/LLM_STORE."
    echo "Make sure you download a .gguf model file and place it in ~/LLM_STORE."
    default_model_url="https://huggingface.co/bartowski/gemma-2-27b-it-GGUF/resolve/main/gemma-2-27b-it-Q5_K_M.gguf"

    read -p "Would you like to download the default model (gemma-2-27b-it-Q5_K_M.gguf, approx. 20GB)? [y/n] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        mkdir -p ~/LLM_STORE
        echo "Downloading default model to ~/LLM_STORE..."
        wget --content-disposition "$default_model_url" -O ~/LLM_STORE/gemma-2-27b-it-Q5_K_M.gguf
        
        # Check if download succeeded
        if [ $? -eq 0 ] && [ -f ~/LLM_STORE/gemma-2-27b-it-Q5_K_M.gguf ]; then
            echo "Download completed successfully."
        else
            echo "Error: Failed to download the model. Please check your internet connection or try again."
            exit 1
        fi
    else
        echo "Exiting..."
        exit 1
    fi
fi

if ! docker image inspect krirag-api >/dev/null 2>&1 || ! docker image inspect krirag-ui >/dev/null 2>&1; then
    if [ -f api.tar ] && [ -f ui.tar ]; then
        echo "Loading offline docker images..."
        FILESIZE_GB=$(du -h api.tar | cut -f1)
        echo "Loading API ($FILESIZE_GB)..."
        docker load -i api.tar
        FILESIZE_GB=$(du -h ui.tar | cut -f1)
        echo "Loading UI ($FILESIZE_GB)..."
        docker load -i ui.tar
    else
        echo "Error: api.tar/ui.tar not found."
        exit 1
    fi
else
    echo "Docker images already exist. Skipping load."
fi

docker network create krirag-net

cleanup_container() {
    local CONTAINER_NAME=$1

    if docker ps --filter "name=$CONTAINER_NAME" --format "{{.Names}}" | grep -q "^$CONTAINER_NAME$"; then
        echo "Stopping existing container: $CONTAINER_NAME..."
        docker stop $CONTAINER_NAME
    fi

    if docker ps -a --filter "name=$CONTAINER_NAME" --format "{{.Names}}" | grep -q "^$CONTAINER_NAME$"; then
        echo "Removing existing container: $CONTAINER_NAME..."
        docker rm $CONTAINER_NAME
    fi
}

API_NAME="krirag-api"
UI_NAME="krirag-ui"
NGPU="100" # Number of GPU layers, just max it at 100
N_CONTEXT_LEN="4096"

cleanup_container $API_NAME
docker run -d \
    --gpus all \
    --name $API_NAME \
    --network krirag-net \
    -p 8502:8502 \
    -v ~/LLM_STORE:/models \
    krirag-api \
    -m "models/$MODEL_NAME" \
    --port 8502 -n $N_CONTEXT_LEN -ngl $NGPU

cleanup_container $UI_NAME
docker run \
    --gpus all \
    --name $UI_NAME \
    --network krirag-net \
    -p 8501:8501 \
    krirag-ui
