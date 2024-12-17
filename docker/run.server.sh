#!/bin/bash
if ! docker network ls | grep -q krirag-net; then
  docker network create krirag-net
else
  echo "Connecting to existing krirag docker net."
fi
if ! docker images | grep -q krirag-api; then
  docker load < api.tar
else
  echo "krirag-api image is already loaded."
fi

CONTAINER_NAME="krirag-api"

# stop and remove the old one
if docker ps --filter "name=$CONTAINER_NAME" --format "{{.Names}}" | grep -q "^$CONTAINER_NAME$"; then
  echo "Stopping existing container: $CONTAINER_NAME..."
  docker stop $CONTAINER_NAME
else
  echo "No running container named $CONTAINER_NAME found."
fi

if docker ps -a --filter "name=$CONTAINER_NAME" --format "{{.Names}}" | grep -q "^$CONTAINER_NAME$"; then
  echo "Removing existing container: $CONTAINER_NAME..."
  docker rm $CONTAINER_NAME
else
  echo "No stopped container named $CONTAINER_NAME found."
fi

MODEL_PATH="/models/gemma-2-9b-it-Q5_K_M.gguf"
MODEL_PATH="/models/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"
MODEL_PATH="/models/Phi-3-medium-4k-instruct-Q5_K_M.gguf"
MODEL_PATH="/models/gemma-2-27b-it-Q5_K_M.gguf"
NGPU="100" # Number of GPU layers, just max it at 100
N_CONTEXT_LEN="4096"

# Run the Llama.cpp server
docker run \
  -v ~/LLM_STORE:/models \
  -p 8502:8502 \
  --name $CONTAINER_NAME \
  --network krirag-net \
  --gpus all \
  $CONTAINER_NAME \
  -m "$MODEL_PATH" \
  --port 8502 -n $N_CONTEXT_LEN -ngl $NGPU
