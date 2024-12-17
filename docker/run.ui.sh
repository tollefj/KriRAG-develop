#!/bin/bash

if ! docker images | grep -q "krirag-ui"; then
  docker load -i ui.tar
else
  echo "Image krirag-ui is already loaded."
fi

CONTAINER_NAME="krirag-ui"

if ! docker network ls | grep -q krirag-net; then
  docker network create krirag-net
else
  echo "Connecting to existing krirag docker net."
fi

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

docker run --gpus all --name $CONTAINER_NAME --network krirag-net -p 8501:8501 krirag-ui