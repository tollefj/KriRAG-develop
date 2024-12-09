#!/bin/bash
if ! docker network ls | grep -q krirag-net; then
  docker network create krirag-net
fi

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

cleanup_container krirag-api
cleanup_container krirag-ui
