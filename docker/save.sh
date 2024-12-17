#!/bin/bash
if [ ! -f api.tar ]; then
    echo "Saving API docker image..."
    docker save -o api.tar krirag-api
fi

if [ ! -f ui.tar ]; then
    echo "Saving UI docker image..."
    docker save -o ui.tar krirag-ui
fi
