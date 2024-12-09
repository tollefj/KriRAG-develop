#!/bin/bash
if [ ! -f api.tar ]; then
    docker save -o api.tar krirag-api
fi

if [ ! -f ui.tar ]; then
    docker save -o ui.tar krirag-ui
fi
