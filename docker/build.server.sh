#!/bin/bash
git clone git@github.com:ggerganov/llama.cpp.git
docker build -t krirag-api . -f docker/krirag.api.dockerfile