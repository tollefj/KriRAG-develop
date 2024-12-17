#!/bin/bash
cd src
python3 install.py
cd ..
docker build -t krirag-ui . -f docker/krirag.cpu.dockerfile