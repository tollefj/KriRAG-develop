FROM python:3.12-slim
LABEL maintainer="tollefj"
WORKDIR /app

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     git \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir --upgrade pip
COPY src/requirements.cpu.txt /app/
RUN python3 -m pip install --no-cache-dir -r requirements.cpu.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY src/ /app
EXPOSE 8501
CMD ["bash", "-c", "streamlit run ui.py"]
