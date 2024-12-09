FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
# FROM semitechnologies/transformers-inference:sentence-transformers-all-MiniLM-L6-v2
LABEL maintainer="tollefj"
WORKDIR /app

# RUN apt-get update && apt-get install -y \
#     python3-pip \
#     python3-dev \
#     && apt-get clean

COPY src/requirements.txt /app/
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY src/ /app
RUN python3 install.py

EXPOSE 8501
CMD ["bash", "-c", "streamlit run ui.py"]
