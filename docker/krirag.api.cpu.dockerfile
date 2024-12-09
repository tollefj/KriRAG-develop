ARG UBUNTU_VERSION=22.04
FROM ubuntu:$UBUNTU_VERSION AS build

RUN apt update && apt install -y build-essential git libcurl4-openssl-dev

WORKDIR /app
COPY llama.cpp/ .

ENV LLAMA_CURL=1

RUN make -j$(nproc) llama-server

FROM ubuntu:$UBUNTU_VERSION AS runtime

RUN apt update && apt install -y libcurl4-openssl-dev libgomp1 curl

COPY --from=build /app/llama-server /llama-server

ENV LC_ALL=C.utf8
ENV LLAMA_ARG_HOST=0.0.0.0

ENTRYPOINT [ "/llama-server" ]
