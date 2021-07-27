FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

RUN apt-get update && apt-get upgrade -y --allow-unauthenticated && \
    apt-get install -y --no-install-recommends --allow-unauthenticated \
    build-essential \
    git \
    python2.7

RUN apt install python-pip -y

RUN pip install --upgrade pip
RUN pip install -U numpy

RUN ["mkdir", "results"]

ADD . /

VOLUME /results
