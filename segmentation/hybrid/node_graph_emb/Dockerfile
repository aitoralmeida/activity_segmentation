FROM nvidia/cuda:8.0-cudnn5-runtime-ubuntu16.04

RUN apt-get update && apt-get install -y --allow-unauthenticated \
    build-essential libssl-dev libffi-dev \
    libxml2-dev libxslt1-dev zlib1g-dev \
    python-minimal \
    python-pip \
    python-tk \
    python-dev

RUN pip install --upgrade "pip < 21.0"

RUN pip install networkx==2.2
RUN pip install scipy==1.1.0
RUN pip install setuptools==40.6.3
RUN pip install numpy==1.15.4
RUN pip install tensorflow==1.4.0
RUN pip install h5py
RUN pip install Keras==2.0.8