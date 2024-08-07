# NVIDIA CUDA 베이스 이미지 사용 (Ubuntu 20.04 기반)
FROM nvidia/cuda:12.5.0-devel-ubuntu20.04

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive

# GPG 키 및 저장소 업데이트
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    gnupg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC \
    && apt-get update


# 필요한 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    wget \
    curl \
    build-essential \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    && rm -rf /var/lib/apt/lists/*

# NVIDIA 저장소 추가 및 cuDNN 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl \
    && curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub | apt-key add - \
    && echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    libcudnn8 \
    libcudnn8-dev \
    && apt-mark hold libcudnn8 \
    && rm -rf /var/lib/apt/lists/*
    
# pip 설치
RUN wget https://bootstrap.pypa.io/get-pip.py && python3.10 get-pip.py

# 나머지 Dockerfile 내용은 그대로 유지...

# 소스 코드 복사
COPY ./keras-benchmarks /workspace/keras-benchmarks
COPY ./SQuAD2_sampled.json /workspace/keras-benchmarks/SQuAD2_sampled.json

# 복사된 파일 확인
RUN ls -al /workspace/keras-benchmarks

# pip 업그레이드
RUN python3.10 -m pip install --upgrade pip

# TensorFlow GPU 버전 및 필요한 패키지 설치
RUN python3.10 -m pip install tensorflow==2.15.0  # TensorFlow GPU 버전
RUN python3.10 -m pip install keras==3.2.0 keras-nlp
RUN python3.10 -m pip install -r /workspace/keras-benchmarks/requirements/hmchoi.txt
RUN python3.10 -m pip install -e /workspace/keras-benchmarks

# 설치된 패키지 확인
RUN python3.10 -m pip list

# PYTHONPATH 환경 변수 설정
ENV PYTHONPATH="/workspace/keras-benchmarks:${PYTHONPATH}"

# 작업 디렉토리 설정
WORKDIR /workspace/

# 실행 명령어 설정
CMD ["bash", "-c", "export PYTHONPATH=$PYTHONPATH:/workspace/keras-benchmarks && bash /workspace/keras-benchmarks/shell/run.sh"]
