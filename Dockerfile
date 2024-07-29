# 베이스 이미지 선택 (Ubuntu 20.04)
FROM ubuntu:20.04

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive

# 필요한 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg2 \
    curl \
    ca-certificates \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# pip 설치
RUN wget https://bootstrap.pypa.io/get-pip.py && python3.10 get-pip.py

# NVIDIA 패키지 레포지토리 추가
RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin | tee /etc/apt/preferences.d/cuda-repository-pin-600 && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-515.43.04-1_amd64.deb -O && \
    dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-515.43.04-1_amd64.deb && \
    apt-key add /var/cuda-repo-ubuntu2004-11-8-local/7fa2af80.pub && \
    apt-get update && \
    apt-get install -y cuda

# NVIDIA Container Toolkit 설치
RUN curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add - && \
    curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu20.04/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list && \
    apt-get update && apt-get install -y nvidia-container-toolkit

# NVML 경로를 LD_LIBRARY_PATH에 추가
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}"

# 소스 코드 복사
COPY ./keras-benchmarks /workspace/keras-benchmarks
COPY ./SQuAD2_sampled.json /workspace/keras-benchmarks/SQuAD2_sampled.json

# 복사된 파일 확인
RUN ls -al /workspace/keras-benchmarks

# pip 업그레이드
RUN python3.10 -m pip install --upgrade pip

# keras 및 필요한 패키지 설치
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
