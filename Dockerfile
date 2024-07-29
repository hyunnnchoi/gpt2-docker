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
