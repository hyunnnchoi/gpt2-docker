# 베이스 이미지 선택 (Python 3.10 사용)
FROM python:3.10.12

# 작업 디렉토리 설정
WORKDIR /workspace/keras-benchmarks

# 필요한 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg2 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# NVIDIA 패키지 레포지토리 추가
RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
    && apt-get update

# NVIDIA Container Toolkit 설치
RUN apt-get install -y --no-install-recommends nvidia-container-toolkit

# NVML 경로를 LD_LIBRARY_PATH에 추가
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}"

# 소스 코드 복사
COPY ./keras-benchmarks /workspace/keras-benchmarks

# 복사된 파일 확인
RUN ls -al /workspace/keras-benchmarks/requirements/

# pip 업그레이드
RUN pip install --upgrade pip

# keras-tensorflow 설치
RUN pip install --no-cache-dir -r /workspace/keras-benchmarks/requirements/keras-common.txt

# 필요한 패키지 설치
RUN bash /workspace/keras-benchmarks/shell/install.sh

# PYTHONPATH 환경 변수 설정
ENV PYTHONPATH="/workspace/keras-benchmarks:${PYTHONPATH}"

# 실행 명령어 설정
ENTRYPOINT ["bash", "-c"]
CMD ["bash /workspace/keras-benchmarks/shell/install.sh && bash /workspace/keras-benchmarks/shell/run.sh"]
