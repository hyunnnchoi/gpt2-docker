#!/bin/bash

pip install --upgrade pip
pip install -r /workspace/keras-benchmarks/requirements/hmchoi.txt
pip install keras==3.2.0 #3.0.5
pip install keras-nlp
pip install -e /workspace/keras-benchmarks

echo "Installed libraries from .txt"
