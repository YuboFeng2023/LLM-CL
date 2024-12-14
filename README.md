
# Improving Event Representation Learning via Generating and Utilizing Synthetic Data

We are pleased to release the official implementation of our paper titled "Improving Event Representation Learning via Generating and Utilizing Synthetic Data", which was submitted to the journal of Information Processing & Management.

## News

- Dec 12 2024, the code, dataset and checkpoints are coming soon!
- Dec 14 2024, the code has been released, the dataset and checkpoints are coming soon!

## Quick Start

### Installation

To run a docker container:

```bash
docker run ubuntu:22.04
```

To install pip requirements:

```bash
pip3 install \
  texar-pytorch \
  torch==1.13.1 \
  tensorflow==2.14 \
  numpy==1.26.4 \
  nltk \
  faiss-cpu \
  tiktoken \
  jupyter \
  matplotlib \
  openai \
  scipy \
  scikit-learn
```

### Synthetic

```bash
python3 syn-dat.py \
  --anchor /data/train.json \
  --atomic /data/atomic/v4_atomic_all.csv \
  --output /data/out \
  --api-key API-KEY \
  --prompt analogy-reasoner.j2
```

### Train

```bash
python3 main.py \
  --do-train \
  --output-dir /data/out
```

### Test

```bash
python3 main.py \
  --do-eval \
  --checkpoint /data/checkpoint.pt
```
