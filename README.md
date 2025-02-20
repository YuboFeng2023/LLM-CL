
# Improving Event Representation Learning via Generating and Utilizing Synthetic Data

We are pleased to release the official implementation of our paper titled "Improving Event Representation Learning via Generating and Utilizing Synthetic Data", which was submitted to the journal of Information Processing & Management.

## News

- Dec 12 2024, the code, dataset and checkpoints are coming soon!
- Dec 14 2024, the code has been released, the dataset and checkpoints are coming soon!
- Jan 27 2025, the paper has been accepted by *Information Processing & Management*! 🎉🎉🎉

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

### Acknowledgement

The code is developed based on [SWCC](https://github.com/imgaojun/SWCC4Event). We appreciate all the authors who made their code public, which greatly facilitates this project.

### Citation

@article{feng2025improving,
  title = {Improving event representation learning via generating and utilizing synthetic data},
  author = {Yubo Feng and Lishuang Li and Xueyang Qin and Beibei Zhang},
  journal = {Information Processing & Management},
  volume = {62},
  number = {4},
  pages = {104083},
  year = {2025},
  issn = {0306-4573},
  doi = {https://doi.org/10.1016/j.ipm.2025.104083},
  url = {https://www.sciencedirect.com/science/article/pii/S0306457325000251},
}
