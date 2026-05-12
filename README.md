



# FedMuon: Accelerating Federated Learning with Matrix Orthogonalization

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Ray-Distributed%20FL-028CF0.svg" alt="Ray">
  <img src="https://img.shields.io/badge/Task-Vision%20%7C%20Language-green.svg" alt="Tasks">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey.svg" alt="License">
</p>

<p align="center">
  <b>A structure-aware federated optimizer for large vision and language models.</b><br>
  FedMuon stabilizes non-IID federated training by coupling <b>matrix-orthogonalized local updates</b>,
  <b>local-global alignment</b>, and <b>cross-round momentum aggregation</b>.
</p>

---

* 一张4090或者两张2080ti即可训练！！发顶会！！代码问题或者讨论+vx 15653218567

* 我的其他论文也都是这一套代码配置，均可复现！

* 个人主页：https://junkangliu0.github.io/
## Overview

Federated learning usually relies on element-wise local optimizers such as SGD or AdamW. These optimizers treat matrix-shaped parameters as flattened vectors and may amplify ill-conditioned directions during multi-step local training, especially when client data are heterogeneous.

**FedMuon** introduces matrix orthogonalization into federated optimization. It first studies **Local Muon**, where each client applies Muon-style orthogonalized updates locally, and then addresses the instability of Local Muon under non-IID data with two mechanisms:

- **Local-Global Alignment**: aligns client-side orthogonalized updates with the global update direction to reduce client drift.
- **Momentum Aggregation**: aggregates client momentum states across communication rounds to avoid momentum reinitialization.
- **SVD Momentum Compression**: optionally communicates a low-rank approximation of momentum states to reduce communication overhead.

The repository provides Ray-based federated simulations for vision models and LoRA fine-tuning support for large language models.

---

## Highlights

- **Structure-aware federated optimization** for matrix-shaped model parameters.
- **Fast and stable convergence** under both IID and non-IID federated settings.
- **Non-IID correction** through global-direction alignment and momentum sharing.
- **Communication-efficient variant** via SVD-compressed momentum aggregation.
- **Broad model coverage**, including CNNs, Vision Transformers, Swin Transformers, RoBERTa, and GPT-style models.
- **Ray-based parallel simulation** for scalable client-server federated learning experiments.

---

## Method at a Glance

At communication round `r`, each selected client starts from the global model and the aggregated momentum state. For local step `k`, FedMuon computes a momentum matrix, applies a Muon-style LMO/orthogonalization step, and corrects the local direction using the global update estimate:

```text
M_i^{r,k+1} = beta M_i^{r,k} + (1 - beta) G_i^{r,k}
q_i^{r,k}   = lmo(M_i^{r,k+1})
x_i^{r,k+1} = x_i^{r,k} + eta [(1 - alpha) q_i^{r,k} - alpha Delta_G^r]
```

After local training, clients send model updates and momentum states to the server. The server aggregates model deltas and updates the global momentum. The SVD variant transmits a compressed momentum representation instead of the full matrix state.

---

## Repository Structure

```text
.
├── main_FedMuon.py          # Vision federated training entry point
├── new_llm.py               # Language / GLUE LoRA training entry point, if included
├── dirichlet_data.py        # Dirichlet non-IID partitioning
├── dataset.py               # Tiny-ImageNet dataset wrapper
├── model.py                 # Swin Transformer backbones
├── vit_model.py             # ViT backbones
├── models/
│   ├── resnet.py            # ResNet with GN variants
│   ├── resnet_bn.py         # ResNet with BN variants
│   └── DeiTTiny.py          # DeiT-Tiny backbone
├── data/                    # Dataset root
├── log/                     # Training logs
├── checkpoint/              # Checkpoints
└── plot/                    # Saved curves / numpy results
```

---

## Installation

```bash
conda create -n fedmuon python=3.8 -y
conda activate fedmuon

pip install torch torchvision
pip install numpy matplotlib filelock tensorboardX ray==1.0.0
pip install peft transformers
```

Recommended package versions used by the original implementation:

```text
python >= 3.8
torch >= 2.0
torchvision >= 0.15
ray == 1.0.0
tensorboardX == 2.6.2.2
peft == 0.13.2
transformers == 4.46.3
```

---

## Datasets

### Vision

The vision entry point supports:

| Dataset argument | Dataset | Notes |
|---|---|---|
| `CIFAR10` | CIFAR-10 | Automatically downloaded by torchvision |
| `CIFAR100` | CIFAR-100 | Automatically downloaded by torchvision |
| `imagenet` | Tiny-ImageNet-200 | Place under `./data/tiny-imagenet-200` |

For non-IID experiments, client partitions are generated with a Dirichlet distribution:

```text
alpha_value = 0.6  # mild heterogeneity
alpha_value = 0.1  # strong heterogeneity
```

Generated partition files are cached with names such as:

```text
num_workers_100-alpha_value_0.1-data_CIFAR100
```

### Language

For LoRA-based language experiments, the paper evaluates GLUE tasks and OpenWebText with RoBERTa / GPT-style models. Use the language training script if it is included in your repository.

---

## Quick Start

### FedMuon on CIFAR-100, Dirichlet-0.1

```bash
python main_FedMuon.py \
  --alg FedMuon \
  --data_name CIFAR100 \
  --CNN deit_tiny \
  --lr 3e-4 \
  --epoch 301 \
  --num_workers 100 \
  --selection 0.1 \
  --alpha_value 0.1 \
  --batch_size 50 \
  --E 5 \
  --K 50 \
  --lr_decay 2 \
  --gamma 0.5 \
  --alpha 10 \
  --beta1 0.9 \
  --beta2 0.999 \
  --rho 0.01 \
  --pix 32 \
  --lora 0 \
  --pre 1 \
  --gpu 0 \
  --num_gpus_per 0.1 \
  --p 1 \
  --preprint 10 \
  --normalization BN \
  --extname fedmuon_cifar100_dir01_deit
```

### Local Muon baseline

```bash
python main_FedMuon.py \
  --alg Local_Muon \
  --data_name CIFAR100 \
  --CNN deit_tiny \
  --lr 3e-4 \
  --epoch 301 \
  --num_workers 100 \
  --selection 0.1 \
  --alpha_value 0.1 \
  --batch_size 50 \
  --E 5 \
  --K 50 \
  --lr_decay 2 \
  --gamma 0.5 \
  --alpha 10 \
  --beta1 0.9 \
  --beta2 0.999 \
  --rho 0.01 \
  --pix 32 \
  --lora 0 \
  --pre 1 \
  --gpu 0 \
  --num_gpus_per 0.1 \
  --p 1 \
  --preprint 10 \
  --normalization BN \
  --extname local_muon_cifar100_dir01_deit
```

### FedAvg and AdamW baselines

```bash
python main_FedMuon.py \
  --alg FedAvg \
  --data_name CIFAR100 \
  --CNN deit_tiny \
  --lr 1e-1 \
  --epoch 301 \
  --num_workers 100 \
  --selection 0.1 \
  --alpha_value 0.1 \
  --batch_size 50 \
  --E 5 \
  --K 50 \
  --lr_decay 2 \
  --gpu 0 \
  --num_gpus_per 0.1 \
  --p 1 \
  --preprint 10 \
  --normalization BN \
  --extname fedavg_cifar100_dir01_deit
```

```bash
python main_FedMuon.py \
  --alg FedAvg_adamw \
  --data_name CIFAR100 \
  --CNN deit_tiny \
  --lr 3e-4 \
  --epoch 301 \
  --num_workers 100 \
  --selection 0.1 \
  --alpha_value 0.1 \
  --batch_size 50 \
  --E 5 \
  --K 50 \
  --lr_decay 2 \
  --gamma 0.5 \
  --alpha 10 \
  --beta1 0.9 \
  --beta2 0.999 \
  --rho 0.01 \
  --pix 32 \
  --lora 0 \
  --pre 1 \
  --gpu 0 \
  --num_gpus_per 0.1 \
  --p 1 \
  --preprint 10 \
  --normalization BN \
  --extname fedadamw_cifar100_dir01_deit
```


## ResNet-18
```bash
python main_FedMuon.py --alg FedMuon --lr 3e-2 --data_name CIFAR100 --alpha_value 0.1 --alpha 0.5 --epoch 301 --extname FedMuon_resnet18 --lr_decay 2 --gamma 0.5 --CNN resnet18 --E 5 --batch_size 50 --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --pre 1 --num_workers 100 --preprint 10 --beta1 0.9 --beta2 0.999 --rho 0.01 --pix 32 --lora 0 --K 50
```

```bash
python main_FedMuon.py --alg Local_Muon --lr 3e-2 --data_name CIFAR100 --alpha_value 0.1 --alpha 0.5 --epoch 301 --extname LocalMuon_resnet18 --lr_decay 2 --gamma 0.5 --CNN resnet18 --E 5 --batch_size 50 --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --pre 1 --num_workers 100 --preprint 10 --beta1 0.9 --beta2 0.999 --rho 0.01 --pix 32 --lora 0 --K 50
```

```bash
python main_FedMuon.py --alg FedAvg --lr 1e-1 --data_name CIFAR100 --alpha_value 0.1 --alpha 0.5 --epoch 301 --extname FedAvg_resnet18 --lr_decay 2 --gamma 0.5 --CNN resnet18 --E 5 --batch_size 50 --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --pre 1 --num_workers 100 --preprint 10 --beta1 0.9 --beta2 0.999 --rho 0.01 --pix 32 --lora 0 --K 50
```

```bash
python main_FedMuon.py --alg FedAvg_adamw --lr 3e-4 --data_name CIFAR100 --alpha_value 0.1 --alpha 0.5 --epoch 301 --extname FedAvgAdamW_resnet18 --lr_decay 2 --gamma 0.5 --CNN resnet18 --E 5 --batch_size 50 --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --pre 1 --num_workers 100 --preprint 10 --beta1 0.9 --beta2 0.999 --rho 0.01 --pix 32 --lora 0 --K 50
```

---

## Supported Algorithms

The current training script includes the following algorithm choices:

| Argument | Description |
|---|---|
| `FedMuon` | Proposed matrix-orthogonalized FL optimizer with momentum aggregation and local-global alignment |
| `Local_Muon` | Local Muon baseline without FedMuon correction |
| `FedAvg` | Local SGD / FedAvg baseline |
| `FedAvg_adamw` | AdamW-style local baseline |
| `FedAdam` | Server-side adaptive FedAdam baseline |
| `FedAdamW` | AdamW-based federated baseline |
| `FedCM` | Federated client-momentum baseline |
| `SCAFFOLD` | Control-variate correction baseline |
| `FedLADA` | Adaptive moment aggregation baseline |
| `Local_Soap` | SOAP-style local optimizer baseline |

> Note: `FedMuon_SVD` is implemented as a communication-efficient momentum-compression variant in the worker dispatch. If your local branch does not expose it in the main algorithm allow-list, add it before running `--alg FedMuon_SVD`.

---

## Supported Models

| Model argument | Architecture |
|---|---|
| `lenet5` | LeNet-style CNN |
| `resnet10`, `resnet18`, `resnet34`, `resnet50` | ResNet variants |
| `resnet18pre`, `resnet50pre`, `resnet101pre` | ImageNet-pretrained ResNet variants |
| `deit_tiny` | DeiT-Tiny |
| `VIT-B`, `VIT-L` | Vision Transformer backbones |
| `swin_tiny`, `swin_small`, `swin_base`, `swin_large` | Swin Transformer backbones |

LoRA is available for Transformer-style vision backbones and pretrained ResNet classifiers through `--lora 1`.

---

## Important Arguments

| Argument | Default | Description |
|---|---:|---|
| `--alg` | `FedLESAM` | Federated algorithm name. Use `FedMuon` for the proposed method. |
| `--data_name` | `CIFAR100` | Dataset name: `CIFAR10`, `CIFAR100`, or `imagenet`. |
| `--CNN` | `lenet5` | Model architecture. |
| `--lr` | `0.1` | Client learning rate. |
| `--epoch` | `1001` | Number of communication rounds. |
| `--num_workers` | `100` | Number of simulated clients. |
| `--selection` | `0.1` | Client participation ratio per round. |
| `--alpha_value` | `0.1` | Dirichlet concentration parameter for non-IID partitioning. |
| `--batch_size` | `50` | Client mini-batch size. |
| `--E` | `5` | Local epochs / local update budget. |
| `--K` | `50` | Maximum local steps per round. |
| `--lr_decay` | `0.998` | Learning-rate decay setting. |
| `--gpu` | `0` | Visible GPU device IDs. |
| `--num_gpus_per` | `1` | GPU fraction assigned to each Ray worker. |
| `--p` | `10` | Parallelism factor for client updates. |
| `--preprint` | `10` | Evaluation interval. |
| `--lora` | `0` | Enable LoRA fine-tuning. |
| `--r` | `16` | LoRA rank. |
| `--pix` | `224` | Input image resolution. Use `32` for CIFAR-style training. |
| `--pre` | `1` | Use pretrained weights when available. |
| `--normalization` | `BN` | Normalization type for ResNet variants. |
| `--datapath` | `./data` | Dataset root. |

---

## Paper Results

### CIFAR-100, 100 clients, 10% participation, batch size 50, K = 50

| Method | ResNet-18 Dir-0.6 | ResNet-18 Dir-0.1 | ViT-Tiny Dir-0.6 | ViT-Tiny Dir-0.1 |
|---|---:|---:|---:|---:|
| FedAvg | 64.08 ± 0.18 | 60.25 ± 0.20 | 32.36 ± 0.08 | 27.14 ± 0.12 |
| Local AdamW | 62.84 ± 0.08 | 58.97 ± 0.10 | 40.47 ± 0.09 | 37.86 ± 0.11 |
| Local Muon | 71.66 ± 0.15 | 66.71 ± 0.15 | 46.69 ± 0.15 | 40.53 ± 0.17 |
| **FedMuon** | **74.12 ± 0.18** | **73.05 ± 0.16** | **50.22 ± 0.14** | **48.18 ± 0.12** |

### GLUE with RoBERTa-Base + LoRA, 20 clients, 20% participation, K = 50

| Method | Average Accuracy |
|---|---:|
| FedAvg | 76.73 |
| Local AdamW | 78.77 |
| Local Muon | 80.17 |
| **FedMuon** | **81.00** |

---

## Outputs

The training script automatically writes logs, checkpoints, and curve files.

```text
log/         # Training logs, e.g., alg-dataset-lr-workers-batch-E-lr_decay.txt
checkpoint/  # Checkpoints, e.g., ckpt-{alg}-{lr}-{extname}-{alpha_value}-{timestamp}/
plot/        # Saved numpy arrays for accuracy / loss curves
runs/        # TensorBoard summaries
```

During training, the script reports:

```text
Iter r: accuracy, train loss, test loss, learning rate, algorithm, model, data split
```

Visualize TensorBoard logs with:

```bash
tensorboard --logdir runs
```

---

## Reproducibility Checklist

- Use the same client partition by keeping `num_workers`, `alpha_value`, and `data_name` unchanged.
- Keep `selection`, `batch_size`, `E`, and `K` fixed when comparing FL algorithms.
- Use the same backbone and input size across methods.
- Run multiple seeds and report mean ± standard deviation.
- For fair non-IID comparison, reuse cached Dirichlet partition files.
- For Ray simulation, tune `--num_gpus_per` and `--p` according to available GPU memory.

---

## Citation

```bibtex
@inproceedings{fedmuon2026,
  title     = {FedMuon: Accelerating Federated Learning with Matrix Orthogonalization for Large Models},
  author    = {Anonymous Authors},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

---

## Acknowledgements

This implementation builds on PyTorch, Ray, torchvision, PEFT, and Transformers. We thank the open-source community for providing reliable tools for scalable federated learning research.
