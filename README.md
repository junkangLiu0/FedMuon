# FedMuon

# FedMuon: Accelerating Federated Learning with Matrix Orthogonalization

This repository provides the official implementation of the paper
**“FedMuon: Accelerating Federated Learning with Matrix Orthogonalization”**,
which introduces a communication-efficient and fast-converging federated learning framework that incorporates **orthogonalized updates** for local models and **matrix-based momentum aggregation** on the server.

---

## 🌟 Key Features

* **Matrix Orthogonalization**: Prevents gradient interference between clients by orthogonalizing parameter updates.
* **Accelerated Convergence**: Combines momentum aggregation with block-wise orthogonal projections to stabilize updates.
* **Communication-Efficient**: Reduces redundant information transmission in cross-device federated settings.
* **Flexible Framework**: Built on **Ray (v1.0.0)** for scalable client–server simulations.
* **Extensible Design**: Supports multiple optimization methods including `FedMuon`, `FedAvg`, `FedAdam`, and `FedMomentum`.

---

## 🛠 Environment Setup

### Requirements

python 3.8 !!!
torch==2.4.1\
torchvision==0.19.1\
numpy\
ray==1.0.0\
tensorboardX==2.6.2.2\
peft==0.13.2\
transformers==4.46.3



---

## 📂 Dataset Preparation

The code automatically downloads the dataset if it’s not found locally.
Supported datasets:

* **MNIST**
* **EMNIST**
* **CIFAR-10**
* **CIFAR-100**

You can also manually place datasets in the `./data/` directory.

---

## 🚀 Quick Start

### Run FedMuon on CIFAR-100

```bash
python  main_FedMuon.py --alg FedMuon --lr 3e-4 --data_name CIFAR100 --alpha_value 0.1 --alpha  10  --epoch 301  --extname FedMuon --lr_decay 2 --gamma 0.5  --CNN   deit_tiny --E 5 --batch_size 50   --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --pre 1 --num_workers 100 --preprint 10 --beta1 0.9 --beta2 0.999 --rho 0.01 --pix 32 --lora 0 --K 50
python  main_FedMuon.py --alg Local_Muon --lr 3e-4 --data_name CIFAR100 --alpha_value 0.1 --alpha  10  --epoch 301  --extname FedMuon --lr_decay 2 --gamma 0.5  --CNN   deit_tiny --E 5 --batch_size 50   --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --pre 1 --num_workers 100 --preprint 10 --beta1 0.9 --beta2 0.999 --rho 0.01 --pix 32 --lora 0 --K 50
python  main_FedMuon.py --alg FedAvg --lr 1e-1 --data_name CIFAR100 --alpha_value 0.1 --alpha  10  --epoch 301  --extname FedMuon --lr_decay 2 --gamma 0.5  --CNN   deit_tiny --E 5 --batch_size 50   --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --pre 1 --num_workers 100 --preprint 10 --beta1 0.9 --beta2 0.999 --rho 0.01 --pix 32 --lora 0 --K 50
python  main_FedMuon.py --alg FedAvg_adamw --lr 3e-4 --data_name CIFAR100 --alpha_value 0.1 --alpha  10  --epoch 301  --extname FedMuon --lr_decay 2 --gamma 0.5  --CNN   deit_tiny --E 5 --batch_size 50   --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --pre 1 --num_workers 100 --preprint 10 --beta1 0.9 --beta2 0.999 --rho 0.01 --pix 32 --lora 0 --K 50

```

### Run FedAvg for Comparison



---

## ⚙️ Argument Overview

| Argument         | Description                                                | Default     |
| ---------------- | ---------------------------------------------------------- | ----------- |
| `--alg`          | Federated algorithm (`FedMuon`, `FedAvg`, `FedAdam`, etc.) | `FedAvg`    |
| `--data_name`    | Dataset name (`MNIST`, `EMNIST`, `CIFAR10`, `CIFAR100`)    | `MNIST`     |
| `--model`        | Model architecture (`lenet5`, `resnet10`, `resnet18`)      | `lenet5`    |
| `--num_workers`  | Number of simulated clients                                | `100`       |
| `--selection`    | Fraction of clients selected per round                     | `0.1`       |
| `--E`            | Local epochs per client                                    | `1`         |
| `--batch_size`   | Local batch size                                           | `50`        |
| `--lr`           | Learning rate                                              | `0.1`       |
| `--lr_decay`     | Learning rate decay factor                                 | `1.0`       |
| `--alpha_value`  | Dirichlet parameter controlling non-IID degree             | `0.6`       |
| `--gpu`          | GPU index(es)                                              | `'0'`       |
| `--extname`      | Extra name tag for output                                  | `'default'` |
| `--check`        | Resume training from checkpoint                            | `0`         |
| `--num_gpus_per` | GPU fraction allocated to each Ray worker                  | `1.0`       |

---

## Parameter Reference

### Core Federated Learning Parameters
| Parameter | Description |
|-----------|-------------|
| `--alg` | Algorithm choice: `FedAvg`, `FedAdamW`, `FedCM`, `SCAFFOLD`, etc. |
| `--lr` | Client learning rate |
| `--lr_decay` | Learning rate decay strategy (1=exponential, 2=cosine annealing) |
| `--gamma` | Momentum parameter for certain algorithms |
| `--alpha` | Weight decay coefficient for AdamW optimizer |

### Data Parameters
| Parameter | Description |
|-----------|-------------|
| `--data_name` | Dataset: `CIFAR10`, `CIFAR100`, `imagenet`, `QQP`, `MNLI`, etc. |
| `--alpha_value` | Dirichlet distribution parameter for non-IID data splitting (0.1=highly non-IID, 1=IID) |
| `--num_workers` | Total number of clients |
| `--selection` | Fraction of clients selected per round (0.1=10%) |

### Model Parameters
| Parameter | Description |
|-----------|-------------|
| `--CNN` | Model architecture: `resnet18`, `swin_tiny`, `deit_tiny`, `roberta_base` |
| `--pre` | Use pretrained weights (1=True, 0=False) |
| `--normalization` | Normalization type: `BN` (BatchNorm) or `GN` (GroupNorm) |
| `--pix` | Input image size (32 for CIFAR, 224 for ImageNet) |

### Training Parameters
| Parameter | Description |
|-----------|-------------|
| `--epoch` | Total communication rounds |
| `--E` | Local epochs per client |
| `--batch_size` | Client batch size |
| `--K` | Maximum local steps per round (overrides E if smaller) |
| `--p` | Parallelism factor for client updates |

### LoRA Parameters
| Parameter | Description |
|-----------|-------------|
| `--lora` | Enable LoRA fine-tuning (1=True, 0=False) |
| `--r` | LoRA rank |
| `--lora_alpha` | LoRA scaling parameter |

### Optimization Parameters
| Parameter | Description |
|-----------|-------------|
| `--beta1` | Adam optimizer β1 parameter |
| `--beta2` | Adam optimizer β2 parameter |
| `--rho` | SAM optimizer perturbation radius |
| `--optimizer` | Base optimizer: `SGD` or `AdamW` |

### System Parameters
| Parameter | Description |
|-----------|-------------|
| `--gpu` | GPU device IDs (e.g., "0,1,2") |
| `--num_gpus_per` | GPU fraction per client (0.2=20% of a GPU) |
| `--print` | Print detailed logs (1=True, 0=False) |
| `--preprint` | Evaluation frequency (in epochs) |

---

## Output Files

- **Logs**: `./log/alg-dataset-lr-workers-batch-epochs-lr_decay.txt`
- **Checkpoints**: `./checkpoint/ckpt-alg-lr-extname-alpha_value-timestamp/`
- **Plots**: `./plot/alg-dataset-...-timestamp.npy` (contains accuracy/loss arrays)
- **Models**: `./model/model-alg-...-timestamp.pth`

---

## Notes

1. **LoRA Usage**: When `--lora 1`, only LoRA parameters are trainable by default
2. **Pretrained Models**: Automatically downloads required pretrained weights
3. **Data Splitting**: Uses Dirichlet distribution for non-IID splits when `--alpha_value < 1`
4. **Memory**: Adjust `--num_gpus_per` based on your GPU memory capacity

For transformer training with GLUE tasks, use `new_llm.py` with appropriate `--data_name` (QQP, MNLI, SST2, etc.).


## 🧠 Algorithm Highlights

### 1. **Matrix Orthogonalization Layer**

Local client updates are decomposed and orthogonalized using matrix operations, ensuring that aggregated updates capture **independent learning directions** across clients.

### 2. **Server Momentum Aggregation**

The central server maintains a **momentum term** over aggregated gradients, smoothing oscillations and promoting faster convergence.

### 3. **Ray-Based Simulation**

Clients (`DataWorker`) and the server (`ParameterServer`) are implemented as **Ray remote actors**, allowing parallel training and realistic communication simulation even on a single machine.

---

## 📊 Logging and Checkpoints

* Logs are automatically written to the `./log/` directory.
* Checkpoints are saved to:

  ```
  ./checkpoint/ckpt-{alg}-{lr}-{dataset}-{alpha_value}.pt
  ```
* Training curves (accuracy, loss) are stored in `.npy` format in the `./plot/` directory.
* TensorBoard logs can be visualized with:

  ```bash
  tensorboard --logdir runs
  ```

---

## 🔬 Reproducibility Tips

1. The script automatically sets fixed random seeds for Python, NumPy, and PyTorch (`seed=42`).
2. Keep `alpha_value` and `selection` consistent for fair comparisons.
3. Repeat experiments 3–5 times and report average ± std accuracy.
4. Use consistent hardware (single GPU or same Ray configuration).

---

## 📘 Citation

If you find this code useful for your research, please cite:

```bibtex
@inproceedings{fedmuon2025,
  title={FedMuon: Accelerating Federated Learning with Matrix Orthogonalization},
  author={Your Name and Coauthors},
  booktitle={Proceedings of ...},
  year={2025}
}
```

---

Would you like me to generate and attach this as a downloadable `README.md` file (UTF-8 encoded) with formatting preserved and example commands aligned?
