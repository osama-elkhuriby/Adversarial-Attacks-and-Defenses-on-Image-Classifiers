# Exploring Adversarial Robustness of Lightweight Image Classifiers  
## From MNIST Baselines to CIFAR-10 Adversarial Training

This repository contains the implementation, experiments, and reproducibility materials for a deep learning project on adversarial robustness in image classification. The project studies how lightweight convolutional neural networks behave under standard white-box adversarial attacks and how adversarial training changes the clean-accuracy versus robust-accuracy trade-off.

---

## Abstract

Deep neural networks can achieve strong accuracy on clean image classification benchmarks, but they may fail under small, carefully designed adversarial perturbations. This project first validates an adversarial evaluation pipeline on MNIST using a lightweight CNN and then extends the study to CIFAR-10, a more challenging RGB image dataset. We evaluate a clean CIFAR-10 CNN baseline under Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD), then compare it with three defended variants: FGSM adversarial training (FGSM-AT), PGD adversarial training (PGD-AT), and Mixed Adversarial Training (Mixed-AT). The results show that clean accuracy alone is not sufficient for evaluating model reliability. The clean baseline reaches 78.22% CIFAR-10 clean accuracy but collapses to 2.42% under FGSM and 0.00% under PGD at ε = 8/255. Adversarial training substantially improves robust accuracy, while introducing a clear robustness–accuracy trade-off.

---

## Research Question

How vulnerable are lightweight CNN image classifiers to FGSM and PGD attacks, and to what extent can adversarial training improve robustness while preserving clean accuracy?

---

## Main Contributions

- Implemented standard FGSM and PGD adversarial attacks in PyTorch.
- Validated the attack pipeline on MNIST as a preliminary benchmark.
- Extended the project to CIFAR-10 as the final robustness benchmark.
- Trained and evaluated four CIFAR-10 variants:
  - Clean baseline CNN
  - FGSM-AT
  - PGD-AT
  - Mixed-AT
- Compared clean accuracy, FGSM robust accuracy, and PGD robust accuracy under matched attack settings.
- Added a small PGD-AT ablation study over training steps and epochs.
- Provided reproducibility scripts, result tables, generated figures, and a final IEEE-style report.

---

## Project Structure

```text
Adversarial-Attacks-and-Defenses-on-Image-Classifiers/
├── data/                              # Auto-downloaded datasets; not committed
├── CHECKPOINT_DIR/                    # Saved checkpoints; not committed
├── results/                           # JSON metrics and evaluation outputs
├── reports/
│   ├── figures/                       # Figures used in the report and presentation
│   └── report.tex                     # Final LaTeX report
├── src/
│   ├── attacks/
│   │   ├── fgsm.py                    # FGSM implementation
│   │   └── pgd.py                     # PGD implementation
│   └── models/
│       ├── cnn.py                     # MNIST CNN
│       └── cifar_cnn.py               # CIFAR-10 CNN
├── scripts/
│   ├── training/
│   │   ├── train_MNIST_clean.py
│   │   ├── training cnn on clear cifar10.py
│   │   ├── train_adversarial_MNIST.py
│   │   ├── train_adversarial_cifar.py
│   │   └── mixed_CIFAR.py
│   ├── evaluation/
│   │   ├── evaluate_fgsm.py
│   │   ├── evaluate_pgd.py
│   │   ├── evaluate_all_cifar.py
│   │   ├── ablation_study.py
│   │   └── ablation_study_cifar.py
│   └── visualization/
│       ├── plot_results_cifar.py
│       └── other visualization utilities
├── dl-project-linux.yml               # Conda environment for Linux/macOS
├── dl-project-windows.yml             # Conda environment for Windows
├── .gitignore
└── README.md
```

---

## Datasets

| Dataset | Image Type | Classes | Standard Split | Project Role |
|---|---:|---:|---:|---|
| MNIST | 1 × 28 × 28 grayscale | 10 | 60,000 train / 10,000 test | Preliminary validation |
| CIFAR-10 | 3 × 32 × 32 RGB | 10 | 50,000 train / 10,000 test | Final benchmark |

For the project protocol, MNIST training data is split into 50,000 training and 10,000 validation samples. CIFAR-10 training data is split into 45,000 training and 5,000 validation samples using seed 42. The CIFAR-10 test set contains 10,000 images and is used for final clean, FGSM, and PGD evaluation.

Datasets are downloaded automatically through Torchvision. Raw datasets and checkpoints should not be committed to GitHub.

---

## Model Architectures

### MNIST CNN

The MNIST baseline is a lightweight two-convolution CNN with ReLU activations, max-pooling, and fully connected layers. It is used to validate the basic attack and evaluation pipeline.

### CIFAR-10 CNN

The CIFAR-10 model is a lightweight CNN with three convolutional blocks, batch normalization, ReLU activations, max-pooling, dropout, and a final 10-class classifier. The goal is not to reach state-of-the-art CIFAR-10 accuracy, but to provide a controlled and reproducible robustness evaluation.

---

## Attacks Implemented

### FGSM

FGSM is a one-step white-box attack:

```text
x_adv = x + ε · sign(∇_x J(θ, x, y))
```

It is computationally efficient and useful as a baseline robustness attack.

### PGD

PGD is an iterative projected attack:

```text
x_{t+1} = Π_{Bε(x)}(x_t + α · sign(∇_x J(θ, x_t, y)))
```

PGD is stronger than FGSM because it performs multiple projected gradient steps inside the allowed perturbation budget.

---

## Defense Strategies

The final CIFAR-10 experiments compare the clean baseline with three adversarial training variants.

| Defense | Training Attack | Training Setting |
|---|---|---|
| FGSM-AT | FGSM | ε = 8/255 |
| PGD-AT | PGD | ε = 8/255, α = 2/255, 7 steps |
| Mixed-AT | Clean + FGSM + PGD | Mixed mini-batches with clean, FGSM, and PGD samples |

Adversarial training improves robustness but typically reduces clean accuracy. Mixed-AT is included to study whether combining clean and adversarial examples improves the clean–robust balance.

---

## Experimental Protocol

| Hyperparameter | Baseline | FGSM-AT | PGD-AT | Mixed-AT |
|---|---:|---:|---:|---:|
| Optimizer | Adam | Adam | Adam | Adam |
| Learning rate | 0.001 | 0.001 | 0.001 | 0.001 |
| Batch size | 64 | 64 | 64 | 64 |
| Epochs | 20 | 15 | 15 | 20 |
| AT ε | — | 8/255 | 8/255 | 8/255 |
| PGD α | — | — | 2/255 | 2/255 |
| PGD steps | — | — | 7 | 7 |
| Seed | 42 | 42 | 42 | 42 |

Evaluation uses the same CIFAR-10 test set for all models. Clean accuracy, FGSM robust accuracy, PGD robust accuracy, and training loss are reported.

---

## Main CIFAR-10 Results

At ε = 8/255:

| Model | Clean Accuracy | FGSM 8/255 | PGD 8/255 |
|---|---:|---:|---:|
| Baseline | 78.22% | 2.42% | 0.00% |
| FGSM-AT | 58.66% | 36.87% | 32.62% |
| PGD-AT | 59.53% | 35.89% | 32.60% |
| Mixed-AT | 65.20% | 35.27% | 29.35% |

The clean baseline performs best on unperturbed images, but collapses under adversarial evaluation. FGSM-AT and PGD-AT provide the strongest PGD robustness, while Mixed-AT achieves the best clean accuracy among defended models.

---

## FGSM Robustness Sweep

| Model | 2/255 | 4/255 | 8/255 | 16/255 |
|---|---:|---:|---:|---:|
| Baseline | 19.12% | 7.28% | 2.42% | 1.68% |
| FGSM-AT | 52.51% | 46.67% | 36.87% | 22.22% |
| PGD-AT | 52.70% | 46.57% | 35.89% | 20.91% |
| Mixed-AT | 56.62% | 48.53% | 35.27% | 18.67% |

---

## PGD Robustness Sweep

| Model | 4/255 | 8/255 | 16/255 |
|---|---:|---:|---:|
| Baseline | 0.18% | 0.00% | 0.00% |
| FGSM-AT | 45.58% | 32.62% | 11.75% |
| PGD-AT | 45.70% | 32.60% | 11.95% |
| Mixed-AT | 46.59% | 29.35% | 7.05% |

PGD is significantly stronger than FGSM for the clean baseline. At ε = 8/255 and above, the baseline is essentially unusable under PGD evaluation.

---

## PGD-AT Ablation Study

| Setting | Epochs | PGD Steps | Clean / Robust Accuracy |
|---|---:|---:|---:|
| Steps = 3 | 15 | 3 | 66.46% / 25.84% |
| Steps = 7 | 15 | 7 | 57.46% / 32.27% |
| Epochs = 10 | 10 | 7 | 54.87% / 32.43% |
| Epochs = 15 | 15 | 7 | 57.46% / 32.27% |

Increasing PGD training steps improves robust accuracy but reduces clean accuracy. In this small ablation, PGD step count has a stronger effect on the clean–robust trade-off than the small change in training duration.

---

## Setup and Installation

### Requirements

- Python 3.10
- PyTorch
- TorchVision
- NumPy
- Matplotlib
- Pandas
- scikit-learn
- tqdm
- Pillow

A CUDA-capable GPU is recommended for CIFAR-10 adversarial training. MNIST can run comfortably on CPU.

### Clone Repository

```bash
git clone https://github.com/osama-elkhuriby/Adversarial-Attacks-and-Defenses-on-Image-Classifiers.git
cd Adversarial-Attacks-and-Defenses-on-Image-Classifiers
```

### Linux / macOS

```bash
conda env create -f dl-project-linux.yml
conda activate dl-project
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

### Windows

```bash
conda env create -f dl-project-windows.yml
conda activate dl-project
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

---

## Reproducibility Commands

Each script should be executed from the folder shown below.

| Stage | Folder | Script |
|---|---|---|
| MNIST baseline | `scripts/training/` | `train_MNIST_clean.py` |
| CIFAR-10 baseline | `scripts/training/` | `training cnn on clear cifar10.py` |
| Defense training | `scripts/training/` | `train_adversarial_cifar.py` |
| Mixed-AT training | `scripts/training/` | `mixed_CIFAR.py` |
| Evaluation | `scripts/evaluation/` | `evaluate_all_cifar.py` |
| Visualization | `scripts/visualization/` | `plot_results_cifar.py` |

Example:

```bash
cd scripts/training
python train_adversarial_cifar.py
```

For files with spaces in their names, use quotes:

```bash
cd scripts/training
python "training cnn on clear cifar10.py"
```

---

## Expected Outputs

| Artifact | Purpose |
|---|---|
| Full evaluation JSON | Final clean, FGSM, and PGD accuracy values for all evaluated models |
| Baseline metrics JSON | Per-epoch loss and accuracy for the clean baseline |
| FGSM-AT metrics JSON | Per-epoch loss and accuracy for FGSM adversarial training |
| PGD-AT metrics JSON | Per-epoch loss and accuracy for PGD adversarial training |
| Mixed-AT metrics JSON | Per-epoch loss and accuracy for Mixed-AT |
| Model checkpoints | Saved model weights |
| `reports/figures/` | Exported plots used in the final report and presentation |

Large generated artifacts such as datasets, compressed archives, and checkpoints should be excluded from version control.

---

## Key Findings

1. Clean accuracy alone is not enough for evaluating image classifiers.
2. The CIFAR-10 clean baseline reaches 78.22% accuracy, but drops to 0.00% under PGD at ε = 8/255.
3. Adversarial training substantially improves robustness.
4. FGSM-AT and PGD-AT give the strongest PGD robustness.
5. Mixed-AT gives the best clean accuracy among defended models.
6. Robustness improvements come with a clear clean-accuracy cost.

---

## Limitations and Future Work

This project is a course-scale reproducible study, not a full publication-level robustness benchmark. Future work should include:

- Evaluating stronger architectures such as ResNet or WideResNet.
- Adding stronger attacks such as AutoAttack, DeepFool, and Carlini-Wagner.
- Running multiple random seeds and reporting mean ± standard deviation.
- Testing more ablations on model capacity, dropout, augmentation, training epsilon, and perturbation budgets.
- Profiling runtime, memory usage, and GPU/CPU efficiency.
- Comparing against published robust CIFAR-10 baselines.
- Further investigating Mixed-AT by varying clean/FGSM/PGD mixing ratios.

---

## References

- C. Szegedy et al., “Intriguing properties of neural networks,” ICLR, 2014.
- I. J. Goodfellow, J. Shlens, and C. Szegedy, “Explaining and harnessing adversarial examples,” ICLR, 2015.
- A. Madry et al., “Towards deep learning models resistant to adversarial attacks,” ICLR, 2018.
- N. Carlini and D. Wagner, “Towards evaluating the robustness of neural networks,” IEEE Symposium on Security and Privacy, 2017.
- Y. LeCun et al., “Gradient-based learning applied to document recognition,” Proceedings of the IEEE, 1998.
- A. Krizhevsky, “Learning multiple layers of features from tiny images,” University of Toronto, 2009.
- F. Croce and M. Hein, “Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks,” ICML, 2020.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
