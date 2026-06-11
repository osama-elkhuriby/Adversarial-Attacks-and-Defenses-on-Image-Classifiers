# Development Log — Adversarial Attacks and Defenses on Image Classifiers

## Project 5 | Queen's University | Deep Learning Course

**Project Title:**  
Exploring Adversarial Robustness of Lightweight Image Classifiers: From MNIST Baselines to CIFAR-10 Adversarial Training

**Team Members:**
- Osama Elkhuribi
- Mohamed Abdelkhalek
- Mohamed Abdel Majid

---

## Week 1 — Project Setup, Literature Review, and MNIST Baseline

### Session 1
**Date:** 2026-05-20

**Work done:**
- Read and discussed the core adversarial robustness papers:
  - Goodfellow et al. (2015): Fast Gradient Sign Method (FGSM).
  - Kurakin et al. (2017): Adversarial machine learning at scale.
  - Madry et al. (2018): Projected Gradient Descent (PGD) and adversarial training.
- Defined the initial project scope:
  - Start with MNIST as a controlled preliminary benchmark.
  - Use CIFAR-10 later as a more challenging final benchmark if time and compute allow.
- Assigned initial team responsibilities:
  - Osama Elkhuribi: model architecture and training pipeline.
  - Mohamed Abdelkhalek: FGSM and PGD attack implementation.
  - Mohamed Abdel Majid: visualization, reporting, README, and final integration.

**Decisions made:**
- Use PyTorch for all experiments because the team is more comfortable with it.
- Store metrics in JSON format to make results reproducible without rerunning every experiment.
- Fix random seeds for reproducibility across machines.
- Use MNIST first because it trains quickly and allows fast validation of the attack pipeline.

**Issues:**
- No major issues in this session.

---

### Session 2
**Date:** 2026-05-21

**Work done:**
- Created the initial project directory structure:

```text
src/models/
src/attacks/
scripts/training/
scripts/evaluation/
scripts/visualization/
results/
reports/figures/
data/
```

- Started implementing the MNIST baseline CNN in:

```text
src/models/cnn.py
```

- Defined the MNIST CNN architecture:
  - Conv(32) + ReLU + MaxPool
  - Conv(64) + ReLU + MaxPool
  - FC(128) + ReLU
  - FC(10) output logits

**Decisions made:**
- Use `CrossEntropyLoss` with raw logits, without applying softmax inside the model.
- Split MNIST as:
  - 50,000 training samples
  - 10,000 validation samples
  - 10,000 held-out test samples
- Keep the test set untouched for final clean and adversarial evaluation.

**Issues:**
- No major issues.

---

### Session 3
**Date:** 2026-05-24

**Work done:**
- Completed the MNIST clean training script:

```text
scripts/training/train_MNIST_clean.py
```

- Training configuration:
  - Optimizer: Adam
  - Learning rate: 0.001
  - Batch size: 64
  - Epochs: 5
  - Seed: 42
- Saved model checkpoints and exported per-epoch metrics to JSON.
- Completed the first MNIST baseline run.

**MNIST baseline results:**
- Final clean test accuracy: 98.76%
- Best test accuracy during training: approximately 98.87%

**Decisions made:**
- Save training metrics to JSON for reproducibility.
- Keep the final epoch checkpoint for adversarial evaluation.
- Use MNIST as the preliminary validation stage, not the final benchmark.

**Issues:**
- Initial environment was missing `tqdm`.
- Fixed by adding the missing dependency to the environment setup.

---

## Week 2 — FGSM, PGD, MNIST Evaluation, and Midterm Report

### Session 4
**Date:** 2026-05-25

**Work done:**
- Implemented FGSM from scratch in PyTorch:

```text
src/attacks/fgsm.py
```

- FGSM formula:

```text
x_adv = x + ε · sign(∇_x J(θ, x, y))
```

- Evaluated the MNIST baseline under FGSM using the full 10,000-sample test set.

**MNIST FGSM results:**

| Attack | Epsilon | Accuracy |
|---|---:|---:|
| Clean | 0 | 98.76% |
| FGSM | 0.05 | 94.45% |
| FGSM | 0.10 | 81.74% |
| FGSM | 0.20 | 25.74% |
| FGSM | 0.30 | 5.13% |

**Decisions made:**
- Evaluate all attacks on the full test set instead of a subset.
- Always call `model.eval()` before evaluation.
- Clamp adversarial images to the valid pixel range `[0, 1]`.

**Issues:**
- Needed to explicitly set `requires_grad=True` on input tensors before computing input gradients.
- Fixed the FGSM implementation and verified the expected monotonic drop as epsilon increases.

---

### Session 5
**Date:** 2026-05-26

**Work done:**
- Implemented PGD from scratch in PyTorch:

```text
src/attacks/pgd.py
```

- PGD was implemented as iterative FGSM with:
  - Random initialization inside the epsilon ball.
  - Gradient sign update.
  - Projection back into the allowed perturbation ball.
  - Pixel clamping to `[0, 1]`.

**MNIST PGD results:**

| Attack | Setting | Accuracy |
|---|---|---:|
| PGD | ε = 0.10, α = 0.01, 10 iterations | 77.81% |
| PGD | ε = 0.20, α = 0.01, 20 iterations | 2.33% |
| PGD | ε = 0.30, α = 0.01, 40 iterations | 0.00% |

**Decisions made:**
- Treat PGD as the stronger evaluation attack because it performs multiple projected gradient steps.
- Use random initialization for PGD to avoid underestimating attack strength.

**Issues:**
- Early PGD results were unexpectedly weak.
- Root cause: missing random initialization.
- Added uniform random initialization in `[-ε, ε]`, which produced the expected stronger PGD behavior.

---

### Session 6
**Date:** 2026-05-29

**Work done:**
- Generated preliminary MNIST qualitative visualizations showing original images, adversarial examples, and perturbations.
- Prepared the midterm report in IEEE two-column format.
- Midterm report sections included:
  - Abstract
  - Introduction
  - Related Work
  - Data and Preprocessing
  - Methodology
  - Preliminary Results
  - Planned Work
  - Team Contributions

**Decisions made:**
- The midterm report would focus on MNIST baseline vulnerability.
- CIFAR-10 and adversarial training would be moved to the final stage.
- Final work would include adversarial training and robustness–accuracy trade-off analysis.

**Issues:**
- No major issues in this session.

---

## Week 3 — Adversarial Training and CIFAR-10 Extension

### Session 7
**Date:** 2026-06-02

**Work done:**
- Implemented MNIST adversarial training support:

```text
scripts/training/train_adversarial_MNIST.py
```

- Created evaluation support for attack sweeps and full robustness testing.
- Added plotting utilities for MNIST robustness curves and visual examples.
- Started designing the final CIFAR-10 extension.

**Decisions made:**
- Use adversarial training as the main empirical defense.
- Compare clean training against FGSM adversarial training and PGD adversarial training.
- Extend beyond MNIST to CIFAR-10 because MNIST is too simple to support stronger final conclusions.

**Issues:**
- MNIST adversarial training was manageable on CPU.
- CIFAR-10 experiments were expected to require more compute.

---

### Session 8
**Date:** 2026-06-03

**Work done:**
- Designed a lightweight CIFAR-10 CNN architecture:

```text
src/models/cifar_cnn.py
```

- CIFAR-10 architecture:
  - Conv(32) + BatchNorm + ReLU
  - Conv(64) + BatchNorm + ReLU + MaxPool
  - Conv(128) + BatchNorm + ReLU + MaxPool
  - FC(256) + ReLU + Dropout(0.25)
  - FC(10) output logits
- Created CIFAR-10 training and evaluation scripts.
- Started restructuring the project so scripts are separated from reusable library modules.

**Main CIFAR-10 scripts:**

```text
scripts/training/training cnn on clear cifar10.py
scripts/training/train_adversarial_cifar.py
scripts/evaluation/evaluate_all_cifar.py
scripts/visualization/plot_results_cifar.py
```

**Decisions made:**
- Use CIFAR-10 as the final benchmark.
- Use common CIFAR-10 adversarial perturbation budgets:
  - ε = 8/255 for main evaluation
  - α = 2/255 for PGD
- Keep the model lightweight for computational feasibility.

**Issues:**
- CIFAR-10 training was slower than MNIST.
- Large checkpoint files were excluded from GitHub to avoid repository size problems.

---

## Week 4 — Final CIFAR-10 Experiments, Mixed-AT, and Report Finalization

### Session 9
**Date:** 2026-06-05

**Work done:**
- Completed CIFAR-10 clean baseline training.
- Final baseline configuration:
  - Optimizer: Adam
  - Learning rate: 0.001
  - Batch size: 64
  - Epochs: 20
  - Seed: 42
- Evaluated the baseline under clean, FGSM, and PGD conditions.

**Final baseline results at ε = 8/255:**
- Clean accuracy: 78.22%
- FGSM accuracy: 2.42%
- PGD accuracy: 0.00%

**Decisions made:**
- Use CIFAR-10 as the final benchmark because it is more realistic and challenging than MNIST.
- Report both clean accuracy and robust accuracy because clean accuracy alone is not sufficient for robustness evaluation.

**Issues:**
- Baseline clean accuracy was reasonable, but adversarial robustness collapsed almost completely under PGD.
- This confirmed the central motivation of the project.

---

### Session 10
**Date:** 2026-06-06

**Work done:**
- Completed FGSM adversarial training on CIFAR-10.
- Completed PGD adversarial training on CIFAR-10.
- Main training script:

```text
scripts/training/train_adversarial_cifar.py
```

**Adversarial training settings:**
- FGSM-AT:
  - ε = 8/255
- PGD-AT:
  - ε = 8/255
  - α = 2/255
  - PGD steps = 7
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 64
- Epochs: 15
- Seed: 42

**Final FGSM-AT results at ε = 8/255:**
- Clean accuracy: 58.66%
- FGSM accuracy: 36.87%
- PGD accuracy: 32.62%

**Final PGD-AT results at ε = 8/255:**
- Clean accuracy: 59.53%
- FGSM accuracy: 35.89%
- PGD accuracy: 32.60%

**Decisions made:**
- Use the same evaluation protocol for baseline, FGSM-AT, and PGD-AT.
- Keep PGD-AT training to 7 PGD steps to balance robustness and compute cost.

**Issues:**
- PGD-AT was significantly slower than clean training because each batch required several inner-loop PGD steps.
- Training was kept computationally feasible by limiting the PGD training steps.

---

### Session 11
**Date:** 2026-06-07

**Work done:**
- Implemented Mixed Adversarial Training:

```text
scripts/training/mixed_CIFAR.py
```

- Mixed-AT combines:
  - Clean images
  - FGSM adversarial examples
  - PGD adversarial examples
- Trained Mixed-AT for 20 epochs.
- Evaluated Mixed-AT using the same evaluation protocol as the other models.

**Final Mixed-AT results at ε = 8/255:**
- Clean accuracy: 65.20%
- FGSM accuracy: 35.27%
- PGD accuracy: 29.35%

**Decisions made:**
- Add Mixed-AT as an extension beyond the original baseline/FGSM-AT/PGD-AT comparison.
- Include Mixed-AT in the final report because it achieved the best clean accuracy among defended models while preserving competitive robustness.

**Issues:**
- Mixed-AT increased training cost because both FGSM and PGD adversarial examples were generated inside the training loop.
- Mixed-AT improved clean accuracy compared with FGSM-AT and PGD-AT but was slightly weaker under PGD evaluation.

---

### Session 12
**Date:** 2026-06-08

**Work done:**
- Ran the final CIFAR-10 FGSM robustness sweep.
- Ran the final CIFAR-10 PGD robustness sweep.
- Generated final figures using:

```text
scripts/visualization/plot_results_cifar.py
```

- Generated qualitative CIFAR-10 visualizations for FGSM and PGD progression.

**Final FGSM sweep results:**

| Model | 2/255 | 4/255 | 8/255 | 16/255 |
|---|---:|---:|---:|---:|
| Baseline | 19.12% | 7.28% | 2.42% | 1.68% |
| FGSM-AT | 52.51% | 46.67% | 36.87% | 22.22% |
| PGD-AT | 52.70% | 46.57% | 35.89% | 20.91% |
| Mixed-AT | 56.62% | 48.53% | 35.27% | 18.67% |

**Final PGD sweep results:**

| Model | 4/255 | 8/255 | 16/255 |
|---|---:|---:|---:|
| Baseline | 0.18% | 0.00% | 0.00% |
| FGSM-AT | 45.58% | 32.62% | 11.75% |
| PGD-AT | 45.70% | 32.60% | 11.95% |
| Mixed-AT | 46.59% | 29.35% | 7.05% |

**Decisions made:**
- Include both FGSM and PGD sweeps in the final report.
- Use the sweeps to show robustness sensitivity as the perturbation budget increases.
- Treat PGD as the stronger robustness evaluation attack.

**Issues:**
- Needed to verify that all figures and tables matched the saved JSON metrics.
- Regenerated figures from saved outputs to avoid manual mismatch.

---

### Session 13
**Date:** 2026-06-09

**Work done:**
- Ran a small PGD-AT ablation study on CIFAR-10.
- Tested:
  - PGD training steps
  - Number of training epochs
- Added the ablation results to the appendix.

**PGD-AT ablation results:**

| Setting | Epochs | PGD Steps | Clean / Robust Accuracy |
|---|---:|---:|---:|
| Steps = 3 | 15 | 3 | 66.46% / 25.84% |
| Steps = 7 | 15 | 7 | 57.46% / 32.27% |
| Epochs = 10 | 10 | 7 | 54.87% / 32.43% |
| Epochs = 15 | 15 | 7 | 57.46% / 32.27% |

**Decisions made:**
- Include the ablation as appendix material.
- Use the ablation to support the robustness–accuracy trade-off discussion.
- Leave larger ablations such as ResNet, WideResNet, dropout removal, augmentation changes, and multiple random seeds for future work.

**Issues:**
- Increasing PGD training steps improved robust accuracy but reduced clean accuracy.
- This supported the final conclusion that adversarial training improves robustness but introduces a clean–robust trade-off.

---

### Session 14
**Date:** 2026-06-10

**Work done:**
- Finalized the IEEE-style final report.
- Added:
  - Final CIFAR-10 results
  - FGSM and PGD sweeps
  - Training curves
  - Qualitative figures
  - PGD-AT ablation
  - Limitations and future work
  - Reproducibility section
  - Team contributions
  - Dataset links for MNIST and CIFAR-10
  - Use of GenAI tools statement
- Updated the GitHub README into a report-style README.
- Cleaned the GitHub repository structure.
- Ensured large generated artifacts such as datasets and checkpoints are excluded from version control.
- Prepared and revised the final PowerPoint presentation.

**Final deliverables:**
- IEEE-style final report PDF.
- Final PowerPoint presentation.
- GitHub repository with code, README, and reproducibility instructions.
- Saved JSON metrics and generated figures.

**Use of GenAI tools:**
- OpenAI ChatGPT (GPT-5.5) was used as an assistance tool for:
  - Language polishing
  - LaTeX organization
  - Presentation clarity
  - Minor code modification suggestions
  - Debugging suggestions
  - Code-structure refinements
- All suggested code changes were reviewed, executed, and validated by the team.
- All technical decisions, code execution, experimental results, numerical values, and final analysis remain the responsibility of the authors.

**Decisions made:**
- The final report emphasizes that clean accuracy alone is insufficient for evaluating image classifiers.
- The final conclusion highlights:
  - FGSM-AT and PGD-AT provide the strongest PGD robustness.
  - Mixed-AT provides the best clean accuracy among defended models.
  - Adversarial training introduces a clear robustness–accuracy trade-off.
- The final presentation follows the same structure as the report:
  1. Motivation and problem
  2. Data and methodology
  3. Results and analysis
  4. Limitations and future work

**Issues solved:**
- Fixed LaTeX table and figure placement issues.
- Fixed broken references such as `Table ??`.
- Improved the reproduction scripts table.
- Verified consistency between:
  - Report tables
  - Report figures
  - JSON outputs
  - PowerPoint numbers
  - README results
- Confirmed that the final report, presentation, GitHub README, and development log are aligned.

---

## Final Project Summary

This project demonstrates that lightweight CNN image classifiers can achieve strong clean accuracy while remaining highly vulnerable to adversarial perturbations.

On CIFAR-10, the clean baseline achieved 78.22% clean accuracy but collapsed to 2.42% under FGSM and 0.00% under PGD at ε = 8/255. Adversarial training substantially improved robustness. FGSM-AT achieved 36.87% FGSM accuracy and 32.62% PGD accuracy, while PGD-AT achieved 35.89% FGSM accuracy and 32.60% PGD accuracy at the same perturbation budget. Mixed-AT achieved the best clean accuracy among defended models at 65.20% while maintaining competitive robustness.

The final conclusion is that clean accuracy alone is not sufficient for evaluating image classifiers. Robustness must be measured under adversarial conditions, and adversarial training introduces a clear accuracy–robustness trade-off.

---

## Final Repository Scripts

### Training

```text
scripts/training/train_MNIST_clean.py
scripts/training/training cnn on clear cifar10.py
scripts/training/train_adversarial_MNIST.py
scripts/training/train_adversarial_cifar.py
scripts/training/mixed_CIFAR.py
```

### Evaluation

```text
scripts/evaluation/evaluate_fgsm.py
scripts/evaluation/evaluate_pgd.py
scripts/evaluation/evaluate_all_cifar.py
scripts/evaluation/ablation_study.py
scripts/evaluation/ablation_study_cifar.py
```

### Visualization

```text
scripts/visualization/plot_results_cifar.py
```

---

## Final Notes

- Generated datasets and model checkpoints should not be committed to GitHub.
- Final reported results are based on saved JSON metrics and generated figures.
- The report, presentation, README, and development log should all use the same final result values.
- The final GitHub repository should include clear setup, training, evaluation, and visualization instructions.
