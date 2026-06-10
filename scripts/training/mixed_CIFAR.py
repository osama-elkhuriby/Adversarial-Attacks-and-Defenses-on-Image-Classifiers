import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from models.cifar_cnn import CifarCNN
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack
from Evaluationfunction import evaluate



SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# H.P:
BATCH_SIZE = 64
LR = 0.001
EPOCHS = 20

DATA_DIR = "data"      # Root directory for MNIST data
CHECKPOINT_DIR = "results/CHECKPOINT_new"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "Mixed_CIFAR.pth")
METRICS_DIR = "results/METRICS_DIR_new"
METRICS_PATH = os.path.join(METRICS_DIR, "Mixed_CIFAR_metrics.json")

# Adversarial settings
EPS = 8 / 255
FGSM_RATIO = 0.33
PGD_RATIO = 0.33
CLEAN_RATIO = 0.34

PGD_ALPHA = 2 / 255
PGD_ITERS = 7

TRAIN_SIZE = 45000
VAL_SIZE = 5000



# Mixed Adversarial Training
def train_mixed():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

    test_transform = transforms.ToTensor()

    full_train = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=test_transform)
    test_set = datasets.CIFAR10(root=DATA_DIR, train=False,download=True, transform=test_transform)

    train_set, val_set = random_split(
        full_train, [TRAIN_SIZE, VAL_SIZE],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    model = CifarCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    metrics = {"epochs": []}

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):

            images, labels = images.to(device), labels.to(device)

         
            # Split batch into 3 parts
           
            bsz = images.size(0)
            n_clean = int(bsz * CLEAN_RATIO)
            n_fgsm = int(bsz * FGSM_RATIO)
            n_pgd = bsz - n_clean - n_fgsm

            clean_x = images[:n_clean]
            clean_y = labels[:n_clean]

            fgsm_x = images[n_clean:n_clean+n_fgsm]
            fgsm_y = labels[n_clean:n_clean+n_fgsm]

            pgd_x = images[n_clean+n_fgsm:]
            pgd_y = labels[n_clean+n_fgsm:]

            model.eval()

            adv_fgsm = fgsm_attack(model, fgsm_x, fgsm_y, EPS)
            adv_pgd = pgd_attack(model, pgd_x, pgd_y,
                                 epsilon=EPS,
                                 alpha=PGD_ALPHA,
                                 iters=PGD_ITERS)

            model.train()

            
            # Combine all data
            final_x = torch.cat([clean_x, adv_fgsm, adv_pgd], dim=0)
            final_y = torch.cat([clean_y, fgsm_y, pgd_y], dim=0)
            #shuffeling
            perm = torch.randperm(final_x.size(0), device=final_x.device)
            final_x = final_x[perm]
            final_y = final_y[perm]
            optimizer.zero_grad()
            outputs = model(final_x)
            loss = criterion(outputs, final_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        val_acc = evaluate(model, val_loader, device, label="Val")
        test_acc = evaluate(model, test_loader, device, label="Test")

        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, val={val_acc:.2f}, test={test_acc:.2f}")

        
        metrics["epochs"].append({
            "epoch": epoch + 1,
            "avg_loss": avg_loss,
            "val_accuracy": val_acc,
            "test_accuracy": test_acc
        })

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"\nModel checkpoint saved to: {CHECKPOINT_PATH}")
    # Save Metrics 
    os.makedirs(METRICS_DIR, exist_ok=True)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Training metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    train_mixed()