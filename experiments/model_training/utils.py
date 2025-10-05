import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

global DEVICE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

global ACTIVATION_CLASSES

ACTIVATION_CLASSES = (
    nn.ReLU,
    nn.Sigmoid,
    nn.Tanh,
    nn.Hardtanh,
    nn.LeakyReLU,
    nn.ELU,
    nn.SELU,
    nn.GELU,
    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.MaxPool3d,
    nn.AvgPool1d,
    nn.AvgPool2d,
    nn.AvgPool3d,
)

# combos = [
"""
    # === MNIST fully connected ===
    {"dataset": "mnist", "arch": "fcn3x50", "defense": "standard"},
    {"dataset": "mnist", "arch": "fcn3x100", "defense": "standard"},
    {"dataset": "mnist", "arch": "fcn5x100", "defense": "diffai", "eps": 0.3},
    {"dataset": "mnist", "arch": "fcn6x100", "defense": "standard"},
    {"dataset": "mnist", "arch": "fcn9x100", "defense": "standard"},
    {"dataset": "mnist", "arch": "fcn6x200", "defense": "standard"},
    {"dataset": "mnist", "arch": "fcn9x200", "defense": "standard"},
    {"dataset": "mnist", "arch": "fcn6x500", "defense": "standard"},
    {"dataset": "mnist", "arch": "fcn6x500", "defense": "pgd", "eps": 0.1},
    {"dataset": "mnist", "arch": "fcn6x500", "defense": "pgd", "eps": 0.3},
    {"dataset": "mnist", "arch": "fcn4x1024", "defense": "standard"},
    # === MNIST conv family ===
    {"dataset": "mnist", "arch": "convsmall", "defense": "standard"},
    {"dataset": "mnist", "arch": "convsmall", "defense": "pgd", "eps": 0.3},
    {"dataset": "mnist", "arch": "convsmall", "defense": "diffai", "eps": 0.3},
    {"dataset": "mnist", "arch": "convmed", "defense": "standard"},
    {"dataset": "mnist", "arch": "convmed", "defense": "pgd", "eps": 0.1},
    {"dataset": "mnist", "arch": "convmed", "defense": "pgd", "eps": 0.3},
    {"dataset": "mnist", "arch": "convmed", "defense": "diffai", "eps": 0.3},
    {
        "dataset": "mnist",
        "arch": "convmed",
        "defense": "standard",
        "use_maxpool": True,
    },  # ConvMaxpool
    # === CIFAR10 fully connected ===
    {"dataset": "cifar10", "arch": "fcn4x100", "defense": "standard"},
    {"dataset": "cifar10", "arch": "fcn6x100", "defense": "standard"},
    {"dataset": "cifar10", "arch": "fcn9x200", "defense": "standard"},
    {"dataset": "cifar10", "arch": "fcn6x500", "defense": "standard"},
    {
        "dataset": "cifar10",
        "arch": "fcn6x500",
        "defense": "pgd",
        "eps": 0.0078,
    },  # ~2/255
    {
        "dataset": "cifar10",
        "arch": "fcn6x500",
        "defense": "pgd",
        "eps": 0.0313,
    },  # 8/255
    {"dataset": "cifar10", "arch": "fcn7x1024", "defense": "standard"},

    # === CIFAR10 conv family ===
    {"dataset": "cifar10", "arch": "convsmall", "defense": "standard"},
    {"dataset": "cifar10", "arch": "convsmall", "defense": "pgd", "eps": 0.0313},
    {"dataset": "cifar10", "arch": "convsmall", "defense": "diffai", "eps": 0.0313},
    """
combos = [
    {"dataset": "cifar10", "arch": "convmed", "defense": "standard"},
    {"dataset": "cifar10", "arch": "convmed", "defense": "pgd", "eps": 0.0078},
    {"dataset": "cifar10", "arch": "convmed", "defense": "pgd", "eps": 0.0313},
    {"dataset": "cifar10", "arch": "convmed", "defense": "diffai", "eps": 0.0313},
    {
        "dataset": "cifar10",
        "arch": "convmed",
        "defense": "standard",
        "use_maxpool": True,
    },  # ConvMaxpool
]


def count_units_and_activations(model: nn.Module):

    num_units = sum(p.numel() for p in model.parameters())

    num_activations = 0
    for m in model.modules():
        if isinstance(m, ACTIVATION_CLASSES):
            num_activations += 1

    return num_units, num_activations


def num_params(model):
    return sum(p.numel() for p in model.parameters())


def save_onnx(model, path, in_ch, img_size):
    model.eval()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dummy = torch.randn(1, in_ch, img_size, img_size, device=DEVICE)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
    )
    print(f"[ONNX] Saved to {path}")


@dataclass
class TrainConfig:
    dataset: str
    arch: str
    defense: str
    epochs: int
    batch_size: int
    lr: float
    eps: float
    pgd_steps: int
    pgd_step_factor: float
    mix_alpha: float
    use_maxpool: bool
    save_dir: str


def get_loaders(dataset, batch_size):
    if dataset.lower() == "mnist":
        transform = transforms.Compose([transforms.ToTensor()])
        train_set = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_set = datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )
        in_ch, img = 1, 28
        num_classes = 10
    elif dataset.lower() == "cifar10":
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        transform_test = transforms.Compose([transforms.ToTensor()])
        train_set = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
        test_set = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )
        in_ch, img = 3, 32
        num_classes = 10
    else:
        raise ValueError("dataset must be mnist or cifar10")

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    return train_loader, test_loader, in_ch, img, num_classes


def pgd_linf(model, x, y, eps, steps, step_size, rand_start=True):
    model.eval()
    x_adv = x.detach()
    if rand_start:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    x_adv.requires_grad_(True)
    ce = nn.CrossEntropyLoss()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logits = model(x_adv)
        loss = ce(logits, y)
        loss.backward()
        grad = x_adv.grad.detach()
        with torch.no_grad():
            x_adv = x_adv + step_size * torch.sign(grad)
            x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        x_adv.requires_grad_(True)

    return x_adv.detach()


def eval_clean_and_pgd(model, loader, eps, steps, step_size):
    model.eval()
    ce = nn.CrossEntropyLoss()
    tot, clean_correct, rob_correct = 0, 0, 0
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        with torch.no_grad():
            logits = model(x)
            total_loss += ce(logits, y).item() * x.size(0)
            clean_correct += (logits.argmax(1) == y).sum().item()

        if eps > 0.0:
            with torch.enable_grad():
                x_adv = pgd_linf(
                    model,
                    x,
                    y,
                    eps=eps,
                    steps=steps,
                    step_size=step_size,
                    rand_start=True,
                )
            with torch.no_grad():
                logits_adv = model(x_adv)
                rob_correct += (logits_adv.argmax(1) == y).sum().item()

        tot += x.size(0)

    clean_acc = clean_correct / tot
    rob_acc = rob_correct / tot if eps > 0 else 0.0
    avg_loss = total_loss / tot
    return clean_acc, rob_acc, avg_loss


def train_one_epoch(model, loader, optimizer, defense, cfg: TrainConfig, epoch):
    model.train()
    ce = nn.CrossEntropyLoss()
    total, correct, total_loss = 0, 0, 0.0

    # For DiffAI-style curriculum: grow eps over time
    if defense == "diffai":
        # cosine or linear ramp of epsilon
        t = epoch / max(1, cfg.epochs - 1)
        eps_eff = cfg.eps * t  # linear ramp
        step_size = max(1e-6, eps_eff * cfg.pgd_step_factor)
        adv_ratio = cfg.mix_alpha
    elif defense == "pgd":
        eps_eff = cfg.eps
        step_size = max(1e-6, cfg.eps * cfg.pgd_step_factor)
        adv_ratio = 1.0
    else:
        eps_eff = 0.0
        step_size = 0.0
        adv_ratio = 0.0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        if adv_ratio > 0.0 and eps_eff > 0.0:
            # mixed batch (DiffAI) or full adv (PGD)
            n = x.size(0)
            adv_n = int(round(adv_ratio * n))
            if adv_n == 0 and defense == "pgd":
                adv_n = n
            if adv_n > 0:
                x_adv_part = pgd_linf(
                    model,
                    x[:adv_n],
                    y[:adv_n],
                    eps=eps_eff,
                    steps=cfg.pgd_steps,
                    step_size=step_size,
                    rand_start=True,
                )
                x_mix = (
                    torch.cat([x_adv_part, x[adv_n:]], dim=0)
                    if adv_n < n
                    else x_adv_part
                )
                y_mix = y
            else:
                x_mix, y_mix = x, y
        else:
            x_mix, y_mix = x, y

        optimizer.zero_grad(set_to_none=True)
        logits = model(x_mix)
        loss = ce(logits, y_mix)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            correct += (pred == y_mix).sum().item()
            total += y_mix.size(0)
            total_loss += loss.item() * y_mix.size(0)

    return correct / total, total_loss / total
