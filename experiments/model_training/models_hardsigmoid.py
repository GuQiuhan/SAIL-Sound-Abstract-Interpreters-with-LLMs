import argparse
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils import *

date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f"training_{date_str}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
)


def hardsigmoid():
    return nn.Hardsigmoid()


class FCNHardSigmoid(nn.Module):
    """Fully-connected depth x width with HardSigmoid."""

    def __init__(self, in_dim, depth, width, num_classes=10):
        super().__init__()
        layers = [nn.Flatten(), nn.Linear(in_dim, width), hardsigmoid()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), hardsigmoid()]
        layers += [nn.Linear(width, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ConvSmallHardSigmoid(nn.Module):
    """ConvSmall: Conv -> HSig -> Conv -> HSig -> FC -> HSig -> FC"""

    def __init__(self, in_ch, num_classes=10, use_maxpool=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, 32, 3, padding=1),
            hardsigmoid(),
            nn.Conv2d(32, 64, 3, padding=1),
            hardsigmoid(),
        ]
        if use_maxpool:
            layers += [nn.MaxPool2d(2)]
        self.feat = nn.Sequential(*layers)

        side = 32 if in_ch == 3 else 28
        if use_maxpool:
            side //= 2  # 只有一层 MaxPool2d(2)
        in_features = 64 * side * side

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 256),
            hardsigmoid(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.feat(x)
        return self.classifier(x)


class ConvMedHardSigmoid(nn.Module):
    """ConvMed: 3 conv blocks + FC with HardSigmoid."""

    def __init__(self, in_ch, num_classes=10, use_maxpool=False):
        super().__init__()
        blocks = []
        ch = [in_ch, 64, 128, 128]
        for i in range(3):
            blocks += [nn.Conv2d(ch[i], ch[i + 1], 3, padding=1), hardsigmoid()]
            if use_maxpool and i < 2:
                blocks += [nn.MaxPool2d(2)]
        self.feat = nn.Sequential(*blocks)
        side = 32 if in_ch == 3 else 28
        shrink = 4 if use_maxpool else 1
        side //= shrink
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch[-1] * side * side, 512),
            hardsigmoid(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.feat(x))


def build_model(arch, in_ch, img, num_classes, use_maxpool):
    if arch.startswith("fcn"):
        depth, width = 6, 500
        try:
            spec = arch.replace("fcn", "")
            parts = spec.split("x")
            if len(parts) == 2:
                depth = int(parts[0])
                width = int(parts[1])
        except:
            pass
        in_dim = in_ch * img * img
        return FCNHardSigmoid(in_dim, depth, width, num_classes)
    elif arch == "convsmall":
        return ConvSmallHardSigmoid(in_ch, num_classes, use_maxpool=use_maxpool)
    elif arch == "convmed":
        return ConvMedHardSigmoid(in_ch, num_classes, use_maxpool=use_maxpool)
    else:
        raise ValueError("Unknown arch. Try: fcn6x500, fcn9x200, convsmall, convmed")


def main():
    parser = argparse.ArgumentParser(
        description="HardSigmoid Networks: Standard/PGD/DiffAI Training"
    )
    parser.add_argument(
        "--dataset", type=str, default="mnist", choices=["mnist", "cifar10"]
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="fcn6x500",
        help="fcn{depth}x{width} | convsmall | convmed",
    )
    parser.add_argument(
        "--defense", type=str, default="standard", choices=["standard", "pgd", "diffai"]
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--eps",
        type=float,
        default=0.3,
        help="Linf epsilon (MNIST: 0.1/0.3, CIFAR: 8/255≈0.0313)",
    )
    parser.add_argument("--pgd_steps", type=int, default=10)
    parser.add_argument(
        "--pgd_step_factor",
        type=float,
        default=0.25,
        help="step_size=eps*factor (≈ eps/4)",
    )
    parser.add_argument(
        "--mix_alpha", type=float, default=0.5, help="adv ratio in diffai (0~1)"
    )
    parser.add_argument("--use_maxpool", action="store_true")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    cfg = TrainConfig(
        dataset=args.dataset,
        arch=args.arch,
        defense=args.defense,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        eps=args.eps,
        pgd_steps=args.pgd_steps,
        pgd_step_factor=args.pgd_step_factor,
        mix_alpha=args.mix_alpha,
        use_maxpool=args.use_maxpool,
        save_dir=args.save_dir,
    )

    # Data
    train_loader, test_loader, in_ch, img, num_classes = get_loaders(
        cfg.dataset, cfg.batch_size
    )

    # Model
    model = build_model(cfg.arch, in_ch, img, num_classes, cfg.use_maxpool).to(DEVICE)
    units, acts = count_units_and_activations(model)
    logging.info(
        f"[Model] {cfg.arch} params={num_params(model):,} | #units={units:,} | #activation layers={acts}"
    )

    # Opt
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)

    # Train
    for epoch in range(cfg.epochs):
        train_acc, train_loss = train_one_epoch(
            model, train_loader, optimizer, cfg.defense, cfg, epoch
        )
        step_size = max(1e-6, cfg.eps * cfg.pgd_step_factor)
        clean_acc, rob_acc, val_loss = eval_clean_and_pgd(
            model,
            test_loader,
            eps=(cfg.eps if cfg.defense in ["pgd", "diffai"] else 0.0),
            steps=cfg.pgd_steps,
            step_size=step_size,
        )
        logging.info(
            f"[Epoch {epoch+1:03d}/{cfg.epochs}] "
            f"train_acc={train_acc:.4f} train_loss={train_loss:.4f} | "
            f"val_clean={clean_acc:.4f} val_rob={rob_acc:.4f} val_loss={val_loss:.4f}"
        )

    os.makedirs(cfg.save_dir, exist_ok=True)
    tag = f"{cfg.dataset}_{cfg.arch}_hardsigmoid_{cfg.defense}_eps{cfg.eps}".replace(
        "/", "-"
    )
    onnx_path = os.path.join(cfg.save_dir, f"{tag}.onnx")

    save_onnx(model, onnx_path, in_ch, img)


def run_all(combos):
    for cfg_dict in combos:
        cfg = TrainConfig(
            dataset=cfg_dict.get("dataset"),
            arch=cfg_dict.get("arch"),
            defense=cfg_dict.get("defense"),
            epochs=20,
            batch_size=128,
            lr=1e-3,
            eps=cfg_dict.get("eps", 0.3),
            pgd_steps=10,
            pgd_step_factor=0.25,
            mix_alpha=0.5,
            use_maxpool=cfg_dict.get("use_maxpool", False),
            save_dir="./checkpoints/hardsigmoid/",
        )

        logging.info("=" * 80)
        logging.info(
            f"🚀 Training start: activation=HardSigmoid, dataset={cfg.dataset}, arch={cfg.arch}, "
            f"defense={cfg.defense}, eps={cfg.eps}, use_maxpool={cfg.use_maxpool}"
        )
        logging.info("=" * 80)

        train_loader, test_loader, in_ch, img, num_classes = get_loaders(
            cfg.dataset, cfg.batch_size
        )

        model = build_model(cfg.arch, in_ch, img, num_classes, cfg.use_maxpool).to(
            DEVICE
        )
        units, acts = count_units_and_activations(model)
        logging.info(
            f"[Model] {cfg.arch} params={num_params(model):,} | #units={units:,} | #activation layers={acts}"
        )

        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)

        for epoch in range(cfg.epochs):
            train_acc, train_loss = train_one_epoch(
                model, train_loader, optimizer, cfg.defense, cfg, epoch
            )
            step_size = max(1e-6, cfg.eps * cfg.pgd_step_factor)
            clean_acc, rob_acc, val_loss = eval_clean_and_pgd(
                model,
                test_loader,
                eps=(cfg.eps if cfg.defense in ["pgd", "diffai"] else 0.0),
                steps=cfg.pgd_steps,
                step_size=step_size,
            )
            logging.info(
                f"[Epoch {epoch+1:03d}/{cfg.epochs}] "
                f"train_acc={train_acc:.4f} train_loss={train_loss:.4f} | "
                f"val_clean={clean_acc:.4f} val_rob={rob_acc:.4f} val_loss={val_loss:.4f}"
            )

        os.makedirs(cfg.save_dir, exist_ok=True)
        tag = (
            f"{cfg.dataset}_{cfg.arch}_hardsigmoid_{cfg.defense}_eps{cfg.eps}".replace(
                "/", "-"
            )
        )
        onnx_path = os.path.join(cfg.save_dir, f"{tag}.onnx")
        save_onnx(model, onnx_path, in_ch, img)


if __name__ == "__main__":

    run_all(combos)
