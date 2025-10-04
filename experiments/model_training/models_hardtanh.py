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


def hardtanh():
    return nn.Hardtanh(min_val=-1.0, max_val=1.0, inplace=False)


class FCNHardtanh(nn.Module):
    """
    Fully-connected: depth x width (e.g., 6 x 500) with Hardtanh.
    """

    def __init__(self, in_dim, depth, width, num_classes=10):
        super().__init__()
        layers = [nn.Flatten(), nn.Linear(in_dim, width), hardtanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), hardtanh()]
        layers += [nn.Linear(width, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ConvSmallHardtanh(nn.Module):
    """
    ConvSmall: roughly aligned with common baselines
    Conv -> HT -> Conv -> HT -> FC -> HT -> FC
    """

    def __init__(self, in_ch, num_classes=10, use_maxpool=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, 32, 3, padding=1),
            hardtanh(),
            nn.Conv2d(32, 64, 3, padding=1),
            hardtanh(),
        ]
        if use_maxpool:
            layers += [nn.MaxPool2d(2)]
        self.feat = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                64 * (14 if use_maxpool else 28) * (14 if use_maxpool else 28), 256
            ),
            hardtanh(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.feat(x)
        return self.classifier(x)


class ConvMedHardtanh(nn.Module):
    """
    ConvMed: 3 conv blocks + 1 or 2 FC as needed.
    """

    def __init__(self, in_ch, num_classes=10, use_maxpool=False):
        super().__init__()
        blocks = []
        ch = [in_ch, 64, 128, 128]
        for i in range(3):
            blocks += [nn.Conv2d(ch[i], ch[i + 1], 3, padding=1), hardtanh()]
            if use_maxpool and i < 2:
                blocks += [nn.MaxPool2d(2)]
        self.feat = nn.Sequential(*blocks)
        side = 32 if in_ch == 3 else 28
        # estimate spatial size after optional pooling on first two blocks
        shrink = 4 if use_maxpool else 1
        side //= shrink
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch[-1] * side * side, 512),
            hardtanh(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.feat(x))


def build_model(arch, in_ch, img, num_classes, use_maxpool):
    if arch.startswith("fcn"):
        # fcn{depth}x{width}, e.g., fcn6x500
        # default: fcn6x500
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
        return FCNHardtanh(in_dim, depth, width, num_classes)
    elif arch == "convsmall":
        return ConvSmallHardtanh(in_ch, num_classes, use_maxpool=use_maxpool)
    elif arch == "convmed":
        return ConvMedHardtanh(in_ch, num_classes, use_maxpool=use_maxpool)
    else:
        raise ValueError("Unknown arch. Try: fcn6x500, fcn9x200, convsmall, convmed")


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


def main():
    parser = argparse.ArgumentParser(
        description="Hardtanh Networks: Standard/PGD/DiffAI Training"
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
    # parser.add_argument("--export_onnx", action="store_true")
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
        # export_onnx=args.export_onnx
    )

    # Data
    train_loader, test_loader, in_ch, img, num_classes = get_loaders(
        cfg.dataset, cfg.batch_size
    )

    # Model
    model = build_model(cfg.arch, in_ch, img, num_classes, cfg.use_maxpool).to(DEVICE)
    units, acts = count_units_and_activations(model)
    # print(f"[Model] {cfg.arch} params={num_params(model):,} | #units={units:,} | #activation layers={acts}")
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
        # print(f"[Epoch {epoch+1:03d}/{cfg.epochs}] "
        #      f"train_acc={train_acc:.4f} train_loss={train_loss:.4f} | "
        #      f"val_clean={clean_acc:.4f} val_rob={rob_acc:.4f} val_loss={val_loss:.4f}")
        logging.info(
            f"[Epoch {epoch+1:03d}/{cfg.epochs}] "
            f"train_acc={train_acc:.4f} train_loss={train_loss:.4f} | "
            f"val_clean={clean_acc:.4f} val_rob={rob_acc:.4f} val_loss={val_loss:.4f}"
        )

    # Save checkpoint
    # os.makedirs(cfg.save_dir, exist_ok=True)
    # tag = f"{cfg.dataset}_{cfg.arch}_hardtanh_{cfg.defense}_eps{cfg.eps}".replace("/", "-")
    # ckpt_path = os.path.join(cfg.save_dir, f"{tag}.pth")
    # torch.save(model.state_dict(), ckpt_path)
    # print(f"[CKPT] Saved to {ckpt_path}")

    os.makedirs(cfg.save_dir, exist_ok=True)
    tag = f"{cfg.dataset}_{cfg.arch}_hardtanh_{cfg.defense}_eps{cfg.eps}".replace(
        "/", "-"
    )
    onnx_path = os.path.join(cfg.save_dir, f"{tag}.onnx")
    save_onnx(model, onnx_path, in_ch, img)


def run_all():

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
            save_dir="./checkpoints",
        )

        # print("="*80)
        # print(f"🚀 Training start: dataset={cfg.dataset}, arch={cfg.arch}, "
        #     f"defense={cfg.defense}, eps={cfg.eps}, use_maxpool={cfg.use_maxpool}")
        # print("="*80)

        logging.info("=" * 80)
        logging.info(
            f"🚀 Training start: activation=Hardtanh, dataset={cfg.dataset}, arch={cfg.arch}, "
            f"defense={cfg.defense}, eps={cfg.eps}, use_maxpool={cfg.use_maxpool}"
        )
        logging.info("=" * 80)

        # Data
        train_loader, test_loader, in_ch, img, num_classes = get_loaders(
            cfg.dataset, cfg.batch_size
        )

        # Model
        model = build_model(cfg.arch, in_ch, img, num_classes, cfg.use_maxpool).to(
            DEVICE
        )
        units, acts = count_units_and_activations(model)
        # print(f"[Model] {cfg.arch} params={num_params(model):,} | #units={units:,} | #activation layers={acts}")
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
            # print(f"[Epoch {epoch+1:03d}/{cfg.epochs}] "
            #    f"train_acc={train_acc:.4f} train_loss={train_loss:.4f} | "
            #    f"val_clean={clean_acc:.4f} val_rob={rob_acc:.4f} val_loss={val_loss:.4f}")
            logging.info(
                f"[Epoch {epoch+1:03d}/{cfg.epochs}] "
                f"train_acc={train_acc:.4f} train_loss={train_loss:.4f} | "
                f"val_clean={clean_acc:.4f} val_rob={rob_acc:.4f} val_loss={val_loss:.4f}"
            )

        os.makedirs(cfg.save_dir, exist_ok=True)
        tag = f"{cfg.dataset}_{cfg.arch}_hardtanh_{cfg.defense}_eps{cfg.eps}".replace(
            "/", "-"
        )
        onnx_path = os.path.join(cfg.save_dir, f"{tag}.onnx")
        save_onnx(model, onnx_path, in_ch, img)


if __name__ == "__main__":
    # main()
    run_all()
