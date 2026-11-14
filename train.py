"""
Training 5-fold MSLD v2.0.

Cara pakai:
    python train.py --model ConvNeXtTiny            # semua fold
    python train.py --model ConvNeXtTiny --folds 1  # coba 1 fold dulu
    python train.py --model SimpleCNN --epochs 100

Hasil:
    checkpoints/<model>/fold{k}.pth      bobot terbaik (jangan di-push)
    Results/<model>/fold{k}_history.csv
    Results/<model>/fold{k}_curve.png
    Results/<model>/train_summary.csv

Nama class di Models/: SimpleCNN (CNN.py), ModifiedCNN (ModifiedCNN.py),
ConvNeXtTiny (ConvNeXt.py), semua pakai argumen num_classes.
"""

import argparse
import importlib
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from Utils.getData import BASE_DIR, CLASSES, get_dataloaders

MODELS = {
    "SimpleCNN":    ("Models.CNN", "SimpleCNN"),
    "ModifiedCNN":  ("Models.ModifiedCNN", "ModifiedCNN"),
    "ConvNeXtTiny": ("Models.ConvNeXt", "ConvNeXtTiny"),
}

# default tiap model, bisa diganti lewat argumen
DEFAULTS = {
    "SimpleCNN":    {"epochs": 30, "lr": 1e-3, "optimizer": "adam",  "weight_decay": 0.0},
    "ModifiedCNN":  {"epochs": 30, "lr": 1e-3, "optimizer": "adam",  "weight_decay": 0.0},
    "ConvNeXtTiny": {"epochs": 15, "lr": 1e-4, "optimizer": "adamw", "weight_decay": 0.05},
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_model(name, num_classes=len(CLASSES)):
    module_name, class_name = MODELS[name]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)(num_classes=num_classes)


def make_optimizer(model, cfg):
    if cfg["optimizer"] == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                 weight_decay=cfg["weight_decay"])
    return torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])


def run_epoch(model, loader, criterion, device, optimizer=None):
    # ada optimizer = mode training, kalau None = evaluasi aja
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * y.size(0)
            correct += (out.argmax(dim=1) == y).sum().item()
            n += y.size(0)

    return total_loss / n, correct / n


def plot_history(hist, title, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(hist["epoch"], hist["train_loss"], label="train")
    ax1.plot(hist["epoch"], hist["val_loss"], label="valid")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_title("Loss")
    ax1.legend()

    ax2.plot(hist["epoch"], hist["train_acc"], label="train")
    ax2.plot(hist["epoch"], hist["val_acc"], label="valid")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train_fold(args, cfg, fold, device):
    set_seed(args.seed + fold)
    train_loader, valid_loader, _, _ = get_dataloaders(
        fold, args.model, batch_size=args.batch_size,
        num_workers=args.num_workers, root=args.root)

    model = get_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, cfg)

    ckpt_dir = os.path.join(args.ckpt_dir, args.model)
    res_dir = os.path.join(args.results_dir, args.model)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"fold{fold}.pth")

    history, best_acc, best_epoch = [], -1.0, 0
    t0 = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        te = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, valid_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc,
                        "seconds": time.time() - te})

        # save kalau val acc lebih bagus dari sebelumnya
        if va_acc > best_acc:
            best_acc, best_epoch = va_acc, epoch
            torch.save(model.state_dict(), ckpt_path)

        print(f"[{args.model} fold{fold}] epoch {epoch:3d}/{cfg['epochs']} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"valid loss {va_loss:.4f} acc {va_acc:.4f} | {time.time() - te:.0f}s")

    hist = pd.DataFrame(history)
    hist.to_csv(os.path.join(res_dir, f"fold{fold}_history.csv"), index=False)
    plot_history(hist, f"{args.model} - fold {fold}",
                 os.path.join(res_dir, f"fold{fold}_curve.png"))

    minutes = (time.time() - t0) / 60
    print(f"[{args.model} fold{fold}] selesai {minutes:.1f} menit | "
          f"valid acc terbaik {best_acc:.4f} (epoch {best_epoch})")
    return {"fold": fold, "best_epoch": best_epoch, "best_val_acc": best_acc,
            "train_minutes": minutes}


def main():
    p = argparse.ArgumentParser(description="Training 5-fold MSLD v2.0")
    p.add_argument("--model", required=True, choices=MODELS.keys())
    p.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--root", default=BASE_DIR)
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--results_dir", default="Results")
    args = p.parse_args()

    cfg = dict(DEFAULTS[args.model])
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.lr is not None:
        cfg["lr"] = args.lr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | model: {args.model} | folds: {args.folds} | config: {cfg}")

    summary = [train_fold(args, cfg, k, device) for k in args.folds]

    res_dir = os.path.join(args.results_dir, args.model)
    pd.DataFrame(summary).to_csv(os.path.join(res_dir, "train_summary.csv"), index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
