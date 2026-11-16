"""
Evaluasi model dari train.py di folder Test tiap fold, terus dirata-rata.

Cara pakai:
    python test.py --model ConvNeXtTiny
    python test.py --model ConvNeXtTiny --folds 1

Metrik: accuracy, precision/recall/F1 (macro), ROC-AUC (macro OvR), confusion matrix.

File yang disimpan di Results/<model>/:
    fold{k}_confusion_matrix.png, fold{k}_roc.png, fold{k}_report.txt
    confusion_matrix_total.png   gabungan semua fold
    metrics_per_fold.csv, metrics_summary.csv (mean & std), f1_per_class.csv
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)

from train import MODELS, get_model, get_preprocess
from Utils.getData import BASE_DIR, CLASSES, get_dataloaders


@torch.no_grad()
def predict(model, loader, device):
    # balikin y_true, y_pred, y_prob (N x 6)
    model.eval()
    ys, probs = [], []
    for x, y in loader:
        out = model(x.to(device))
        # SimpleCNN (M9) udah ada softmax di akhir, yang lain masih logits
        already_prob = (out.min() >= 0) and torch.allclose(
            out.sum(dim=1), torch.ones(out.size(0), device=out.device), atol=1e-3)
        p = out if already_prob else F.softmax(out, dim=1)
        ys.append(y.numpy())
        probs.append(p.cpu().numpy())
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(probs)
    return y_true, y_prob.argmax(axis=1), y_prob


def plot_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(
        ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(y_true, y_prob, title, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, cls in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_prob[:, i])
        ax.plot(fpr, tpr, label=f"{cls} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate_fold(args, fold, device, res_dir):
    _, _, test_loader, _ = get_dataloaders(
        fold, args.preprocess, batch_size=args.batch_size,
        num_workers=args.num_workers, root=args.root)

    model = get_model(args.model).to(device)
    ckpt = os.path.join(args.ckpt_dir, args.model, f"fold{fold}.pth")
    model.load_state_dict(torch.load(ckpt, map_location=device))

    y_true, y_pred, y_prob = predict(model, test_loader, device)
    labels = list(range(len(CLASSES)))

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    metrics = {
        "fold": fold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc_score(y_true, y_prob, multi_class="ovr",
                                 average="macro", labels=labels),
    }

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plot_confusion(cm, f"{args.model} - fold {fold}",
                   os.path.join(res_dir, f"fold{fold}_confusion_matrix.png"))
    plot_roc(y_true, y_prob, f"{args.model} - ROC fold {fold}",
             os.path.join(res_dir, f"fold{fold}_roc.png"))

    report = classification_report(y_true, y_pred, labels=labels,
                                   target_names=CLASSES, digits=4, zero_division=0)
    with open(os.path.join(res_dir, f"fold{fold}_report.txt"), "w") as f:
        f.write(report)

    # F1 per kelas
    _, _, f1_cls, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1_row = {"fold": fold, **{c: v for c, v in zip(CLASSES, f1_cls)}}

    print(f"\n===== {args.model} fold{fold} =====")
    print(report)
    return metrics, f1_row, cm


def main():
    p = argparse.ArgumentParser(description="Evaluasi 5-fold MSLD v2.0")
    p.add_argument("--model", required=True, choices=MODELS.keys())
    p.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--img_size", type=int, default=None,
                   help="harus sama dengan --img_size waktu training")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--root", default=BASE_DIR)
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--results_dir", default="Results")
    args = p.parse_args()
    args.preprocess = get_preprocess(args.model, args.img_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res_dir = os.path.join(args.results_dir, args.model)
    os.makedirs(res_dir, exist_ok=True)

    all_metrics, all_f1, cm_total = [], [], None
    for k in args.folds:
        m, f1_row, cm = evaluate_fold(args, k, device, res_dir)
        all_metrics.append(m)
        all_f1.append(f1_row)
        cm_total = cm if cm_total is None else cm_total + cm

    per_fold = pd.DataFrame(all_metrics)
    per_fold.to_csv(os.path.join(res_dir, "metrics_per_fold.csv"), index=False)
    pd.DataFrame(all_f1).to_csv(os.path.join(res_dir, "f1_per_class.csv"), index=False)

    cols = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    summary = pd.DataFrame({"mean": per_fold[cols].mean(), "std": per_fold[cols].std()})
    summary.to_csv(os.path.join(res_dir, "metrics_summary.csv"))

    plot_confusion(cm_total, f"{args.model} - total {len(args.folds)} fold",
                   os.path.join(res_dir, "confusion_matrix_total.png"))

    print(f"\n===== {args.model}: metrik per fold =====")
    print(per_fold.round(4).to_string(index=False))
    print(f"\n===== {args.model}: rata-rata {len(args.folds)} fold =====")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
