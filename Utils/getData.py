"""
Loader data MSLD v2.0 (Mpox Skin Lesion Dataset).

Split per fold:
  train = Original fold{k}/Train + Augmented fold{k}_AUG/Train
  valid = Original fold{k}/Valid
  test  = Original fold{k}/Test
Valid & test tidak ikut training.

Jalankan `python Utils/getData.py` buat bikin tabel + grafik distribusi data.
"""

import os
from collections import Counter

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

BASE_DIR = ("/kaggle/input/datasets/joydippaul/"
            "mpox-skin-lesion-dataset-version-20-msld-v20")

# urut abjad, index = label (Chickenpox=0 ... Monkeypox=5)
CLASSES = ["Chickenpox", "Cowpox", "HFMD", "Healthy", "Measles", "Monkeypox"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# setting preprocessing default. Tiap model bisa nimpa lewat atribut
# class PREPROCESS di file modelnya sendiri, contoh:
#     class SimpleCNN(nn.Module):
#         PREPROCESS = {"size": 100, "bgr": True, "normalize": False}
# key yang nggak ditulis di model otomatis pakai nilai default di bawah
DEFAULT_PREPROCESS = {"size": 224, "bgr": False, "normalize": False}


def get_split_dirs(fold, root=BASE_DIR):
    orig = os.path.join(root, "Original Images", "Original Images",
                        "FOLDS", f"fold{fold}")
    aug = os.path.join(root, "Augmented Images", "Augmented Images",
                       "FOLDS_AUG", f"fold{fold}_AUG")
    return {
        "train_orig": os.path.join(orig, "Train"),
        "train_aug":  os.path.join(aug, "Train"),
        "valid":      os.path.join(orig, "Valid"),
        "test":       os.path.join(orig, "Test"),
    }


def list_images(split_dir):
    # ambil semua (path, label) dari 6 folder kelas
    samples = []
    for label, cls in enumerate(CLASSES):
        cls_dir = os.path.join(split_dir, cls)
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append((os.path.join(cls_dir, fname), label))
    return samples


def get_samples(fold, root=BASE_DIR):
    d = get_split_dirs(fold, root)
    train = list_images(d["train_orig"]) + list_images(d["train_aug"])
    valid = list_images(d["valid"])
    test = list_images(d["test"])
    return train, valid, test


class MpoxDataset(Dataset):
    def __init__(self, samples, preprocess=None):
        cfg = {**DEFAULT_PREPROCESS, **(preprocess or {})}
        self.samples = samples
        self.bgr = cfg["bgr"]

        steps = [transforms.Resize((cfg["size"], cfg["size"])),
                 transforms.ToTensor()]  # jadi tensor CxHxW, 0-1
        if cfg["normalize"]:
            steps.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
        self.transform = transforms.Compose(steps)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        if self.bgr:
            x = x[[2, 1, 0]]  # RGB -> BGR
        return x, label


def get_dataloaders(fold, preprocess=None, batch_size=32, num_workers=2, root=BASE_DIR):
    # balikin train_loader, valid_loader, test_loader, CLASSES
    # preprocess = dict setting dari model (lihat DEFAULT_PREPROCESS)
    train, valid, test = get_samples(fold, root)
    pin = torch.cuda.is_available()

    train_loader = DataLoader(MpoxDataset(train, preprocess), batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=pin)
    valid_loader = DataLoader(MpoxDataset(valid, preprocess), batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(MpoxDataset(test, preprocess), batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=pin)

    print(f"[fold{fold}] train={len(train)} valid={len(valid)} test={len(test)}")
    return train_loader, valid_loader, test_loader, CLASSES


def count_distribution(folds=(1, 2, 3, 4, 5), root=BASE_DIR):
    # hitung jumlah gambar per kelas di tiap fold & split
    rows = []
    for k in folds:
        for split, split_dir in get_split_dirs(k, root).items():
            cnt = Counter(CLASSES[lbl] for _, lbl in list_images(split_dir))
            rows.append({"fold": k, "split": split, **{c: cnt[c] for c in CLASSES}})
    return rows


if __name__ == "__main__":
    import argparse
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = argparse.ArgumentParser(description="Tabel dan grafik distribusi data per kelas")
    p.add_argument("--root", default=BASE_DIR)
    p.add_argument("--out_dir", default="Results")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    df = pd.DataFrame(count_distribution(root=a.root))
    df["total"] = df[CLASSES].sum(axis=1)
    df.to_csv(os.path.join(a.out_dir, "distribusi_data.csv"), index=False)
    print(df.to_string(index=False))

    # rata-rata 5 fold per split
    mean_df = df.groupby("split")[CLASSES].mean().loc[
        ["train_orig", "train_aug", "valid", "test"]]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, (split, row) in zip(axes, mean_df.iterrows()):
        ax.bar(CLASSES, row.values)
        ax.set_title(split)
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel("jumlah gambar (rata-rata 5 fold)")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, "distribusi_data.png"), dpi=150)
    print("Tersimpan:", os.path.join(a.out_dir, "distribusi_data.png"))
