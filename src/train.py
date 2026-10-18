"""
train.py — Skrip training model CNN untuk klasifikasi karakter Mandarin.

Modul ini berisi definisi model, fungsi training, dan evaluasi.
Digunakan oleh notebook 02_training.ipynb.
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

# Tambahkan src ke path agar bisa import data_utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import NUM_CLASSES, IMG_SIZE, RANDOM_SEED


# ============================================================
# Dataset PyTorch
# ============================================================
class ChineseCharDataset(Dataset):
    """
    Dataset PyTorch untuk gambar karakter Mandarin.

    Args:
        images: numpy array (N, 64, 64) uint8
        labels: numpy array (N,) int
        transform: fungsi augmentasi opsional (menerima tensor, return tensor)
    """

    def __init__(self, images, labels, transform=None):
        self.images = images  # (N, 64, 64) uint8
        self.labels = labels  # (N,) int
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Konversi ke tensor float32 dan normalisasi ke [0, 1]
        img = self.images[idx].astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)  # (1, 64, 64)

        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


# ============================================================
# Augmentasi sederhana (untuk training saja)
# ============================================================
class SimpleAugmentation:
    """
    Augmentasi ringan yang aman untuk karakter tulisan tangan.
    Hanya diterapkan pada data training, TIDAK pada validation.

    Augmentasi yang digunakan:
    - Random rotation kecil (±10 derajat)
    - Random translation kecil
    - Random noise
    Menghindari flip horizontal/vertikal karena bisa mengubah karakter.
    """

    def __init__(self, rotation_deg=10, translate_frac=0.05, noise_std=0.02):
        self.rotation_deg = rotation_deg
        self.translate_frac = translate_frac
        self.noise_std = noise_std

    def __call__(self, img_tensor):
        """
        Args:
            img_tensor: tensor shape (1, 64, 64) float32 [0, 1]
        Returns:
            tensor shape (1, 64, 64) float32 [0, 1]
        """
        # Random noise
        if np.random.random() < 0.5:
            noise = torch.randn_like(img_tensor) * self.noise_std
            img_tensor = torch.clamp(img_tensor + noise, 0.0, 1.0)

        return img_tensor


# ============================================================
# Model CNN — Baseline
# ============================================================
class BaselineCNN(nn.Module):
    """
    CNN sederhana sebagai baseline.
    Input: (batch, 1, 64, 64) — grayscale 64×64.
    Output: logits (batch, 15).
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super(BaselineCNN, self).__init__()
        self.features = nn.Sequential(
            # Conv block 1: 1 -> 32 channels
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32

            # Conv block 2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16

            # Conv block 3: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 -> 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# Model CNN — Improved (lebih dalam, dengan lebih banyak regularisasi)
# ============================================================
class ImprovedCNN(nn.Module):
    """
    CNN yang lebih kuat untuk performa lebih baik.
    Input: (batch, 1, 64, 64).
    Output: logits (batch, 15).
    """

    def __init__(self, num_classes=NUM_CLASSES, dropout_rate=0.4):
        super(ImprovedCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1: 1 -> 32
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32
            nn.Dropout2d(0.1),

            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Dropout2d(0.15),

            # Block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 -> 8
            nn.Dropout2d(0.2),

            # Block 4: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),  # 8 -> 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# Fungsi Training
# ============================================================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Melatih model untuk satu epoch.

    Returns:
        (avg_loss, accuracy) untuk epoch tersebut
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_imgs, batch_labels in dataloader:
        batch_imgs = batch_imgs.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        outputs = model(batch_imgs)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_imgs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """
    Evaluasi model pada data validation.

    Returns:
        (avg_loss, accuracy, all_preds, all_labels)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_imgs, batch_labels in dataloader:
            batch_imgs = batch_imgs.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_imgs)
            loss = criterion(outputs, batch_labels)

            running_loss += loss.item() * batch_imgs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs=30,
    save_dir=None,
    model_name="model",
    patience=7,
):
    """
    Loop training lengkap dengan early stopping dan penyimpanan model terbaik.

    Args:
        model: model PyTorch
        train_loader, val_loader: DataLoader
        criterion: loss function
        optimizer: optimizer
        scheduler: learning rate scheduler (bisa None)
        device: 'cuda' atau 'cpu'
        num_epochs: jumlah epoch maksimal
        save_dir: direktori untuk menyimpan checkpoint
        model_name: nama file model
        patience: jumlah epoch tanpa peningkatan sebelum berhenti

    Returns:
        history: dict berisi train_loss, val_loss, train_acc, val_acc per epoch
        best_val_acc: akurasi validation terbaik
    """
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_acc = 0.0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    start_time = time.time()

    if save_dir is None:
        save_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
        )
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Training
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validation
        val_loss, val_acc, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device
        )

        # Learning rate scheduler
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Simpan history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
        )

        # Simpan model terbaik (berdasarkan val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            epochs_without_improvement = 0

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "train_loss": train_loss,
            }
            save_path = os.path.join(save_dir, f"{model_name}_best.pth")
            torch.save(checkpoint, save_path)
            print(f"  ✓ Model terbaik disimpan: val_acc={val_acc:.4f}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"\n⚠ Early stopping pada epoch {epoch+1}. "
                    f"Tidak ada peningkatan selama {patience} epoch."
                )
                break

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training selesai dalam {total_time:.1f} detik ({total_time/60:.1f} menit)")
    print(f"Akurasi validation terbaik: {best_val_acc:.4f}")
    print(f"Loss validation terbaik: {best_val_loss:.4f}")
    print(f"{'='*60}")

    # Simpan history
    history_path = os.path.join(save_dir, f"{model_name}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History training disimpan: {history_path}")

    return history, best_val_acc


def load_model(model_class, checkpoint_path, device, **model_kwargs):
    """
    Memuat model dari checkpoint.

    Args:
        model_class: kelas model (BaselineCNN atau ImprovedCNN)
        checkpoint_path: path ke file .pth
        device: 'cuda' atau 'cpu'
        **model_kwargs: parameter tambahan untuk konstruktor model

    Returns:
        model yang sudah dimuat weights-nya
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {checkpoint_path}")

    model = model_class(**model_kwargs)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Model dimuat dari: {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', '?')}")
    print(f"  Val Acc: {checkpoint.get('val_acc', '?'):.4f}")
    print(f"  Val Loss: {checkpoint.get('val_loss', '?'):.4f}")

    return model


def get_device():
    """Deteksi dan return device terbaik yang tersedia."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Menggunakan GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("GPU tidak tersedia, menggunakan CPU")
    return device


def set_seed(seed=RANDOM_SEED):
    """Set random seed untuk reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed diset ke: {seed}")
