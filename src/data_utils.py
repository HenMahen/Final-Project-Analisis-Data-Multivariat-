"""
data_utils.py — Fungsi utilitas data untuk proyek klasifikasi karakter Mandarin.

Modul ini berisi fungsi-fungsi yang digunakan bersama oleh notebook EDA,
training, dan prediction untuk memastikan konsistensi preprocessing.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image


# ============================================================
# Konfigurasi path default
# ============================================================
# Path relatif terhadap root proyek (Final Project/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data_latih")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train")
TRAIN_LABELS_CSV = os.path.join(DATA_DIR, "train_labels.csv")
CLASSES_CSV = os.path.join(DATA_DIR, "classes.csv")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
METRICS_DIR = os.path.join(OUTPUTS_DIR, "metrics")
PREDICTIONS_DIR = os.path.join(OUTPUTS_DIR, "predictions")

# Parameter gambar
IMG_SIZE = 64       # ukuran gambar 64x64
NUM_CLASSES = 15    # label 0-14
RANDOM_SEED = 42


def load_labels(csv_path=None):
    """
    Memuat file CSV label training.

    Returns:
        pd.DataFrame dengan kolom id, label, character, value, writer_id
    """
    if csv_path is None:
        csv_path = TRAIN_LABELS_CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File label tidak ditemukan: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def load_classes(csv_path=None):
    """
    Memuat file CSV pemetaan kelas.

    Returns:
        pd.DataFrame dengan kolom label, character, value, pinyin, arti, dll.
    """
    if csv_path is None:
        csv_path = CLASSES_CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File classes tidak ditemukan: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def load_single_image(img_path):
    """
    Memuat satu gambar sebagai numpy array grayscale.

    Args:
        img_path: path lengkap ke file gambar

    Returns:
        numpy array shape (H, W) dengan dtype uint8, atau None jika gagal
    """
    try:
        img = Image.open(img_path).convert("L")  # grayscale
        return np.array(img, dtype=np.uint8)
    except Exception as e:
        print(f"[WARNING] Gagal membaca gambar: {img_path} — {e}")
        return None


def load_images_and_labels(df, img_dir=None):
    """
    Memuat seluruh gambar dan label berdasarkan DataFrame.

    Args:
        df: DataFrame dengan kolom 'id' dan 'label'
        img_dir: direktori gambar (default: TRAIN_IMG_DIR)

    Returns:
        images: numpy array shape (N, 64, 64) dtype uint8
        labels: numpy array shape (N,) dtype int
        valid_indices: list indeks baris yang berhasil dimuat
        failed_ids: list id gambar yang gagal dimuat
    """
    if img_dir is None:
        img_dir = TRAIN_IMG_DIR

    images = []
    labels = []
    valid_indices = []
    failed_ids = []

    for idx, row in df.iterrows():
        img_id = row["id"]
        img_path = os.path.join(img_dir, f"{img_id}.png")
        img = load_single_image(img_path)

        if img is not None:
            images.append(img)
            labels.append(int(row["label"]))
            valid_indices.append(idx)
        else:
            failed_ids.append(img_id)

    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.int64)

    return images, labels, valid_indices, failed_ids


def preprocess_for_model(images, normalize=True):
    """
    Preprocessing gambar untuk input model CNN.
    Konsisten antara training dan inference.

    Args:
        images: numpy array shape (N, 64, 64) dtype uint8
        normalize: jika True, normalisasi ke [0, 1]

    Returns:
        numpy array shape (N, 1, 64, 64) dtype float32
        (format: batch, channel, height, width — sesuai PyTorch)
    """
    # Tambah dimensi channel: (N, H, W) -> (N, 1, H, W)
    x = images.astype(np.float32)
    if normalize:
        x = x / 255.0
    x = x[:, np.newaxis, :, :]  # (N, 1, 64, 64)
    return x


def preprocess_single_image(img_array, normalize=True):
    """
    Preprocessing satu gambar untuk inference.

    Args:
        img_array: numpy array shape (64, 64) dtype uint8

    Returns:
        numpy array shape (1, 1, 64, 64) dtype float32
    """
    x = img_array.astype(np.float32)
    if normalize:
        x = x / 255.0
    x = x[np.newaxis, np.newaxis, :, :]  # (1, 1, 64, 64)
    return x


def get_label_to_character_map(classes_df=None):
    """
    Membuat dictionary pemetaan label (int) -> karakter (str).

    Returns:
        dict, contoh: {0: '零', 1: '一', ...}
    """
    if classes_df is None:
        classes_df = load_classes()
    return dict(zip(classes_df["label"], classes_df["character"]))


def validate_submission_csv(csv_path, expected_count=None):
    """
    Memvalidasi file submission CSV sebelum di-submit.

    Args:
        csv_path: path ke file predictions.csv
        expected_count: jumlah baris yang diharapkan (opsional)

    Returns:
        (is_valid, list_of_issues)
    """
    issues = []

    if not os.path.exists(csv_path):
        return False, ["File tidak ditemukan"]

    df = pd.read_csv(csv_path)

    # Cek kolom
    if list(df.columns) != ["id", "label"]:
        issues.append(f"Kolom harus persis ['id', 'label'], ditemukan: {list(df.columns)}")

    # Cek missing values
    if df.isnull().any().any():
        issues.append(f"Terdapat nilai kosong: {df.isnull().sum().to_dict()}")

    # Cek duplikat ID
    dup = df["id"].duplicated().sum()
    if dup > 0:
        issues.append(f"Terdapat {dup} ID duplikat")

    # Cek rentang label
    if "label" in df.columns:
        invalid = df[(df["label"] < 0) | (df["label"] > 14)]
        if len(invalid) > 0:
            issues.append(f"Terdapat {len(invalid)} label di luar rentang 0-14")

        # Cek tipe data label (harus integer)
        if not pd.api.types.is_integer_dtype(df["label"]):
            issues.append("Label harus bertipe integer")

    # Cek jumlah baris
    if expected_count is not None and len(df) != expected_count:
        issues.append(f"Jumlah baris {len(df)}, diharapkan {expected_count}")

    is_valid = len(issues) == 0
    return is_valid, issues
