"""
predict.py — Skrip prediksi/inference untuk submission kompetisi Codabench.

Modul ini memuat model terbaik dan menghasilkan prediksi pada gambar baru.
Digunakan oleh notebook 03_prediction_submission.ipynb.
"""

import os
import sys
import glob
import zipfile
import numpy as np
import pandas as pd
import torch

# Tambahkan src ke path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import (
    load_single_image,
    preprocess_single_image,
    validate_submission_csv,
    NUM_CLASSES,
    MODELS_DIR,
    PREDICTIONS_DIR,
)
from train import ImprovedCNN, BaselineCNN, get_device


def predict_folder(
    model,
    img_dir,
    device,
    batch_size=64,
):
    """
    Melakukan prediksi pada seluruh gambar di dalam folder.

    Args:
        model: model PyTorch yang sudah dimuat
        img_dir: path ke folder berisi gambar PNG
        device: torch device
        batch_size: ukuran batch untuk prediksi

    Returns:
        results_df: DataFrame dengan kolom 'id' dan 'label'
    """
    # Kumpulkan semua file PNG
    png_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))

    if len(png_files) == 0:
        print(f"⚠ Tidak ada file PNG ditemukan di: {img_dir}")
        return None

    print(f"Ditemukan {len(png_files)} gambar di {img_dir}")

    ids = []
    predictions = []
    failed = []

    model.eval()

    # Proses dalam batch
    batch_imgs = []
    batch_ids = []

    for img_path in png_files:
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        img = load_single_image(img_path)

        if img is None:
            failed.append(img_id)
            continue

        # Preprocessing: sama dengan saat training
        img_tensor = torch.from_numpy(
            img.astype(np.float32) / 255.0
        ).unsqueeze(0)  # (1, 64, 64)

        batch_imgs.append(img_tensor)
        batch_ids.append(img_id)

        # Jika batch sudah penuh, prediksi
        if len(batch_imgs) >= batch_size:
            batch_tensor = torch.stack(batch_imgs).to(device)  # (B, 1, 64, 64)
            with torch.no_grad():
                outputs = model(batch_tensor)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
            ids.extend(batch_ids)
            predictions.extend(preds.tolist())
            batch_imgs = []
            batch_ids = []

    # Sisa batch terakhir
    if len(batch_imgs) > 0:
        batch_tensor = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            outputs = model(batch_tensor)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
        ids.extend(batch_ids)
        predictions.extend(preds.tolist())

    if failed:
        print(f"⚠ {len(failed)} gambar gagal dibaca: {failed[:10]}...")

    results_df = pd.DataFrame({"id": ids, "label": predictions})
    results_df["label"] = results_df["label"].astype(int)

    print(f"✓ Prediksi selesai: {len(results_df)} gambar berhasil diprediksi")
    return results_df


def save_predictions(df, output_path=None):
    """
    Menyimpan DataFrame prediksi ke CSV.

    Args:
        df: DataFrame dengan kolom 'id' dan 'label'
        output_path: path file output (default: outputs/predictions/predictions.csv)
    """
    if output_path is None:
        output_path = os.path.join(PREDICTIONS_DIR, "predictions.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✓ Prediksi disimpan ke: {output_path}")
    return output_path


def create_submission_zip(csv_path, zip_path=None):
    """
    Membuat file ZIP untuk submission Codabench.
    File predictions.csv ditempatkan langsung di root ZIP (bukan di dalam folder).

    Args:
        csv_path: path ke file predictions.csv
        zip_path: path file ZIP output

    Returns:
        path ke file ZIP yang dibuat
    """
    if zip_path is None:
        zip_path = os.path.join(PREDICTIONS_DIR, "submission.zip")

    # Validasi CSV dulu
    is_valid, issues = validate_submission_csv(csv_path)
    if not is_valid:
        print("✗ CSV tidak valid untuk submission:")
        for issue in issues:
            print(f"  - {issue}")
        return None

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, "predictions.csv")  # langsung di root ZIP

    print(f"✓ File submission ZIP dibuat: {zip_path}")
    return zip_path
