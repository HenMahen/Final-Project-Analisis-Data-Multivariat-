# Klasifikasi Karakter Mandarin 

Proyek machine learning untuk mengklasifikasikan gambar tulisan tangan karakter Mandarin ke dalam 15 kelas (label 0–14). Dikembangkan sebagai tugas mata kuliah Analisis Data Multivariat (ADM).

## Deskripsi Kompetisi

- **Platform:** Codabench
- **Tugas:** Klasifikasi gambar karakter Mandarin tulisan tangan
- **Kelas:** 15 karakter angka Mandarin (零, 一, 二, 三, 四, 五, 六, 七, 八, 九, 十, 百, 千, 万, 亿)
- **Fase 1:** Prediksi pada gambar validasi (`val_*.png`)
- **Fase 2:** Prediksi pada gambar test (`test_*.png`)

## Struktur Folder

```
Final Project/
├── data_latih/              # Dataset training
│   ├── train/               # ~10.500 gambar PNG grayscale 64×64
│   ├── train_labels.csv     # Label: id, label, character, value, writer_id
│   └── classes.csv          # Pemetaan 15 kelas karakter
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory Data Analysis
│   ├── 02_training.ipynb               # Training & evaluasi model
│   └── 03_prediction_submission.ipynb  # Pipeline inference & submission
├── models/                  # Checkpoint model tersimpan
├── outputs/
│   ├── figures/             # Visualisasi (grafik, confusion matrix)
│   ├── metrics/             # Hasil evaluasi (JSON, txt)
│   └── predictions/         # File prediksi (CSV, ZIP)
├── src/
│   ├── data_utils.py        # Utilitas data (loading, preprocessing)
│   ├── train.py             # Definisi model & fungsi training
│   └── predict.py           # Pipeline inference & submission
├── requirements.txt
└── README.md
```

## Instalasi Dependensi

**GPU (Rekomendasi):**

```bash
# Uninstall torch CPU jika sudah terinstall
pip uninstall torch -y

# Install PyTorch dengan CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Install dependencies lainnya
pip install numpy pandas matplotlib seaborn Pillow scikit-learn
```

**CPU only:**

```bash
pip install -r requirements.txt
```

**Catatan:** Training dengan GPU (RTX 2050) ~10-20x lebih cepat daripada CPU.

## Cara Menjalankan

### 1. Exploratory Data Analysis

Buka dan jalankan notebook:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

Output: grafik distribusi di `outputs/figures/`, statistik di output cell.

### 2. Training Model

Buka dan jalankan notebook:

```bash
jupyter notebook notebooks/02_training.ipynb
```

Training akan:

- Melatih 2 model: BaselineCNN dan ImprovedCNN
- Menggunakan GPU otomatis jika tersedia (10-20x lebih cepat)
- Menyimpan checkpoint terbaik ke `models/`
- Early stopping otomatis

Estimasi waktu:

- **GPU (RTX 2050):** 30-60 menit
- **CPU:** 2-4 jam

Hasil training akan tersimpan di:

- **Checkpoint model:** `models/<nama_model>_best.pth`
- **History training:** `models/<nama_model>_history.json`
- **Konfigurasi inference:** `models/inference_config.json`
- **Visualisasi:** `outputs/figures/`
- **Metrik evaluasi:** `outputs/metrics/`

### 3. Prediksi & Submission

Buka `notebooks/03_prediction_submission.ipynb`:

1. Ubah variabel `TARGET_DIR` ke folder data evaluasi resmi
2. Jalankan notebook dari atas ke bawah
3. File `predictions.csv` dan `submission.zip` akan dibuat di `outputs/predictions/`

## Strategi Pembagian Data (Validation)

Kami menggunakan **GroupShuffleSplit berdasarkan `writer_id`** dengan rasio 80% train / 20% validation.

**Alasan:** Satu writer menulis banyak karakter. Jika gambar dari writer yang sama ada di training dan validation, model bisa menghafal gaya tulisan (bukan mengenali karakter), sehingga evaluasi menjadi terlalu optimistis. Dengan memisahkan berdasarkan writer, kita mendapatkan evaluasi yang lebih realistis.

**Keterbatasan:**

- Beberapa kelas mungkin kurang terwakili di validation jika writer tertentu tidak menulis semua karakter
- Pembagian tidak stratified per kelas, tetapi berdasarkan grup writer

## Format Submission Codabench

File submission berupa ZIP yang berisi `predictions.csv` **langsung di root ZIP** (bukan di dalam folder):

```
submission.zip
└── predictions.csv    ← langsung di root
```

Format `predictions.csv`:

```csv
id,label
val_00001,5
val_00002,12
...
```

- Header: persis `id,label`
- Label: integer 0–14
- Setiap ID target muncul tepat satu kali

## Arsitektur Model

### BaselineCNN (Baseline)

- 3 blok konvolusi (Conv2d + BatchNorm + ReLU + MaxPool)
- Fully connected layers dengan Dropout
- Parameter: ~2.1 juta

### ImprovedCNN (Model Utama)

- 4 blok konvolusi, masing-masing dengan 2 layer Conv2d
- BatchNorm2d + Dropout2d per blok
- AdaptiveAvgPool2d pada blok terakhir
- FC layers yang lebih lebar dengan Dropout
- Parameter: ~3.5 juta

## Catatan

- **Status saat ini:** Fase training. Data evaluasi resmi belum tersedia.
- **Preprocessing:** Gambar grayscale 64×64, normalisasi ke [0, 1].
- **Augmentasi:** Hanya noise ringan pada training. Tidak ada flip karena bisa mengubah makna karakter.
- **Reproducibility:** Random seed = 42 digunakan secara konsisten.
