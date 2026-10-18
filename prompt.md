# Tugas: Bangun Proyek Machine Learning Klasifikasi Chinese Characters — Fokus Fase Training

Saya sedang mengikuti kompetisi Analisis Data Multivariat di Codabench untuk mengklasifikasikan gambar karakter Mandarin ke dalam 15 kelas, yaitu label `0–14`.

**Kondisi saat ini masih fase training.** Dataset gambar evaluasi resmi untuk Fase 1 (`val_...`) dan Fase 2 (`test_...`) belum tersedia. Oleh karena itu, prioritas utama adalah membangun model yang akurat, mengevaluasinya secara benar, menyimpan model terbaik, dan menyiapkan pipeline prediksi yang dapat digunakan nanti.

Buat seluruh proyek secara langsung di workspace saat ini. Jangan hanya memberikan tutorial atau contoh kode dalam jawaban.

## 1. Dataset yang tersedia

Dataset training memiliki struktur kurang lebih seperti berikut:

- `train/`: sekitar 10.500 gambar PNG grayscale berukuran 64×64 piksel.
- 
- `train_labels.csv`: label gambar, dengan kolom `id`, `label`, `character`, `value`, dan `writer_id`.
- `classes.csv`: pemetaan kelas karakter.

Sebelum menulis kode, periksa struktur folder dan file yang benar-benar tersedia. Sesuaikan implementasi dengan dataset aktual, bukan asumsi.

Ketentuan:

- Jangan mengarang data, label, nama file, atau hasil eksperimen.
- Periksa apakah ID pada CSV cocok dengan nama file gambar.
- Pastikan seluruh label berada pada rentang `0–14`.
- Periksa jumlah data per kelas, data yang hilang, ID duplikat, gambar yang rusak, dan distribusi `writer_id`.
- Jangan menganggap file validasi atau test kompetisi sudah tersedia jika memang belum ada.

## 2. Struktur proyek

Buat struktur proyek yang rapi dan tidak berlebihan seperti berikut:

```text
project/
├── data_latih/
│   ├── train/
│   ├── train_labels.csv
│   └── classes.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_training.ipynb
│   └── 03_prediction_submission.ipynb
├── models/
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── predictions/
├── src/
│   ├── data_utils.py
│   ├── train.py
│   └── predict.py
├── requirements.txt
└── README.md
```

Jangan memindahkan atau menyalin dataset berukuran besar tanpa alasan yang jelas. Jika dataset sudah berada di lokasi lain, gunakan path yang sesuai dan dokumentasikan konfigurasi path-nya.

## 3. Notebook 1 — Exploratory Data Analysis (EDA)

Buat `01_eda.ipynb` dengan sel-sel terpisah dan urutan yang logis.

Isi minimal:

1. Import library dan konfigurasi path.
2. Memuat dataset serta label.
3. Menampilkan dimensi dataset dan lima baris pertama.
4. Memeriksa tipe data, missing value, ID duplikat, dan validitas label.
5. Memeriksa distribusi jumlah gambar per kelas.
6. Memeriksa distribusi jumlah gambar per `writer_id`.
7. Menampilkan contoh gambar untuk setiap kelas beserta labelnya.
8. Menampilkan beberapa gambar dari writer yang berbeda.
9. Memeriksa ukuran gambar, jumlah channel, rentang nilai piksel, dan gambar yang gagal dibaca.
10. Menyimpan visualisasi penting ke `outputs/figures/`.

Berikan interpretasi sederhana dalam bahasa Indonesia pada Markdown notebook. Jelaskan temuan berdasarkan hasil aktual, bukan teks interpretasi yang sudah ditentukan sebelumnya.

## 4. Notebook 2 — Training dan evaluasi model

Buat `02_training.ipynb`. Ini adalah prioritas utama proyek.

### A. Persiapan data

- Muat gambar dan label berdasarkan ID yang cocok.
- Pastikan urutan gambar dan label tetap konsisten.
- Terapkan preprocessing yang sesuai dengan format gambar asli.
- Periksa apakah gambar berupa goresan putih pada latar hitam atau format lainnya.
- Hindari transformasi yang dapat mengubah karakter menjadi kelas berbeda.
- Jangan melakukan augmentasi pada validation data.

### B. Pembagian training dan validation

Gunakan pemisahan berdasarkan `writer_id`, misalnya `GroupShuffleSplit` atau `GroupKFold`.

Alasannya: satu writer dapat memiliki banyak gambar. Jika gambar dari writer yang sama tersebar ke training dan validation, hasil evaluasi dapat terlalu optimistis karena model sudah melihat gaya tulisan orang tersebut.

Pastikan:

- Tidak ada `writer_id` yang sama di kedua kelompok.
- Distribusi kelas pada kedua kelompok diperiksa.
- Jika pembagian yang dipilih menyebabkan kelas tertentu tidak muncul di validation, deteksi dan tangani masalah tersebut secara benar.
- Jelaskan strategi pembagian data dan keterbatasannya.

Jangan gunakan gambar evaluasi resmi sebagai validation internal.

### C. Model baseline dan eksperimen

Mulai dengan baseline yang masuk akal, kemudian tingkatkan secara bertahap.

Prioritas:

1. Baseline sederhana yang cepat dilatih.
2. CNN untuk klasifikasi gambar.
3. Jika waktu dan resource memungkinkan, eksperimen variasi arsitektur, regularisasi, data augmentation, learning rate, dan batch size.

Gunakan framework yang sesuai dengan lingkungan, seperti PyTorch atau TensorFlow/Keras. Periksa ketersediaan GPU sebelum menentukan konfigurasi training.

Jangan menggunakan model yang terlalu kompleks tanpa alasan. Hindari download pretrained weights atau dataset eksternal yang membutuhkan internet jika tidak tersedia atau tidak diperlukan.

Semua eksperimen harus reproducible dengan random seed dan konfigurasi yang terdokumentasi.

### D. Evaluasi model

Ukur dan tampilkan minimal:

- Accuracy pada validation internal.
- Confusion matrix.
- Precision, recall, dan F1-score per kelas.
- Classification report.
- Kurva training loss dan validation loss.
- Kurva training accuracy dan validation accuracy.
- Waktu training dan ukuran model jika memungkinkan.

Simpan hasil evaluasi aktual ke `outputs/metrics/` dan visualisasi ke `outputs/figures/`.

Jangan mengklaim akurasi tertentu sebelum model benar-benar dijalankan. Jelaskan indikasi overfitting atau underfitting berdasarkan hasil aktual.

### E. Pemilihan dan penyimpanan model

- Simpan checkpoint model terbaik berdasarkan metrik validation yang ditentukan.
- Simpan konfigurasi preprocessing, label mapping, dan parameter penting yang diperlukan untuk inference.
- Pastikan model tersimpan dapat dimuat kembali.
- Jika memungkinkan, buat skrip kecil untuk menguji loading model dan prediksi terhadap beberapa gambar.
- Jangan menimpa model terbaik dengan model yang performanya lebih buruk.
- Dokumentasikan nama file model dan metrik validation terbaik.

Jangan mengklaim eksperimen berhasil jika proses training gagal atau belum dijalankan.

## 5. Notebook 3 — Persiapan prediksi dan submission untuk nanti

Buat `03_prediction_submission.ipynb`, tetapi **jangan menjalankan pembuatan submission resmi sekarang**, karena folder validasi dan test kompetisi belum tersedia.

Notebook ini harus disiapkan agar nanti dapat digunakan untuk:

- Fase 1: gambar dengan ID `val_...`.
- Fase 2: gambar dengan ID `test_...`.

Buat pipeline inference yang:

1. Memuat checkpoint model terbaik.
2. Membaca seluruh gambar pada folder target.
3. Mengambil ID dari nama file tanpa ekstensi `.png`.
4. Melakukan preprocessing yang sama dengan saat training.
5. Menghasilkan prediksi label berupa integer `0–14`.
6. Menjaga urutan ID dan prediksi tetap konsisten.
7. Menghasilkan `predictions.csv` dengan header persis `id,label`.
8. Memvalidasi CSV sebelum submission.

Validasi wajib saat data resmi nanti tersedia:

- Setiap ID target tercantum tepat satu kali.
- Tidak ada ID yang hilang atau berlebih.
- Jumlah baris sesuai jumlah gambar target.
- Tidak ada nilai label di luar rentang `0–14`.
- Tidak ada nilai kosong atau label non-integer.
- Nama kolom persis `id,label`.

Jika seluruh pemeriksaan lolos, sediakan fungsi untuk membuat ZIP yang berisi `predictions.csv` langsung di root ZIP, bukan di dalam folder.

Namun, selama fase training sekarang:

- Jangan mengarang ID `val_...` atau `test_...`.
- Jangan membuat submission palsu.
- Jangan membuat CSV kosong seolah-olah sudah siap dikumpulkan.
- Jangan membuat ZIP submission jika data target resmi belum tersedia.
- Jangan mengunggah submission ke Codabench secara otomatis.

Jika folder target belum tersedia, notebook harus berhenti dengan pesan yang jelas dan informatif, bukan error yang membingungkan.

## 6. Kualitas kode dan notebook

- Pisahkan kode ke dalam beberapa sel berdasarkan tahapan kerja.
- Hindari satu sel raksasa yang mengerjakan semuanya.
- Gunakan nama variabel yang jelas.
- Tambahkan komentar dan Markdown dalam bahasa Indonesia yang mudah dipahami mahasiswa pemula.
- Jelaskan alasan di balik langkah penting, bukan hanya apa yang dilakukan kodenya.
- Gunakan fungsi reusable agar preprocessing saat training dan inference konsisten.
- Tangani path yang tidak ditemukan, gambar rusak, dan konfigurasi yang tidak sesuai dengan pesan error yang jelas.
- Hindari redundansi dan abstraksi berlebihan.
- Pastikan notebook dapat dijalankan berurutan dari atas ke bawah.
- Jangan menyimpan dataset atau file sementara yang tidak diperlukan ke repository.

## 7. README dan dependensi

Buat `README.md` dalam bahasa Indonesia yang berisi:

- Deskripsi proyek dan kompetisi.
- Struktur folder.
- Instalasi dependensi.
- Cara menjalankan EDA.
- Cara menjalankan training.
- Cara mengevaluasi model.
- Lokasi checkpoint dan hasil metrik.
- Cara menyiapkan inference untuk Fase 1 dan Fase 2 ketika data resmi dirilis.
- Format submission yang diwajibkan Codabench.
- Catatan mengenai pemisahan validation berdasarkan `writer_id`.

Buat `requirements.txt` berisi dependensi yang benar-benar digunakan. Hindari mencantumkan library yang tidak diperlukan.

## 8. Pemeriksaan akhir

Setelah membuat proyek:

1. Periksa seluruh file dan struktur folder.
2. Jalankan pemeriksaan sintaks atau smoke test yang memungkinkan.
3. Jalankan EDA dan training jika lingkungan serta waktu memungkinkan.
4. Pastikan preprocessing training dan inference konsisten.
5. Pastikan checkpoint dapat dimuat kembali.
6. Pastikan notebook tidak bergantung pada file validation/test yang belum tersedia.
7. Catat kendala lingkungan, misalnya GPU atau dependensi yang tidak tersedia.

Jika training belum selesai dijalankan, katakan dengan jelas bahwa hasil model dan metrik belum tersedia. Jangan mengarang angka akurasi, grafik, atau hasil eksperimen.

## 9. Laporan akhir

Setelah pekerjaan selesai, laporkan secara ringkas:

- File yang berhasil dibuat.
- Dataset dan path yang benar-benar digunakan.
- Arsitektur model yang diterapkan.
- Strategi pembagian training dan validation.
- Status eksekusi training.
- Akurasi dan metrik validation aktual jika tersedia.
- Lokasi checkpoint terbaik.
- Tes yang sudah berhasil dan yang belum berhasil.
- Langkah berikutnya yang harus dilakukan saat data evaluasi resmi dirilis.

**Prioritas utama sekarang adalah memperoleh model yang bagus dan evaluasi internal yang dapat dipercaya. Submission Codabench merupakan tahap berikutnya, setelah data evaluasi resmi tersedia.**
