# Kode – UTS Deep Learning

Kode untuk tiga task: klasifikasi multimodal (Task 1), regresi RUL (Task 2), dan clustering multimodal (Task 3).

## Struktur

```
laporan.pdf                  laporan Task 1–3
prediksi_klasifikasi.csv     id_unit,kelas
prediksi_rul.csv             id_unit,rul
cluster.csv                  id_sampel,cluster
kontribusi.md                peran anggota, link PR, penggunaan AI
kode/
├── README.md
├── requirements.txt
├── run_all.sh               satu perintah untuk reproduksi
├── task1_multimodal_klasifikasi.ipynb
├── task2_time_series_regression.ipynb
├── task3_multimodal_clustering.ipynb
└── outputs/                 bobot model dan hasil antara (cache)
```

## Persiapan

1. Python 3.11+ (diuji pada: **ISI versi**).
2. Pasang dependensi:
   ```bash
   pip install -r kode/requirements.txt
   ```
   Untuk GPU NVIDIA, pasang PyTorch versi CUDA terlebih dahulu (https://pytorch.org/get-started/locally/).
3. Letakkan data dari dosen di folder `dataset/` pada root repo (tidak ikut di-commit, ada di `.gitignore`):
   ```
   dataset/klasifikasi_train.npz, klasifikasi_test.npz,
           klasifikasi_train_meta.csv, klasifikasi_test_meta.csv
   dataset/rul_train.csv.gz, rul_test.csv.gz
   dataset/clustering.npz (atau clustering_tanpa_gambar.npz), clustering_meta.csv
   ```

## Reproduksi semua angka di laporan (satu perintah)

Jalankan dari root repo:

```bash
bash kode/run_all.sh
```

Ketiga CSV langsung ditulis ke **root repo** (bukan ke `kode/outputs/`). Skrip ini mengeksekusi ketiga notebook berurutan (Task 1 → 2 → 3) dengan seed tetap (`SEED = 42`, Task 1 memakai tiga seed untuk eksperimen robustness) dan menulis:

| Output | Dibuat oleh |
|---|---|
| `prediksi_klasifikasi.csv` | Task 1, bagian 12 |
| `prediksi_rul.csv` | Task 2, bagian 11–12 |
| `cluster.csv` | Task 3, bagian 15 |
| tabel dan grafik laporan | `kode/outputs/` dan notebook yang sudah dieksekusi (`kode/executed/`) |

Perkiraan waktu: **ISI** menit (GPU: **ISI**).

## Bobot model

Bobot akhir ada di `kode/outputs/` (semuanya di bawah 100 MB):

- Task 2: `CNN1D_best.pt`, `LSTM_best.pt`
- Task 3: `spec_ae.pt`, `spec_contrastive.pt`
- Task 1: **ISI** (letak bobot model final)

Notebook memakai cache ini secara default. Untuk melatih ulang dari nol:

- Task 3: set `RETRAIN = True` di sel konfigurasi.
- Task 1: set `EPOCHS` lewat variabel lingkungan, misalnya `EPOCHS=14`.
- Task 2: hapus file `.pt` terkait di `kode/outputs/`, lalu jalankan ulang.

File besar yang bisa dibuat ulang (`spektrum_bersih.npy`, `emb_*.npy`) tidak ikut di repo; notebook Task 3 membuatnya kembali.

## Catatan

- Task 3 memakai `dataset/embedding_gambar.npz` (embedding gambar dari Anggota 1) bila ada. Jika tidak ada, notebook melatih sendiri image encoder contrastive dari array `gambar` di `clustering.npz` (`IMG_FALLBACK_TRAIN = True`), sehingga `cluster.csv` final tetap dibuat. Sumber yang dipakai dicetak sebagai `Sumber embedding gambar`; sebutkan di laporan.
- Semua hyperparameter dipusatkan di sel **Konfigurasi** tiap notebook.
