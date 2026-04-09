# Penjelasan Folder `Data-set`

Dokumen ini menjelaskan isi folder `Data-set`, alur eksperimen, arti output, dan cara membaca hasil untuk kebutuhan skripsi/laporan.

## 1) Isi Utama Folder

- `diabetes_prediction_dataset.csv`  
  Dataset utama dari Kaggle (100K baris) untuk prediksi diabetes.

- `diabetes_fl_blockchain.py`  
  Script utama eksperimen:
  - preprocessing data
  - baseline centralized ML
  - federated learning + blockchain verification
  - simulasi klien jahat
  - ekspor metrik, log keamanan, ledger, dan visualisasi

- File dataset lain (`diabetes.csv`, BRFSS, dan lain-lain)  
  Bisa dipakai untuk eksperimen tambahan/komparasi, tetapi pipeline saat ini fokus pada `diabetes_prediction_dataset.csv`.

## 2) Alur Proses Model

1. **Preprocessing**
   - hapus duplikat
   - encoding kolom kategorikal (`gender`, `smoking_history`)
   - split train-test (stratified)
   - oversampling kelas minoritas pada train
   - standardisasi fitur

2. **Baseline Centralized**
   - Logistic Regression
   - Random Forest

3. **Federated Learning + Blockchain**
   - data train dibagi ke beberapa klien (IID split)
   - tiap klien melatih model lokal
   - hash update model dicatat ke blockchain (commitment)
   - server verifikasi hash sebelum agregasi
   - update invalid ditolak (deteksi anomali/manipulasi)
   - agregasi robust (trimmed mean fallback weighted)

4. **Skenario Uji**
   - `fl_bc_normal` (tanpa klien jahat)
   - `fl_bc_1_malicious` (1 klien jahat)
   - `fl_bc_2_malicious` (2 klien jahat)

## 3) Struktur Output yang Dihasilkan

Output berada di `Data-set/output` dan dibagi per bagian:

- `1_baseline/`
  - `centralized_results.csv`: hasil metrik model centralized

- `2_federated_history/`
  - `*_history.csv`: histori metrik per ronde untuk tiap skenario FL+Blockchain

- `3_security_logs/`
  - `*_security.csv`: jumlah klien accepted/rejected per ronde

- `4_blockchain_ledger/`
  - `*_ledger.json`: isi chain per skenario

- `5_summary/`
  - `summary_results.csv`: ringkasan gabungan semua pendekatan
  - `summary_centralized.csv`: ringkasan khusus centralized
  - `summary_federated_blockchain.csv`: ringkasan khusus FL+Blockchain

- `6_visualizations/`
  - gambar perbandingan dan tren metrik

## 4) Daftar Visualisasi dan Fungsinya

Di `output/6_visualizations`:

1. `01_perbandingan_akhir_semua_pendekatan.png`  
   Perbandingan semua metrik akhir antar pendekatan/skenario.

2. `02_tren_f1_per_ronde.png`  
   Perkembangan F1 dari ronde awal sampai akhir.

3. `03_tren_auc_per_ronde.png`  
   Perkembangan AUC per ronde.

4. `04_keamanan_accepted_vs_rejected.png`  
   Visual deteksi keamanan (accepted vs rejected) pada skenario 2 klien jahat.

5. `05_perbandingan_accuracy.png`  
   Fokus komparasi akurasi antar pendekatan.

6. `06_heatmap_perbandingan_metrik.png`  
   Heatmap agar pembacaan performa antar metrik/skenario lebih cepat.

7. `07_confusion_matrix_random_forest.png`  
   Confusion matrix model centralized terbaik (Random Forest).

8. `08_ringkasan_keamanan_antar_skenario.png`  
   Ringkasan accepted/rejected ronde akhir antar skenario serangan.

## 5) Cara Menafsirkan Hasil

- **Akurasi/F1/AUC lebih tinggi** menandakan kualitas prediksi lebih baik.
- **Accepted tinggi + rejected sesuai jumlah klien jahat** menandakan mekanisme verifikasi blockchain bekerja.
- **`ledger_valid = True`** menandakan integritas chain tidak rusak.
- Jika FL lebih rendah dari centralized, itu normal karena setting privasi/distribusi data biasanya menambah trade-off performa.

## 6) Cara Menjalankan Ulang

Dari folder `Data-set`:

```bash
python diabetes_fl_blockchain.py
```

Semua output akan diperbarui otomatis di struktur folder `output`.

## 7) Catatan untuk Skripsi

- Gunakan:
  - `summary_results.csv` untuk tabel utama hasil
  - file di `6_visualizations` untuk gambar perbandingan
  - `*_security.csv` dan `*_ledger.json` untuk bukti aspek keamanan blockchain
- Narasi yang bisa dipakai:
  - centralized = benchmark performa maksimum
  - FL+Blockchain = kompromi performa demi privasi, kolaborasi, dan ketahanan terhadap update berbahaya
