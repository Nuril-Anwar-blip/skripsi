"""
================================================================================
SISTEM PREDIKSI DIABETES TIPE 2 DENGAN FEDERATED LEARNING DAN BLOCKCHAIN
================================================================================

Deskripsi Utama:
Sistem ini mengimplementasikan machine learning dengan federated learning yang 
dikombinasikan dengan blockchain untuk keamanan dan keberlanjutan (sustainability).
Sistem ini dirancang khusus untuk prediksi diabetes tipe 2 dengan keunggulan:
- Data pasien tidak dikirim ke server pusat (privasi terjamin)
- Proses training terjadi di dalam device pasien sendiri
- Blockchain memastikan integritas dan transparansi model update
- Kompleks dan fungsional untuk semua kalangan pasien diabetes tipe 2

Fitur Utama:
1. Preprocessing data diabetes (cleaning, encoding, oversampling)
2. Federated Learning dengan multiple clients (device)
3. Blockchain security untuk integritas parameter model
4. Deteksi serangan poisoning attack
5. Visualisasi komprehensif dengan grafik dan heatmap

Author: Sistem ML Skripsi
Tanggal: 2024/2025
================================================================================
"""

import os
import sys

# Path konfigurasi
BASE_DIR = r"d:\Kampus\Semester 6\Skripsi\Keperluan Skripsi"
PROJECT_NAME = "FL_Blockchain_Diabetes_Type2"
PROJECT_DIR = os.path.join(BASE_DIR, PROJECT_NAME)

# Buat direktori project
os.makedirs(PROJECT_DIR, exist_ok=True)
os.makedirs(os.path.join(PROJECT_DIR, "terminal"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_DIR, "terminal", "output"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_DIR, "notebook"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_DIR, "data"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_DIR, "img"), exist_ok=True)

print(f"Project folder dibuat: {PROJECT_DIR}")
print("Struktur folder:")
print(f"  - {PROJECT_DIR}/terminal/")
print(f"  - {PROJECT_DIR}/terminal/output/")
print(f"  - {PROJECT_DIR}/notebook/")
print(f"  - {PROJECT_DIR}/data/")
print(f"  - {PROJECT_DIR}/img/")