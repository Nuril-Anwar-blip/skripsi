"""
================================================================================
SISTEM PREDIKSI DIABETES DENGAN FEDERATED LEARNING DAN BLOCKCHAIN
================================================================================

File: main.py
Deskripsi: File utama untuk menjalankan semua modul secara berurutan
             - Preprocessing data
             - Centralized ML (baseline)
             - Federated Learning
             - Blockchain + FL
             - Perbandingan hasil

Author: Sistem ML Skripsi
Tanggal: 2024
================================================================================

Cara Penggunaan:
    python main.py
    
Atau jalankan satu per satu:
    python 01_preprocessing.py
    python 02_centralized_ml.py
    python 03_federated_learning.py
    python 04_blockchain_security.py
================================================================================
"""

import os
import sys

# ============================================================================
# KONFIGURASI
# ============================================================================

# Folder output
BASE_DIR = r"d:\Kampus\Semester 6\Skripsi\Keperluan Skripsi"
OUTPUT_DIR = os.path.join(BASE_DIR, 'Data-set', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def print_header():
    """Mencetak header program."""
    print(""")
    ==============================================================================
    SISTEM PREDIKSI DIABETES DENGAN
    FEDERATED LEARNING DAN BLOCKCHAIN
    
    Skripsi - Machine Learning Terdistribusi dengan Keamanan
    ==============================================================================
    """)


def jalankan_preprocessing():
    """Menjalankan modul preprocessing."""
    print("\n" + "=" * 70)
    print("BAGIAN 1: PREPROCESSING DATA")
    print("=" * 70)
    print("""
    Pada bagian ini akan dilakukan:
    1. Load dataset diabetes dari folder Data-set
    2. Hapus duplikat
    3. Hapus outlier dengan metode IQR
    4. Encoding variabel kategorikal
    5. Oversampling untuk data tidak seimbang
    6. Standarisasi fitur numerik
    7. Simpan data untuk digunakan di modul lain
    """)
    
    from preprocessing import jalankan_preprocessing
    data = jalankan_preprocessing()
    
    print("\n[BERHASIL] Preprocessing selesai!")
    return data


def jalankan_centralized_ml():
    """Menjalankan modul Centralized ML."""
    print("\n" + "=" * 70)
    print("BAGIAN 2: CENTRALIZED MACHINE LEARNING")
    print("=" * 70)
    print("""
    Pada bagian ini akan dilakukan:
    1. Load data yang sudah di-preprocess
    2. Latih 4 model ML:
       - Logistic Regression
       - Random Forest
       - K-Nearest Neighbors (KNN)
       - Gradient Boosting
    3. Evaluasi dengan metrik:
       - Accuracy, Precision, Recall, F1-Score, AUC-ROC
    4. Visualisasi confusion matrix
    5. Simpan hasil ke CSV
    """)
    
    from centralized_ml import jalankan_centralized_ml
    hasil = jalankan_centralized_ml()
    
    print("\n[BERHASIL] Centralized ML selesai!")
    return hasil


def jalankan_federated_learning():
    """Menjalankan modul Federated Learning."""
    print("\n" + "=" * 70)
    print("BAGIAN 3: FEDERATED LEARNING")
    print("=" * 70)
    print("""
    Pada bagian ini akan dilakukan:
    1. Simulasi Federated Learning dengan multiple clients
    2. Implementasi Federated Averaging (FedAvg)
    3. Tiga skenario:
       - IID dengan 5 klien
       - Non-IID dengan 5 klien
       - IID dengan 10 klien
    4. Analisis konvergensi model
    5. Simpan hasil ke CSV
    
    Catatan: FL menjaga privasi data pasien karena hanya
    bobot model yang dikirim, bukan data asli.
    """)
    
    from federated_learning import jalankan_semua_skenario
    hasil = jalankan_semua_skenario()
    
    print("\n[BERHASIL] Federated Learning selesai!")
    return hasil


def jalankan_blockchain_fl():
    """Menjalankan modul Blockchain + FL."""
    print("\n" + "=" * 70)
    print("BAGIAN 4: BLOCKCHAIN + FEDERATED LEARNING")
    print("=" * 70)
    print("""
    Pada bagian ini akan dilakukan:
    1. Implementasi blockchain untuk integritas data
    2. Verifikasi hash parameter model
    3. Deteksi poisoning attack
    4. Tiga skenario:
       - Normal (tanpa serangan)
       - 1 klien jahat
       - 2 klien jahat
    5. Audit trail untuk setiap ronde
    6. Simpan ledger ke JSON
    
    Catatan: Blockchain memastikan tidak ada manipulasi
    parameter model oleh klien yang tidak jujur.
    """)
    
    from blockchain_security import jalankan_semua_skenario_bc
    hasil = jalankan_semua_skenario_bc()
    
    print("\n[BERHASIL] Blockchain + FL selesai!")
    return hasil


def buat_perbandingan_akhir():
    """Membuat perbandingan akhir semua pendekatan."""
    print("\n" + "=" * 70)
    print("BAGIAN 5: PERBANDINGAN AKHIR")
    print("=" * 70)
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Load semua hasil
    try:
        df_central = pd.read_csv(os.path.join(OUTPUT_DIR, 'centralized_results.csv'), index_col=0)
    except:
        print("[WARNING] Hasil centralized tidak ditemukan")
        return
    
    try:
        df_fl_iid5 = pd.read_csv(os.path.join(OUTPUT_DIR, 'fl_iid5.csv'))
        df_fl_noniid = pd.read_csv(os.path.join(OUTPUT_DIR, 'fl_noniid.csv'))
        df_fl_iid10 = pd.read_csv(os.path.join(OUTPUT_DIR, 'fl_iid10.csv'))
    except:
        print("[WARNING] Hasil FL tidak ditemukan")
        return
    
    try:
        df_bc_normal = pd.read_csv(os.path.join(OUTPUT_DIR, 'bc_normal.csv'))
        df_bc_1jahat = pd.read_csv(os.path.join(OUTPUT_DIR, 'bc_1_jahat.csv'))
        df_bc_2jahat = pd.read_csv(os.path.join(OUTPUT_DIR, 'bc_2_jahat.csv'))
    except:
        print("[WARNING] Hasil blockchain tidak ditemukan")
        return
    
    # Bangun DataFrame perbandingan
    rows = []
    
    # Centralized
    for nama, row in df_central.iterrows():
        rows.append({
            'Pendekatan': 'Centralized',
            'Skenario': nama,
            'Accuracy': row['Accuracy'],
            'Precision': row['Precision'],
            'Recall': row['Recall'],
            'F1-Score': row['F1-Score'],
            'AUC-ROC': row['AUC-ROC']
        })
    
    # FL
    for nama, df in [('FL IID 5K', df_fl_iid5), 
                     ('FL Non-IID 5K', df_fl_noniid), 
                     ('FL IID 10K', df_fl_iid10)]:
        r = df.iloc[-1]
        rows.append({
            'Pendekatan': 'FL',
            'Skenario': nama,
            'Accuracy': r['Accuracy'],
            'Precision': r['Precision'],
            'Recall': r['Recall'],
            'F1-Score': r['F1-Score'],
            'AUC-ROC': r['AUC-ROC']
        })
    
    # FL + Blockchain
    for nama, df in [('Normal', df_bc_normal), 
                     ('1 Jahat', df_bc_1jahat), 
                     ('2 Jahat', df_bc_2jahat)]:
        r = df.iloc[-1]
        rows.append({
            'Pendekatan': 'FL+Blockchain',
            'Skenario': nama,
            'Accuracy': r['Accuracy'],
            'Precision': r['Precision'],
            'Recall': r['Recall'],
            'F1-Score': r['F1-Score'],
            'AUC-ROC': r['AUC-ROC']
        })
    
    df_final = pd.DataFrame(rows)
    
    print("\n" + "=" * 70)
    print("TABEL PERBANDINGAN AKHIR")
    print("=" * 70)
    print(df_final.to_string(index=False))
    
    # Simpan ke CSV
    df_final.to_csv(os.path.join(OUTPUT_DIR, 'HASIL_LENGKAP.csv'), index=False)
    print(f"\n[INFO] Hasil disimpan ke: HASIL_LENGKAP.csv")
    
    # Plot perbandingan
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Perbandingan Akhir - Semua Pendekatan', fontsize=13)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    x = np.arange(len(metrics))
    
    # Plot 1: Centralized vs FL
    ax = axes[0]
    sub = df_final[df_final['Pendekatan'].isin(['Centralized', 'FL'])]
    colors = ['#2196F3', '#7E57C2', '#43A047', '#FF5722', '#26C6DA', '#EC407A']
    w = 0.12
    
    for i, (_, row) in enumerate(sub.iterrows()):
        ax.bar(x + i*w, [row[m] for m in metrics], w,
               label=f"{row['Pendekatan']} | {row['Skenario']}",
               color=colors[i % len(colors)], alpha=0.85)
    
    ax.set_xticks(x + w * 2.5)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_title('Centralized vs FL')
    
    # Plot 2: FL + Blockchain
    ax = axes[1]
    sub = df_final[df_final['Pendekatan'] == 'FL+Blockchain']
    colors = ['#43A047', '#E53935', '#FF9800']
    
    for i, (_, row) in enumerate(sub.iterrows()):
        ax.bar(x + i*w, [row[m] for m in metrics], w,
               label=f"{row['Pendekatan']} | {row['Skenario']}",
               color=colors[i], alpha=0.85)
    
    ax.set_xticks(x + w)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_title('FL + Blockchain')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'perbandingan_akhir.png'), dpi=150)
    plt.close()
    
    print("[INFO] Plot perbandingan disimpan: perbandingan_akhir.png")
    
    return df_final


def main():
    """Fungsi utama untuk menjalankan semua modul."""
    print_header()
    
    # Langsung jalankan semua modul (mode auto)
    print("\n[MODE AUTO] Menjalankan semua modul...")
    print("[INFO] Pastikan dataset ada di folder Data-set/")
    
    try:
        jalankan_preprocessing()
        jalankan_centralized_ml()
        jalankan_federated_learning()
        jalankan_blockchain_fl()
        buat_perbandingan_akhir()
        
        print("\n" + "=" * 70)
        print("SEMUA MODUL SELESAI!")
        print("=" * 70)
        print("""
        Hasil tersimpan di folder Data-set/output/:
        - centralized_results.csv
        - fl_iid5.csv, fl_noniid.csv, fl_iid10.csv
        - bc_normal.csv, bc_1_jahat.csv, bc_2_jahat.csv
        - ledger_*.json
        - HASIL_LENGKAP.csv
        - *.png (visualisasi)
        """)
        
    except Exception as e:
        print(f"\n[ERROR] Terjadi kesalahan: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# EKSEKUSI UTAMA
# ============================================================================

if __name__ == "__main__":
    main()
