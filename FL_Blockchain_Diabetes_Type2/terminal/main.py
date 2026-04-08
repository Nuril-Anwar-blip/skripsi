"""
================================================================================
SISTEM PREDIKSI DIABETES TIPE 2 DENGAN FEDERATED LEARNING DAN BLOCKCHAIN
================================================================================

File: main.py
Deskripsi: Main orchestration script untuk menjalankan seluruh sistem
             - Preprocessing data dari MULTIPLE DATASETS
             - Centralized ML (baseline comparison)
             - Federated Learning
             - Blockchain Security
             - Visualization and reporting

Author: Sistem ML Skripsi
Tanggal: 2024/2025
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import modul
from multi_dataset_preprocessing import load_all_datasets, eksplorasi_semua_dataset

# ============================================================================
# KONFIGURASI
# ============================================================================

BASE_DIR = r"d:\Kampus\Semester 6\Skripsi\Keperluan Skripsi"
OUTPUT_DIR = os.path.join(BASE_DIR, 'FL_Blockchain_Diabetes_Type2', 'terminal', 'output')
IMG_DIR = os.path.join(BASE_DIR, 'FL_Blockchain_Diabetes_Type2', 'img')


def buat_laporan_perbandingan():
    """
    Membuat laporan perbandingan akhir dari semua pendekatan.
    """
    print("\n" + "=" * 80)
    print("MEMBUAT LAPORAN PERBANDINGAN AKHIR")
    print("=" * 80)
    
    # Load semua hasil
    try:
        centralized = pd.read_csv(os.path.join(OUTPUT_DIR, 'centralized_results.csv'))
        fl_results = pd.read_csv(os.path.join(OUTPUT_DIR, 'fl_results.csv'))
        bc_results = pd.read_csv(os.path.join(OUTPUT_DIR, 'bc_results.csv'))
    except:
        print("[WARNING] File hasil tidak ditemukan. Pastikan semua modul sudah dijalankan.")
        return
    
    # Gabungkan semua hasil
    all_results = []
    
    # Centralized results
    for _, row in centralized.iterrows():
        all_results.append({
            'Pendekatan': 'Centralized',
            'Skenario': row['Model'],
            'Accuracy': row['Accuracy'],
            'Precision': row['Precision'],
            'Recall': row['Recall'],
            'F1-Score': row['F1-Score'],
            'AUC-ROC': row['AUC-ROC']
        })
    
    # FL results
    for _, row in fl_results.iterrows():
        all_results.append({
            'Pendekatan': 'FL',
            'Skenario': row['skenario'],
            'Accuracy': row['Accuracy'],
            'Precision': row['Precision'],
            'Recall': row['Recall'],
            'F1-Score': row['F1-Score'],
            'AUC-ROC': row['AUC-ROC']
        })
    
    # Blockchain results
    for _, row in bc_results.iterrows():
        all_results.append({
            'Pendekatan': 'FL+Blockchain',
            'Skenario': row['skenario'],
            'Accuracy': row['Accuracy'],
            'Precision': row['Precision'],
            'Recall': row['Recall'],
            'F1-Score': row['F1-Score'],
            'AUC-ROC': row['AUC-ROC']
        })
    
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(os.path.join(OUTPUT_DIR, 'HASIL_LENGKAP.csv'), index=False)
    
    print("\n[INFO] Hasil Lengkap:")
    print(df_all.to_string(index=False))
    
    # Visualisasi perbandingan akhir
    visualisasi_perbandingan_akhir(df_all)
    
    return df_all


def visualisasi_perbandingan_akhir(df_all, save_path=None):
    """
    Membuat visualisasi perbandingan akhir semua pendekatan.
    """
    if save_path is None:
        save_path = os.path.join(IMG_DIR, 'perbandingan_akhir.png')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Group by approach
    approaches = df_all['Pendekatan'].unique()
    metrics = ['Accuracy', 'F1-Score']
    
    # Plot 1: Accuracy comparison
    ax = axes[0]
    approach_data = df_all.groupby('Pendekatan')['Accuracy'].mean()
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars = ax.bar(approach_data.index, approach_data.values, color=colors, edgecolor='black')
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Rata-rata Accuracy per Pendekatan', fontsize=14, fontweight='bold')
    ax.set_ylim(0.8, 1.0)
    for bar, val in zip(bars, approach_data.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: F1-Score comparison
    ax = axes[1]
    approach_data = df_all.groupby('Pendekatan')['F1-Score'].mean()
    bars = ax.bar(approach_data.index, approach_data.values, color=colors, edgecolor='black')
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('Rata-rata F1-Score per Pendekatan', fontsize=14, fontweight='bold')
    ax.set_ylim(0.4, 0.9)
    for bar, val in zip(bars, approach_data.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Perbandingan Akhir: Centralized vs FL vs FL+Blockchain', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n[SAVE] Grafik perbandingan akhir disimpan: {save_path}")


def jalankan_sistem():
    """
    Menjalankan seluruh sistem secara berurutan.
    Menggunakan SEMUA 5 DATASET yang tersedia.
    """
    print("\n" + "=" * 80)
    print("=" * 80)
    print("SISTEM PREDIKSI DIABETES TIPE 2")
    print("Federated Learning + Blockchain + Privacy-Preserving")
    print("=".join(["=" * 80]))
    print("\nKEUNGGULAN SISTEM:")
    print("1. Data pasien TIDAK PERNAH dikirim ke server pusat")
    print("2. Training terjadi di dalam device pasien (on-device)")
    print("3. Blockchain memastikan integritas model update")
    print("4. Deteksi otomatis serangan poisoning attack")
    print("5. Transparansi dan audit trail untuk setiap update")
    print("=" * 80)
    print("\nMENGGUNAKAN 5 DATASET BERBEDA:")
    print("  1. diabetes_prediction_dataset.csv (100k)")
    print("  2. diabetes_012_health_indicators_BRFSS2015.csv (253k)")
    print("  3. diabetes_binary_health_indicators_BRFSS2015.csv (253k)")
    print("  4. diabetes_binary_5050split_health_indicators_BRFSS2015.csv (70k)")
    print("  5. diabetes.csv (768)")
    print("  TOTAL: ~678rb record!")
    print("=" * 80)
    
    # Langkah 1: Load dan eksplorasi SEMUA dataset
    print("\n\n" + "=" * 80)
    print("LANGKAH 1: LOAD DAN EKSPLORASI SEMUA DATASET")
    print("=" * 80)
    datasets = load_all_datasets()
    eksplorasi_semua_dataset(datasets)
    
    # Langkah 2: Preprocessing untuk dataset utama
    print("\n\n" + "=" * 80)
    print("LANGKAH 2: PREPROCESSING SEMUA DATASET")
    print("=" * 80)
    from multi_dataset_preprocessing import jalankan_multi_dataset_preprocessing
    jalankan_multi_dataset_preprocessing()
    
    # Langkah 3: Centralized ML (baseline)
    print("\n\n" + "=" * 80)
    print("LANGKAH 3: CENTRALIZED ML (BASELINE)")
    print("=" * 80)
    from centralized_ml import jalankan_centralized_ml
    jalankan_centralized_ml()
    
    # Langkah 4: Federated Learning
    print("\n\n" + "=" * 80)
    print("LANGKAH 4: FEDERATED LEARNING")
    print("=" * 80)
    from federated_learning import jalankan_semua_skenario_fl
    jalankan_semua_skenario_fl()
    
    # Langkah 5: Blockchain Security
    print("\n\n" + "=" * 80)
    print("LANGKAH 5: BLOCKCHAIN SECURITY")
    print("=" * 80)
    from blockchain_security import jalankan_skenario_blockchain
    jalankan_skenario_blockchain()
    
    # Langkah 6: Laporan Akhir
    print("\n\n" + "=" * 80)
    print("LANGKAH 6: LAPORAN AKHIR")
    print("=" * 80)
    df_all = buat_laporan_perbandingan()
    
    print("\n" + "=" * 80)
    print("SEMUA PROSES SELESAI!")
    print("=" * 80)
    print("\nOUTPUT TERSIMPAN DI:")
    print(f"  - Data: {OUTPUT_DIR}")
    print(f"  - Gambar: {IMG_DIR}")
    print("\nFILE OUTPUT:")
    print("  - X_train.npy, X_test.npy, y_train.npy, y_test.npy")
    print("  - centralized_results.csv")
    print("  - fl_results.csv, fl_iid5.csv, fl_noniid.csv, fl_iid10.csv")
    print("  - bc_results.csv, ledger_normal.json, ledger_1_jahat.json, ledger_2_jahat.json")
    print("  - HASIL_LENGKAP.csv")
    print("\nGAMBAR OUTPUT:")
    print("  - perbandingan_semua_dataset.png")
    print("  - distribusi_diabetes.png")
    print("  - correlation_matrix.png")
    print("  - cm_*.png, roc_curve_comparison.png, feature_importance.png")
    print("  - centralized_comparison.png")
    print("  - fl_konvergensi_*.png")
    print("  - bc_attack_detection.png")
    print("  - perbandingan_akhir.png")
    
    return df_all


if __name__ == "__main__":
    results = jalankan_sistem()