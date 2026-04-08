"""
================================================================================
SISTEM PREDIKSI DIABETES DENGAN FEDERATED LEARNING DAN BLOCKCHAIN
================================================================================

File: 02_centralized_ml.py
Deskripsi: Modul Machine Learning Terpusat (Centralized ML)
             - Melatih 4 model ML: Logistic Regression, Random Forest, KNN, Gradient Boosting
             - Evaluasi menggunakan berbagai metrik (Accuracy, Precision, Recall, F1, AUC)
             - Confusion Matrix visualization
             - Menyimpan model dan hasil

Author: Sistem ML Skripsi
Tanggal: 2024
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================================
# KONFIGURASI
# ============================================================================

# Path ke data yang sudah di-preprocess
BASE_DIR = r"d:\Kampus\Semester 6\Skripsi\Keperluan Skripsi"
OUTPUT_DIR = os.path.join(BASE_DIR, 'Data-set', 'output')
RANDOM_STATE = 42

# Buat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)


def muat_data():
    """
    Memuat data yang sudah di-preprocess dari file .npy.
    
    Returns:
    --------
    dict
        Dictionary berisi X_train, X_test, y_train, y_test
        
    Notes:
    ------
    Data dimuat dari file yang dibuat oleh 01_preprocessing.py
    Jika file tidak ditemukan, akan dicoba menjalankan preprocessing terlebih dahulu.
    """
    print("=" * 70)
    print("MEMUAT DATA PREPROCESSED")
    print("=" * 70)
    
    # Cek apakah file ada
    files = ['X_train.npy', 'X_test.npy', 'y_train.npy', 'y_test.npy']
    for f in files:
        path = os.path.join(OUTPUT_DIR, f)
        if not os.path.exists(path):
            print(f"[WARNING] File {f} tidak ditemukan!")
            print("          Menjalankan preprocessing terlebih dahulu...")
            import subprocess
            subprocess.run(['python', '01_preprocessing.py'])
            return muat_data()
    
    # Load data
    X_train = np.load(os.path.join(OUTPUT_DIR, 'X_train.npy'))
    X_test = np.load(os.path.join(OUTPUT_DIR, 'X_test.npy'))
    y_train = np.load(os.path.join(OUTPUT_DIR, 'y_train.npy'))
    y_test = np.load(os.path.join(OUTPUT_DIR, 'y_test.npy'))
    
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


def buat_model():
    """
    Membuat dictionary berisi model-model yang akan dilatih.
    
    Returns:
    --------
    dict
        Dictionary dengan nama model sebagai key dan objek model sebagai value
        
    Notes:
    ------
    Model yang digunakan:
    1. Logistic Regression: Model linear sederhana, baik untuk baseline
    2. Random Forest: Ensemble method berbasis decision tree
    3. KNN (K-Nearest Neighbors): Model berbasis jarak
    4. Gradient Boosting: Ensemble method sequential
    
    Setiap model dikonfigurasi dengan parameter yang sudah di-tune
    untuk dataset diabetes.
    """
    print("\n" + "=" * 70)
    print("MEMBUAT MODEL ML")
    print("=" * 70)
    
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,           # Maksimum iterasi untuk konvergensi
            random_state=RANDOM_STATE,
            solver='lbfgs'           # Solver untuk optimasi
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,        # Jumlah pohon dalam forest
            max_depth=16,            # Kedalaman maksimum pohon
            random_state=RANDOM_STATE,
            n_jobs=-1                # Gunakan semua core CPU
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=10,          # Jumlah tetangga terdekat
            metric='euclidean'       # Jarak Euclidean
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,        # Jumlah boosting stage
            learning_rate=0.1,       # Learning rate
            max_depth=5,             # Kedalaman maksimum
            random_state=RANDOM_STATE
        )
    }
    
    print("Model yang akan dilatih:")
    for nama in models.keys():
        print(f"  - {nama}")
    
    return models


def latih_model(models, X_train, y_train):
    """
    Melatih semua model pada data training.
    
    Parameters:
    -----------
    models : dict
        Dictionary berisi objek model
    X_train : np.array
        Fitur training
    y_train : np.array
        Label training
        
    Returns:
    --------
    dict
        Dictionary berisi model yang sudah dilatih
        
    Notes:
    ------
    Proses training:
    1. Iterasi melalui setiap model
    2. Fit model pada data training
    3. Hitung akurasi pada training data
    4. Tampilkan progress
    
    Waktu training tergantung pada kompleksitas model:
    - Logistic Regression: cepat
    - KNN: cepat (hanya menyimpan data)
    - Random Forest: sedang
    - Gradient Boosting: lebih lambat
    """
    print("\n" + "=" * 70)
    print(" MELATIH MODEL")
    print("=" * 70)
    
    models_latih = {}
    
    for nama, model in models.items():
        print(f"\nMelatih {nama}...")
        
        # Latih model
        model.fit(X_train, y_train)
        
        # Hitung training accuracy
        y_pred_train = model.predict(X_train)
        train_acc = accuracy_score(y_train, y_pred_train)
        
        print(f"  Training Accuracy: {train_acc:.4f}")
        
        models_latih[nama] = model
    
    print("\n[INFO] Semua model berhasil dilatih!")
    
    return models_latih


def evaluasi_model(models, X_test, y_test):
    """
    Mengevaluasi semua model pada data testing.
    
    Parameters:
    -----------
    models : dict
        Dictionary berisi model yang sudah dilatih
    X_test : np.array
        Fitur testing
    y_test : np.array
        Label testing
        
    Returns:
    --------
    pd.DataFrame
        DataFrame berisi hasil evaluasi semua model
        
    Notes:
    ------
    Metrik evaluasi yang digunakan:
    1. Accuracy: Proporsi prediksi yang benar
    2. Precision: Proporsi prediksi positif yang benar
    3. Recall: Proporsi data positif yang diprediksi benar
    4. F1-Score: Harmonic mean dari Precision dan Recall
    5. AUC-ROC: Area under ROC Curve
    
    Classification report dan confusion matrix juga ditampilkan
    untuk analisis lebih detail.
    """
    print("\n" + "=" * 70)
    print("EVALUASI MODEL")
    print("=" * 70)
    
    hasil = {}
    
    for nama, model in models.items():
        print(f"\n{'='*50}")
        print(f"Model: {nama}")
        print('='*50)
        
        # Prediksi
        y_pred = model.predict(X_test)
        
        # Probabilitas prediksi (untuk AUC-ROC)
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred
        
        # Hitung metrik
        metrik = {
            'Accuracy': round(accuracy_score(y_test, y_pred), 4),
            'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
            'Recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
            'F1-Score': round(f1_score(y_test, y_pred, zero_division=0), 4),
            'AUC-ROC': round(roc_auc_score(y_test, y_prob), 4)
        }
        
        hasil[nama] = metrik
        
        # Tampilkan classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                    target_names=['Non-Diabetes', 'Diabetes']))
        
        # Tampilkan confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)
        
        # Visualisasi confusion matrix
        plt.figure(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=['Non-Diabetes', 'Diabetes']
        )
        disp.plot(cmap='Blues', values_format='d')
        plt.title(f'Confusion Matrix - {nama}')
        plt.tight_layout()
        
        # Simpan gambar
        filename = f"cm_{nama.replace(' ', '_')}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
        plt.close()
        print(f"\n[INFO] Confusion matrix disimpan: {filename}")
    
    # Buat DataFrame hasil
    df_hasil = pd.DataFrame(hasil).T
    
    print("\n" + "=" * 70)
    print("RINGKASAN HASIL EVALUASI")
    print("=" * 70)
    print(df_hasil.to_string())
    
    return df_hasil


def plot_perbandingan_model(df_hasil):
    """
    Membuat visualisasi perbandingan performa model.
    
    Parameters:
    -----------
    df_hasil : pd.DataFrame
        DataFrame berisi hasil evaluasi
        
    Notes:
    ------
    Visualisasi yang dibuat:
    1. Bar chart perbandingan metrik antar model
    2. Heatmap korelasi antar metrik
    """
    print("\n" + "=" * 70)
    print("MEMBUAT VISUALISASI PERBANDINGAN")
    print("=" * 70)
    
    # Plot 1: Bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrik = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    x = np.arange(len(metrik))
    width = 0.2
    
    warna = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
    
    for i, (nama, row) in enumerate(df_hasil.iterrows()):
        values = [row[m] for m in metrik]
        ax.bar(x + i*width, values, width, label=nama, color=warna[i], alpha=0.8)
    
    ax.set_xlabel('Metrik')
    ax.set_ylabel('Nilai')
    ax.set_title('Perbandingan Performa Model ML - Centralized')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metrik)
    ax.legend(loc='lower right')
    ax.set_ylim(0.5, 1.05)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'centralized_comparison.png'), dpi=150)
    plt.close()
    
    print("[INFO] Visualisasi perbandingan disimpan: centralized_comparison.png")
    
    # Plot 2: Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_hasil, annot=True, cmap='YlGnBu', fmt='.3f',
                linewidths=0.5, cbar_kws={'label': 'Nilai'})
    plt.title('Heatmap Perbandingan Model ML')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'centralized_heatmap.png'), dpi=150)
    plt.close()
    
    print("[INFO] Heatmap disimpan: centralized_heatmap.png")


def jalankan_centralized_ml():
    """
    Fungsi utama untuk menjalankan ML terpusat.
    
    Returns:
    --------
    dict
        Dictionary berisi model yang sudah dilatih dan hasil evaluasi
        
    Notes:
    ------
    Urutan proses:
    1. Muat data dari file .npy
    2. Buat model yang akan digunakan
    3. Latih semua model
    4. Evaluasi semua model
    5. Simpan hasil ke CSV
    6. Buat visualisasi
    """
    print("\n" + "=" * 70)
    print("CENTRALIZED MACHINE LEARNING")
    print("=" * 70)
    print("Modul ini melatih model ML secara terpusat (centralized)")
    print("Semua data dikumpulkan di satu tempat untuk training.")
    
    # 1. Muat data
    data = muat_data()
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    # 2. Buat model
    models = buat_model()
    
    # 3. Latih model
    models_latih = latih_model(models, X_train, y_train)
    
    # 4. Evaluasi model
    df_hasil = evaluasi_model(models_latih, X_test, y_test)
    
    # 5. Simpan hasil ke CSV
    df_hasil.to_csv(os.path.join(OUTPUT_DIR, 'centralized_results.csv'))
    print(f"\n[INFO] Hasil disimpan ke: centralized_results.csv")
    
    # 6. Visualisasi
    plot_perbandingan_model(df_hasil)
    
    # Tentukan model terbaik
    best_model = df_hasil['F1-Score'].idxmax()
    print(f"\n{'='*70}")
    print(f"MODEL TERBAIK: {best_model}")
    print(f"{'='*70}")
    print(df_hasil.loc[best_model].to_string())
    
    return {
        'models': models_latih,
        'results': df_hasil,
        'best_model': best_model
    }


# ============================================================================
# EKSEKUSI UTAMA
# ============================================================================

if __name__ == "__main__":
    print(__doc__)
    hasil = jalankan_centralized_ml()
    print("\n" + "=" * 70)
    print("CENTRALIZED ML SELESAI!")
    print("=" * 70)
    print("\nHasil dapat dibandingkan dengan:")
    print("  - 03_federated_learning.py (Federated Learning)")
    print("  - 04_blockchain_security.py (Blockchain + FL)")
