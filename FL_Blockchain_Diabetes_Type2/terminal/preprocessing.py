"""
================================================================================
SISTEM PREDIKSI DIABETES TIPE 2 DENGAN FEDERATED LEARNING DAN BLOCKCHAIN
================================================================================

File: preprocessing.py
Deskripsi: Modul preprocessing data untuk prediksi diabetes tipe 2
             - Load dataset dari folder Data-set
             - Cleaning data (hapus duplikat, outlier)
             - Encoding variabel kategorikal
             - Oversampling untuk menangani data tidak seimbang
             - Standarisasi fitur numerik
             - Feature engineering untuk diabetes

Author: Sistem ML Skripsi
Tanggal: 2024/2025
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import resample
from scipy import stats

# ============================================================================
# KONFIGURASI DAN PATH
# ============================================================================

BASE_DIR = r"d:\Kampus\Semester 6\Skripsi\Keperluan Skripsi"
DATA_PATH = os.path.join(BASE_DIR, 'Data-set', 'diabetes_prediction_dataset.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'FL_Blockchain_Diabetes_Type2', 'terminal', 'output')
IMG_DIR = os.path.join(BASE_DIR, 'FL_Blockchain_Diabetes_Type2', 'img')

# Konfigurasi preprocessing
RANDOM_STATE = 42
TEST_SIZE = 0.10
OVERSEED = 42

# Buat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)


def load_dataset(path):
    """
    Memuat dataset diabetes dari file CSV.
    
    Parameters:
    -----------
    path : str
        Path lengkap ke file CSV dataset
        
    Returns:
    --------
    pd.DataFrame
        DataFrame berisi dataset diabetes
    """
    print("=" * 80)
    print("LOAD DATASET - PREDIKSI DIABETES TIPE 2")
    print("=" * 80)
    print(f"Path: {path}")
    
    df = pd.read_csv(path)
    
    print(f"\n[INFO] Dataset berhasil dimuat!")
    print(f"       - Jumlah baris: {df.shape[0]:,}")
    print(f"       - Jumlah kolom: {df.shape[1]}")
    print(f"       - Missing value: {df.isnull().sum().sum()}")
    
    return df


def eksplorasi_data(df):
    """
    Melakukan eksplorasi awal terhadap dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame dataset
        
    Returns:
    --------
    dict
        Dictionary berisi statistik dan informasi dataset
    """
    print("\n" + "=" * 80)
    print("EKSPLORASI DATA AWAL")
    print("=" * 80)
    
    # Info dataset
    print("\n[INFO] Struktur Dataset:")
    print(df.info())
    
    # Statistik deskriptif
    print("\n[INFO] Statistik Deskriptif:")
    print(df.describe())
    
    # Distribusi target
    print("\n[INFO] Distribusi Diabetes:")
    print(df['diabetes'].value_counts())
    print(f"   - Persentase Diabetes: {df['diabetes'].mean()*100:.2f}%")
    
    # Missing values
    print("\n[INFO] Missing Values:")
    print(df.isnull().sum())
    
    # Visualisasi distribusi target
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pie chart
    colors = ['#3498db', '#e74c3c']
    axes[0].pie(df['diabetes'].value_counts(), 
                labels=['Non-Diabetes', 'Diabetes'],
                autopct='%1.1f%%', colors=colors, explode=(0, 0.05))
    axes[0].set_title('Distribusi Diabetes dalam Dataset', fontsize=14, fontweight='bold')
    
    # Count plot
    sns.countplot(x='diabetes', data=df, ax=axes[1], palette=colors)
    axes[1].set_title('Jumlah Pasien Diabetes vs Non-Diabetes', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Diabetes (0=No, 1=Yes)')
    axes[1].set_ylabel('Jumlah')
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'distribusi_diabetes.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[SAVE] Grafik disimpan: {IMG_DIR}/distribusi_diabetes.png")
    
    return {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing': df.isnull().sum().to_dict()
    }


def bersihkan_duplikat(df):
    """
    Menghapus baris duplikat dari dataset.
    """
    print("\n" + "=" * 80)
    print("CLEANING: HAPUS DUPLIKAT")
    print("=" * 80)
    
    jumlah_sebelum = len(df)
    df_clean = df.drop_duplicates()
    jumlah_setelah = len(df_clean)
    
    print(f"Sebelum: {jumlah_sebelum:,} baris")
    print(f"Setelah: {jumlah_setelah:,} baris")
    print(f"Dihapus: {jumlah_sebelum - jumlah_setelah:,} duplikat")
    
    return df_clean


def hapus_outlier_iqr(df, kolom_numerik):
    """
    Menghapus outlier menggunakan metode Interquartile Range (IQR).
    """
    print("\n" + "=" * 80)
    print("CLEANING: HAPUS OUTLIER DENGAN IQR")
    print("=" * 80)
    print(f"Kolom yang dicek: {kolom_numerik}")
    
    def filter_iqr_per_kelas(data, cols):
        Q1 = data[cols].quantile(0.25)
        Q3 = data[cols].quantile(0.75)
        IQR = Q3 - Q1
        
        batas_bawah = Q1 - 1.5 * IQR
        batas_atas = Q3 + 1.5 * IQR
        
        mask = ~((data[cols] < batas_bawah) | (data[cols] > batas_atas)).any(axis=1)
        
        return data[mask]
    
    jumlah_sebelum = len(df)
    
    # Filter outlier untuk setiap kelas secara terpisah
    df_diabetes = df[df['diabetes'] == 1].copy()
    df_non_diabetes = df[df['diabetes'] == 0].copy()
    
    df_diabetes_clean = filter_iqr_per_kelas(df_diabetes, kolom_numerik)
    df_non_diabetes_clean = filter_iqr_per_kelas(df_non_diabetes, kolom_numerik)
    
    df_clean = pd.concat([df_diabetes_clean, df_non_diabetes_clean]).reset_index(drop=True)
    
    jumlah_setelah = len(df_clean)
    
    print(f"Sebelum: {jumlah_sebelum:,} baris")
    print(f"Setelah: {jumlah_setelah:,} baris")
    print(f"Dihapus: {jumlah_sebelum - jumlah_setelah:,} outlier")
    print(f"  - Diabetes: {len(df_diabetes):,} -> {len(df_diabetes_clean):,}")
    print(f"  - Non-Diabetes: {len(df_non_diabetes):,} -> {len(df_non_diabetes_clean):,}")
    
    return df_clean


def feature_engineering(df):
    """
    Melakukan feature engineering untuk meningkatkan prediksi diabetes.
    """
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING")
    print("=" * 80)
    
    # BMI Category
    def categorize_bmi(bmi):
        if bmi < 18.5:
            return 0  # Underweight
        elif bmi < 25:
            return 1  # Normal
        elif bmi < 30:
            return 2  # Overweight
        else:
            return 3  # Obese
    
    df['bmi_category'] = df['bmi'].apply(categorize_bmi)
    
    # Age Category
    def categorize_age(age):
        if age < 30:
            return 0  # Young
        elif age < 45:
            return 1  # Middle-aged
        elif age < 60:
            return 2  # Senior
        else:
            return 3  # Elderly
    
    df['age_category'] = df['age'].apply(categorize_age)
    
    # Risk Score (kombinasi faktor risiko)
    df['risk_score'] = (
        df['hypertension'] * 2 + 
        df['heart_disease'] * 2 + 
        (df['bmi'] > 30).astype(int) * 2 +
        (df['HbA1c_level'] > 6.5).astype(int) * 3 +
        (df['blood_glucose_level'] > 126).astype(int) * 3
    )
    
    # High Risk Indicator
    df['high_risk'] = ((df['HbA1c_level'] >= 6.5) | (df['blood_glucose_level'] >= 126)).astype(int)
    
    print("Fitur baru ditambahkan:")
    print("  - bmi_category: Kategori BMI (0=Underweight, 1=Normal, 2=Overweight, 3=Obese)")
    print("  - age_category: Kategori usia (0=Young, 1=Middle-aged, 2=Senior, 3=Elderly)")
    print("  - risk_score: Skor risiko kombinasi")
    print("  - high_risk: Indikator risiko tinggi")
    
    return df


def encoding_kategorikal(df):
    """
    Mengubah variabel kategorikal menjadi numerik menggunakan Label Encoding.
    """
    print("\n" + "=" * 80)
    print("ENCODING: VARIABEL KATEGORIKAL")
    print("=" * 80)
    
    kolom_kategorikal = ['gender', 'smoking_history']
    encoders = {}
    
    for col in kolom_kategorikal:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        
        print(f"\nKolom: {col}")
        print(f"  Kategori asli: {list(le.classes_)}")
        print(f"  Nilai numerik: {list(range(len(le.classes_)))}")
    
    return df, encoders


def oversampling_data(X_train, y_train, random_state=42):
    """
    Melakukan oversampling pada kelas minoritas menggunakan resampling.
    """
    print("\n" + "=" * 80)
    print("OVERSMAPLING: MENANGANI DATA TIDAK SEIMBANG")
    print("=" * 80)
    
    jumlah_kelas = np.bincount(y_train)
    print(f"Sebelum oversampling:")
    print(f"  - Non-Diabetes (0): {jumlah_kelas[0]:,}")
    print(f"  - Diabetes (1): {jumlah_kelas[1]:,}")
    print(f"  - Rasio: 1:{jumlah_kelas[0]//jumlah_kelas[1] if jumlah_kelas[1] > 0 else 'inf'}")
    
    idx_non_diabetes = np.where(y_train == 0)[0]
    idx_diabetes = np.where(y_train == 1)[0]
    
    idx_diabetes_oversample = resample(
        idx_diabetes,
        replace=True,
        n_samples=len(idx_non_diabetes),
        random_state=random_state
    )
    
    idx_balanced = np.random.permutation(
        np.concatenate([idx_non_diabetes, idx_diabetes_oversample])
    )
    
    X_train_balanced = X_train[idx_balanced]
    y_train_balanced = y_train[idx_balanced]
    
    jumlah_baru = np.bincount(y_train_balanced)
    print(f"\nSetelah oversampling:")
    print(f"  - Non-Diabetes (0): {jumlah_baru[0]:,}")
    print(f"  - Diabetes (1): {jumlah_baru[1]:,}")
    print(f"  - Total training: {len(y_train_balanced):,}")
    
    return X_train_balanced, y_train_balanced


def standarisasi_fitur(X_train, X_test):
    """
    Melakukan standarisasi pada fitur numerik.
    """
    print("\n" + "=" * 80)
    print("STANDARISASI: FITUR NUMERIK")
    print("=" * 80)
    
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Standarisasi berhasil!")
    print(f"  - Mean fitur training: {X_train_scaled.mean(axis=0)[:3].round(4)} ...")
    print(f"  - Std fitur training: {X_train_scaled.std(axis=0)[:3].round(4)} ...")
    
    return X_train_scaled, X_test_scaled, scaler


def bagi_fitur_target(df):
    """
    Memisahkan fitur (X) dan target (y) dari dataframe.
    """
    print("\n" + "=" * 80)
    print("PEMISAHAN FITUR DAN TARGET")
    print("=" * 80)
    
    # Fitur yang digunakan untuk prediksi diabetes
    fitur = [
        'age', 'hypertension', 'heart_disease', 'bmi',
        'HbA1c_level', 'blood_glucose_level', 'gender', 'smoking_history',
        'bmi_category', 'age_category', 'risk_score', 'high_risk'
    ]
    
    X = df[fitur].values
    y = df['diabetes'].values
    
    print(f"Fitur: {fitur}")
    print(f"Jumlah fitur: {len(fitur)}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Distribusi target: {np.bincount(y)}")
    
    return X, y, fitur


def visualisasi_correlation_matrix(X, feature_names, save_path):
    """
    Membuat visualisasi correlation matrix.
    """
    print("\n" + "=" * 80)
    print("VISUALISASI: CORRELATION MATRIX")
    print("=" * 80)
    
    # Hitung korelasi
    corr_matrix = np.corrcoef(X.T)
    
    # Plot heatmap
    plt.figure(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, 
                annot=True, 
                fmt='.2f', 
                cmap='RdBu_r',
                mask=mask,
                xticklabels=feature_names,
                yticklabels=feature_names,
                square=True,
                linewidths=0.5)
    plt.title('Correlation Matrix - Fitur Diabetes', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVE] Correlation matrix disimpan: {save_path}")
    
    return corr_matrix


def jalankan_preprocessing():
    """
    Fungsi utama untuk menjalankan seluruh proses preprocessing.
    """
    print("\n" + "=" * 80)
    print("MEMULAI PREPROCESSING DATA DIABETES TIPE 2")
    print("=" * 80)
    print("=" * 80)
    print("Federated Learning + Blockchain untuk Privasi Pasien")
    print("=" * 80)
    
    # 1. Load dataset
    df = load_dataset(DATA_PATH)
    
    # 2. Eksplorasi data
    eksplorasi_data(df)
    
    # 3. Hapus duplikat
    df = bersihkan_duplikat(df)
    
    # 4. Feature engineering
    df = feature_engineering(df)
    
    # 5. Hapus outlier
    kolom_numerik = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    df = hapus_outlier_iqr(df, kolom_numerik)
    
    # 6. Encoding kategorikal
    df, encoders = encoding_kategorikal(df)
    
    # 7. Bagi fitur dan target
    X, y, nama_fitur = bagi_fitur_target(df)
    
    # 8. Split train/test
    print("\n" + "=" * 80)
    print("SPLIT DATA: TRAIN/TEST")
    print("=" * 80)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=y
    )
    
    print(f"Training: {X_train.shape[0]:,} sampel")
    print(f"Testing: {X_test.shape[0]:,} sampel")
    print(f"Train/Test ratio: {(1-TEST_SIZE)/TEST_SIZE:.1f}")
    
    # 9. Oversampling
    X_train, y_train = oversampling_data(X_train, y_train, OVERSEED)
    
    # 10. Standarisasi
    X_train, X_test, scaler = standarisasi_fitur(X_train, X_test)
    
    # 11. Visualisasi correlation matrix
    corr_path = os.path.join(IMG_DIR, 'correlation_matrix.png')
    visualisasi_correlation_matrix(X_train, nama_fitur, corr_path)
    
    # 12. Simpan data
    print("\n" + "=" * 80)
    print("MENYIMPAN DATA PREPROCESSED")
    print("=" * 80)
    
    np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(OUTPUT_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'), y_test)
    
    print(f"Data tersimpan di: {OUTPUT_DIR}")
    print("  - X_train.npy")
    print("  - X_test.npy")
    print("  - y_train.npy")
    print("  - y_test.npy")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'nama_fitur': nama_fitur,
        'scaler': scaler,
        'encoders': encoders
    }


if __name__ == "__main__":
    data = jalankan_preprocessing()
    print("\n" + "=" * 80)
    print("PREPROCESSING SELESAI!")
    print("=" * 80)