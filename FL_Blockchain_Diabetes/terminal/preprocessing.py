"""
================================================================================
SISTEM PREDIKSI DIABETES DENGAN FEDERATED LEARNING DAN BLOCKCHAIN
================================================================================

File: 01_preprocessing.py
Deskripsi: Modul preprocessing data untuk prediksi diabetes
             - Load dataset dari folder Data-set
             - Cleaning data (hapus duplikat, outlier)
             - Encoding variabel kategorikal
             - Oversampling untuk menangani data tidak seimbang
             - Standarisasi fitur numerik

Author: Sistem ML Skripsi
Tanggal: 2024
================================================================================
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# ============================================================================
# KONFIGURASI DAN PATH
# ============================================================================

# Path ke dataset diabetes - menggunakan dataset dari folder Data-set
BASE_DIR = r"d:\Kampus\Semester 6\Skripsi\Keperluan Skripsi"
DATA_PATH = os.path.join(BASE_DIR, 'Data-set', 'diabetes_prediction_dataset.csv')

# Konfigurasi preprocessing
RANDOM_STATE = 42    # Seed untuk reproduktibilitas hasil
TEST_SIZE = 0.10     # 10% data untuk testing
OUTPUT_DIR = os.path.join(BASE_DIR, 'FL_Blockchain_Diabetes', 'terminal', 'output')

# Buat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)


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
        
    Notes:
    ------
    Dataset ini berisi informasi medis pasien termasuk:
    - age: Usia pasien
    - gender: Jenis kelamin
    - bmi: Body Mass Index
    - HbA1c_level: Tingkat HbA1c (hemoglobin terikat glukosa)
    - blood_glucose_level: Kadar glukosa darah
    - hypertension: Tekanan darah tinggi
    - heart_disease: Penyakit jantung
    - smoking_history: Riwayat merokok
    - diabetes: Label target (1 = diabetes, 0 = non-diabetes)
    """
    print("=" * 70)
    print("LOAD DATASET")
    print("=" * 70)
    print(f"Path: {path}")
    
    df = pd.read_csv(path)
    
    print(f"\n[INFO] Dataset berhasil dimuat!")
    print(f"       - Jumlah baris: {df.shape[0]}")
    print(f"       - Jumlah kolom: {df.shape[1]}")
    print(f"       - Missing value: {df.isnull().sum().sum()}")
    
    return df


def bersihkan_duplikat(df):
    """
    Menghapus baris duplikat dari dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame yang akan dibersihkan
        
    Returns:
    --------
    pd.DataFrame
        DataFrame tanpa duplikat
        
    Notes:
    ------
    Duplikat dapat terjadi karena:
    - Input data yang sama dari berbagai sumber
    - Error saat pengumpulan data
    - Duplikasi rekam medis
    """
    print("\n" + "=" * 70)
    print("CLEANING: HAPUS DUPLIKAT")
    print("=" * 70)
    
    jumlah_sebelum = len(df)
    df_clean = df.drop_duplicates()
    jumlah_setelah = len(df_clean)
    
    print(f"Sebelum: {jumlah_sebelum} baris")
    print(f"Setelah: {jumlah_setelah} baris")
    print(f"Dihapus: {jumlah_sebelum - jumlah_setelah} duplikat")
    
    return df_clean


def hapus_outlier_iqr(df, kolom_numerik):
    """
    Menghapus outlier menggunakan metode Interquartile Range (IQR).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame yang akan dibersihkan
    kolom_numerik : list
        List nama kolom numerik yang akan dicek outliernya
        
    Returns:
    --------
    pd.DataFrame
        DataFrame tanpa outlier
        
    Notes:
    ------
    Metode IQR bekerja dengan:
    1. Menghitung Q1 (kuartil 25%) dan Q3 (kuartil 75%)
    2. Menghitung IQR = Q3 - Q1
    3. Batas bawah = Q1 - 1.5 * IQR
    4. Batas atas = Q3 + 1.5 * IQR
    5. Data di luar batas ini dianggap outlier
    
    Outlier ditangani secara terpisah untuk setiap kelas (diabetes/non-diabetes)
    untuk mencegah penghapusan data minoritas yang penting.
    """
    print("\n" + "=" * 70)
    print("CLEANING: HAPUS OUTLIER DENGAN IQR")
    print("=" * 70)
    print(f"Kolom yang dicek: {kolom_numerik}")
    
    def filter_iqr_per_kelas(data, cols):
        """
        Filter outlier per kelas untuk menjaga proporsi data.
        """
        Q1 = data[cols].quantile(0.25)
        Q3 = data[cols].quantile(0.75)
        IQR = Q3 - Q1
        
        # Hitung batas bawah dan atas
        batas_bawah = Q1 - 1.5 * IQR
        batas_atas = Q3 + 1.5 * IQR
        
        # Buat mask untuk data yang valid
        mask = ~((data[cols] < batas_bawah) | (data[cols] > batas_atas)).any(axis=1)
        
        return data[mask]
    
    jumlah_sebelum = len(df)
    
    # Filter outlier untuk setiap kelas secara terpisah
    df_diabetes = df[df['diabetes'] == 1].copy()
    df_non_diabetes = df[df['diabetes'] == 0].copy()
    
    df_diabetes_clean = filter_iqr_per_kelas(df_diabetes, kolom_numerik)
    df_non_diabetes_clean = filter_iqr_per_kelas(df_non_diabetes, kolom_numerik)
    
    # Gabungkan kembali
    df_clean = pd.concat([df_diabetes_clean, df_non_diabetes_clean]).reset_index(drop=True)
    
    jumlah_setelah = len(df_clean)
    
    print(f"Sebelum: {jumlah_sebelum} baris")
    print(f"Setelah: {jumlah_setelah} baris")
    print(f"Dihapus: {jumlah_sebelum - jumlah_setelah} outlier")
    print(f"  - Diabetes: {len(df_diabetes)} -> {len(df_diabetes_clean)}")
    print(f"  - Non-Diabetes: {len(df_non_diabetes)} -> {len(df_non_diabetes_clean)}")
    
    return df_clean


def encoding_kategorikal(df):
    """
    Mengubah variabel kategorikal menjadi numerik menggunakan Label Encoding.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame yang akan diencoding
        
    Returns:
    --------
    pd.DataFrame
        DataFrame dengan variabel kategorikal yang sudah diencoding
    dict
        Dictionary berisi encoder untuk setiap kolom
        
    Notes:
    ------
    Label Encoding mengubah kategori menjadi angka:
    - gender: ['Female', 'Male', 'Other'] -> [0, 1, 2]
    - smoking_history: ['No Info', 'current', 'ever', 'former', 'never', 'not current'] -> [0, 1, 2, 3, 4, 5]
    
    Alternatif lain adalah One-Hot Encoding, tetapi Label Encoding
    lebih efisien untuk model tree-based.
    """
    print("\n" + "=" * 70)
    print("ENCODING: VARIABEL KATEGORIKAL")
    print("=" * 70)
    
    # Kolom kategorikal yang akan diencoding
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
    
    Parameters:
    -----------
    X_train : np.array
        Fitur training
    y_train : np.array
        Label training
    random_state : int
        Seed untuk reproduktibilitas
        
    Returns:
    --------
    np.array, np.array
        Fitur dan label setelah oversampling
        
    Notes:
    ------
    Oversampling diperlukan karena dataset diabetes biasanya tidak seimbang:
    - Jumlah pasien non-diabetes jauh lebih banyak daripada diabetes
    - Tanpa oversampling, model akan cenderung memprediksi non-diabetes
    
    Metode yang digunakan adalah Random Over-sampling dengan replacement,
    sehingga kelas minoritas (diabetes) akan diperbanyak agar sama dengan
    kelas mayoritas (non-diabetes).
    """
    print("\n" + "=" * 70)
    print("OVERSMAPLING: MENANGANI DATA TIDAK SEIMBANG")
    print("=" * 70)
    
    # Hitung jumlah setiap kelas
    jumlah_kelas = np.bincount(y_train)
    print(f"Sebelum oversampling:")
    print(f"  - Non-Diabetes (0): {jumlah_kelas[0]}")
    print(f"  - Diabetes (1): {jumlah_kelas[1]}")
    print(f"  - Rasio: 1:{jumlah_kelas[0]//jumlah_kelas[1] if jumlah_kelas[1] > 0 else 'inf'}")
    
    # Indeks setiap kelas
    idx_non_diabetes = np.where(y_train == 0)[0]
    idx_diabetes = np.where(y_train == 1)[0]
    
    # Oversampling kelas diabetes agar sama dengan non-diabetes
    idx_diabetes_oversample = resample(
        idx_diabetes,
        replace=True,  # Sampling dengan replacement
        n_samples=len(idx_non_diabetes),  # Samakan jumlah
        random_state=random_state
    )
    
    # Gabungkan indeks
    idx_balanced = np.random.permutation(
        np.concatenate([idx_non_diabetes, idx_diabetes_oversample])
    )
    
    # Data setelah oversampling
    X_train_balanced = X_train[idx_balanced]
    y_train_balanced = y_train[idx_balanced]
    
    jumlah_baru = np.bincount(y_train_balanced)
    print(f"\nSetelah oversampling:")
    print(f"  - Non-Diabetes (0): {jumlah_baru[0]}")
    print(f"  - Diabetes (1): {jumlah_baru[1]}")
    print(f"  - Total training: {len(y_train_balanced)}")
    
    return X_train_balanced, y_train_balanced


def standarisasi_fitur(X_train, X_test):
    """
    Melakukan standarisasi pada fitur numerik.
    
    Parameters:
    -----------
    X_train : np.array
        Fitur training
    X_test : np.array
        Fitur testing
        
    Returns:
    --------
    np.array, np.array, StandardScaler
        Fitur training dan test yang sudah distandarisasi, beserta scaler
        
    Notes:
    ------
    Standarisasi mengubah data sehingga:
    - Mean = 0
    - Standard deviation = 1
    
    Rumus: z = (x - mean) / std
    
    Ini penting karena:
    1. Fitur dengan skala berbeda akan diperlakukan sama oleh model
    2. Konvergensi model (terutama gradient-based) menjadi lebih cepat
    3. Menghindari dominasi fitur dengan nilai besar
    """
    print("\n" + "=" * 70)
    print("STANDARISASI: FITUR NUMERIK")
    print("=" * 70)
    
    scaler = StandardScaler()
    
    # Fit pada training data, transform both train and test
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Standarisasi berhasil!")
    print(f"  - Mean fitur training: {X_train_scaled.mean(axis=0)[:3].round(4)} ...")
    print(f"  - Std fitur training: {X_train_scaled.std(axis=0)[:3].round(4)} ...")
    
    return X_train_scaled, X_test_scaled, scaler


def bagi_fitur_target(df):
    """
    Memisahkan fitur (X) dan target (y) dari dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame yang berisi fitur dan target
        
    Returns:
    --------
    np.array, np.array, list
        Fitur, target, dan nama fitur
        
    Notes:
    ------
    Fitur yang digunakan untuk prediksi diabetes:
    - age: Usia pasien
    - hypertension: Apakah memiliki tekanan darah tinggi
    - heart_disease: Apakah memiliki penyakit jantung
    - bmi: Body Mass Index
    - HbA1c_level: Tingkat HbA1c
    - blood_glucose_level: Kadar glukosa darah
    - gender: Jenis kelamin (sudah diencoding)
    - smoking_history: Riwayat merokok (sudah diencoding)
    """
    print("\n" + "=" * 70)
    print("PEMISAHAN FITUR DAN TARGET")
    print("=" * 70)
    
    # Definisi fitur yang digunakan
    fitur = [
        'age', 'hypertension', 'heart_disease', 'bmi',
        'HbA1c_level', 'blood_glucose_level', 'gender', 'smoking_history'
    ]
    
    X = df[fitur].values
    y = df['diabetes'].values
    
    print(f"Fitur: {fitur}")
    print(f"Jumlah fitur: {len(fitur)}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Distribusi target: {np.bincount(y)}")
    
    return X, y, fitur


def jalankan_preprocessing():
    """
    Fungsi utama untuk menjalankan seluruh proses preprocessing.
    
    Returns:
    --------
    dict
        Dictionary berisi semua data yang sudah diproses
        
    Notes:
    ------
    Urutan preprocessing:
    1. Load dataset
    2. Hapus duplikat
    3. Hapus outlier dengan IQR
    4. Encoding variabel kategorikal
    5. Bagi fitur dan target
    6. Split train/test
    7. Oversampling pada training data
    8. Standarisasi fitur
    9. Simpan data untuk digunakan di modul lain
    """
    print("\n" + "=" * 70)
    print("MEMULAI PREPROCESSING DATA DIABETES")
    print("=" * 70)
    
    # 1. Load dataset
    df = load_dataset(DATA_PATH)
    
    # Tampilkan info awal
    print("\n[INFO] Kolom dalam dataset:")
    for col in df.columns:
        print(f"  - {col}: {df[col].dtype}")
    
    print("\n[INFO] Distribusi label diabetes:")
    print(df['diabetes'].value_counts())
    
    # 2. Hapus duplikat
    df = bersihkan_duplikat(df)
    
    # 3. Hapus outlier
    kolom_numerik = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    df = hapus_outlier_iqr(df, kolom_numerik)
    
    # 4. Encoding kategorikal
    df, encoders = encoding_kategorikal(df)
    
    # 5. Bagi fitur dan target
    X, y, nama_fitur = bagi_fitur_target(df)
    
    # 6. Split train/test
    print("\n" + "=" * 70)
    print("SPLIT DATA: TRAIN/TEST")
    print("=" * 70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=y  # Stratified split untuk menjaga proporsi kelas
    )
    
    print(f"Training: {X_train.shape[0]} sampel")
    print(f"Testing: {X_test.shape[0]} sampel")
    print(f"Train/Test ratio: {(1-TEST_SIZE)/TEST_SIZE:.1f}")
    
    # 7. Oversampling
    X_train, y_train = oversampling_data(X_train, y_train, RANDOM_STATE)
    
    # 8. Standarisasi
    X_train, X_test, scaler = standarisasi_fitur(X_train, X_test)
    
    # 9. Simpan data
    print("\n" + "=" * 70)
    print("MENYIMPAN DATA")
    print("=" * 70)
    
    np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(OUTPUT_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'), y_test)
    
    print(f"Data tersimpan di: {OUTPUT_DIR}")
    print("  - X_train.npy")
    print("  - X_test.npy")
    print("  - y_train.npy")
    print("  - y_test.npy")
    
    # Return semua data untuk digunakan di modul lain
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'nama_fitur': nama_fitur,
        'scaler': scaler,
        'encoders': encoders
    }


# ============================================================================
# EKSEKUSI UTAMA
# ============================================================================

if __name__ == "__main__":
    print(__doc__)
    data = jalankan_preprocessing()
    print("\n" + "=" * 70)
    print("PREPROCESSING SELESAI!")
    print("=" * 70)
    print("\nData siap untuk digunakan dalam:")
    print("  - 02_centralized_ml.py (ML Terpusat)")
    print("  - 03_federated_learning.py (Federated Learning)")
    print("  - 04_blockchain_security.py (Blockchain + FL)")
