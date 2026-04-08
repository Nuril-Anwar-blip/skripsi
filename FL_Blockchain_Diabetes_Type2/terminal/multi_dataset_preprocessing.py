"""
================================================================================
SISTEM PREDIKSI DIABETES TIPE 2 DENGAN FEDERATED LEARNING DAN BLOCKCHAIN
================================================================================

File: multi_dataset_preprocessing.py
Deskripsi: Modul preprocessing untuk MULTIPLE DATASETS
             - Mengelola 5+ dataset diabetes yang berbeda
             - Analisis komparatif antar dataset
             - Feature engineering untuk setiap dataset
             - Visualisasi detail untuk setiap dataset

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

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# ============================================================================
# KONFIGURASI DAN PATH
# ============================================================================

BASE_DIR = r"d:\Kampus\Semester 6\Skripsi\Keperluan Skripsi"
DATA_DIR = os.path.join(BASE_DIR, 'Data-set')
OUTPUT_DIR = os.path.join(BASE_DIR, 'FL_Blockchain_Diabetes_Type2', 'terminal', 'output')
IMG_DIR = os.path.join(BASE_DIR, 'FL_Blockchain_Diabetes_Type2', 'img')

RANDOM_STATE = 42
TEST_SIZE = 0.10

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)


def load_all_datasets():
    """
    Memuat SEMUA dataset diabetes yang tersedia.
    
    Dataset yang tersedia:
    1. diabetes_prediction_dataset.csv - Dataset utama (~100k records)
    2. diabetes_012_health_indicators_BRFSS2015.csv - BRFSS 2015 (3 kelas)
    3. diabetes_binary_health_indicators_BRFSS2015.csv - BRFSS 2015 (2 kelas)
    4. diabetes_binary_5050split_health_indicators_BRFSS2015.csv - Balanced split
    5. diabetes.csv - Dataset tambahan
    """
    print("=" * 80)
    print("MEMUAT SEMUA DATASET DIABETES")
    print("=" * 80)
    
    datasets = {}
    
    # 1. Diabetes Prediction Dataset (utama)
    print("\n[1] Loading diabetes_prediction_dataset.csv...")
    df1 = pd.read_csv(os.path.join(DATA_DIR, 'diabetes_prediction_dataset.csv'))
    datasets['prediction'] = {
        'name': 'Diabetes Prediction Dataset',
        'df': df1,
        'rows': len(df1),
        'cols': len(df1.columns),
        'target': 'diabetes',
        'description': 'Dataset prediksi diabetes dari Kaggle dengan fitur klinis'
    }
    print(f"    -> {len(df1):,} rows, {len(df1.columns)} columns")
    
    # 2. BRFSS 2015 - 3 classes (0, 1, 2)
    print("\n[2] Loading diabetes_012_health_indicators_BRFSS2015.csv...")
    df2 = pd.read_csv(os.path.join(DATA_DIR, 'diabetes_012_health_indicators_BRFSS2015.csv'))
    datasets['brfss_012'] = {
        'name': 'BRFSS 2015 (3 Classes)',
        'df': df2,
        'rows': len(df2),
        'cols': len(df2.columns),
        'target': 'Diabetes_012',
        'description': 'Behavioral Risk Factor Surveillance System 2015 - 3 kelas (0=No, 1=Pre-diabetes, 2=Diabetes)'
    }
    print(f"    -> {len(df2):,} rows, {len(df2.columns)} columns")
    
    # 3. BRFSS 2015 - Binary
    print("\n[3] Loading diabetes_binary_health_indicators_BRFSS2015.csv...")
    df3 = pd.read_csv(os.path.join(DATA_DIR, 'diabetes_binary_health_indicators_BRFSS2015.csv'))
    datasets['brfss_binary'] = {
        'name': 'BRFSS 2015 (Binary)',
        'df': df3,
        'rows': len(df3),
        'cols': len(df3.columns),
        'target': 'Diabetes_binary',
        'description': 'Behavioral Risk Factor Surveillance System 2015 - 2 kelas (0=No Diabetes, 1=Diabetes)'
    }
    print(f"    -> {len(df3):,} rows, {len(df3.columns)} columns")
    
    # 4. BRFSS 2015 - 50/50 Split
    print("\n[4] Loading diabetes_binary_5050split_health_indicators_BRFSS2015.csv...")
    df4 = pd.read_csv(os.path.join(DATA_DIR, 'diabetes_binary_5050split_health_indicators_BRFSS2015.csv'))
    datasets['brfss_5050'] = {
        'name': 'BRFSS 2015 (50/50 Split)',
        'df': df4,
        'rows': len(df4),
        'cols': len(df4.columns),
        'target': 'Diabetes_binary',
        'description': 'Balanced 50/50 split dari BRFSS 2015 - jumlah sama untuk diabetes dan non-diabetes'
    }
    print(f"    -> {len(df4):,} rows, {len(df4.columns)} columns")
    
    # 5. Diabetes CSV tambahan
    print("\n[5] Loading diabetes.csv...")
    df5 = pd.read_csv(os.path.join(DATA_DIR, 'diabetes.csv'))
    datasets['diabetes_csv'] = {
        'name': 'Diabetes CSV',
        'df': df5,
        'rows': len(df5),
        'cols': len(df5.columns),
        'target': 'Outcome',
        'description': 'Dataset diabetes tambahan dengan Outcome sebagai target'
    }
    print(f"    -> {len(df5)} rows, {len(df5.columns)} columns")
    
    print("\n" + "=" * 80)
    print(f"TOTAL DATASET: {len(datasets)}")
    print("=" * 80)
    
    return datasets


def eksplorasi_semua_dataset(datasets):
    """
    Melakukan eksplorasi detail untuk SEMUA dataset.
    """
    print("\n" + "=" * 80)
    print("EKSPLORASI DETAIL SEMUA DATASET")
    print("=" * 80)
    
    for key, data in datasets.items():
        df = data['df']
        print(f"\n{'='*60}")
        print(f"Dataset: {data['name']}")
        print(f"{'='*60}")
        print(f"Shape: {df.shape}")
        print(f"\nKolom: {df.columns.tolist()}")
        print(f"\nTipe Data:")
        print(df.dtypes)
        print(f"\nStatistik Deskriptif:")
        print(df.describe().T)
        print(f"\nDistribusi Target:")
        if data['target'] in df.columns:
            print(df[data['target']].value_counts())
        print(f"\nMissing Values:")
        print(df.isnull().sum())
    
    # Visualisasi comparison antar dataset
    visualisasi_perbandingan_dataset(datasets)


def visualisasi_perbandingan_dataset(datasets):
    """
    Membuat visualisasi perbandingan semua dataset.
    """
    print("\n" + "=" * 80)
    print("VISUALISASI: PERBANDINGAN DATASET")
    print("=" * 80)
    
    # Perbandingan jumlah data
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Jumlah records per dataset
    names = [d['name'] for d in datasets.values()]
    rows = [d['rows'] for d in datasets.values()]
    
    ax = axes[0, 0]
    bars = ax.bar(range(len(names)), rows, color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12'])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n[:15] + '...' if len(n) > 15 else n for n in names], rotation=45, ha='right')
    ax.set_ylabel('Jumlah Records')
    ax.set_title('Jumlah Records per Dataset', fontweight='bold')
    for bar, val in zip(bars, rows):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
               f'{val:,}', ha='center', va='bottom', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Jumlah fitur per dataset
    cols = [d['cols'] for d in datasets.values()]
    
    ax = axes[0, 1]
    bars = ax.bar(range(len(names)), cols, color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12'])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n[:15] + '...' if len(n) > 15 else n for n in names], rotation=45, ha='right')
    ax.set_ylabel('Jumlah Fitur')
    ax.set_title('Jumlah Fitur per Dataset', fontweight='bold')
    for bar, val in zip(bars, cols):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               str(val), ha='center', va='bottom', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Distribusi kelas (jika ada)
    ax = axes[1, 0]
    class_dist = []
    for d in datasets.values():
        if d['target'] in d['df'].columns:
            dist = d['df'][d['target']].value_counts().to_dict()
            class_dist.append(dist)
        else:
            class_dist.append({})
    
    # Pie chart untuk dataset pertama yang memiliki target
    for i, (key, d) in enumerate(datasets.items()):
        if d['target'] in d['df'].columns:
            ax.pie(d['df'][d['target']].value_counts(), 
                  labels=[str(x) for x in d['df'][d['target']].unique()],
                  autopct='%1.1f%%', startangle=90)
            ax.set_title(f'Distribusi Kelas: {d["name"][:20]}...', fontweight='bold')
            break
    
    # 4. Missing values comparison
    ax = axes[1, 1]
    missing_counts = [df.isnull().sum().sum() for df in [d['df'] for d in datasets.values()]]
    bars = ax.bar(range(len(names)), missing_counts, color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12'])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n[:15] + '...' if len(n) > 15 else n for n in names], rotation=45, ha='right')
    ax.set_ylabel('Jumlah Missing Values')
    ax.set_title('Missing Values per Dataset', fontweight='bold')
    for bar, val in zip(bars, missing_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
               str(val), ha='center', va='bottom', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Perbandingan Semua Dataset Diabetes', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'perbandingan_semua_dataset.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVE] Grafik perbandingan dataset disimpan: {IMG_DIR}/perbandingan_semua_dataset.png")


def preprocess_brfss_dataset(df, target_col):
    """
    Preprocessing untuk dataset BRFSS.
    """
    print(f"\nPreprocessing {target_col}...")
    
    # Hapus duplikat
    df = df.drop_duplicates()
    print(f"  Setelah hapus duplikat: {len(df):,} rows")
    
    # Kolom numerik untuk outlier detection
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # Hilangkan kolom yang tidak perlu (contoh: répondent ID)
    cols_to_drop = ['RespondentID'] if 'RespondentID' in df.columns else []
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # Encoding variabel kategorikal jika ada
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    # pisahkan fitur dan target - convert ke numpy array
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    
    return X, y


def preprocess_diabetes_csv(df):
    """
    Preprocessing untuk diabetes.csv.
    """
    print("\nPreprocessing diabetes.csv...")
    
    # Hapus duplikat
    df = df.drop_duplicates()
    print(f"  Setelah hapus duplikat: {len(df)} rows")
    
    # Pisahkan fitur dan target - convert ke numpy array
    X = df.drop(columns=['Outcome']).values
    y = df['Outcome'].values
    
    return X, y


def jalankan_multi_dataset_preprocessing():
    """
    Main function untuk memproses semua dataset.
    """
    print("\n" + "=" * 80)
    print("PREPROCESSING MULTIPLE DATASETS")
    print("=" * 80)
    
    # Load semua dataset
    datasets = load_all_datasets()
    
    # Eksplorasi semua dataset
    eksplorasi_semua_dataset(datasets)
    
    # Proses setiap dataset
    processed_datasets = {}
    
    # 1. Diabetes Prediction Dataset
    print("\n" + "=" * 80)
    print("MEMPROSES: Diabetes Prediction Dataset")
    print("=" * 80)
    
    df1 = datasets['prediction']['df'].copy()
    
    # Hapus duplikat
    df1 = df1.drop_duplicates()
    print(f"Setelah hapus duplikat: {len(df1):,} rows")
    
    # Feature engineering
    def categorize_bmi(bmi):
        if bmi < 18.5: return 0
        elif bmi < 25: return 1
        elif bmi < 30: return 2
        else: return 3
    
    df1['bmi_category'] = df1['bmi'].apply(categorize_bmi)
    
    def categorize_age(age):
        if age < 30: return 0
        elif age < 45: return 1
        elif age < 60: return 2
        else: return 3
    
    df1['age_category'] = df1['age'].apply(categorize_age)
    
    df1['risk_score'] = (
        df1['hypertension'] * 2 + 
        df1['heart_disease'] * 2 + 
        (df1['bmi'] > 30).astype(int) * 2 +
        (df1['HbA1c_level'] > 6.5).astype(int) * 3 +
        (df1['blood_glucose_level'] > 126).astype(int) * 3
    )
    
    df1['high_risk'] = ((df1['HbA1c_level'] >= 6.5) | (df1['blood_glucose_level'] >= 126)).astype(int)
    
    # Encoding
    le_gender = LabelEncoder()
    df1['gender'] = le_gender.fit_transform(df1['gender'])
    le_smoking = LabelEncoder()
    df1['smoking_history'] = le_smoking.fit_transform(df1['smoking_history'])
    
    # Fitur
    features = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level',
                'blood_glucose_level', 'gender', 'smoking_history',
                'bmi_category', 'age_category', 'risk_score', 'high_risk']
    
    X = df1[features].values
    y = df1['diabetes'].values
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Oversampling
    idx_0 = np.where(y_train == 0)[0]
    idx_1 = np.where(y_train == 1)[0]
    idx_1_os = resample(idx_1, replace=True, n_samples=len(idx_0), random_state=RANDOM_STATE)
    idx_balanced = np.random.permutation(np.concatenate([idx_0, idx_1_os]))
    X_train = X_train[idx_balanced]
    y_train = y_train[idx_balanced]
    
    # Standarisasi
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Simpan
    np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(OUTPUT_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'), y_test)
    
    print(f"Data tersimpan: {X_train.shape[0]:,} train, {X_test.shape[0]:,} test")
    
    processed_datasets['prediction'] = {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'features': features
    }
    
    # 2. BRFSS Binary
    print("\n" + "=" * 80)
    print("MEMPROSES: BRFSS 2015 Binary Dataset")
    print("=" * 80)
    
    df2 = datasets['brfss_binary']['df'].copy()
    
    # Preprocess
    X2, y2 = preprocess_brfss_dataset(df2, 'Diabetes_binary')
    
    # Split
    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X2, y2, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y2
    )
    
    # Oversampling
    idx_0 = np.where(y2_train == 0)[0]
    idx_1 = np.where(y2_train == 1)[0]
    if len(idx_1) > 0:
        idx_1_os = resample(idx_1, replace=True, n_samples=len(idx_0), random_state=RANDOM_STATE)
        idx_balanced = np.random.permutation(np.concatenate([idx_0, idx_1_os]))
        X2_train = X2_train[idx_balanced]
        y2_train = y2_train[idx_balanced]
    
    # Standarisasi
    scaler2 = StandardScaler()
    X2_train = scaler2.fit_transform(X2_train)
    X2_test = scaler2.transform(X2_test)
    
    print(f"BRFSS Binary: {X2_train.shape[0]:,} train, {X2_test.shape[0]:,} test")
    
    # 3. BRFSS 5050 Split
    print("\n" + "=" * 80)
    print("MEMPROSES: BRFSS 2015 50/50 Split Dataset")
    print("=" * 80)
    
    df3 = datasets['brfss_5050']['df'].copy()
    X3, y3 = preprocess_brfss_dataset(df3, 'Diabetes_binary')
    
    X3_train, X3_test, y3_train, y3_test = train_test_split(
        X3, y3, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y3
    )
    
    scaler3 = StandardScaler()
    X3_train = scaler3.fit_transform(X3_train)
    X3_test = scaler3.transform(X3_test)
    
    print(f"BRFSS 50/50: {X3_train.shape[0]:,} train, {X3_test.shape[0]:,} test")
    
    # 4. Diabetes CSV
    print("\n" + "=" * 80)
    print("MEMPROSES: Diabetes CSV Dataset")
    print("=" * 80)
    
    df4 = datasets['diabetes_csv']['df'].copy()
    X4, y4 = preprocess_diabetes_csv(df4)
    
    X4_train, X4_test, y4_train, y4_test = train_test_split(
        X4, y4, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y4
    )
    
    # Oversampling
    idx_0 = np.where(y4_train == 0)[0]
    idx_1 = np.where(y4_train == 1)[0]
    if len(idx_1) > 0:
        idx_1_os = resample(idx_1, replace=True, n_samples=len(idx_0), random_state=RANDOM_STATE)
        idx_balanced = np.random.permutation(np.concatenate([idx_0, idx_1_os]))
        X4_train = X4_train[idx_balanced]
        y4_train = y4_train[idx_balanced]
    
    scaler4 = StandardScaler()
    X4_train = scaler4.fit_transform(X4_train)
    X4_test = scaler4.transform(X4_test)
    
    print(f"Diabetes CSV: {X4_train.shape[0]} train, {X4_test.shape[0]} test")
    
    print("\n" + "=" * 80)
    print("PREPROCESSING SEMUA DATASET SELESAI!")
    print("=" * 80)
    
    return processed_datasets


if __name__ == "__main__":
    results = jalankan_multi_dataset_preprocessing()
    print("\nSemua dataset telah diproses dan siap untuk training!")