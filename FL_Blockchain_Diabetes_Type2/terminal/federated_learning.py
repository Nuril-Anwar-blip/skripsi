"""
================================================================================
SISTEM PREDIKSI DIABETES TIPE 2 DENGAN FEDERATED LEARNING DAN BLOCKCHAIN
================================================================================

File: federated_learning.py
Deskripsi: Modul Federated Learning (FL)
             - Implementasi Federated Averaging (FedAvg)
             - Simulasi multiple clients (perangkat pasien)
             - Dua skenario: IID dan Non-IID data distribution
             - Analisis konvergensi model
             - Privasi data - tidak ada data yang dikirim ke server

Author: Sistem ML Skripsi
Tanggal: 2024/2025
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)

# ============================================================================
# KONFIGURASI
# ============================================================================

BASE_DIR = r"d:\Kampus\Semester 6\Skripsi\Keperluan Skripsi"
OUTPUT_DIR = os.path.join(BASE_DIR, 'FL_Blockchain_Diabetes_Type2', 'terminal', 'output')
IMG_DIR = os.path.join(BASE_DIR, 'FL_Blockchain_Diabetes_Type2', 'img')
RANDOM_STATE = 42
N_RONDE = 20


def muat_data():
    """
    Memuat data yang sudah di-preprocess.
    """
    print("=" * 80)
    print("MEMUAT DATA PREPROCESSED")
    print("=" * 80)
    
    X_train = np.load(os.path.join(OUTPUT_DIR, 'X_train.npy'))
    X_test = np.load(os.path.join(OUTPUT_DIR, 'X_test.npy'))
    y_train = np.load(os.path.join(OUTPUT_DIR, 'y_train.npy'))
    y_test = np.load(os.path.join(OUTPUT_DIR, 'y_test.npy'))
    
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test


class FLClient:
    """
    Kelas merepresentasikan satu klien dalam Federated Learning.
    
    Setiap klien dalam FL merepresentasikan satu perangkat pasien.
    Data lokal TIDAK PERNAH dikirim ke server - hanya parameter model
    yang dikirim untuk agregasi.
    
    Attributes:
    -----------
    cid : int
        ID unik untuk klien (perangkat pasien)
    X : np.array
        Data fitur lokal klien (tersimpan di perangkat)
    y : np.array
        Label lokal klien (tersimpan di perangkat)
    n : int
        Jumlah sampel data lokal
    model : LogisticRegression
        Model machine learning lokal
    """
    
    def __init__(self, cid, X, y):
        """
        Inisialisasi klien FL.
        
        Parameters:
        -----------
        cid : int
            ID klien
        X : np.array
            Fitur data lokal (dari perangkat pasien)
        y : np.array
            Label data lokal
        """
        self.cid = cid
        self.X = X
        self.y = y
        self.n = len(y)
        
        # Inisialisasi model logistic regression
        self.model = LogisticRegression(
            max_iter=300,
            warm_start=True,
            random_state=RANDOM_STATE
        )
        
        # Inisialisasi awal dengan data dummy
        xi = np.vstack([X[:3], X[:3]])
        yi = np.array([0, 0, 0, 1, 1, 1])
        self.model.fit(xi, yi)
        
        print(f"  [Klien {cid}] Dibuat dengan {self.n} sampel (data lokal di perangkat)")
    
    def get_params(self):
        """
        Mengambil parameter model (bobot) untuk dikirim ke server.
        
        Returns:
        --------
        dict
            Dictionary berisi coef dan intercept model
            
        Notes:
        ------
        Hanya parameter ini yang dikirim ke server, BUKAN data asli.
        Ini menjaga privasi pasien - data tidak pernah meninggalkan perangkat.
        """
        return {
            'coef': self.model.coef_.copy(),
            'intercept': self.model.intercept_.copy()
        }
    
    def set_params(self, params):
        """
        Mengatur parameter model dari server (parameter global).
        """
        self.model.coef_ = params['coef'].copy()
        self.model.intercept_ = params['intercept'].copy()
    
    def latih(self, global_params):
        """
        Melatih model lokal dengan parameter global.
        
        Parameters:
        -----------
        global_params : dict
            Parameter global dari server
            
        Returns:
        --------
        dict
            Dictionary berisi parameter lokal dan jumlah sampel
        """
        # Gunakan parameter global
        self.set_params(global_params)
        
        # Latih pada data lokal (di perangkat pasien)
        self.model.fit(self.X, self.y)
        
        # Hitung akurasi lokal
        acc = accuracy_score(self.y, self.model.predict(self.X))
        
        print(f"    [Klien {self.cid}] n={self.n} | local_acc={acc:.4f}")
        
        # Kembalikan parameter untuk agregasi
        return {'params': self.get_params(), 'n': self.n}


class FLServer:
    """
    Kelas merepresentasikan server dalam Federated Learning.
    
    Server hanya menerima dan mengagregasi parameter model,
    TIDAK PERNAH menerima data asli pasien.
    """
    
    def __init__(self, n_fitur):
        """
        Inisialisasi server FL.
        """
        # Inisialisasi parameter dengan nol
        self.params = {
            'coef': np.zeros((1, n_fitur)),
            'intercept': np.zeros(1)
        }
        self.history = []
    
    def fedavg(self, updates):
        """
        Federated Averaging (FedAvg) - Agregasi parameter dari klien.
        
        Parameters:
        -----------
        updates : list
            List berisi dictionary dari setiap klien
            
        Returns:
        --------
        dict
            Parameter global yang sudah diagregasi
        """
        total = sum(u['n'] for u in updates)
        
        # Inisialisasi accumulator
        coef = np.zeros_like(self.params['coef'])
        intercept = np.zeros_like(self.params['intercept'])
        
        # Weighted average
        for u in updates:
            weight = u['n'] / total
            coef += weight * u['params']['coef']
            intercept += weight * u['params']['intercept']
        
        # Update parameter global
        self.params = {'coef': coef, 'intercept': intercept}
        
        return self.params
    
    def evaluasi(self, X, y):
        """
        Mengevaluasi model global pada data test.
        """
        # Buat model dummy untuk evaluasi
        model = LogisticRegression(max_iter=1)
        model.classes_ = np.array([0, 1])
        model.coef_ = self.params['coef']
        model.intercept_ = self.params['intercept']
        
        # Prediksi
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        
        # Hitung metrik
        return {
            'Accuracy': round(accuracy_score(y, y_pred), 4),
            'Precision': round(precision_score(y, y_pred, zero_division=0), 4),
            'Recall': round(recall_score(y, y_pred, zero_division=0), 4),
            'F1-Score': round(f1_score(y, y_pred, zero_division=0), 4),
            'AUC-ROC': round(roc_auc_score(y, y_prob), 4)
        }


def bagi_data_iid(X, y, n_klien):
    """
    Membagi data secara Independent and Identically Distributed (IID).
    
    Setiap klien mendapat proporsi acak dari data dengan distribusi seimbang.
    """
    print("\n" + "=" * 80)
    print("DISTRIBUSI DATA: IID (Independent and Identically Distributed)")
    print("=" * 80)
    print("Setiap klien mendapat proporsi acak dari data dengan distribusi seimbang.")
    
    # Acak indeks
    idx = np.random.permutation(len(X))
    
    # Bagi menjadi n bagian
    parts = np.array_split(idx, n_klien)
    
    # Return list of (X, y) tuples
    data_clients = [(X[p], y[p]) for p in parts]
    
    for i, (Xc, yc) in enumerate(data_clients):
        print(f"  Klien {i}: {len(yc):,} sampel, Diabetes: {sum(yc):,} ({100*sum(yc)/len(yc):.1f}%)")
    
    return data_clients


def bagi_data_noniid(X, y, n_klien, alpha=0.5):
    """
    Membagi data secara Non-IID (Non-Independent and Identically Distributed).
    
    Setiap klien mungkin hanya memiliki satu kelas atau distribusi berbeda.
    Ini mencerminkan situasi nyata di mana rumah sakit berbeda memiliki pasien berbeda.
    """
    print("\n" + "=" * 80)
    print("DISTRIBUSI DATA: Non-IID (Non-Independent and Identically Distributed)")
    print("=" * 80)
    print("Setiap klien mendapat proporsi tidak seimbang dari data.")
    print(f"Parameter alpha: {alpha} (semakin kecil, semakin tidak seimbang)")
    
    # Inisialisasi list untuk setiap klien
    klien_data = [[] for _ in range(n_klien)]
    
    # Bagi setiap kelas secara terpisah
    for kelas in np.unique(y):
        idx_kelas = np.where(y == kelas)[0]
        np.random.shuffle(idx_kelas)
        
        # Proporsi acak menggunakan Dirichlet
        proporsi = np.random.dirichlet([alpha] * n_klien)
        
        # Hitung batas untuk setiap klien
        batas = (np.cumsum(proporsi) * len(idx_kelas)).astype(int)
        batas = np.concatenate([[0], batas])
        
        # Bagi indeks ke setiap klien
        for k in range(n_klien):
            klien_data[k].extend(idx_kelas[batas[k]:batas[k+1]].tolist())
    
    # Konversi ke array
    data_clients = [(X[np.array(ids)], y[np.array(ids)]) for ids in klien_data]
    
    for i, (Xc, yc) in enumerate(data_clients):
        print(f"  Klien {i}: {len(yc):,} sampel, Diabetes: {sum(yc):,} ({100*sum(yc)/len(yc):.1f}%)")
    
    return data_clients


def jalankan_fl(X_train, y_train, X_test, y_test, n_klien, n_ronde, mode='iid'):
    """
    Menjalankan siklus Federated Learning.
    
    Parameters:
    -----------
    X_train : np.array
        Fitur training
    y_train : np.array
        Label training
    X_test : np.array
        Fitur testing
    y_test : np.array
        Label testing
    n_klien : int
        Jumlah klien (perangkat)
    n_ronde : int
        Jumlah ronde FL
    mode : str
        Mode distribusi data ('iid' atau 'non_iid')
        
    Returns:
    --------
    pd.DataFrame
        DataFrame berisi riwayat evaluasi setiap ronde
    """
    print(f"\n{'='*80}")
    print(f"FEDERATED LEARNING: {mode.upper()} | Klien={n_klien} | Ronde={n_ronde}")
    print('='*80)
    print("Catatan: Data pasien TIDAK PERNAH dikirim ke server!")
    print("         Hanya parameter model yang diaggregasi.")
    
    # Set random seed untuk reproduktibilitas
    np.random.seed(RANDOM_STATE)
    
    # Bagi data ke klien
    if mode == 'iid':
        data_clients = bagi_data_iid(X_train, y_train, n_klien)
    else:
        data_clients = bagi_data_noniid(X_train, y_train, n_klien)
    
    # Buat objek klien
    klien = [FLClient(i, *data_clients[i]) for i in range(n_klien)]
    
    # Buat server
    server = FLServer(X_train.shape[1])
    
    # History evaluasi
    history = []
    
    # Loop ronde
    for r in range(1, n_ronde + 1):
        print(f"\n  --- Ronde {r}/{n_ronde} ---")
        
        # Setiap klien melatih model lokal
        updates = [k.latih(server.params) for k in klien]
        
        # Server agregasi dengan FedAvg
        server.fedavg(updates)
        
        # Evaluasi pada test set
        metrik = server.evaluasi(X_test, y_test)
        metrik['ronde'] = r
        history.append(metrik)
        
        print(f"    [Server] Acc={metrik['Accuracy']:.4f} F1={metrik['F1-Score']:.4f} AUC={metrik['AUC-ROC']:.4f}")
    
    return pd.DataFrame(history)


def visualisasi_konvergensi(history_fl, save_path):
    """
    Membuat visualisasi konvergensi Federated Learning.
    """
    print("\n" + "=" * 80)
    print("VISUALISASI: KONVERGENSI FEDERATED LEARNING")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy
    axes[0, 0].plot(history_fl['ronde'], history_fl['Accuracy'], 'b-o', linewidth=2)
    axes[0, 0].set_xlabel('Ronde')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy per Ronde', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # F1-Score
    axes[0, 1].plot(history_fl['ronde'], history_fl['F1-Score'], 'g-o', linewidth=2)
    axes[0, 1].set_xlabel('Ronde')
    axes[0, 1].set_ylabel('F1-Score')
    axes[0, 1].set_title('F1-Score per Ronde', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # AUC-ROC
    axes[1, 0].plot(history_fl['ronde'], history_fl['AUC-ROC'], 'r-o', linewidth=2)
    axes[1, 0].set_xlabel('Ronde')
    axes[1, 0].set_ylabel('AUC-ROC')
    axes[1, 0].set_title('AUC-ROC per Ronde', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Precision & Recall
    axes[1, 1].plot(history_fl['ronde'], history_fl['Precision'], 'm-o', label='Precision', linewidth=2)
    axes[1, 1].plot(history_fl['ronde'], history_fl['Recall'], 'c-o', label='Recall', linewidth=2)
    axes[1, 1].set_xlabel('Ronde')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Precision & Recall per Ronde', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Federated Learning Convergence Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVE] Grafik konvergensi disimpan: {save_path}")


def jalankan_semua_skenario_fl():
    """
    Menjalankan semua skenario FL dan membandingkan hasilnya.
    """
    print("\n" + "=" * 80)
    print("MENJALANKAN SEMUA SKENARIO FEDERATED LEARNING")
    print("=" * 80)
    
    # Muat data
    X_train, X_test, y_train, y_test = muat_data()
    
    results = []
    
    # Skenario 1: FL IID dengan 5 klien
    print("\n" + "=" * 80)
    print("SKENARIO 1: FL IID 5 Klien")
    print("=" * 80)
    history_iid5 = jalankan_fl(X_train, y_train, X_test, y_test, n_klien=5, n_ronde=N_RONDE, mode='iid')
    final_iid5 = history_iid5.iloc[-1].to_dict()
    final_iid5['skenario'] = 'FL IID 5K'
    results.append(final_iid5)
    
    # Simpan CSV
    history_iid5.to_csv(os.path.join(OUTPUT_DIR, 'fl_iid5.csv'), index=False)
    
    # Visualisasi konvergensi
    visualisasi_konvergensi(history_iid5, os.path.join(IMG_DIR, 'fl_konvergensi_iid5.png'))
    
    # Skenario 2: FL Non-IID dengan 5 klien
    print("\n" + "=" * 80)
    print("SKENARIO 2: FL Non-IID 5 Klien")
    print("=" * 80)
    history_noniid = jalankan_fl(X_train, y_train, X_test, y_test, n_klien=5, n_ronde=N_RONDE, mode='non_iid')
    final_noniid = history_noniid.iloc[-1].to_dict()
    final_noniid['skenario'] = 'FL Non-IID 5K'
    results.append(final_noniid)
    
    # Simpan CSV
    history_noniid.to_csv(os.path.join(OUTPUT_DIR, 'fl_noniid.csv'), index=False)
    
    # Skenario 3: FL IID dengan 10 klien
    print("\n" + "=" * 80)
    print("SKENARIO 3: FL IID 10 Klien")
    print("=" * 80)
    history_iid10 = jalankan_fl(X_train, y_train, X_test, y_test, n_klien=10, n_ronde=N_RONDE, mode='iid')
    final_iid10 = history_iid10.iloc[-1].to_dict()
    final_iid10['skenario'] = 'FL IID 10K'
    results.append(final_iid10)
    
    # Simpan CSV
    history_iid10.to_csv(os.path.join(OUTPUT_DIR, 'fl_iid10.csv'), index=False)
    
    # Gabungkan hasil
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(OUTPUT_DIR, 'fl_results.csv'), index=False)
    
    print("\n" + "=" * 80)
    print("HASIL AKHIR FEDERATED LEARNING")
    print("=" * 80)
    print(df_results.to_string(index=False))
    
    return df_results


if __name__ == "__main__":
    results = jalankan_semua_skenario_fl()
    print("\n" + "=" * 80)
    print("FEDERATED LEARNING SELESAI!")
    print("=" * 80)