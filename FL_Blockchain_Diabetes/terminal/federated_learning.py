"""
================================================================================
SISTEM PREDIKSI DIABETES DENGAN FEDERATED LEARNING DAN BLOCKCHAIN
================================================================================

File: 03_federated_learning.py
Deskripsi: Modul Federated Learning (FL)
             - Implementasi Federated Averaging (FedAvg)
             - Simulasi multiple clients (klien)
             - Dua skenario: IID dan Non-IID data distribution
             - Analisis konvergensi model

Author: Sistem ML Skripsi
Tanggal: 2024
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score
)

# ============================================================================
# KONFIGURASI
# ============================================================================

BASE_DIR = r"d:\Kampus\Semester 6\Skripsi\Keperluan Skripsi"
OUTPUT_DIR = os.path.join(BASE_DIR, 'Data-set', 'output')
RANDOM_STATE = 42
N_RONDE = 20  # Jumlah ronde federated learning


def muat_data():
    """
    Memuat data yang sudah di-preprocess.
    """
    print("=" * 70)
    print("MEMUAT DATA PREPROCESSED")
    print("=" * 70)
    
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
    
    Attributes:
    -----------
    cid : int
        ID unik untuk klien
    X : np.array
        Data fitur lokal klien
    y : np.array
        Label lokal klien
    n : int
        Jumlah sampel data lokal
    model : LogisticRegression
        Model machine learning lokal
        
    Notes:
    ------
    Setiap klien dalam FL memiliki:
    1. Data lokal sendiri (tidak dikirim ke server)
    2. Model lokal yang dilatih pada data lokal
    3. Kemampuan untuk menerima parameter global dari server
    4. Kemampuan untuk mengirim parameter lokal ke server
    
    Privasi data terjaga karena hanya bobot model yang dikirim,
    bukan data asli pasien.
    """
    
    def __init__(self, cid, X, y):
        """
        Inisialisasi klien FL.
        
        Parameters:
        -----------
        cid : int
            ID klien
        X : np.array
            Fitur data lokal
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
            warm_start=True,  # Untuk melanjutkan training
            random_state=RANDOM_STATE
        )
        
        # Inisialisasi awal dengan data dummy untuk memastikan 2 kelas ada
        # Ini diperlukan karena model perlu di-fit sebelum bisa diambil parameternya
        xi = np.vstack([X[:3], X[:3]])
        yi = np.array([0, 0, 0, 1, 1, 1])
        self.model.fit(xi, yi)
        
        print(f"  [Klien {cid}] Dibuat dengan {self.n} sampel")
    
    def get_params(self):
        """
        Mengambil parameter model (bobot) untuk dikirim ke server.
        
        Returns:
        --------
        dict
            Dictionary berisi coef dan intercept model
            
        Notes:
        ------
        Parameter yang diambil:
        - coef: Bobot untuk setiap fitur
        - intercept: Bias model
        
        Hanya parameter ini yang dikirim ke server, BUKAN data asli.
        """
        return {
            'coef': self.model.coef_.copy(),
            'intercept': self.model.intercept_.copy()
        }
    
    def set_params(self, params):
        """
        Mengatur parameter model dari server (parameter global).
        
        Parameters:
        -----------
        params : dict
            Dictionary berisi coef dan intercept dari server
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
            
        Notes:
        ------
        Proses training lokal:
        1. Set parameter global ke model lokal
        2. Latih model pada data lokal
        3. Hitung akurasi lokal
        4. Kembalikan parameter untuk agregasi
        """
        # Gunakan parameter global
        self.set_params(global_params)
        
        # Latih pada data lokal
        self.model.fit(self.X, self.y)
        
        # Hitung akurasi lokal
        acc = accuracy_score(self.y, self.model.predict(self.X))
        
        print(f"    [Klien {self.cid}] n={self.n} | local_acc={acc:.4f}")
        
        # Kembalikan parameter untuk agregasi
        return {'params': self.get_params(), 'n': self.n}


class FLServer:
    """
    Kelas merepresentasikan server dalam Federated Learning.
    
    Attributes:
    -----------
    params : dict
        Parameter global (bobot model)
    history : list
        Riwayat evaluasi setiap ronde
        
    Notes:
    ------
    Responsibilities server:
    1. Menginisialisasi parameter model global
    2. Mendistribusikan parameter ke semua klien
    3. Mengumpulkan update dari klien
    4. Melakukan agregasi dengan FedAvg
    5. Mengevaluasi model global pada data test
    """
    
    def __init__(self, n_fitur):
        """
        Inisialisasi server FL.
        
        Parameters:
        -----------
        n_fitur : int
            Jumlah fitur dalam dataset
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
            List berisi dictionary dari setiap klien dengan:
            - 'params': parameter model
            - 'n': jumlah sampel klien
            
        Returns:
        --------
        dict
            Parameter global yang sudah diagregasi
            
        Notes:
        ------
        Algoritma FedAvg:
        1. Hitung total sampel dari semua klien
        2. Untuk setiap parameter, hitung weighted average:
           - Bobot = (jumlah sampel klien / total sampel)
        3. Jumlahkan weighted average dari semua klien
        
        Ini memastikan klien dengan lebih banyak data
        memiliki pengaruh lebih besar dalam model global.
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
        
        Parameters:
        -----------
        X : np.array
            Fitur test
        y : np.array
            Label test
            
        Returns:
        --------
        dict
            Dictionary berisi metrik evaluasi
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
    
    Parameters:
    -----------
    X : np.array
        Semua fitur
    y : np.array
        Semua label
    n_klien : int
        Jumlah klien
        
    Returns:
    --------
    list
        List of tuples (X, y) untuk setiap klien
        
    Notes:
    ------
    IID distribution berarti:
    - Setiap klien mendapat proporsi acak dari data
    - Distribusi kelas seimbang di setiap klien
    - Ini adalah asumsi ideal dalam FL
    
    Metode: Acak indeks, lalu bagi rata ke semua klien
    """
    print("\n" + "=" * 70)
    print("DISTRIBUSI DATA: IID (Independent and Identically Distributed)")
    print("=" * 70)
    print("Setiap klien mendapat proporsi acak dari data dengan distribusi seimbang.")
    
    # Acak indeks
    idx = np.random.permutation(len(X))
    
    # Bagi menjadi n bagian
    parts = np.array_split(idx, n_klien)
    
    # Return list of (X, y) tuples
    data_clients = [(X[p], y[p]) for p in parts]
    
    for i, (Xc, yc) in enumerate(data_clients):
        print(f"  Klien {i}: {len(yc)} sampel, Diabetes: {sum(yc)} ({100*sum(yc)/len(yc):.1f}%)")
    
    return data_clients


def bagi_data_noniid(X, y, n_klien, alpha=0.5):
    """
    Membagi data secara Non-IID (Non-Independent and Identically Distributed).
    
    Parameters:
    -----------
    X : np.array
        Semua fitur
    y : np.array
        Semua label
    n_klien : int
        Jumlah klien
    alpha : float
        Parameter Dirichlet untuk distribusi tidak seimbang
        
    Returns:
    --------
    list
        List of tuples (X, y) untuk setiap klien
        
    Notes:
    ------
    Non-IID distribution lebih realistis:
    - Setiap klien mungkin hanya memiliki satu kelas
    - Distribusi kelas berbeda antar klien
    - Ini mencerminkan situasi nyata (rs不同医院可能有不同患者群体)
    
    Metode: Dirichlet distribution untuk proporsi acak per kelas per klien
    """
    print("\n" + "=" * 70)
    print("DISTRIBUSI DATA: Non-IID (Non-Independent and Identically Distributed)")
    print("=" * 70)
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
        print(f"  Klien {i}: {len(yc)} sampel, Diabetes: {sum(yc)} ({100*sum(yc)/len(yc):.1f}%)")
    
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
        Jumlah klien
    n_ronde : int
        Jumlah ronde FL
    mode : str
        Mode distribusi data ('iid' atau 'non_iid')
        
    Returns:
    --------
    pd.DataFrame
        DataFrame berisi riwayat evaluasi setiap ronde
        
    Notes:
    ------
    Siklus FL per ronde:
    1. Server mengirim parameter global ke semua klien
    2. Setiap klien melatih model pada data lokal
    3. Klien mengirim parameter ke server
    4. Server melakukan FedAvg untuk mendapat parameter global baru
    5. Server evaluasi pada data test
    
    Proses ini diulang selama n_ronde sampai konvergensi.
    """
    print(f"\n{'='*70}")
    print(f"FEDERATED LEARNING: {mode.upper()} | Klien={n_klien} | Ronde={n_ronde}")
    print('='*70)
    
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


def plot_konvergensi(history_list, labels):
    """
    Membuat plot konvergensi FL.
    
    Parameters:
    -----------
    history_list : list
        List of DataFrame hasil FL
    labels : list
        Label untuk setiap skenario
    """
    print("\n" + "=" * 70)
    print("MEMBUAT PLOT KONVERGENSI")
    print("=" * 70)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Style untuk plot
    styles = ['b-', 'r--', 'g:']
    colors = ['#2196F3', '#E91E63', '#4CAF50']
    
    for i, (hist, label) in enumerate(zip(history_list, labels)):
        # Plot Accuracy
        axes[0].plot(hist['ronde'], hist['Accuracy'], styles[i % len(styles)],
                    label=label, linewidth=2, color=colors[i % len(colors)])
        
        # Plot F1-Score
        axes[1].plot(hist['ronde'], hist['F1-Score'], styles[i % len(styles)],
                    label=label, linewidth=2, color=colors[i % len(colors)])
    
    # Pengaturan plot
    axes[0].set_title('Konvergensi FL - Accuracy', fontsize=12)
    axes[0].set_xlabel('Ronde')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].set_title('Konvergensi FL - F1-Score', fontsize=12)
    axes[1].set_xlabel('Ronde')
    axes[1].set_ylabel('F1-Score')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fl_konvergensi.png'), dpi=150)
    plt.close()
    
    print("[INFO] Plot konvergensi disimpan: fl_konvergensi.png")


def jalankan_semua_skenario():
    """
    Menjalankan semua skenario FL.
    """
    print("\n" + "=" * 70)
    print("FEDERATED LEARNING - SEMUA SKENARIO")
    print("=" * 70)
    print("""
    Skenario yang akan dijalankan:
    1. IID dengan 5 klien - Distribusi seimbang
    2. Non-IID dengan 5 klien - Distribusi tidak seimbang
    3. IID dengan 10 klien - Lebih banyak klien
    
    Setiap skenario akan diuji selama 20 ronde.
    """)
    
    # Muat data
    X_train, X_test, y_train, y_test = muat_data()
    
    # Skenario 1: IID 5 klien
    print("\n" + "#" * 70)
    print("# SKENARIO 1: IID 5 KLIEN")
    print("#" * 70)
    hist_iid5 = jalankan_fl(X_train, y_train, X_test, y_test, 5, N_RONDE, 'iid')
    
    # Skenario 2: Non-IID 5 klien
    print("\n" + "#" * 70)
    print("# SKENARIO 2: Non-IID 5 KLIEN")
    print("#" * 70)
    hist_noniid = jalankan_fl(X_train, y_train, X_test, y_test, 5, N_RONDE, 'non_iid')
    
    # Skenario 3: IID 10 klien
    print("\n" + "#" * 70)
    print("# SKENARIO 3: IID 10 KLIEN")
    print("#" * 70)
    hist_iid10 = jalankan_fl(X_train, y_train, X_test, y_test, 10, N_RONDE, 'iid')
    
    # Simpan hasil
    hist_iid5.to_csv(os.path.join(OUTPUT_DIR, 'fl_iid5.csv'), index=False)
    hist_noniid.to_csv(os.path.join(OUTPUT_DIR, 'fl_noniid.csv'), index=False)
    hist_iid10.to_csv(os.path.join(OUTPUT_DIR, 'fl_iid10.csv'), index=False)
    
    # Plot konvergensi
    plot_konvergensi(
        [hist_iid5, hist_noniid, hist_iid10],
        ['IID 5 Klien', 'Non-IID 5 Klien', 'IID 10 Klien']
    )
    
    # Ringkasan hasil
    print("\n" + "=" * 70)
    print("RINGKASAN HASIL FEDERATED LEARNING")
    print("=" * 70)
    
    for nama, hist in [('IID 5 Klien', hist_iid5), 
                       ('Non-IID 5 Klien', hist_noniid), 
                       ('IID 10 Klien', hist_iid10)]:
        r = hist.iloc[-1]
        print(f"\n{nama}:")
        print(f"  Accuracy:  {r['Accuracy']:.4f}")
        print(f"  F1-Score:  {r['F1-Score']:.4f}")
        print(f"  AUC-ROC:   {r['AUC-ROC']:.4f}")
    
    return {
        'iid5': hist_iid5,
        'noniid': hist_noniid,
        'iid10': hist_iid10
    }


# ============================================================================
# EKSEKUSI UTAMA
# ============================================================================

if __name__ == "__main__":
    print(__doc__)
    hasil = jalankan_semua_skenario()
    print("\n" + "=" * 70)
    print("FEDERATED LEARNING SELESAI!")
    print("=" * 70)
    print("\nHasil dapat dibandingkan dengan:")
    print("  - 02_centralized_ml.py (Centralized ML)")
    print("  - 04_blockchain_security.py (Blockchain + FL)")
