"""
================================================================================
SISTEM PREDIKSI DIABETES DENGAN FEDERATED LEARNING DAN BLOCKCHAIN
================================================================================

File: 04_blockchain_security.py
Deskripsi: Modul Keamanan Blockchain untuk Federated Learning
             - Implementasi Blockchain untuk integritas data
             - Deteksi serangan poisoning attack
             - Verifikasi hash parameter model
             - Audit trail untuk setiap ronde FL

Author: Sistem ML Skripsi
Tanggal: 2024
================================================================================
"""

import os
import json
import time
import hashlib
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
N_RONDE = 20


def muat_data():
    """Memuat data yang sudah di-preprocess."""
    print("=" * 70)
    print("MEMUAT DATA PREPROCESSED")
    print("=" * 70)
    
    X_train = np.load(os.path.join(OUTPUT_DIR, 'X_train.npy'))
    X_test = np.load(os.path.join(OUTPUT_DIR, 'X_test.npy'))
    y_train = np.load(os.path.join(OUTPUT_DIR, 'y_train.npy'))
    y_test = np.load(os.path.join(OUTPUT_DIR, 'y_test.npy'))
    
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


class Block:
    """
    Kelas merepresentasikan satu blok dalam blockchain.
    
    Attributes:
    -----------
    index : int
        Posisi blok dalam chain
    timestamp : str
        Waktu pembuatan blok
    data : dict
        Data yang disimpan dalam blok
    previous_hash : str
        Hash dari blok sebelumnya
    hash : str
        Hash dari blok saat ini
        
    Notes:
    ------
    Setiap blok berisi:
    1. Index: Posisi dalam chain
    2. Timestamp: Waktu pembuatan
    3. Data: Informasi yang disimpan (parameter model, klien, dll)
    4. Previous hash: Link ke blok sebelumnya
    5. Hash: Identifikasi unik blok
    
    Hash dihitung menggunakan SHA-256 dari semua data blok.
    Jika ada perubahan pada data, hash akan berubah.
    """
    
    def __init__(self, index, data, previous_hash):
        """
        Membuat blok baru.
        
        Parameters:
        -----------
        index : int
            Index blok
        data : dict
            Data yang disimpan
        previous_hash : str
            Hash blok sebelumnya
        """
        self.index = index
        self.timestamp = time.strftime('%Y-%m-%dT%H:%M:%S')
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self._compute_hash()
    
    def _compute_hash(self):
        """
        Menghitung hash blok menggunakan SHA-256.
        
        Returns:
        --------
        str
            Hash dalam format hexadecimal
            
        Notes:
        ------
        Hash dihitung dari:
        - Index blok
        - Timestamp
        - Data (tanpa field 'verified' dan 'hash_recv')
        - Previous hash
        
        Menggunakan JSON serialization untuk konsistensi.
        """
        # Buat dictionary untuk di-hash (exclude verified dan hash_recv)
        content = {
            'index': self.index,
            'timestamp': self.timestamp,
            'data': {k: v for k, v in self.data.items() 
                    if k not in ('verified', 'hash_recv')},
            'previous_hash': self.previous_hash
        }
        
        # Convert ke JSON string
        content_str = json.dumps(content, sort_keys=True)
        
        # Hash dengan SHA-256
        return hashlib.sha256(content_str.encode()).hexdigest()


class Blockchain:
    """
    Kelas merepresentasikan ledger blockchain untuk FL.
    
    Attributes:
    -----------
    chain : list
        List berisi semua blok
    index_map : dict
        Mapping untuk akses cepat blok
        
    Notes:
    ------
    Blockchain dalam konteks FL berfungsi untuk:
    1. Menyimpan hash parameter setiap klien di setiap ronde
    2. Memverifikasi integritas parameter yang dikirim
    3. Mendeteksi jika ada manipulasi data (poisoning attack)
    4. Audit trail untuk transparansi
    
    Setiap perubahan pada data akan terdeteksi karena
    hash akan berubah.
    """
    
    def __init__(self):
        """Membuat blockchain dengan genesis block."""
        # Genesis block
        genesis_block = Block(0, {'info': 'Genesis Block'}, '0' * 64)
        self.chain = [genesis_block]
        
        # Index map untuk akses cepat
        self.index_map = {}
    
    def add_block(self, client_id, round_num, model_hash, n_samples):
        """
        Menambahkan blok baru ke chain.
        
        Parameters:
        -----------
        client_id : int
            ID klien
        round_num : int
            Nomor ronde
        model_hash : str
            Hash parameter model
        n_samples : int
            Jumlah sampel klien
        """
        # Buat data blok
        data = {
            'client_id': client_id,
            'round': round_num,
            'model_hash': model_hash,
            'n_samples': n_samples,
            'verified': False  # Akan diverifikasi nanti
        }
        
        # Buat blok baru
        new_block = Block(
            len(self.chain),
            data,
            self.chain[-1].hash
        )
        
        # Tambahkan ke chain
        self.chain.append(new_block)
        
        # Simpan mapping
        key = f"{client_id}_{round_num}"
        self.index_map[key] = {
            'model_hash': model_hash,
            'block_index': new_block.index
        }
    
    def verify(self, client_id, round_num, received_hash):
        """
        Memverifikasi hash yang diterima dengan yang tersimpan.
        
        Parameters:
        -----------
        client_id : int
            ID klien
        round_num : int
            Nomor ronde
        received_hash : str
            Hash yang diterima dari klien
            
        Returns:
        --------
        bool
            True jika hash cocok, False jika tidak
        """
        key = f"{client_id}_{round_num}"
        record = self.index_map.get(key)
        
        if not record:
            return False
        
        # Verifikasi hash
        is_valid = record['model_hash'] == received_hash
        
        # Update status verifikasi di blok
        block_idx = record['block_index']
        self.chain[block_idx].data.update({
            'verified': is_valid,
            'hash_received': received_hash
        })
        
        return is_valid
    
    def check_integrity(self):
        """
        Memeriksa integritas seluruh blockchain.
        
        Returns:
        --------
        bool
            True jika chain utuh, False jika ada manipulasi
            
        Notes:
        ------
        Pemeriksaan integritas:
        1. Untuk setiap blok, hitung ulang hash
        2. Bandingkan dengan hash yang tersimpan
        3. Periksa previous_hash setiap blok
        
        Jika ada yang tidak cocok, chain sudah dimanipulasi.
        """
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            # Cek hash blok saat ini
            if current_block.hash != current_block._compute_hash():
                return False
            
            # Cek link ke blok sebelumnya
            if current_block.previous_hash != previous_block.hash:
                return False
        
        return True
    
    def export_to_json(self, filepath):
        """
        Mengekspor blockchain ke file JSON.
        
        Parameters:
        -----------
        filepath : str
            Path file output
        """
        data = []
        for block in self.chain:
            data.append({
                'index': block.index,
                'timestamp': block.timestamp,
                'data': block.data,
                'previous_hash': block.previous_hash,
                'hash': block.hash
            })
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


def hash_params(params):
    """
    Menghitung hash dari parameter model.
    
    Parameters:
    -----------
    params : dict
        Dictionary berisi coef dan intercept
        
    Returns:
    --------
    str
        Hash dalam format hexadecimal
    """
    hasher = hashlib.sha256()
    
    # Hash coef
    hasher.update(params['coef'].astype(np.float32).tobytes())
    
    # Hash intercept
    hasher.update(params['intercept'].astype(np.float32).tobytes())
    
    return hasher.hexdigest()


class SecureFLClient:
    """
    Klien FL dengan integrasi blockchain untuk keamanan.
    
    Attributes:
    -----------
    cid : int
        ID klien
    X, y : np.array
        Data lokal
    n : int
        Jumlah sampel
    ledger : Blockchain
        Referensi ke blockchain
    is_malicious : bool
        Apakah klien adalah penyerang
    model : LogisticRegression
        Model ML lokal
        
    Notes:
    ------
    Klien aman memiliki fitur:
    1. Mencatat hash SEBELUM mengirim ke server
    2. Bisa disimulasikan sebagai malicious untuk pengujian
    3. Jika malicious, akan memanipulasi parameter sebelum dikirim
    """
    
    def __init__(self, cid, X, y, ledger, is_malicious=False):
        """Inisialisasi klien aman."""
        self.cid = cid
        self.X = X
        self.y = y
        self.n = len(y)
        self.ledger = ledger
        self.is_malicious = is_malicious
        
        # Inisialisasi model
        self.model = LogisticRegression(
            max_iter=300,
            warm_start=True,
            random_state=RANDOM_STATE
        )
        
        # Inisialisasi dengan data dummy
        xi = np.vstack([X[:3], X[:3]])
        yi = np.array([0, 0, 0, 1, 1, 1])
        self.model.fit(xi, yi)
    
    def get_params(self):
        """Ambil parameter model."""
        return {
            'coef': self.model.coef_.copy(),
            'intercept': self.model.intercept_.copy()
        }
    
    def set_params(self, params):
        """Set parameter model."""
        self.model.coef_ = params['coef'].copy()
        self.model.intercept_ = params['intercept'].copy()
    
    def train_and_send(self, global_params, round_num):
        """
        Melatih model dan mengirim ke server dengan keamanan blockchain.
        
        Parameters:
        -----------
        global_params : dict
            Parameter global dari server
        round_num : int
            Nomor ronde saat ini
            
        Returns:
        --------
        dict
            Dictionary berisi parameter dan hash
        """
        # Gunakan parameter global
        self.set_params(global_params)
        
        # Latih pada data lokal
        self.model.fit(self.X, self.y)
        
        # Dapatkan parameter asli
        original_params = self.get_params()
        original_hash = hash_params(original_params)
        
        # Catat hash ke blockchain SEBELUM mengirim
        # Ini penting untuk verifikasi integritas
        self.ledger.add_block(self.cid, round_num, original_hash, self.n)
        
        # Jika malicious, manipulasi parameter
        if self.is_malicious:
            print(f"    [ATTACK] Klien {self.cid} mengirim parameter MANIPULASI!")
            manipulated_params = {
                'coef': original_params['coef'] * -5,  # Balikkan dan perkecil
                'intercept': original_params['intercept'] + 99
            }
            params_to_send = manipulated_params
        else:
            params_to_send = original_params
        
        # Hitung hash dari parameter yang akan dikirim
        sent_hash = hash_params(params_to_send)
        
        return {
            'client_id': self.cid,
            'params': params_to_send,
            'hash': sent_hash,
            'n_samples': self.n
        }


class SecureFLServer:
    """
    Server FL dengan keamanan blockchain.
    
    Attributes:
    -----------
    params : dict
        Parameter global
    ledger : Blockchain
        Referensi ke blockchain
    log : list
        Riwayat verifikasi
    """
    
    def __init__(self, n_features, ledger):
        """Inisialisasi server aman."""
        self.params = {
            'coef': np.zeros((1, n_features)),
            'intercept': np.zeros(1)
        }
        self.ledger = ledger
        self.log = []
    
    def aggregate(self, updates, round_num):
        """
        Mengagregasi parameter dari klien dengan verifikasi.
        
        Parameters:
        -----------
        updates : list
            List update dari klien
        round_num : int
            Nomor ronde
            
        Returns:
        --------
        dict
            Parameter agregasi
        """
        valid_updates = []
        rejected_count = 0
        
        for update in updates:
            # Verifikasi dengan blockchain
            is_valid = self.ledger.verify(
                update['client_id'],
                round_num,
                update['hash']
            )
            
            if is_valid:
                valid_updates.append(update)
                print(f"    [Server] Klien {update['client_id']}: VALID - Diterima")
            else:
                rejected_count += 1
                print(f"    [Server] Klien {update['client_id']}: INVALID - DITOLAK (poisoning attack!)")
        
        # Catat ke log
        self.log.append({
            'round': round_num,
            'accepted': len(valid_updates),
            'rejected': rejected_count
        })
        
        # Jika tidak ada update valid, gunakan parameter lama
        if not valid_updates:
            print(f"    [WARNING] Semua update ditolak! Menggunakan parameter lama.")
            return self.params
        
        # FedAvg
        total = sum(u['n_samples'] for u in valid_updates)
        coef = np.zeros_like(self.params['coef'])
        intercept = np.zeros_like(self.params['intercept'])
        
        for u in valid_updates:
            weight = u['n_samples'] / total
            coef += weight * u['params']['coef']
            intercept += weight * u['params']['intercept']
        
        self.params = {'coef': coef, 'intercept': intercept}
        
        print(f"    [FedAvg] {len(valid_updates)} diterima, {rejected_count} ditolak")
        
        return self.params
    
    def evaluate(self, X, y):
        """Evaluasi model global."""
        model = LogisticRegression(max_iter=1)
        model.classes_ = np.array([0, 1])
        model.coef_ = self.params['coef']
        model.intercept_ = self.params['intercept']
        
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        
        return {
            'Accuracy': round(accuracy_score(y, y_pred), 4),
            'Precision': round(precision_score(y, y_pred, zero_division=0), 4),
            'Recall': round(recall_score(y, y_pred, zero_division=0), 4),
            'F1-Score': round(f1_score(y, y_pred, zero_division=0), 4),
            'AUC-ROC': round(roc_auc_score(y, y_prob), 4)
        }


def bagi_data_iid(X, y, n_clients):
    """Membagi data IID."""
    idx = np.random.permutation(len(X))
    parts = np.array_split(idx, n_clients)
    return [(X[p], y[p]) for p in parts]


def jalankan_fl_bc(X_train, y_train, X_test, y_test, n_clients, n_rounds, n_malicious, label):
    """
    Menjalankan FL dengan keamanan blockchain.
    
    Parameters:
    -----------
    X_train, y_train : np.array
        Data training
    X_test, y_test : np.array
        Data testing
    n_clients : int
        Jumlah klien
    n_rounds : int
        Jumlah ronde
    n_malicious : int
        Jumlah klien malicious
    label : str
        Label skenario
        
    Returns:
    --------
    pd.DataFrame, pd.DataFrame
        History dan log verifikasi
    """
    print(f"\n{'='*70}")
    print(f"FL + BLOCKCHAIN: {label}")
    print(f"Klien: {n_clients}, Malicious: {n_malicious}, Ronde: {n_rounds}")
    print('='*70)
    
    np.random.seed(RANDOM_STATE)
    
    # Bagi data
    data_clients = bagi_data_iid(X_train, y_train, n_clients)
    
    # Buat blockchain
    ledger = Blockchain()
    
    # Buat klien (beberapa malicious)
    clients = [
        SecureFLClient(i, *data_clients[i], ledger, is_malicious=(i < n_malicious))
        for i in range(n_clients)
    ]
    
    # Buat server
    server = SecureFLServer(X_train.shape[1], ledger)
    
    history = []
    
    for r in range(1, n_rounds + 1):
        print(f"\n  --- Ronde {r}/{n_rounds} ---")
        
        # Klien train dan send
        updates = [c.train_and_send(server.params, r) for c in clients]
        
        # Server aggregate dengan verifikasi
        server.aggregate(updates, r)
        
        # Evaluasi
        metrics = server.evaluate(X_test, y_test)
        metrics['round'] = r
        history.append(metrics)
        
        print(f"    [Evaluasi] Acc={metrics['Accuracy']:.4f} F1={metrics['F1-Score']:.4f}")
    
    # Cek integritas blockchain
    is_integrity_ok = ledger.check_integrity()
    print(f"\n  Integritas Blockchain: {'UTUH' if is_integrity_ok else 'RUSAK'}")
    print(f"  Total blok: {len(ledger.chain)}")
    
    # Simpan
    ledger.export_to_json(os.path.join(OUTPUT_DIR, f'ledger_{label}.json'))
    pd.DataFrame(history).to_csv(os.path.join(OUTPUT_DIR, f'bc_{label}.csv'), index=False)
    pd.DataFrame(server.log).to_csv(os.path.join(OUTPUT_DIR, f'log_{label}.csv'), index=False)
    
    return pd.DataFrame(history), pd.DataFrame(server.log)


def plot_attack_detection(log_list, labels):
    """Plot deteksi serangan."""
    print("\n" + "=" * 70)
    print("MEMBUAT PLOT DETEKSI SERANGAN")
    print("=" * 70)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rounds = log_list[0]['round'].values
    width = 0.25
    x = np.arange(len(rounds))
    
    colors = ['#4CAF50', '#FF9800', '#E91E63']
    
    for i, (log, label) in enumerate(zip(log_list, labels)):
        ax.bar(x + i * width, log['rejected'], width, label=label, color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Ronde')
    ax.set_ylabel('Jumlah Serangan Terdeteksi')
    ax.set_title('Deteksi Poisoning Attack per Ronde')
    ax.set_xticks(x + width)
    ax.set_xticklabels(rounds)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'bc_attack_detection.png'), dpi=150)
    plt.close()
    
    print("[INFO] Plot deteksi serangan disimpan: bc_attack_detection.png")


def jalankan_semua_skenario_bc():
    """Menjalankan semua skenario blockchain."""
    print("\n" + "=" * 70)
    print("FEDERATED LEARNING + BLOCKCHAIN - SEMUA SKENARIO")
    print("=" * 70)
    print("""
    Skenario yang akan dijalankan:
    1. Normal - Semua klien jujur
    2. 1 Klien Jahat - Satu klien melakukan poisoning attack
    3. 2 Klien Jahat - Dua klien melakukan poisoning attack
    
    Blockchain akan mendeteksi dan menolak parameter yang dimanipulasi.
    """)
    
    X_train, X_test, y_train, y_test = muat_data()
    
    # Skenario 1: Normal
    print("\n" + "#" * 70)
    print("# SKENARIO 1: NORMAL (Tanpa Serangan)")
    print("#" * 70)
    hist_normal, log_normal = jalankan_fl_bc(
        X_train, y_train, X_test, y_test, 5, N_RONDE, 0, 'normal'
    )
    
    # Skenario 2: 1 malicious
    print("\n" + "#" * 70)
    print("# SKENARIO 2: 1 KLIEN JAHAT")
    print("#" * 70)
    hist_1jahat, log_1jahat = jalankan_fl_bc(
        X_train, y_train, X_test, y_test, 5, N_RONDE, 1, '1_jahat'
    )
    
    # Skenario 3: 2 malicious
    print("\n" + "#" * 70)
    print("# SKENARIO 3: 2 KLIEN JAHAT")
    print("#" * 70)
    hist_2jahat, log_2jahat = jalankan_fl_bc(
        X_train, y_train, X_test, y_test, 5, N_RONDE, 2, '2_jahat'
    )
    
    # Plot deteksi serangan
    plot_attack_detection(
        [log_normal, log_1jahat, log_2jahat],
        ['Normal', '1 Jahat', '2 Jahat']
    )
    
    # Ringkasan
    print("\n" + "=" * 70)
    print("RINGKASAN HASIL FL + BLOCKCHAIN")
    print("=" * 70)
    
    for nama, hist in [('Normal', hist_normal), 
                       ('1 Jahat', hist_1jahat), 
                       ('2 Jahat', hist_2jahat)]:
        r = hist.iloc[-1]
        print(f"\n{nama}:")
        print(f"  Accuracy:  {r['Accuracy']:.4f}")
        print(f"  F1-Score:  {r['F1-Score']:.4f}")
        print(f"  AUC-ROC:   {r['AUC-ROC']:.4f}")
    
    return {
        'normal': hist_normal,
        '1_jahat': hist_1jahat,
        '2_jahat': hist_2jahat
    }


# ============================================================================
# EKSEKUSI UTAMA
# ============================================================================

if __name__ == "__main__":
    print(__doc__)
    hasil = jalankan_semua_skenario_bc()
    print("\n" + "=" * 70)
    print("FL + BLOCKCHAIN SELESAI!")
    print("=" * 70)
    print("\nPerbandingan dengan pendekatan lain:")
    print("  - 02_centralized_ml.py (Centralized ML)")
    print("  - 03_federated_learning.py (Federated Learning)")
