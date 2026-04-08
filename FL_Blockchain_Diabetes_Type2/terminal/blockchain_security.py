"""
================================================================================
SISTEM PREDIKSI DIABETES TIPE 2 DENGAN FEDERATED LEARNING DAN BLOCKCHAIN
================================================================================

File: blockchain_security.py
Deskripsi: Modul Keamanan Blockchain untuk Federated Learning
             - Implementasi Blockchain untuk integritas data
             - Deteksi serangan poisoning attack
             - Verifikasi hash parameter model
             - Audit trail untuk setiap ronde FL
             - Immutable ledger untuk transparansi

Author: Sistem ML Skripsi
Tanggal: 2024/2025
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
OUTPUT_DIR = os.path.join(BASE_DIR, 'FL_Blockchain_Diabetes_Type2', 'terminal', 'output')
IMG_DIR = os.path.join(BASE_DIR, 'FL_Blockchain_Diabetes_Type2', 'img')
RANDOM_STATE = 42
N_RONDE = 20


def muat_data():
    """Memuat data yang sudah di-preprocess."""
    print("=" * 80)
    print("MEMUAT DATA PREPROCESSED")
    print("=" * 80)
    
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
    
    Setiap blok menyimpan:
    - Index: Posisi dalam chain
    - Timestamp: Waktu pembuatan
    - Data: Informasi yang disimpan (parameter model, klien, dll)
    - Previous hash: Link ke blok sebelumnya
    - Hash: Identifikasi unik blok
    """
    
    def __init__(self, index, data, previous_hash):
        """Membuat blok baru."""
        self.index = index
        self.timestamp = time.strftime('%Y-%m-%dT%H:%M:%S')
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self._compute_hash()
    
    def _compute_hash(self):
        """
        Menghitung hash blok menggunakan SHA-256.
        
        Hash dihitung dari:
        - Index blok
        - Timestamp
        - Data
        - Previous hash
        """
        content = {
            'index': self.index,
            'timestamp': self.timestamp,
            'data': {k: v for k, v in self.data.items() 
                    if k not in ('verified', 'hash_recv')},
            'previous_hash': self.previous_hash
        }
        
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()


class Blockchain:
    """
    Kelas merepresentasikan ledger blockchain untuk FL.
    
    Blockchain dalam konteks FL berfungsi untuk:
    1. Menyimpan hash parameter setiap klien di setiap ronde
    2. Memverifikasi integritas parameter yang dikirim
    3. Mendeteksi jika ada manipulasi data (poisoning attack)
    4. Audit trail untuk transparansi
    """
    
    def __init__(self):
        """Membuat blockchain dengan genesis block."""
        genesis_block = Block(0, {'info': 'Genesis Block - FL Diabetes Type 2'}, '0' * 64)
        self.chain = [genesis_block]
        self.index_map = {}
    
    def add_block(self, client_id, round_num, model_hash, n_samples):
        """Menambahkan blok baru ke chain."""
        data = {
            'client_id': client_id,
            'round': round_num,
            'model_hash': model_hash,
            'n_samples': n_samples,
            'verified': False
        }
        
        new_block = Block(
            len(self.chain),
            data,
            self.chain[-1].hash
        )
        
        self.chain.append(new_block)
        
        key = f"{client_id}_{round_num}"
        self.index_map[key] = {
            'model_hash': model_hash,
            'block_index': new_block.index
        }
    
    def verify(self, client_id, round_num, received_hash):
        """Memverifikasi hash yang diterima dengan yang tersimpan."""
        key = f"{client_id}_{round_num}"
        record = self.index_map.get(key)
        
        if not record:
            return False
        
        is_valid = record['model_hash'] == received_hash
        
        block_idx = record['block_index']
        self.chain[block_idx].data.update({
            'verified': is_valid,
            'hash_received': received_hash
        })
        
        return is_valid
    
    def check_integrity(self):
        """Memeriksa integritas seluruh blockchain."""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            if current_block.hash != current_block._compute_hash():
                return False
            
            if current_block.previous_hash != previous_block.hash:
                return False
        
        return True
    
    def export_to_json(self, filepath):
        """Mengekspor blockchain ke file JSON."""
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
    
    def get_verification_stats(self):
        """Mendapatkan statistik verifikasi."""
        total = 0
        verified = 0
        
        for block in self.chain[1:]:  # Skip genesis
            if 'verified' in block.data:
                total += 1
                if block.data['verified']:
                    verified += 1
        
        return {'total': total, 'verified': verified, 'rejected': total - verified}


def hash_params(params):
    """Menghitung hash dari parameter model."""
    hasher = hashlib.sha256()
    hasher.update(params['coef'].astype(np.float32).tobytes())
    hasher.update(params['intercept'].astype(np.float32).tobytes())
    return hasher.hexdigest()


class SecureFLClient:
    """
    Klien FL dengan integrasi blockchain untuk keamanan.
    
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
        
        self.model = LogisticRegression(
            max_iter=300,
            warm_start=True,
            random_state=RANDOM_STATE
        )
        
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
        """Melatih model dan mengirim ke server dengan keamanan blockchain."""
        self.set_params(global_params)
        self.model.fit(self.X, self.y)
        
        original_params = self.get_params()
        original_hash = hash_params(original_params)
        
        # Catat hash ke blockchain SEBELUM mengirim
        self.ledger.add_block(self.cid, round_num, original_hash, self.n)
        
        # Jika malicious, manipulasi parameter
        if self.is_malicious:
            print(f"    [ATTACK] Klien {self.cid} mengirim parameter MANIPULASI!")
            manipulated_params = {
                'coef': original_params['coef'] * -5,
                'intercept': original_params['intercept'] + 99
            }
            params_to_send = manipulated_params
        else:
            params_to_send = original_params
        
        sent_hash = hash_params(params_to_send)
        
        return {
            'client_id': self.cid,
            'params': params_to_send,
            'hash': sent_hash,
            'n_samples': self.n
        }


class SecureFLServer:
    """Server FL dengan keamanan blockchain."""
    
    def __init__(self, n_features, ledger):
        """Inisialisasi server aman."""
        self.params = {
            'coef': np.zeros((1, n_features)),
            'intercept': np.zeros(1)
        }
        self.ledger = ledger
        self.log = []
    
    def aggregate(self, updates, round_num):
        """Mengagregasi parameter dari klien dengan verifikasi."""
        valid_updates = []
        rejected_count = 0
        
        for update in updates:
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
                print(f"    [Server] Klien {update['client_id']}: REJECTED - Hash tidak cocok!")
        
        # Log verifikasi
        self.log.append({
            'round': round_num,
            'total': len(updates),
            'accepted': len(valid_updates),
            'rejected': rejected_count
        })
        
        if not valid_updates:
            print(f"    [WARNING] Tidak ada update valid! Menggunakan parameter lama.")
            return self.params
        
        # FedAvg hanya dari update valid
        total = sum(u['n_samples'] for u in valid_updates)
        coef = np.zeros_like(self.params['coef'])
        intercept = np.zeros_like(self.params['intercept'])
        
        for u in valid_updates:
            weight = u['n_samples'] / total
            coef += weight * u['params']['coef']
            intercept += weight * u['params']['intercept']
        
        self.params = {'coef': coef, 'intercept': intercept}
        
        return self.params
    
    def evaluasi(self, X, y):
        """Mengevaluasi model global."""
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


def bagi_data_iid(X, y, n_klien):
    """Membagi data secara IID."""
    idx = np.random.permutation(len(X))
    parts = np.array_split(idx, n_klien)
    return [(X[p], y[p]) for p in parts]


def jalankan_fl_blockchain(X_train, y_train, X_test, y_test, n_klien, n_ronde, 
                           malicious_clients=None):
    """
    Menjalankan FL dengan keamanan blockchain.
    
    Parameters:
    -----------
    malicious_clients : list
        List ID klien yang merupakan attacker
    """
    print(f"\n{'='*80}")
    print(f"FL + BLOCKCHAIN SECURITY | Klien={n_klien} | Ronde={n_ronde}")
    print('='*80)
    
    if malicious_clients:
        print(f"PERINGATAN: Klien jahat: {malicious_clients}")
    
    np.random.seed(RANDOM_STATE)
    
    # Bagi data
    data_clients = bagi_data_iid(X_train, y_train, n_klien)
    
    # Buat blockchain
    ledger = Blockchain()
    
    # Buat klien
    klien = []
    for i in range(n_klien):
        is_malicious = (malicious_clients and i in malicious_clients)
        client = SecureFLClient(i, *data_clients[i], ledger, is_malicious)
        klien.append(client)
    
    # Buat server
    server = SecureFLServer(X_train.shape[1], ledger)
    
    history = []
    
    for r in range(1, n_ronde + 1):
        print(f"\n  --- Ronde {r}/{n_ronde} ---")
        
        updates = [k.train_and_send(server.params, r) for k in klien]
        server.aggregate(updates, r)
        
        metrik = server.evaluasi(X_test, y_test)
        metrik['ronde'] = r
        history.append(metrik)
        
        print(f"    [Server] Acc={metrik['Accuracy']:.4f} F1={metrik['F1-Score']:.4f}")
    
    return pd.DataFrame(history), ledger


def visualisasi_deteksi_serangan(history_normal, history_1jahat, history_2jahat, save_path):
    """
    Membuat visualisasi deteksi serangan poisoning attack.
    """
    print("\n" + "=" * 80)
    print("VISUALISASI: DETEKSI SERANGAN POISONING ATTACK")
    print("=" * 80)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy comparison
    axes[0].plot(history_normal['ronde'], history_normal['Accuracy'], 'g-o', 
                 label='Normal (0 attacker)', linewidth=2, markersize=6)
    axes[0].plot(history_1jahat['ronde'], history_1jahat['Accuracy'], 'r-s', 
                 label='1 Attacker', linewidth=2, markersize=6)
    axes[0].plot(history_2jahat['ronde'], history_2jahat['Accuracy'], 'm-^', 
                 label='2 Attackers', linewidth=2, markersize=6)
    axes[0].set_xlabel('Ronde', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Dampak Poisoning Attack pada Accuracy', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot F1-Score comparison
    axes[1].plot(history_normal['ronde'], history_normal['F1-Score'], 'g-o', 
                 label='Normal (0 attacker)', linewidth=2, markersize=6)
    axes[1].plot(history_1jahat['ronde'], history_1jahat['F1-Score'], 'r-s', 
                 label='1 Attacker', linewidth=2, markersize=6)
    axes[1].plot(history_2jahat['ronde'], history_2jahat['F1-Score'], 'm-^', 
                 label='2 Attackers', linewidth=2, markersize=6)
    axes[1].set_xlabel('Ronde', fontsize=12)
    axes[1].set_ylabel('F1-Score', fontsize=12)
    axes[1].set_title('Dampak Poisoning Attack pada F1-Score', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Blockchain-Based Attack Detection in Federated Learning', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVE] Grafik deteksi serangan disimpan: {save_path}")


def jalankan_skenario_blockchain():
    """
    Menjalankan semua skenario blockchain security.
    """
    print("\n" + "=" * 80)
    print("MENJALANKAN SKENARIO BLOCKCHAIN SECURITY")
    print("=" * 80)
    
    X_train, X_test, y_train, y_test = muat_data()
    
    results = []
    
    # Skenario 1: Normal (tidak ada attacker)
    print("\n" + "=" * 80)
    print("SKENARIO 1: NORMAL (0 Attacker)")
    print("=" * 80)
    history_normal, ledger_normal = jalankan_fl_blockchain(
        X_train, y_train, X_test, y_test, 
        n_klien=5, n_ronde=N_RONDE, malicious_clients=None
    )
    final_normal = history_normal.iloc[-1].to_dict()
    final_normal['skenario'] = 'BC Normal'
    results.append(final_normal)
    
    # Simpan ledger
    ledger_normal.export_to_json(os.path.join(OUTPUT_DIR, 'ledger_normal.json'))
    history_normal.to_csv(os.path.join(OUTPUT_DIR, 'log_normal.csv'), index=False)
    
    # Skenario 2: 1 Attacker
    print("\n" + "=" * 80)
    print("SKENARIO 2: 1 ATTACKER (Poisoning Attack)")
    print("=" * 80)
    history_1jahat, ledger_1jahat = jalankan_fl_blockchain(
        X_train, y_train, X_test, y_test, 
        n_klien=5, n_ronde=N_RONDE, malicious_clients=[1]
    )
    final_1jahat = history_1jahat.iloc[-1].to_dict()
    final_1jahat['skenario'] = 'BC 1 Jahat'
    results.append(final_1jahat)
    
    ledger_1jahat.export_to_json(os.path.join(OUTPUT_DIR, 'ledger_1_jahat.json'))
    history_1jahat.to_csv(os.path.join(OUTPUT_DIR, 'log_1_jahat.csv'), index=False)
    
    # Skenario 3: 2 Attackers
    print("\n" + "=" * 80)
    print("SKENARIO 3: 2 ATTACKERS (Poisoning Attack)")
    print("=" * 80)
    history_2jahat, ledger_2jahat = jalankan_fl_blockchain(
        X_train, y_train, X_test, y_test, 
        n_klien=5, n_ronde=N_RONDE, malicious_clients=[1, 3]
    )
    final_2jahat = history_2jahat.iloc[-1].to_dict()
    final_2jahat['skenario'] = 'BC 2 Jahat'
    results.append(final_2jahat)
    
    ledger_2jahat.export_to_json(os.path.join(OUTPUT_DIR, 'ledger_2_jahat.json'))
    history_2jahat.to_csv(os.path.join(OUTPUT_DIR, 'log_2_jahat.csv'), index=False)
    
    # Visualisasi deteksi serangan
    visualisasi_deteksi_serangan(
        history_normal, history_1jahat, history_2jahat,
        os.path.join(IMG_DIR, 'bc_attack_detection.png')
    )
    
    # Gabungkan hasil
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(OUTPUT_DIR, 'bc_results.csv'), index=False)
    
    print("\n" + "=" * 80)
    print("HASIL AKHIR BLOCKCHAIN SECURITY")
    print("=" * 80)
    print(df_results.to_string(index=False))
    
    return df_results


if __name__ == "__main__":
    results = jalankan_skenario_blockchain()
    print("\n" + "=" * 80)
    print("BLOCKCHAIN SECURITY SELESAI!")
    print("=" * 80)