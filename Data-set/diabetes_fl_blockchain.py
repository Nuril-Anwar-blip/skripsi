import os, hashlib, json, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.model_selection  import train_test_split
from sklearn.utils            import resample
from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.metrics          import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

# ===========================================================================
# KONFIGURASI
# ===========================================================================
DATA_PATH    = 'diabetes_prediction_dataset.csv'
RANDOM_STATE = 42
TEST_SIZE    = 0.10
N_KLIEN      = 5
N_RONDE      = 20
os.makedirs('output', exist_ok=True)

# ===========================================================================
# BAGIAN 1 — PREPROCESSING
# ===========================================================================
print("\n" + "="*55)
print("BAGIAN 1 : PREPROCESSING")
print("="*55)

df = pd.read_csv(DATA_PATH)
print(f"Data awal     : {df.shape}")
print(f"Missing value : {df.isnull().sum().sum()}")
print(f"Label diabetes:\n{df['diabetes'].value_counts()}")

# Hapus duplikat
df.drop_duplicates(inplace=True)

# Hapus outlier IQR (per kelas)
NUM = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']

def iqr_filter(data, cols):
    Q1, Q3 = data[cols].quantile(0.25), data[cols].quantile(0.75)
    IQR    = Q3 - Q1
    mask   = ~((data[cols] < Q1 - 1.5*IQR) | (data[cols] > Q3 + 1.5*IQR)).any(axis=1)
    return data[mask]

df = pd.concat([
    iqr_filter(df[df['diabetes']==1].copy(), NUM),
    iqr_filter(df[df['diabetes']==0].copy(), NUM)
]).reset_index(drop=True)
print(f"Setelah IQR   : {df.shape}")

# Label Encoding
le = LabelEncoder()
for col in ['gender', 'smoking_history']:
    df[col] = le.fit_transform(df[col].astype(str))

# Fitur & target
FITUR = ['age','hypertension','heart_disease','bmi',
         'HbA1c_level','blood_glucose_level','gender','smoking_history']
X, y  = df[FITUR].values, df['diabetes'].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Oversampling (imbalanced)
idx0    = np.where(y_train==0)[0]
idx1    = np.where(y_train==1)[0]
idx1_up = resample(idx1, replace=True, n_samples=len(idx0), random_state=RANDOM_STATE)
idx_bal = np.random.permutation(np.concatenate([idx0, idx1_up]))
X_train, y_train = X_train[idx_bal], y_train[idx_bal]
print(f"Setelah oversample — Train: {X_train.shape} | Label: {np.bincount(y_train)}")

# Standarisasi
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Simpan untuk FL
np.save('output/X_train.npy', X_train)
np.save('output/X_test.npy',  X_test)
np.save('output/y_train.npy', y_train)
np.save('output/y_test.npy',  y_test)
print("Data tersimpan di output/")

# ===========================================================================
# BAGIAN 2 — CENTRALIZED ML (BASELINE)
# ===========================================================================
print("\n" + "="*55)
print("BAGIAN 2 : CENTRALIZED ML (BASELINE)")
print("="*55)

MODELS = {
    'Logistic Regression' : LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    'Random Forest'       : RandomForestClassifier(n_estimators=100, max_depth=16,
                                                    random_state=RANDOM_STATE, n_jobs=-1),
    'KNN'                 : KNeighborsClassifier(n_neighbors=10),
    'Gradient Boosting'   : GradientBoostingClassifier(n_estimators=100,
                                                        random_state=RANDOM_STATE),
}

hasil_central = {}
for nama, model in MODELS.items():
    model.fit(X_train, y_train)
    yp   = model.predict(X_test)
    yprob= model.predict_proba(X_test)[:,1]
    hasil_central[nama] = {
        'Accuracy' : round(accuracy_score(y_test, yp), 4),
        'Precision': round(precision_score(y_test, yp, zero_division=0), 4),
        'Recall'   : round(recall_score(y_test, yp, zero_division=0), 4),
        'F1-Score' : round(f1_score(y_test, yp, zero_division=0), 4),
        'AUC-ROC'  : round(roc_auc_score(y_test, yprob), 4),
    }
    print(f"\n{nama}")
    print(classification_report(y_test, yp, target_names=['Non-DM','DM']))

    # Confusion matrix
    ConfusionMatrixDisplay(confusion_matrix(y_test, yp),
                           display_labels=['Non-DM','DM']).plot(cmap='Blues')
    plt.title(f'Confusion Matrix — {nama}')
    plt.tight_layout()
    plt.savefig(f'output/cm_{nama.replace(" ","_")}.png', dpi=150)
    plt.close()

df_central = pd.DataFrame(hasil_central).T
print("\nRingkasan Centralized:")
print(df_central)
df_central.to_csv('output/hasil_centralized.csv')

# ===========================================================================
# BAGIAN 3 — FEDERATED LEARNING
# ===========================================================================
print("\n" + "="*55)
print("BAGIAN 3 : FEDERATED LEARNING (FedAvg)")
print("="*55)

class FLClient:
    def __init__(self, cid, X, y):
        self.cid      = cid
        self.X        = X
        self.y        = y
        self.n        = len(y)
        self.model    = LogisticRegression(max_iter=300, warm_start=True,
                                            random_state=RANDOM_STATE)
        # inisialisasi awal (pastikan 2 kelas)
        xi = np.vstack([X[:3], X[:3]])
        yi = np.array([0,0,0,1,1,1])
        self.model.fit(xi, yi)

    def get_params(self):
        return {'coef': self.model.coef_.copy(),
                'intercept': self.model.intercept_.copy()}

    def set_params(self, p):
        self.model.coef_      = p['coef'].copy()
        self.model.intercept_ = p['intercept'].copy()

    def latih(self, global_params):
        self.set_params(global_params)
        self.model.fit(self.X, self.y)
        acc = accuracy_score(self.y, self.model.predict(self.X))
        print(f"  [Klien {self.cid}] n={self.n} | local_acc={acc:.4f}")
        return {'params': self.get_params(), 'n': self.n}


class FLServer:
    def __init__(self, n_fitur):
        self.params = {'coef': np.zeros((1, n_fitur)),
                       'intercept': np.zeros(1)}

    def fedavg(self, updates):
        total = sum(u['n'] for u in updates)
        coef  = np.zeros_like(self.params['coef'])
        inter = np.zeros_like(self.params['intercept'])
        for u in updates:
            w      = u['n'] / total
            coef  += w * u['params']['coef']
            inter += w * u['params']['intercept']
        self.params = {'coef': coef, 'intercept': inter}

    def evaluasi(self, X, y):
        m             = LogisticRegression(max_iter=1)
        m.classes_    = np.array([0,1])
        m.coef_       = self.params['coef']
        m.intercept_  = self.params['intercept']
        yp   = m.predict(X)
        yprob= m.predict_proba(X)[:,1]
        return {
            'Accuracy' : round(accuracy_score(y, yp), 4),
            'Precision': round(precision_score(y, yp, zero_division=0), 4),
            'Recall'   : round(recall_score(y, yp, zero_division=0), 4),
            'F1-Score' : round(f1_score(y, yp, zero_division=0), 4),
            'AUC-ROC'  : round(roc_auc_score(y, yprob), 4),
        }


def bagi_iid(X, y, n):
    idx   = np.random.permutation(len(X))
    parts = np.array_split(idx, n)
    return [(X[p], y[p]) for p in parts]


def bagi_noniid(X, y, n, alpha=0.5):
    klien = [[] for _ in range(n)]
    for kls in np.unique(y):
        idx = np.where(y==kls)[0]; np.random.shuffle(idx)
        prop = np.random.dirichlet([alpha]*n)
        batas = (np.cumsum(prop)*len(idx)).astype(int)
        batas = np.concatenate([[0], batas])
        for k in range(n):
            klien[k].extend(idx[batas[k]:batas[k+1]].tolist())
    return [(X[np.array(ids)], y[np.array(ids)]) for ids in klien]


def jalankan_fl(X_tr, y_tr, X_te, y_te, n_klien, n_ronde, mode='iid'):
    print(f"\n  Mode={mode} | Klien={n_klien} | Ronde={n_ronde}")
    np.random.seed(RANDOM_STATE)
    data  = bagi_iid(X_tr, y_tr, n_klien) if mode=='iid' \
            else bagi_noniid(X_tr, y_tr, n_klien)
    klien = [FLClient(i, *data[i]) for i in range(n_klien)]
    srv   = FLServer(X_tr.shape[1])
    hist  = []
    for r in range(1, n_ronde+1):
        print(f"\n  --- Ronde {r}/{n_ronde} ---")
        updates = [k.latih(srv.params) for k in klien]
        srv.fedavg(updates)
        m = srv.evaluasi(X_te, y_te)
        hist.append({'ronde': r, **m})
        print(f"  [Server] Acc={m['Accuracy']:.4f} F1={m['F1-Score']:.4f} AUC={m['AUC-ROC']:.4f}")
    return pd.DataFrame(hist)


# Jalankan 3 skenario FL
hist_iid5   = jalankan_fl(X_train, y_train, X_test, y_test, 5,  N_RONDE, 'iid')
hist_noniid = jalankan_fl(X_train, y_train, X_test, y_test, 5,  N_RONDE, 'non_iid')
hist_iid10  = jalankan_fl(X_train, y_train, X_test, y_test, 10, N_RONDE, 'iid')

hist_iid5.to_csv('output/fl_iid5.csv',   index=False)
hist_noniid.to_csv('output/fl_noniid.csv', index=False)
hist_iid10.to_csv('output/fl_iid10.csv',  index=False)

print("\nHasil akhir FL:")
for nama, hist in [('IID 5K', hist_iid5),('NonIID 5K', hist_noniid),('IID 10K', hist_iid10)]:
    r = hist.iloc[-1]
    print(f"  {nama:12s} Acc={r['Accuracy']:.4f} F1={r['F1-Score']:.4f} AUC={r['AUC-ROC']:.4f}")

# Plot konvergensi
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
for df_h, lbl, ls in [(hist_iid5,'IID 5K','b-'), (hist_noniid,'Non-IID 5K','r--'),
                       (hist_iid10,'IID 10K','g:')]:
    a1.plot(df_h['ronde'], df_h['Accuracy'],  ls, label=lbl, lw=1.8)
    a2.plot(df_h['ronde'], df_h['F1-Score'],  ls, label=lbl, lw=1.8)
for ax, tl in [(a1,'Accuracy'),(a2,'F1-Score')]:
    ax.set_title(f'Konvergensi FL — {tl}'); ax.set_xlabel('Ronde')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('output/fl_konvergensi.png', dpi=150)
plt.close()
print("Plot konvergensi tersimpan.")

# ===========================================================================
# BAGIAN 4 — BLOCKCHAIN SECURITY
# ===========================================================================
print("\n" + "="*55)
print("BAGIAN 4 : BLOCKCHAIN + FL (Keamanan)")
print("="*55)


# ── Blockchain ───────────────────────────────────────────────────────────────
class Block:
    def __init__(self, index, data, prev_hash):
        self.index     = index
        self.ts        = time.strftime('%Y-%m-%dT%H:%M:%S')
        self.data      = data
        self.prev_hash = prev_hash
        self.hash      = self._hash()

    def _hash(self):
        konten = json.dumps({
            'index': self.index, 'ts': self.ts,
            'data' : {k:v for k,v in self.data.items()
                      if k not in ('verified','hash_recv')},
            'prev' : self.prev_hash
        }, sort_keys=True)
        return hashlib.sha256(konten.encode()).hexdigest()


class Blockchain:
    def __init__(self):
        genesis = Block(0, {'info':'Genesis'}, '0'*64)
        self.chain  = [genesis]
        self._idx   = {}

    def tambah(self, cid, ronde, mhash, n):
        blok = Block(len(self.chain),
                     {'cid':cid,'ronde':ronde,'mhash':mhash,'n':n,'verified':False},
                     self.chain[-1].hash)
        self.chain.append(blok)
        self._idx[f'{cid}_{ronde}'] = {'mhash':mhash, 'bi':blok.index}

    def verifikasi(self, cid, ronde, recv_hash):
        rec = self._idx.get(f'{cid}_{ronde}')
        if not rec:
            return False
        ok = rec['mhash'] == recv_hash
        self.chain[rec['bi']].data.update({'verified':ok,'hash_recv':recv_hash})
        return ok

    def integritas(self):
        for i in range(1, len(self.chain)):
            b, pb = self.chain[i], self.chain[i-1]
            if b.hash != b._hash() or b.prev_hash != pb.hash:
                return False
        return True

    def export(self, path):
        with open(path,'w') as f:
            json.dump([{'index':b.index,'ts':b.ts,'data':b.data,
                        'prev':b.prev_hash,'hash':b.hash}
                       for b in self.chain], f, indent=2)


def hash_params(p):
    h = hashlib.sha256()
    h.update(p['coef'].astype(np.float32).tobytes())
    h.update(p['intercept'].astype(np.float32).tobytes())
    return h.hexdigest()


# ── Klien & server aman ───────────────────────────────────────────────────────
class FLClientSecure:
    def __init__(self, cid, X, y, ledger, jahat=False):
        self.cid    = cid
        self.X      = X
        self.y      = y
        self.n      = len(y)
        self.ledger = ledger
        self.jahat  = jahat
        self.model  = LogisticRegression(max_iter=300, warm_start=True,
                                          random_state=RANDOM_STATE)
        xi = np.vstack([X[:3],X[:3]]); yi = np.array([0,0,0,1,1,1])
        self.model.fit(xi, yi)

    def get_params(self):
        return {'coef':self.model.coef_.copy(),'intercept':self.model.intercept_.copy()}

    def set_params(self, p):
        self.model.coef_=p['coef'].copy(); self.model.intercept_=p['intercept'].copy()

    def latih_kirim(self, global_params, ronde):
        self.set_params(global_params)
        self.model.fit(self.X, self.y)
        params_asli = self.get_params()
        mhash       = hash_params(params_asli)

        # Catat hash ke blockchain SEBELUM (mungkin) dimanipulasi
        self.ledger.tambah(self.cid, ronde, mhash, self.n)

        if self.jahat:
            # Manipulasi bobot setelah hash dicatat → server akan menolak
            params_kirim = {'coef': params_asli['coef']*-5,
                            'intercept': params_asli['intercept']+99}
            print(f"  [ATTACK] Klien {self.cid} mengirim bobot MANIPULASI!")
        else:
            params_kirim = params_asli

        return {'cid':self.cid, 'params':params_kirim,
                'hash':hash_params(params_kirim), 'n':self.n}


class FLServerSecure:
    def __init__(self, n_fitur, ledger):
        self.params = {'coef':np.zeros((1,n_fitur)),'intercept':np.zeros(1)}
        self.ledger = ledger
        self.log    = []

    def agregasi(self, updates, ronde):
        valid, ditolak = [], 0
        for u in updates:
            if self.ledger.verifikasi(u['cid'], ronde, u['hash']):
                valid.append(u)
                print(f"  [Server] Klien {u['cid']}: VALID — diterima")
            else:
                ditolak += 1
                print(f"  [Server] Klien {u['cid']}: INVALID — DITOLAK (poisoning!)")
        self.log.append({'ronde':ronde,'diterima':len(valid),'ditolak':ditolak})
        if not valid:
            return
        total = sum(u['n'] for u in valid)
        coef  = np.zeros_like(self.params['coef'])
        inter = np.zeros_like(self.params['intercept'])
        for u in valid:
            w=u['n']/total; coef+=w*u['params']['coef']; inter+=w*u['params']['intercept']
        self.params = {'coef':coef,'intercept':inter}
        print(f"  [FedAvg] {len(valid)} diterima | {ditolak} ditolak")

    def evaluasi(self, X, y):
        m=LogisticRegression(max_iter=1); m.classes_=np.array([0,1])
        m.coef_=self.params['coef']; m.intercept_=self.params['intercept']
        yp=m.predict(X); yprob=m.predict_proba(X)[:,1]
        return {'Accuracy':round(accuracy_score(y,yp),4),
                'Precision':round(precision_score(y,yp,zero_division=0),4),
                'Recall':round(recall_score(y,yp,zero_division=0),4),
                'F1-Score':round(f1_score(y,yp,zero_division=0),4),
                'AUC-ROC':round(roc_auc_score(y,yprob),4)}


def jalankan_fl_bc(X_tr, y_tr, X_te, y_te, n_klien, n_ronde, n_jahat, label):
    print(f"\n  Skenario: {label} | Klien jahat: {n_jahat}")
    np.random.seed(RANDOM_STATE)
    data   = bagi_iid(X_tr, y_tr, n_klien)
    ledger = Blockchain()
    klien  = [FLClientSecure(i, *data[i], ledger, jahat=(i<n_jahat))
              for i in range(n_klien)]
    srv    = FLServerSecure(X_tr.shape[1], ledger)
    hist   = []
    for r in range(1, n_ronde+1):
        print(f"\n  --- Ronde {r}/{n_ronde} ---")
        updates = [k.latih_kirim(srv.params, r) for k in klien]
        srv.agregasi(updates, r)
        m = srv.evaluasi(X_te, y_te)
        hist.append({'ronde':r,**m})
        print(f"  [Evaluasi] Acc={m['Accuracy']:.4f} F1={m['F1-Score']:.4f}")
    print(f"\n  Integritas blockchain: {'UTUH' if ledger.integritas() else 'RUSAK'}")
    print(f"  Total blok: {len(ledger.chain)}")
    ledger.export(f"output/ledger_{label}.json")
    pd.DataFrame(hist).to_csv(f"output/bc_{label}.csv", index=False)
    pd.DataFrame(srv.log).to_csv(f"output/log_{label}.csv", index=False)
    return pd.DataFrame(hist), pd.DataFrame(srv.log)


hist_bc_normal, log_normal = jalankan_fl_bc(
    X_train, y_train, X_test, y_test, N_KLIEN, N_RONDE, 0, 'normal')
hist_bc_1jahat, log_1jahat = jalankan_fl_bc(
    X_train, y_train, X_test, y_test, N_KLIEN, N_RONDE, 1, '1_jahat')
hist_bc_2jahat, log_2jahat = jalankan_fl_bc(
    X_train, y_train, X_test, y_test, N_KLIEN, N_RONDE, 2, '2_jahat')

# ===========================================================================
# BAGIAN 5 — PERBANDINGAN AKHIR
# ===========================================================================
print("\n" + "="*55)
print("BAGIAN 5 : RINGKASAN PERBANDINGAN AKHIR")
print("="*55)

rows = []
for nm, row in df_central.iterrows():
    rows.append({'Pendekatan':'Centralized', 'Skenario':nm, **row})
for nm, hist in [('FL IID 5K',hist_iid5),('FL Non-IID 5K',hist_noniid),('FL IID 10K',hist_iid10)]:
    r = hist.iloc[-1]
    rows.append({'Pendekatan':'FL', 'Skenario':nm,
                 'Accuracy':r['Accuracy'],'Precision':r['Precision'],
                 'Recall':r['Recall'],'F1-Score':r['F1-Score'],'AUC-ROC':r['AUC-ROC']})
for nm, hist in [('Normal',hist_bc_normal),('1 Jahat',hist_bc_1jahat),('2 Jahat',hist_bc_2jahat)]:
    r = hist.iloc[-1]
    rows.append({'Pendekatan':'FL+Blockchain', 'Skenario':nm,
                 'Accuracy':r['Accuracy'],'Precision':r['Precision'],
                 'Recall':r['Recall'],'F1-Score':r['F1-Score'],'AUC-ROC':r['AUC-ROC']})

df_final = pd.DataFrame(rows)
pd.set_option('display.max_colwidth', 25)
print(df_final.to_string(index=False))
df_final.to_csv('output/HASIL_LENGKAP.csv', index=False)

# Plot perbandingan akhir
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Perbandingan Akhir — Semua Pendekatan', fontsize=13)
metrik = ['Accuracy','Precision','Recall','F1-Score','AUC-ROC']
x      = np.arange(len(metrik))

for ax, pend, warna in [
    (axes[0], ['Centralized','FL'], ['#2196F3','#7E57C2','#43A047','#FF5722','#26C6DA','#EC407A','#8BC34A']),
    (axes[1], ['FL+Blockchain'],    ['#43A047','#E53935','#FF9800'])
]:
    sub  = df_final[df_final['Pendekatan'].isin(pend)]
    w    = 0.8 / len(sub)
    for i, (_, row) in enumerate(sub.iterrows()):
        ax.bar(x + i*w, [row[m] for m in metrik], w,
               label=f"{row['Pendekatan']} | {row['Skenario']}",
               color=warna[i % len(warna)], alpha=0.85, edgecolor='white')
    ax.set_xticks(x + w*(len(sub)-1)/2)
    ax.set_xticklabels(metrik, fontsize=9)
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('output/perbandingan_akhir.png', dpi=150)
plt.close()

# Plot deteksi serangan
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Keamanan Blockchain — Deteksi Poisoning Attack', fontsize=12)
for h, lbl, ls in [(hist_bc_normal,'Normal','g-'),
                    (hist_bc_1jahat,'1 Klien Jahat','b--'),
                    (hist_bc_2jahat,'2 Klien Jahat','r:')]:
    a1.plot(h['ronde'], h['F1-Score'], ls, lw=2, label=lbl)
a1.set_title('F1-Score per Ronde (dengan serangan)')
a1.set_xlabel('Ronde'); a1.legend(fontsize=9); a1.grid(alpha=0.3)

a2.bar(log_1jahat['ronde'], log_1jahat['diterima'], color='steelblue', label='Diterima', alpha=0.8)
a2.bar(log_1jahat['ronde'], log_1jahat['ditolak'],
       bottom=log_1jahat['diterima'], color='crimson', label='Ditolak', alpha=0.8)
a2.set_title('Klien Diterima/Ditolak per Ronde (1 Jahat)')
a2.set_xlabel('Ronde'); a2.set_ylabel('Jumlah Klien')
a2.legend(fontsize=9); a2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('output/deteksi_serangan.png', dpi=150)
plt.close()

print("\nSelesai! Semua output di folder 'output/'")
print("\nFile yang dihasilkan:")
for f in sorted(os.listdir('output')):
    print(f"  output/{f}")