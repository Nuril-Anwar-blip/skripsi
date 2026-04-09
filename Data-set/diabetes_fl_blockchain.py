import hashlib
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample

warnings.filterwarnings("ignore")


DATA_PATH = "diabetes_prediction_dataset.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.1
N_CLIENTS = 5
N_ROUNDS = 8
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_SUBDIRS = {
    "eda": os.path.join(OUTPUT_DIR, "0_eda_visualizations"),
    "baseline": os.path.join(OUTPUT_DIR, "1_baseline"),
    "history": os.path.join(OUTPUT_DIR, "2_federated_history"),
    "security": os.path.join(OUTPUT_DIR, "3_security_logs"),
    "ledger": os.path.join(OUTPUT_DIR, "4_blockchain_ledger"),
    "summary": os.path.join(OUTPUT_DIR, "5_summary"),
    "visual": os.path.join(OUTPUT_DIR, "6_visualizations"),
}
for subdir in OUTPUT_SUBDIRS.values():
    os.makedirs(subdir, exist_ok=True)


def metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "auc": round(roc_auc_score(y_true, y_prob), 4),
    }


def generate_eda_visuals(df_raw):
    num_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
    cat_cols = ["gender", "smoking_history"]

    # Missing values per kolom
    plt.figure(figsize=(10, 4))
    miss = df_raw.isnull().sum()
    plt.bar(miss.index, miss.values)
    plt.xticks(rotation=20)
    plt.title("Missing Values per Kolom")
    plt.ylabel("Jumlah")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_SUBDIRS["eda"], "01_missing_values.png"), dpi=180)
    plt.close()

    # Distribusi target
    plt.figure(figsize=(6, 4))
    target_count = df_raw["diabetes"].value_counts().sort_index()
    plt.bar(["Non-Diabetes (0)", "Diabetes (1)"], target_count.values, color=["#3B82F6", "#EF4444"])
    plt.title("Distribusi Label Diabetes")
    plt.ylabel("Jumlah Sampel")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_SUBDIRS["eda"], "02_distribusi_target.png"), dpi=180)
    plt.close()

    # Statistik numerik (boxplot)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.flatten()
    for i, c in enumerate(num_cols):
        axes[i].boxplot(df_raw[c].dropna())
        axes[i].set_title(f"Boxplot {c}")
        axes[i].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_SUBDIRS["eda"], "03_boxplot_numerik.png"), dpi=180)
    plt.close()

    # Histogram numerik
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.flatten()
    for i, c in enumerate(num_cols):
        axes[i].hist(df_raw[c].dropna(), bins=30, alpha=0.85)
        axes[i].set_title(f"Histogram {c}")
        axes[i].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_SUBDIRS["eda"], "04_histogram_numerik.png"), dpi=180)
    plt.close()

    # Distribusi kategori
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, c in enumerate(cat_cols):
        vc = df_raw[c].astype(str).value_counts().head(8)
        axes[i].bar(vc.index, vc.values)
        axes[i].set_title(f"Distribusi {c}")
        axes[i].tick_params(axis="x", rotation=25)
        axes[i].grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_SUBDIRS["eda"], "05_distribusi_kategorikal.png"), dpi=180)
    plt.close()

    # Korelasi numerik
    corr_cols = num_cols + ["hypertension", "heart_disease", "diabetes"]
    corr = df_raw[corr_cols].corr()
    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=25)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Heatmap Korelasi Fitur Numerik")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_SUBDIRS["eda"], "06_heatmap_korelasi.png"), dpi=180)
    plt.close()


def preprocess():
    df = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)
    for col in ["gender", "smoking_history"]:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    features = [
        "age",
        "hypertension",
        "heart_disease",
        "bmi",
        "HbA1c_level",
        "blood_glucose_level",
        "gender",
        "smoking_history",
    ]
    X = df[features].values
    y = df["diabetes"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    idx0 = np.where(y_train == 0)[0]
    idx1 = np.where(y_train == 1)[0]
    idx1_up = resample(idx1, replace=True, n_samples=len(idx0), random_state=RANDOM_STATE)
    idx_bal = np.random.RandomState(RANDOM_STATE).permutation(np.concatenate([idx0, idx1_up]))
    X_train, y_train = X_train[idx_bal], y_train[idx_bal]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def run_centralized(X_train, X_test, y_train, y_test):
    models = {
        "logistic_regression": LogisticRegression(max_iter=800, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=16, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }
    rows = []
    fitted_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        rows.append({"approach": "centralized", "scenario": name, **metrics(y_test, y_pred, y_prob)})
    return pd.DataFrame(rows), fitted_models


def split_iid(X, y, n_clients):
    idx = np.random.RandomState(RANDOM_STATE).permutation(len(X))
    parts = np.array_split(idx, n_clients)
    return [(X[p], y[p]) for p in parts]


class Block:
    def __init__(self, index, data, prev_hash):
        self.index = index
        self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.data = data
        self.prev_hash = prev_hash
        self.hash = self._compute_hash()

    def _compute_hash(self):
        immutable_data = {
            "client_id": self.data.get("client_id"),
            "round": self.data.get("round"),
            "model_hash": self.data.get("model_hash"),
            "event": self.data.get("event"),
        }
        payload = json.dumps(
            {
                "index": self.index,
                "timestamp": self.timestamp,
                "data": immutable_data,
                "prev_hash": self.prev_hash,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Blockchain:
    def __init__(self):
        self.chain = [Block(0, {"event": "genesis"}, "0" * 64)]
        self.index = {}

    def record_commitment(self, client_id, round_no, model_hash):
        data = {"client_id": client_id, "round": round_no, "model_hash": model_hash, "verified": False}
        block = Block(len(self.chain), data, self.chain[-1].hash)
        self.chain.append(block)
        self.index[(client_id, round_no)] = block.index

    def verify(self, client_id, round_no, received_hash):
        key = (client_id, round_no)
        if key not in self.index:
            return False
        block = self.chain[self.index[key]]
        ok = block.data["model_hash"] == received_hash
        block.data["verified"] = ok
        block.data["received_hash"] = received_hash
        return ok

    def is_valid_chain(self):
        for i in range(1, len(self.chain)):
            if self.chain[i].prev_hash != self.chain[i - 1].hash:
                return False
            if self.chain[i].hash != self.chain[i]._compute_hash():
                return False
        return True


def hash_params(params):
    h = hashlib.sha256()
    h.update(params["coef"].astype(np.float32).tobytes())
    h.update(params["intercept"].astype(np.float32).tobytes())
    return h.hexdigest()


class FLClient:
    def __init__(self, client_id, X_local, y_local, malicious=False):
        self.client_id = client_id
        self.X_local = X_local
        self.y_local = y_local
        self.n = len(y_local)
        self.malicious = malicious
        self.model = LogisticRegression(max_iter=300, random_state=RANDOM_STATE, warm_start=True)
        seed_x = np.vstack([X_local[:2], X_local[:2]])
        seed_y = np.array([0, 0, 1, 1])
        self.model.fit(seed_x, seed_y)

    def set_params(self, params):
        self.model.coef_ = params["coef"].copy()
        self.model.intercept_ = params["intercept"].copy()

    def get_params(self):
        return {"coef": self.model.coef_.copy(), "intercept": self.model.intercept_.copy()}

    def train_and_send(self, global_params, blockchain, round_no):
        self.set_params(global_params)
        self.model.fit(self.X_local, self.y_local)
        honest_params = self.get_params()
        honest_hash = hash_params(honest_params)
        blockchain.record_commitment(self.client_id, round_no, honest_hash)

        if self.malicious:
            tampered = {
                "coef": honest_params["coef"] * -4.0,
                "intercept": honest_params["intercept"] + 15.0,
            }
            sent_params = tampered
        else:
            sent_params = honest_params
        return {"client_id": self.client_id, "n": self.n, "params": sent_params, "sent_hash": hash_params(sent_params)}


class FLServer:
    def __init__(self, n_features):
        self.params = {"coef": np.zeros((1, n_features)), "intercept": np.zeros(1)}

    def aggregate_weighted(self, valid_updates):
        total = sum(u["n"] for u in valid_updates)
        coef = np.zeros_like(self.params["coef"])
        intercept = np.zeros_like(self.params["intercept"])
        for u in valid_updates:
            w = u["n"] / total
            coef += w * u["params"]["coef"]
            intercept += w * u["params"]["intercept"]
        self.params = {"coef": coef, "intercept": intercept}

    def aggregate_robust(self, valid_updates, trim_ratio=0.2):
        coefs = np.array([u["params"]["coef"].reshape(-1) for u in valid_updates])
        inters = np.array([u["params"]["intercept"].reshape(-1) for u in valid_updates])
        n = len(coefs)
        k = int(n * trim_ratio)
        if n <= 2 or k == 0:
            self.aggregate_weighted(valid_updates)
            return
        coef_sorted = np.sort(coefs, axis=0)
        inter_sorted = np.sort(inters, axis=0)
        coef_trimmed = coef_sorted[k : n - k]
        inter_trimmed = inter_sorted[k : n - k]
        self.params = {
            "coef": np.mean(coef_trimmed, axis=0).reshape(1, -1),
            "intercept": np.mean(inter_trimmed, axis=0),
        }

    def evaluate(self, X_test, y_test):
        model = LogisticRegression(max_iter=1)
        model.classes_ = np.array([0, 1])
        model.coef_ = self.params["coef"]
        model.intercept_ = self.params["intercept"]
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        return metrics(y_test, y_pred, y_prob)


def run_fl_blockchain(X_train, X_test, y_train, y_test, n_clients, n_rounds, n_malicious):
    clients_data = split_iid(X_train, y_train, n_clients)
    blockchain = Blockchain()
    clients = [
        FLClient(i, clients_data[i][0], clients_data[i][1], malicious=(i < n_malicious))
        for i in range(n_clients)
    ]
    server = FLServer(X_train.shape[1])
    history = []
    security_log = []

    for round_no in range(1, n_rounds + 1):
        updates = [c.train_and_send(server.params, blockchain, round_no) for c in clients]
        valid_updates = []
        rejected = 0
        for u in updates:
            if blockchain.verify(u["client_id"], round_no, u["sent_hash"]):
                valid_updates.append(u)
            else:
                rejected += 1
        if valid_updates:
            server.aggregate_robust(valid_updates, trim_ratio=0.2)
        m = server.evaluate(X_test, y_test)
        history.append({"round": round_no, **m, "accepted": len(valid_updates), "rejected": rejected})
        security_log.append({"round": round_no, "accepted": len(valid_updates), "rejected": rejected})

    return pd.DataFrame(history), pd.DataFrame(security_log), blockchain


def main():
    raw_df = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)
    generate_eda_visuals(raw_df)
    X_train, X_test, y_train, y_test = preprocess()
    central_df, central_models = run_centralized(X_train, X_test, y_train, y_test)
    central_df.to_csv(
        os.path.join(OUTPUT_SUBDIRS["baseline"], "centralized_results.csv"),
        index=False,
    )

    all_rows = [central_df]
    for scenario, bad in [("fl_bc_normal", 0), ("fl_bc_1_malicious", 1), ("fl_bc_2_malicious", 2)]:
        hist, sec_log, chain = run_fl_blockchain(
            X_train, X_test, y_train, y_test, N_CLIENTS, N_ROUNDS, bad
        )
        hist.to_csv(os.path.join(OUTPUT_SUBDIRS["history"], f"{scenario}_history.csv"), index=False)
        sec_log.to_csv(os.path.join(OUTPUT_SUBDIRS["security"], f"{scenario}_security.csv"), index=False)
        with open(
            os.path.join(OUTPUT_SUBDIRS["ledger"], f"{scenario}_ledger.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump([b.__dict__ for b in chain.chain], f, indent=2)
        final = hist.iloc[-1].to_dict()
        all_rows.append(
            pd.DataFrame(
                [
                    {
                        "approach": "federated_blockchain",
                        "scenario": scenario,
                        "accuracy": final["accuracy"],
                        "precision": final["precision"],
                        "recall": final["recall"],
                        "f1": final["f1"],
                        "auc": final["auc"],
                        "accepted_clients_last_round": int(final["accepted"]),
                        "rejected_clients_last_round": int(final["rejected"]),
                        "ledger_valid": chain.is_valid_chain(),
                    }
                ]
            )
        )

    summary = pd.concat(all_rows, ignore_index=True)
    summary.to_csv(os.path.join(OUTPUT_SUBDIRS["summary"], "summary_results.csv"), index=False)
    summary[summary["approach"] == "centralized"].to_csv(
        os.path.join(OUTPUT_SUBDIRS["summary"], "summary_centralized.csv"),
        index=False,
    )
    summary[summary["approach"] == "federated_blockchain"].to_csv(
        os.path.join(OUTPUT_SUBDIRS["summary"], "summary_federated_blockchain.csv"),
        index=False,
    )

    # Visualisasi untuk perbandingan
    fed_only = summary[summary["approach"] == "federated_blockchain"].copy()
    cen_only = summary[summary["approach"] == "centralized"].copy()
    metric_cols = ["accuracy", "precision", "recall", "f1", "auc"]

    # 1) Bar chart perbandingan semua skenario akhir
    plt.figure(figsize=(11, 5))
    chart_df = pd.concat([cen_only[["scenario"] + metric_cols], fed_only[["scenario"] + metric_cols]])
    chart_df = chart_df.set_index("scenario")
    chart_df.plot(kind="bar", ax=plt.gca(), width=0.85)
    plt.title("Perbandingan Akhir Semua Pendekatan")
    plt.ylabel("Score")
    plt.ylim(0.35, 1.02)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_SUBDIRS["visual"], "01_perbandingan_akhir_semua_pendekatan.png"), dpi=180)
    plt.close()

    # 2) Line chart tren F1 per ronde untuk tiap skenario FL+Blockchain
    plt.figure(figsize=(10, 5))
    for scenario in ["fl_bc_normal", "fl_bc_1_malicious", "fl_bc_2_malicious"]:
        path_hist = os.path.join(OUTPUT_SUBDIRS["history"], f"{scenario}_history.csv")
        hist_df = pd.read_csv(path_hist)
        plt.plot(hist_df["round"], hist_df["f1"], marker="o", linewidth=1.8, label=scenario)
    plt.title("Tren F1-Score per Ronde (FL + Blockchain)")
    plt.xlabel("Round")
    plt.ylabel("F1-Score")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_SUBDIRS["visual"], "02_tren_f1_per_ronde.png"), dpi=180)
    plt.close()

    # 3) Line chart tren AUC per ronde
    plt.figure(figsize=(10, 5))
    for scenario in ["fl_bc_normal", "fl_bc_1_malicious", "fl_bc_2_malicious"]:
        path_hist = os.path.join(OUTPUT_SUBDIRS["history"], f"{scenario}_history.csv")
        hist_df = pd.read_csv(path_hist)
        plt.plot(hist_df["round"], hist_df["auc"], marker="s", linewidth=1.8, label=scenario)
    plt.title("Tren AUC per Ronde (FL + Blockchain)")
    plt.xlabel("Round")
    plt.ylabel("AUC")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_SUBDIRS["visual"], "03_tren_auc_per_ronde.png"), dpi=180)
    plt.close()

    # 4) Stacked bar accepted vs rejected (keamanan)
    plt.figure(figsize=(10, 5))
    sec_df = pd.read_csv(os.path.join(OUTPUT_SUBDIRS["security"], "fl_bc_2_malicious_security.csv"))
    plt.bar(sec_df["round"], sec_df["accepted"], label="accepted", alpha=0.85)
    plt.bar(sec_df["round"], sec_df["rejected"], bottom=sec_df["accepted"], label="rejected", alpha=0.85)
    plt.title("Accepted vs Rejected Client per Ronde (2 Malicious)")
    plt.xlabel("Round")
    plt.ylabel("Jumlah Client")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_SUBDIRS["visual"], "04_keamanan_accepted_vs_rejected.png"), dpi=180)
    plt.close()

    # 5) Plot khusus perbandingan centralized vs FL+blockchain (accuracy)
    plt.figure(figsize=(9, 5))
    acc_compare = pd.DataFrame(
        [
            {"group": "centralized_logreg", "accuracy": float(cen_only[cen_only["scenario"] == "logistic_regression"]["accuracy"].iloc[0])},
            {"group": "centralized_rf", "accuracy": float(cen_only[cen_only["scenario"] == "random_forest"]["accuracy"].iloc[0])},
            {"group": "fl_bc_normal", "accuracy": float(fed_only[fed_only["scenario"] == "fl_bc_normal"]["accuracy"].iloc[0])},
            {"group": "fl_bc_1_malicious", "accuracy": float(fed_only[fed_only["scenario"] == "fl_bc_1_malicious"]["accuracy"].iloc[0])},
            {"group": "fl_bc_2_malicious", "accuracy": float(fed_only[fed_only["scenario"] == "fl_bc_2_malicious"]["accuracy"].iloc[0])},
        ]
    )
    plt.bar(acc_compare["group"], acc_compare["accuracy"], color=["#3B82F6", "#1D4ED8", "#10B981", "#F59E0B", "#EF4444"])
    plt.title("Perbandingan Accuracy Tiap Pendekatan")
    plt.ylabel("Accuracy")
    plt.ylim(0.80, 0.95)
    plt.xticks(rotation=15)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_SUBDIRS["visual"], "05_perbandingan_accuracy.png"), dpi=180)
    plt.close()

    # 6) Heatmap metrik agar cepat dibaca per pendekatan
    heatmap_df = summary[["scenario", "accuracy", "precision", "recall", "f1", "auc"]].copy()
    heatmap_df = heatmap_df.set_index("scenario")
    plt.figure(figsize=(9, 4.8))
    im = plt.imshow(heatmap_df.values, aspect="auto", cmap="YlGnBu", vmin=0.35, vmax=1.0)
    plt.colorbar(im, fraction=0.03, pad=0.04, label="Score")
    plt.xticks(range(len(heatmap_df.columns)), heatmap_df.columns, rotation=15)
    plt.yticks(range(len(heatmap_df.index)), heatmap_df.index)
    plt.title("Heatmap Perbandingan Metrik")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_SUBDIRS["visual"], "06_heatmap_perbandingan_metrik.png"), dpi=180)
    plt.close()

    # 7) Confusion matrix model terbaik centralized (Random Forest)
    best_model = central_models["random_forest"]
    y_pred_rf = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_rf, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Diabetes", "Diabetes"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix - Centralized Random Forest")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_SUBDIRS["visual"], "07_confusion_matrix_random_forest.png"), dpi=180)
    plt.close()

    # 8) Rekap keamanan antar skenario (accepted/rejected pada ronde akhir)
    security_compare = pd.DataFrame(
        [
            {
                "scenario": row["scenario"],
                "accepted_last_round": row.get("accepted_clients_last_round", np.nan),
                "rejected_last_round": row.get("rejected_clients_last_round", np.nan),
            }
            for _, row in fed_only.iterrows()
        ]
    )
    plt.figure(figsize=(8, 5))
    x = np.arange(len(security_compare))
    w = 0.35
    plt.bar(x - w / 2, security_compare["accepted_last_round"], width=w, label="accepted")
    plt.bar(x + w / 2, security_compare["rejected_last_round"], width=w, label="rejected")
    plt.xticks(x, security_compare["scenario"], rotation=10)
    plt.title("Perbandingan Keamanan Antar Skenario (Ronde Akhir)")
    plt.ylabel("Jumlah Client")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_SUBDIRS["visual"], "08_ringkasan_keamanan_antar_skenario.png"), dpi=180)
    plt.close()

    # 9) Feature importance random forest
    importances = pd.DataFrame(
        {
            "feature": [
                "age",
                "hypertension",
                "heart_disease",
                "bmi",
                "HbA1c_level",
                "blood_glucose_level",
                "gender",
                "smoking_history",
            ],
            "importance": central_models["random_forest"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    plt.figure(figsize=(8, 5))
    plt.barh(importances["feature"], importances["importance"])
    plt.gca().invert_yaxis()
    plt.title("Feature Importance - Random Forest")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_SUBDIRS["visual"], "09_feature_importance_random_forest.png"), dpi=180)
    plt.close()

    # Laporan informatif otomatis
    report_path = os.path.join(OUTPUT_DIR, "LAPORAN_INFORMATIF_DATASET.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Laporan Informatif Data-set\n\n")
        f.write("Dokumen ini dibuat otomatis dari pipeline `diabetes_fl_blockchain.py`.\n\n")
        f.write("## Ringkasan Dataset\n")
        f.write(f"- Jumlah baris: {len(raw_df)}\n")
        f.write(f"- Jumlah kolom: {raw_df.shape[1]}\n")
        f.write(f"- Missing values total: {int(raw_df.isnull().sum().sum())}\n")
        f.write(f"- Distribusi label diabetes: {raw_df['diabetes'].value_counts().to_dict()}\n\n")
        f.write("## Struktur Output\n")
        for name, path in OUTPUT_SUBDIRS.items():
            f.write(f"- `{name}`: `{path}`\n")
        f.write("\n## Hasil Ringkas\n")
        f.write("```text\n")
        f.write(summary.to_string(index=False))
        f.write("\n```\n")
        f.write("\n\n## Daftar Visualisasi\n")
        f.write("- EDA: `output/0_eda_visualizations`\n")
        f.write("- Evaluasi model: `output/6_visualizations`\n")
        f.write("- Untuk pembahasan skripsi, gunakan heatmap metrik, confusion matrix, dan keamanan accepted/rejected.\n")
    print(summary.to_string(index=False))
    print(f"\nOutput tersimpan di: {OUTPUT_DIR}")
    print("Struktur:")
    for name, path in OUTPUT_SUBDIRS.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()