# Laporan Informatif Data-set

Dokumen ini dibuat otomatis dari pipeline `diabetes_fl_blockchain.py`.

## Ringkasan Dataset
- Jumlah baris: 96146
- Jumlah kolom: 9
- Missing values total: 0
- Distribusi label diabetes: {0: 87664, 1: 8482}

## Struktur Output
- `eda`: `output\0_eda_visualizations`
- `baseline`: `output\1_baseline`
- `history`: `output\2_federated_history`
- `security`: `output\3_security_logs`
- `ledger`: `output\4_blockchain_ledger`
- `summary`: `output\5_summary`
- `visual`: `output\6_visualizations`

## Hasil Ringkas
```text
            approach            scenario  accuracy  precision  recall     f1    auc  accepted_clients_last_round  rejected_clients_last_round ledger_valid
         centralized logistic_regression    0.8809     0.4152  0.8573 0.5594 0.9560                          NaN                          NaN          NaN
         centralized       random_forest    0.9274     0.5584  0.8455 0.6726 0.9723                          NaN                          NaN          NaN
federated_blockchain        fl_bc_normal    0.8810     0.4154  0.8573 0.5597 0.9560                          5.0                          0.0         True
federated_blockchain   fl_bc_1_malicious    0.8811     0.4157  0.8573 0.5599 0.9560                          4.0                          1.0         True
federated_blockchain   fl_bc_2_malicious    0.8812     0.4159  0.8573 0.5601 0.9560                          3.0                          2.0         True
```


## Daftar Visualisasi
- EDA: `output/0_eda_visualizations`
- Evaluasi model: `output/6_visualizations`
- Untuk pembahasan skripsi, gunakan heatmap metrik, confusion matrix, dan keamanan accepted/rejected.
