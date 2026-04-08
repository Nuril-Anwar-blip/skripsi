"""
================================================================================
SISTEM PREDIKSI DIABETES TIPE 2 DENGAN FEDERATED LEARNING DAN BLOCKCHAIN
================================================================================

File: centralized_ml.py
Deskripsi: Modul Machine Learning Terpusat untuk Perbandingan
             - Multiple ML models (Logistic Regression, Random Forest, KNN, Gradient Boosting)
             - Evaluasi komprehensif dengan berbagai metrik
             - Confusion matrix visualization
             - Feature importance analysis
             - Comparison dengan Federated Learning

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

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

# ============================================================================
# KONFIGURASI
# ============================================================================

BASE_DIR = r"d:\Kampus\Semester 6\Skripsi\Keperluan Skripsi"
OUTPUT_DIR = os.path.join(BASE_DIR, 'FL_Blockchain_Diabetes_Type2', 'terminal', 'output')
IMG_DIR = os.path.join(BASE_DIR, 'FL_Blockchain_Diabetes_Type2', 'img')
RANDOM_STATE = 42


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
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test


def latih_model(model, X_train, y_train, model_name):
    """
    Melatih model machine learning.
    """
    print(f"\nMelatih {model_name}...")
    model.fit(X_train, y_train)
    return model


def evaluasi_model(model, X_test, y_test, model_name):
    """
    Mengevaluasi model dan mengembalikan metrik.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'Model': model_name,
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'Recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
        'F1-Score': round(f1_score(y_test, y_pred, zero_division=0), 4),
        'AUC-ROC': round(roc_auc_score(y_test, y_prob), 4)
    }
    
    return metrics, y_pred, y_prob


def visualisasi_confusion_matrix(y_test, y_pred, model_name, save_path):
    """
    Membuat visualisasi confusion matrix.
    """
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Diabetes', 'Diabetes'],
                yticklabels=['Non-Diabetes', 'Diabetes'])
    plt.xlabel('Prediksi', fontsize=12)
    plt.ylabel('Aktual', fontsize=12)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVE] Confusion matrix disimpan: {save_path}")


def visualisasi_roc_curve(models_results, save_path):
    """
    Membuat visualisasi ROC Curve untuk semua model.
    """
    plt.figure(figsize=(10, 8))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
    
    for i, (model_name, y_prob, y_test) in enumerate(models_results):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, color=colors[i], linewidth=2,
                label=f'{model_name} (AUC = {auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve Comparison - Diabetes Prediction', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVE] ROC curve disimpan: {save_path}")


def visualisasi_feature_importance(rf_model, feature_names, save_path):
    """
    Membuat visualisasi feature importance dari Random Forest.
    """
    importance = rf_model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importance)), importance[indices], color='#3498db', edgecolor='black')
    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Fitur', fontsize=12)
    plt.ylabel('Importance', fontsize=12)
    plt.title('Feature Importance - Random Forest', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVE] Feature importance disimpan: {save_path}")


def visualisasi_perbandingan_model(df_results, save_path):
    """
    Membuat visualisasi perbandingan performa model.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    
    for idx, (metric, color) in enumerate(zip(metrics, colors)):
        ax = axes[idx // 2, idx % 2]
        bars = ax.bar(df_results['Model'], df_results[metric], color=color, edgecolor='black')
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, val in zip(bars, df_results[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=10)
        
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Model Performance Comparison - Centralized ML', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVE] Perbandingan model disimpan: {save_path}")


def jalankan_centralized_ml():
    """
    Menjalankan semua model ML terpusat untuk perbandingan.
    """
    print("\n" + "=" * 80)
    print("CENTRALIZED MACHINE LEARNING - PERBANDINGAN MODEL")
    print("=" * 80)
    
    # Muat data
    X_train, X_test, y_train, y_test = muat_data()
    
    # Definisi model
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=300, random_state=RANDOM_STATE
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=RANDOM_STATE
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=5
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=RANDOM_STATE
        )
    }
    
    # Latih dan evaluasi semua model
    results = []
    models_results = []
    
    for model_name, model in models.items():
        # Latih model
        model = latih_model(model, X_train, y_train, model_name)
        
        # Evaluasi model
        metrics, y_pred, y_prob = evaluasi_model(model, X_test, y_test, model_name)
        results.append(metrics)
        models_results.append((model_name, y_prob, y_test))
        
        # Tampilkan hasil
        print(f"\n{'-'*40}")
        print(f"Model: {model_name}")
        print(f"{'-'*40}")
        print(f"Accuracy:  {metrics['Accuracy']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"Recall:    {metrics['Recall']:.4f}")
        print(f"F1-Score:  {metrics['F1-Score']:.4f}")
        print(f"AUC-ROC:   {metrics['AUC-ROC']:.4f}")
        
        # Confusion matrix
        cm_path = os.path.join(IMG_DIR, f'cm_{model_name.replace(" ", "_")}.png')
        visualisasi_confusion_matrix(y_test, y_pred, model_name, cm_path)
    
    # DataFrame hasil
    df_results = pd.DataFrame(results)
    
    # Visualisasi ROC Curve
    roc_path = os.path.join(IMG_DIR, 'roc_curve_comparison.png')
    visualisasi_roc_curve(models_results, roc_path)
    
    # Visualisasi Feature Importance (dari Random Forest)
    rf_model = models['Random Forest']
    feature_names = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level',
                    'blood_glucose_level', 'gender', 'smoking_history',
                    'bmi_category', 'age_category', 'risk_score', 'high_risk']
    fi_path = os.path.join(IMG_DIR, 'feature_importance.png')
    visualisasi_feature_importance(rf_model, feature_names, fi_path)
    
    # Visualisasi perbandingan model
    comp_path = os.path.join(IMG_DIR, 'centralized_comparison.png')
    visualisasi_perbandingan_model(df_results, comp_path)
    
    # Simpan hasil ke CSV
    df_results.to_csv(os.path.join(OUTPUT_DIR, 'centralized_results.csv'), index=False)
    
    print("\n" + "=" * 80)
    print("HASIL AKHIR CENTRALIZED ML")
    print("=" * 80)
    print(df_results.to_string(index=False))
    
    return df_results


if __name__ == "__main__":
    results = jalankan_centralized_ml()
    print("\n" + "=" * 80)
    print("CENTRALIZED ML SELESAI!")
    print("=" * 80)