# -*- coding: utf-8 -*-
"""
=============================================================
  NSL-KDD — Deep Learning IDS Pipeline
  Phát hiện xâm nhập mạng bằng DNN, Autoencoder
=============================================================
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

# UTF-8 support cho Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, Model
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# ============================================================
# CẤU HÌNH
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "NSL-KDD")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ATTACK_MAP = {
    "normal": "Normal",
    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS",
    "processtable": "DoS", "mailbomb": "DoS",
    # Probe
    "satan": "Probe", "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "mscan": "Probe", "saint": "Probe",
    # R2L
    "guess_passwd": "R2L", "ftp_write": "R2L", "imap": "R2L", "phf": "R2L",
    "multihop": "R2L", "warezmaster": "R2L", "warezclient": "R2L", "spy": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "snmpguess": "R2L", "snmpgetattack": "R2L",
    "httptunnel": "R2L", "sendmail": "R2L", "named": "R2L", "worm": "R2L",
    # U2R
    "buffer_overflow": "U2R", "loadmodule": "U2R", "rootkit": "U2R", "perl": "U2R",
    "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R",
}

CATEGORIES = ["Normal", "DoS", "Probe", "R2L", "U2R"]

# Màu sắc chung
COLORS = {
    "DNN":         "#2563eb",
    "Autoencoder": "#d97706",
    "Random Forest":   "#dc2626",
    "Decision Tree":   "#64748b",
}
METRIC_COLORS = ["#2563eb", "#059669", "#d97706", "#dc2626"]


# ============================================================
# 1. TẢI & TIỀN XỬ LÝ DỮ LIỆU
# ============================================================
def load_and_preprocess():
    """Đọc và tiền xử lý NSL-KDD."""
    print("\n" + "=" * 60)
    print("  📊  BƯỚC 1: TẢI & TIỀN XỬ LÝ DỮ LIỆU")
    print("=" * 60)

    df_train = pd.read_csv(os.path.join(DATA_DIR, "KDDTrain+.csv"))
    df_test = pd.read_csv(os.path.join(DATA_DIR, "KDDTest+.csv"))

    print(f"  📁 Train: {df_train.shape[0]:,} × {df_train.shape[1]} cột")
    print(f"  📁 Test:  {df_test.shape[0]:,} × {df_test.shape[1]} cột")

    # Loại bỏ difficulty_level
    for df in (df_train, df_test):
        if "difficulty_level" in df.columns:
            df.drop("difficulty_level", axis=1, inplace=True)

    # Nhãn nhóm tấn công
    df_train["attack_cat"] = df_train["label"].map(ATTACK_MAP).fillna("Unknown")
    df_test["attack_cat"]  = df_test["label"].map(ATTACK_MAP).fillna("Unknown")

    # Nhãn nhị phân
    df_train["binary"] = (df_train["attack_cat"] != "Normal").astype(int)
    df_test["binary"]  = (df_test["attack_cat"]  != "Normal").astype(int)

    print(f"\n  📈 Phân bố nhóm tấn công (Train):")
    for cat, cnt in df_train["attack_cat"].value_counts().items():
        print(f"      {cat:10s}: {cnt:>7,} ({cnt/len(df_train)*100:5.1f}%)")

    # Label Encoding cho categorical
    cat_cols = ["protocol_type", "service", "flag"]
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([df_train[col], df_test[col]])
        le.fit(combined)
        df_train[col] = le.transform(df_train[col])
        df_test[col]  = le.transform(df_test[col])
        print(f"      ✅ {col}: {len(le.classes_)} loại → encoded")

    # Tách features & labels
    drop_cols = ["label", "attack_cat", "binary"]
    X_train = df_train.drop(columns=drop_cols).values.astype("float32")
    X_test  = df_test.drop(columns=drop_cols).values.astype("float32")

    y_train_bin = df_train["binary"].values
    y_test_bin  = df_test["binary"].values

    # Encode multi-class (5 lớp)
    le_multi = LabelEncoder()
    le_multi.fit(CATEGORIES)
    # Gộp Unknown → lớp gần nhất (DoS) để tránh lỗi
    train_cats = df_train["attack_cat"].replace("Unknown", "DoS")
    test_cats  = df_test["attack_cat"].replace("Unknown", "DoS")
    y_train_multi = le_multi.transform(train_cats)
    y_test_multi  = le_multi.transform(test_cats)

    # MinMaxScaler
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    n_features = X_train.shape[1]
    print(f"\n  ✅ Features: {n_features}")
    print(f"  ✅ Binary: Normal / Attack")
    print(f"  ✅ Multi-class: {CATEGORIES}")

    return (X_train, y_train_bin, y_train_multi,
            X_test, y_test_bin, y_test_multi,
            le_multi, n_features)


# ============================================================
# 2. XÂY DỰNG MÔ HÌNH
# ============================================================
def build_dnn(n_features, num_classes=1, task="binary"):
    """Deep Neural Network — Fully Connected."""
    model = keras.Sequential(name="DNN")
    model.add(layers.Input(shape=(n_features,)))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

    if task == "binary":
        model.add(layers.Dense(1, activation="sigmoid"))
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    else:
        model.add(layers.Dense(num_classes, activation="softmax"))
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_autoencoder(n_features):
    """Autoencoder cho phát hiện bất thường."""
    inp = layers.Input(shape=(n_features,))
    # Encoder
    x = layers.Dense(32, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    encoded = layers.Dense(8, activation="relu")(x)
    # Decoder
    x = layers.Dense(16, activation="relu")(encoded)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    decoded = layers.Dense(n_features, activation="sigmoid")(x)

    autoencoder = Model(inp, decoded, name="Autoencoder")
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder


# ============================================================
# 3. HUẤN LUYỆN & ĐÁNH GIÁ
# ============================================================
def get_callbacks():
    return [
        callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=0),
    ]


def train_supervised(model, X_tr, y_tr, name, epochs=50, batch_size=512):
    """Huấn luyện mô hình supervised (DNN / CNN / LSTM)."""
    print(f"\n  🔹 {name}...")
    t0 = time.time()
    history = model.fit(
        X_tr, y_tr,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=get_callbacks(),
        verbose=0,
    )
    train_time = time.time() - t0
    ep = len(history.history["loss"])
    print(f"      ⏱ {train_time:.1f}s  ({ep} epochs)")
    print(f"      Loss {history.history['loss'][-1]:.4f}  |  Val Loss {history.history['val_loss'][-1]:.4f}")
    print(f"      Acc  {history.history['accuracy'][-1]:.4f}  |  Val Acc  {history.history['val_accuracy'][-1]:.4f}")
    return history, train_time


def evaluate(model, X_te, y_te, task="binary"):
    """Đánh giá mô hình, trả về dict metrics + y_pred."""
    if task == "binary":
        y_pred = (model.predict(X_te, verbose=0) > 0.5).astype(int).flatten()
    else:
        y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)

    return {
        "y_pred":    y_pred,
        "accuracy":  accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred, average="weighted", zero_division=0),
        "recall":    recall_score(y_te, y_pred, average="weighted", zero_division=0),
        "f1":        f1_score(y_te, y_pred, average="weighted", zero_division=0),
    }


def run_dl_models(X_train, y_train, X_test, y_test, n_features,
                  task="binary", num_classes=1):
    """Huấn luyện & đánh giá DNN, CNN, LSTM cho một task."""
    print(f"\n{'=' * 60}")
    print(f"  🤖  HUẤN LUYỆN DEEP LEARNING — {task.upper()}")
    print(f"{'=' * 60}")

    builders = {
        "DNN":    (build_dnn,  X_train, X_test),
    }

    results = {}
    histories = {}

    for name, (builder, X_tr, X_te) in builders.items():
        model = builder(n_features, num_classes=num_classes, task=task)
        ep = 30
        bs = 512
        hist, t = train_supervised(model, X_tr, y_train, f"{name} ({task})", epochs=ep, batch_size=bs)
        metrics = evaluate(model, X_te, y_test, task=task)
        metrics["train_time"] = t
        results[name] = metrics
        histories[name] = hist

        print(f"      ── Test ──  Acc {metrics['accuracy']:.4f}  "
              f"Prec {metrics['precision']:.4f}  "
              f"Rec {metrics['recall']:.4f}  "
              f"F1 {metrics['f1']:.4f}")

    return results, histories


def run_autoencoder(X_train, y_train_bin, X_test, y_test_bin, n_features):
    """Huấn luyện Autoencoder chỉ trên dữ liệu Normal."""
    print(f"\n{'=' * 60}")
    print(f"  🔬  HUẤN LUYỆN AUTOENCODER (Anomaly Detection)")
    print(f"{'=' * 60}")

    # Chỉ lấy dữ liệu Normal để huấn luyện
    X_normal = X_train[y_train_bin == 0]
    print(f"  📁 Dữ liệu Normal cho training: {X_normal.shape[0]:,} bản ghi")

    ae = build_autoencoder(n_features)

    print(f"\n  🔹 Autoencoder (Binary — Anomaly Detection)...")
    t0 = time.time()
    history = ae.fit(
        X_normal, X_normal,
        epochs=50,
        batch_size=512,
        validation_split=0.2,
        callbacks=get_callbacks(),
        verbose=0,
    )
    train_time = time.time() - t0
    ep = len(history.history["loss"])
    print(f"      ⏱ {train_time:.1f}s  ({ep} epochs)")
    print(f"      MSE Loss {history.history['loss'][-1]:.6f}")

    # Tính ngưỡng từ reconstruction error trên tập Normal
    recon_normal = ae.predict(X_normal, verbose=0)
    mse_normal = np.mean(np.square(X_normal - recon_normal), axis=1)
    threshold = np.mean(mse_normal) + 2 * np.std(mse_normal)
    print(f"      Ngưỡng phát hiện: {threshold:.6f}")

    # Dự đoán trên tập test
    recon_test = ae.predict(X_test, verbose=0)
    mse_test = np.mean(np.square(X_test - recon_test), axis=1)
    y_pred = (mse_test > threshold).astype(int)

    acc  = accuracy_score(y_test_bin, y_pred)
    prec = precision_score(y_test_bin, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test_bin, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test_bin, y_pred, average="weighted", zero_division=0)

    print(f"      ── Test ──  Acc {acc:.4f}  Prec {prec:.4f}  Rec {rec:.4f}  F1 {f1:.4f}")

    result = {
        "y_pred": y_pred, "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1, "train_time": train_time,
        "threshold": threshold, "mse_test": mse_test,
    }
    return result, history


def run_ml_baselines(X_train, y_train, X_test, y_test, task="binary"):
    """Huấn luyện ML baselines (RF + DT) để so sánh."""
    print(f"\n{'=' * 60}")
    print(f"  📐  ML BASELINES — {task.upper()}")
    print(f"{'=' * 60}")

    models = {
        "Random Forest":  RandomForestClassifier(n_estimators=100, max_depth=20,
                                                  random_state=42, n_jobs=-1),
        "Decision Tree":  DecisionTreeClassifier(max_depth=20, random_state=42),
    }

    results = {}
    for name, model in models.items():
        print(f"\n  🔹 {name}...")
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        y_pred = model.predict(X_test)
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1val = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results[name] = {
            "y_pred": y_pred, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1val, "train_time": train_time,
        }
        print(f"      ⏱ {train_time:.1f}s")
        print(f"      Acc {acc:.4f}  Prec {prec:.4f}  Rec {rec:.4f}  F1 {f1val:.4f}")

    return results


# ============================================================
# 4. BIỂU ĐỒ
# ============================================================
def plot_training_history(histories, ae_history, task_name="binary"):
    """Vẽ Loss & Accuracy trong quá trình huấn luyện."""
    print(f"\n  📊 Vẽ biểu đồ Training History ({task_name})...")
    n_models = len(histories) + (1 if ae_history else 0)
    fig, axes = plt.subplots(n_models, 2, figsize=(14, 4 * n_models))
    if n_models == 1:
        axes = np.expand_dims(axes, axis=0)
    fig.suptitle(f"Training History — {task_name.title()}", fontsize=15, fontweight="bold", y=1.01)

    all_items = list(histories.items())
    if ae_history:
        all_items.append(("Autoencoder", ae_history))

    for row, (name, hist) in enumerate(all_items):
        h = hist.history
        color = COLORS.get(name, "#333")

        # Loss
        axes[row, 0].plot(h["loss"], label="Train Loss", color=color, linewidth=2)
        axes[row, 0].plot(h["val_loss"], label="Val Loss", color=color, linewidth=2, linestyle="--")
        axes[row, 0].set_title(f"{name} — Loss", fontweight="bold", fontsize=11)
        axes[row, 0].set_xlabel("Epoch")
        axes[row, 0].set_ylabel("Loss")
        axes[row, 0].legend(fontsize=8)
        axes[row, 0].grid(True, alpha=0.3)

        # Accuracy (Autoencoder không có)
        if "accuracy" in h:
            axes[row, 1].plot(h["accuracy"], label="Train Acc", color=color, linewidth=2)
            axes[row, 1].plot(h["val_accuracy"], label="Val Acc", color=color, linewidth=2, linestyle="--")
            axes[row, 1].set_title(f"{name} — Accuracy", fontweight="bold", fontsize=11)
            axes[row, 1].set_ylabel("Accuracy")
            axes[row, 1].legend(fontsize=8)
        else:
            axes[row, 1].text(0.5, 0.5, "Autoencoder\n(Unsupervised — không có Accuracy)",
                              ha="center", va="center", fontsize=12, color="#888",
                              transform=axes[row, 1].transAxes)
            axes[row, 1].set_title(f"{name} — MSE Reconstruction", fontweight="bold", fontsize=11)
        axes[row, 1].set_xlabel("Epoch")
        axes[row, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"06_dl_training_history_{task_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 → {path}")


def plot_model_comparison(all_results, task_name="binary"):
    """Bar chart so sánh tất cả mô hình."""
    print(f"\n  📊 Vẽ biểu đồ Model Comparison ({task_name})...")
    names = list(all_results.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Model Comparison — {task_name.title()}", fontsize=15, fontweight="bold")

    # --- Metrics bar chart ---
    x = np.arange(len(names))
    w = 0.18
    for i, (m, ml) in enumerate(zip(metrics, metric_labels)):
        vals = [all_results[n][m] for n in names]
        axes[0].bar(x + i * w, vals, w, label=ml, color=METRIC_COLORS[i])

    axes[0].set_xticks(x + w * 1.5)
    axes[0].set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    axes[0].set_ylim(0.5, 1.05)
    axes[0].set_ylabel("Score")
    axes[0].set_title("So sánh các Metrics", fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    # --- Training time ---
    times = [all_results[n]["train_time"] for n in names]
    bars_c = [COLORS.get(n, "#333") for n in names]
    bars = axes[1].bar(names, times, color=bars_c)
    axes[1].set_title("Thời gian Huấn luyện (giây)", fontweight="bold")
    axes[1].set_ylabel("Giây")
    for bar, t in zip(bars, times):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{t:.1f}s", ha="center", fontsize=9)
    axes[1].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    sfx = task_name.lower().replace(" ", "_")
    path = os.path.join(OUTPUT_DIR, f"07_dl_model_comparison_{sfx}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 → {path}")


def plot_confusion_matrices(all_results, y_test, task_name="binary", label_names=None):
    """Confusion Matrix cho từng mô hình."""
    print(f"\n  📊 Vẽ Confusion Matrices ({task_name})...")
    names = list(all_results.keys())
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    fig.suptitle(f"Confusion Matrices — {task_name.title()}", fontsize=14, fontweight="bold")
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        cm = confusion_matrix(y_test, all_results[name]["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd", ax=ax, cbar=False,
                    xticklabels=label_names, yticklabels=label_names)
        acc = all_results[name]["accuracy"]
        ax.set_title(f"{name}\nAcc={acc:.4f}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    sfx = task_name.lower().replace(" ", "_")
    path = os.path.join(OUTPUT_DIR, f"08_dl_confusion_matrices_{sfx}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 → {path}")


def plot_dl_vs_ml(dl_results, ml_results, task_name="binary"):
    """So sánh DL vs ML."""
    print(f"\n  📊 Vẽ biểu đồ DL vs ML ({task_name})...")
    all_r = {}
    all_r.update(dl_results)
    all_r.update(ml_results)

    names = list(all_r.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(f"Deep Learning vs Machine Learning — {task_name.title()}",
                 fontsize=15, fontweight="bold")

    x = np.arange(len(names))
    w = 0.18

    for i, (m, ml) in enumerate(zip(metrics, metric_labels)):
        vals = [all_r[n][m] for n in names]
        ax.bar(x + i * w, vals, w, label=ml, color=METRIC_COLORS[i])

    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("Score", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Vạch phân cách DL | ML
    sep = len(dl_results) - 0.5
    ax.axvline(x=sep, color="#aaa", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(sep - 0.1, 0.52, "← Deep Learning", ha="right", fontsize=9, color="#555")
    ax.text(sep + 0.1, 0.52, "Machine Learning →", ha="left", fontsize=9, color="#555")

    plt.tight_layout()
    sfx = task_name.lower().replace(" ", "_")
    path = os.path.join(OUTPUT_DIR, f"09_dl_vs_ml_{sfx}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 → {path}")


def save_summary(results_bin, results_multi):
    """Lưu báo cáo tổng kết."""
    path = os.path.join(OUTPUT_DIR, "10_dl_summary_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write("  NSL-KDD DEEP LEARNING IDS — BÁO CÁO TỔNG KẾT\n")
        f.write("=" * 65 + "\n\n")

        for task_name, results in [("BINARY (Normal vs Attack)", results_bin),
                                   ("MULTI-CLASS (5 nhóm)", results_multi)]:
            f.write(f"\n{'─' * 60}\n  {task_name}\n{'─' * 60}\n")
            f.write(f"  {'Model':<22s} {'Acc':>7s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'Train':>8s}\n")
            f.write(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*8}\n")
            for name, r in results.items():
                f.write(f"  {name:<22s} {r['accuracy']:>7.4f} {r['precision']:>7.4f} "
                        f"{r['recall']:>7.4f} {r['f1']:>7.4f} {r['train_time']:>7.1f}s\n")

        # Best model
        best_name = max(results_bin, key=lambda k: results_bin[k]["f1"])
        best = results_bin[best_name]
        f.write(f"\n\n{'=' * 65}\n")
        f.write(f"  🏆 MÔ HÌNH TỐT NHẤT (Binary): {best_name}\n")
        f.write(f"     F1-Score: {best['f1']:.4f} | Accuracy: {best['accuracy']:.4f}\n")
        f.write(f"     Thời gian train: {best['train_time']:.1f}s\n")
        f.write(f"{'=' * 65}\n")

    print(f"\n  📝 Đã lưu báo cáo → {path}")


# ============================================================
# MAIN
# ============================================================
def main():
    total_start = time.time()

    print("\n" + "🔷" * 30)
    print("  NSL-KDD — DEEP LEARNING IDS PIPELINE")
    print("  DNN  ·  Autoencoder")
    print("🔷" * 30)

    # ── 1. Load data ──
    (X_train, y_train_bin, y_train_multi,
     X_test,  y_test_bin,  y_test_multi,
     le_multi, n_features) = load_and_preprocess()

    # ── 2. Binary — DL ──
    dl_bin, hist_bin = run_dl_models(
        X_train, y_train_bin, X_test, y_test_bin,
        n_features, task="binary", num_classes=1,
    )

    # ── 3. Binary — Autoencoder ──
    ae_result, ae_hist = run_autoencoder(
        X_train, y_train_bin, X_test, y_test_bin, n_features,
    )
    dl_bin["Autoencoder"] = ae_result

    # ── 4. Multi-class — DL ──
    n_classes = len(CATEGORIES)
    dl_multi, hist_multi = run_dl_models(
        X_train, y_train_multi, X_test, y_test_multi,
        n_features, task="multi-class", num_classes=n_classes,
    )

    # ── 5. ML Baselines ──
    ml_bin   = run_ml_baselines(X_train, y_train_bin,   X_test, y_test_bin,   task="binary")
    ml_multi = run_ml_baselines(X_train, y_train_multi, X_test, y_test_multi, task="multi-class")

    # ── 6. Biểu đồ ──
    print(f"\n{'=' * 60}")
    print(f"  📊  VẼ BIỂU ĐỒ")
    print(f"{'=' * 60}")

    # Training history
    plot_training_history(hist_bin, ae_hist, task_name="binary")
    plot_training_history(hist_multi, None, task_name="multi-class")

    # Model comparison (DL only)
    plot_model_comparison(dl_bin, task_name="binary")
    plot_model_comparison(dl_multi, task_name="multi-class")

    # Confusion matrices — gộp DL + ML
    all_bin = {**dl_bin, **ml_bin}
    all_multi = {**dl_multi, **ml_multi}

    plot_confusion_matrices(all_bin, y_test_bin, task_name="binary",
                           label_names=["Normal", "Attack"])
    plot_confusion_matrices(all_multi, y_test_multi, task_name="multi-class",
                           label_names=CATEGORIES)

    # DL vs ML
    plot_dl_vs_ml(dl_bin, ml_bin, task_name="binary")
    plot_dl_vs_ml(dl_multi, ml_multi, task_name="multi-class")

    # ── 7. Classification Reports ──
    best_dl = max(dl_bin, key=lambda k: dl_bin[k]["f1"])
    print(f"\n{'=' * 60}")
    print(f"  📋 CLASSIFICATION REPORT — {best_dl} (Binary)")
    print(f"{'=' * 60}")
    print(classification_report(y_test_bin, dl_bin[best_dl]["y_pred"],
                                target_names=["Normal", "Attack"]))

    best_dl_m = max(dl_multi, key=lambda k: dl_multi[k]["f1"])
    print(f"\n{'=' * 60}")
    print(f"  📋 CLASSIFICATION REPORT — {best_dl_m} (Multi-Class)")
    print(f"{'=' * 60}")
    print(classification_report(y_test_multi, dl_multi[best_dl_m]["y_pred"],
                                target_names=CATEGORIES))

    # ── 8. Summary ──
    save_summary(all_bin, all_multi)

    # ── Done ──
    total_time = time.time() - total_start
    print(f"\n{'🎉' * 20}")
    print(f"  HOÀN TẤT! Tổng thời gian: {total_time:.1f}s")
    print(f"  📂 Kết quả: {OUTPUT_DIR}")
    print(f"  Các file đầu ra:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        sz = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
        print(f"      📄 {f} ({sz:.1f} KB)")
    print(f"{'🎉' * 20}")


if __name__ == "__main__":
    main()
