"""
=============================================================
  NSL-KDD — IDS Anomaly Detection Pipeline for TinyLM
  Tải dataset, phân tích, tiền xử lý, huấn luyện & đánh giá
=============================================================
"""

import os
import sys
import time
import warnings
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# ============================================================
# CẤU HÌNH
# ============================================================
BASE_DIR = r"d:\datasetTinyLM"
DATA_DIR = os.path.join(BASE_DIR, "NSL-KDD")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# URL tải từ GitHub (raw files)
URLS = {
    "KDDTrain+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
    "KDDTest+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt",
}

# 43 cột: 41 features + label + difficulty_level
COLUMNS = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_hot_login", "is_guest_login",
    "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty_level",
]

# Phân loại tấn công → nhóm
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


# ============================================================
# 1. TẢI DATASET
# ============================================================
def download_dataset():
    """Tải NSL-KDD từ GitHub nếu chưa có."""
    print("\n" + "=" * 60)
    print("  📥  BƯỚC 1: TẢI DATASET NSL-KDD")
    print("=" * 60)

    for filename, url in URLS.items():
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  ✅ {filename} đã tồn tại ({size_mb:.2f} MB)")
            continue

        print(f"  ⬇️  Đang tải {filename}...", end=" ", flush=True)
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            with open(filepath, "wb") as f:
                f.write(resp.content)
            size_mb = len(resp.content) / (1024 * 1024)
            print(f"✅ Xong ({size_mb:.2f} MB)")
        except Exception as e:
            print(f"❌ Lỗi: {e}")
            sys.exit(1)


# ============================================================
# 2. ĐỌC & TIỀN XỬ LÝ
# ============================================================
def load_data():
    """Đọc và tiền xử lý NSL-KDD."""
    print("\n" + "=" * 60)
    print("  📊  BƯỚC 2: ĐỌC & TIỀN XỬ LÝ DỮ LIỆU")
    print("=" * 60)

    # Đọc file
    train_path = os.path.join(DATA_DIR, "KDDTrain+.txt")
    test_path = os.path.join(DATA_DIR, "KDDTest+.txt")

    df_train = pd.read_csv(train_path, header=None, names=COLUMNS)
    df_test = pd.read_csv(test_path, header=None, names=COLUMNS)

    print(f"  📁 Train: {df_train.shape[0]:,} bản ghi × {df_train.shape[1]} cột")
    print(f"  📁 Test:  {df_test.shape[0]:,} bản ghi × {df_test.shape[1]} cột")

    # Loại bỏ cột difficulty_level
    df_train.drop("difficulty_level", axis=1, inplace=True)
    df_test.drop("difficulty_level", axis=1, inplace=True)

    # Gắn nhãn nhóm tấn công
    df_train["attack_category"] = df_train["label"].map(ATTACK_MAP).fillna("Unknown")
    df_test["attack_category"] = df_test["label"].map(ATTACK_MAP).fillna("Unknown")

    # Nhãn nhị phân: Normal vs Attack
    df_train["binary_label"] = (df_train["attack_category"] != "Normal").astype(int)
    df_test["binary_label"] = (df_test["attack_category"] != "Normal").astype(int)

    print(f"\n  📈 Phân bố nhóm tấn công (Train):")
    for cat, cnt in df_train["attack_category"].value_counts().items():
        pct = cnt / len(df_train) * 100
        print(f"      {cat:10s}: {cnt:>7,} ({pct:5.1f}%)")

    print(f"\n  📈 Phân bố nhị phân (Train):")
    print(f"      Normal:  {(df_train['binary_label'] == 0).sum():>7,}")
    print(f"      Attack:  {(df_train['binary_label'] == 1).sum():>7,}")

    return df_train, df_test


def preprocess(df_train, df_test):
    """Mã hóa categorical + chuẩn hóa numerical."""
    print("\n  🔧 Tiền xử lý: Mã hóa & Chuẩn hóa...")

    # Các cột categorical
    cat_cols = ["protocol_type", "service", "flag"]

    # Label Encoding cho categorical
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([df_train[col], df_test[col]], axis=0)
        le.fit(combined)
        df_train[col] = le.transform(df_train[col])
        df_test[col] = le.transform(df_test[col])
        le_dict[col] = le
        print(f"      ✅ {col}: {len(le.classes_)} loại → encoded")

    # Tách features & labels
    drop_cols = ["label", "attack_category", "binary_label"]
    X_train = df_train.drop(columns=drop_cols)
    y_train_binary = df_train["binary_label"]
    y_train_multi = df_train["attack_category"]

    X_test = df_test.drop(columns=drop_cols)
    y_test_binary = df_test["binary_label"]
    y_test_multi = df_test["attack_category"]

    # Chuẩn hóa MinMax
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns
    )

    print(f"      ✅ MinMaxScaler applied: {X_train_scaled.shape[1]} features")
    print(f"      ✅ Giá trị min: {X_train_scaled.min().min():.4f}")
    print(f"      ✅ Giá trị max: {X_train_scaled.max().max():.4f}")

    return (
        X_train_scaled, y_train_binary, y_train_multi,
        X_test_scaled, y_test_binary, y_test_multi,
        df_train, df_test,
    )


# ============================================================
# 3. PHÂN TÍCH KHÁM PHÁ (EDA)
# ============================================================
def exploratory_analysis(df_train, df_test):
    """Tạo biểu đồ EDA."""
    print("\n" + "=" * 60)
    print("  📉  BƯỚC 3: PHÂN TÍCH KHÁM PHÁ (EDA)")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("NSL-KDD — Exploratory Data Analysis", fontsize=16, fontweight="bold", y=0.98)

    # 3a. Phân bố nhóm tấn công (Train)
    attack_counts = df_train["attack_category"].value_counts()
    colors_bar = ["#00d4ff", "#ff4757", "#ffd700", "#00ff88", "#ff6b35"]
    axes[0, 0].barh(attack_counts.index, attack_counts.values, color=colors_bar[:len(attack_counts)])
    axes[0, 0].set_title("Phân bố Nhóm Tấn công (Train)", fontweight="bold")
    axes[0, 0].set_xlabel("Số bản ghi")
    for i, (v, name) in enumerate(zip(attack_counts.values, attack_counts.index)):
        axes[0, 0].text(v + 100, i, f"{v:,}", va="center", fontsize=9)

    # 3b. Pie chart Normal vs Attack
    binary_counts = df_train["binary_label"].value_counts()
    labels_pie = ["Normal", "Attack"]
    axes[0, 1].pie(
        binary_counts.values,
        labels=labels_pie,
        autopct="%1.1f%%",
        colors=["#00ff88", "#ff4757"],
        startangle=90,
        explode=(0.05, 0.05),
        shadow=True,
    )
    axes[0, 1].set_title("Tỷ lệ Normal vs Attack (Train)", fontweight="bold")

    # 3c. Top 10 protocol distribution
    proto_atk = df_train.groupby(["protocol_type", "attack_category"]).size().unstack(fill_value=0)
    proto_atk.plot(kind="bar", stacked=True, ax=axes[1, 0], colormap="Set2")
    axes[1, 0].set_title("Protocol × Nhóm Tấn công", fontweight="bold")
    axes[1, 0].set_xlabel("Protocol Type")
    axes[1, 0].set_ylabel("Số bản ghi")
    axes[1, 0].legend(fontsize=7, loc="upper right")
    axes[1, 0].tick_params(axis='x', rotation=0)

    # 3d. So sánh Train vs Test
    categories = ["Normal", "DoS", "Probe", "R2L", "U2R"]
    train_counts = [df_train[df_train["attack_category"] == c].shape[0] for c in categories]
    test_counts = [df_test[df_test["attack_category"] == c].shape[0] for c in categories]
    x = np.arange(len(categories))
    w = 0.35
    axes[1, 1].bar(x - w / 2, train_counts, w, label="Train", color="#00d4ff", alpha=0.8)
    axes[1, 1].bar(x + w / 2, test_counts, w, label="Test", color="#ff6b35", alpha=0.8)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(categories)
    axes[1, 1].set_title("Train vs Test: Phân bố theo nhóm", fontweight="bold")
    axes[1, 1].set_ylabel("Số bản ghi")
    axes[1, 1].legend()

    plt.tight_layout()
    eda_path = os.path.join(OUTPUT_DIR, "01_eda_analysis.png")
    plt.savefig(eda_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 Đã lưu biểu đồ EDA → {eda_path}")


# ============================================================
# 4. HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH
# ============================================================
def train_and_evaluate(X_train, y_train, X_test, y_test, task_name="Binary"):
    """Huấn luyện nhiều mô hình, so sánh kết quả."""
    print(f"\n{'=' * 60}")
    print(f"  🤖  BƯỚC 4: HUẤN LUYỆN MÔ HÌNH ({task_name})")
    print(f"{'=' * 60}")

    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=20, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, n_jobs=-1
        ),
        "Linear SVM": LinearSVC(max_iter=2000, random_state=42),
    }

    results = {}

    for name, model in models.items():
        print(f"\n  🔹 {name}...")
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        t0 = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - t0

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "train_time": train_time,
            "pred_time": pred_time,
        }

        print(f"      Accuracy:  {acc:.4f}")
        print(f"      Precision: {prec:.4f}")
        print(f"      Recall:    {rec:.4f}")
        print(f"      F1-Score:  {f1:.4f}")
        print(f"      ⏱ Train: {train_time:.2f}s | Predict: {pred_time:.4f}s")

    return results


def plot_results(results, y_test, task_name="Binary"):
    """Vẽ biểu đồ so sánh mô hình + confusion matrix."""
    print(f"\n  📊 Vẽ biểu đồ đánh giá ({task_name})...")

    model_names = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]

    # ---- So sánh metrics ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"NSL-KDD — Model Comparison ({task_name})", fontsize=14, fontweight="bold")

    # Bar chart so sánh
    x = np.arange(len(model_names))
    width = 0.2
    colors_metric = ["#00d4ff", "#00ff88", "#ffd700", "#ff4757"]

    for i, metric in enumerate(metrics):
        values = [results[m][metric] for m in model_names]
        axes[0].bar(x + i * width, values, width, label=metric.capitalize(), color=colors_metric[i])

    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels(model_names, rotation=15, ha="right", fontsize=9)
    axes[0].set_ylim(0.5, 1.05)
    axes[0].set_ylabel("Score")
    axes[0].set_title("So sánh các Metrics", fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    # Thời gian huấn luyện
    train_times = [results[m]["train_time"] for m in model_names]
    bars = axes[1].bar(model_names, train_times, color=colors_metric[:len(model_names)])
    axes[1].set_title("Thời gian Huấn luyện (giây)", fontweight="bold")
    axes[1].set_ylabel("Giây")
    for bar, t in zip(bars, train_times):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            f"{t:.2f}s", ha="center", fontsize=9
        )
    axes[1].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    suffix = task_name.lower().replace(" ", "_")
    compare_path = os.path.join(OUTPUT_DIR, f"02_model_comparison_{suffix}.png")
    plt.savefig(compare_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 Đã lưu → {compare_path}")

    # ---- Confusion Matrices ----
    fig, axes = plt.subplots(1, len(model_names), figsize=(5 * len(model_names), 5))
    fig.suptitle(f"Confusion Matrices ({task_name})", fontsize=14, fontweight="bold")

    if len(model_names) == 1:
        axes = [axes]

    for ax, name in zip(axes, model_names):
        cm = confusion_matrix(y_test, results[name]["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd", ax=ax, cbar=False)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, f"03_confusion_matrices_{suffix}.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 Đã lưu → {cm_path}")


def feature_importance(model, feature_names):
    """Vẽ top 15 features quan trọng nhất."""
    print("\n  🎯 Top 15 Features quan trọng nhất (Random Forest)...")

    importances = model.feature_importances_
    indices = np.argsort(importances)[-15:]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        [feature_names[i] for i in indices],
        importances[indices],
        color="#00d4ff",
        edgecolor="#0a0e1a",
    )
    ax.set_title("Top 15 Feature Importance (Random Forest)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()

    fi_path = os.path.join(OUTPUT_DIR, "04_feature_importance.png")
    plt.savefig(fi_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 Đã lưu → {fi_path}")


def save_summary(results_binary, results_multi):
    """Lưu báo cáo tổng kết."""
    summary_path = os.path.join(OUTPUT_DIR, "05_summary_report.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write("  NSL-KDD IDS ANOMALY DETECTION — BÁO CÁO TỔNG KẾT\n")
        f.write("=" * 65 + "\n\n")

        for task_name, results in [("BINARY (Normal vs Attack)", results_binary),
                                   ("MULTI-CLASS (5 nhóm)", results_multi)]:
            f.write(f"\n{'─' * 60}\n")
            f.write(f"  {task_name}\n")
            f.write(f"{'─' * 60}\n")
            f.write(f"  {'Model':<22s} {'Acc':>7s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'Train':>8s}\n")
            f.write(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*8}\n")

            for name, r in results.items():
                f.write(
                    f"  {name:<22s} {r['accuracy']:>7.4f} {r['precision']:>7.4f} "
                    f"{r['recall']:>7.4f} {r['f1']:>7.4f} {r['train_time']:>7.2f}s\n"
                )

        # Best model
        best_name = max(results_binary, key=lambda k: results_binary[k]["f1"])
        best = results_binary[best_name]
        f.write(f"\n\n{'=' * 65}\n")
        f.write(f"  🏆 MÔ HÌNH TỐT NHẤT (Binary): {best_name}\n")
        f.write(f"     F1-Score: {best['f1']:.4f} | Accuracy: {best['accuracy']:.4f}\n")
        f.write(f"     Thời gian train: {best['train_time']:.2f}s\n")
        f.write(f"{'=' * 65}\n")

    print(f"\n  📝 Đã lưu báo cáo → {summary_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "🔷" * 30)
    print("  NSL-KDD — IDS ANOMALY DETECTION PIPELINE")
    print("  Phát hiện bất thường hệ thống xâm nhập mạng")
    print("🔷" * 30)

    # 1. Tải dataset
    download_dataset()

    # 2. Đọc & tiền xử lý
    df_train, df_test = load_data()
    (
        X_train, y_train_binary, y_train_multi,
        X_test, y_test_binary, y_test_multi,
        df_train, df_test,
    ) = preprocess(df_train, df_test)

    # 3. EDA
    exploratory_analysis(df_train, df_test)

    # 4a. Huấn luyện nhị phân (Normal vs Attack)
    results_binary = train_and_evaluate(
        X_train, y_train_binary, X_test, y_test_binary, task_name="Binary"
    )
    plot_results(results_binary, y_test_binary, task_name="Binary")

    # 4b. Huấn luyện đa lớp (5 nhóm: Normal, DoS, Probe, R2L, U2R)
    results_multi = train_and_evaluate(
        X_train, y_train_multi, X_test, y_test_multi, task_name="Multi-Class"
    )
    plot_results(results_multi, y_test_multi, task_name="Multi-Class")

    # 5. Feature importance (từ Random Forest binary)
    rf_model = results_binary["Random Forest"]["model"]
    feature_importance(rf_model, X_train.columns.tolist())

    # 6. Classification report chi tiết (best model)
    best_name = max(results_binary, key=lambda k: results_binary[k]["f1"])
    print(f"\n{'=' * 60}")
    print(f"  📋 CLASSIFICATION REPORT — {best_name} (Binary)")
    print(f"{'=' * 60}")
    print(classification_report(
        y_test_binary,
        results_binary[best_name]["y_pred"],
        target_names=["Normal", "Attack"],
    ))

    best_name_multi = max(results_multi, key=lambda k: results_multi[k]["f1"])
    print(f"\n{'=' * 60}")
    print(f"  📋 CLASSIFICATION REPORT — {best_name_multi} (Multi-Class)")
    print(f"{'=' * 60}")
    print(classification_report(
        y_test_multi,
        results_multi[best_name_multi]["y_pred"],
    ))

    # 7. Lưu báo cáo
    save_summary(results_binary, results_multi)

    # Kết thúc
    print("\n" + "🎉" * 20)
    print("  HOÀN TẤT! Kết quả đã được lưu tại:")
    print(f"  📂 {OUTPUT_DIR}")
    print("  Các file đầu ra:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(fpath) / 1024
        print(f"      📄 {f} ({size:.1f} KB)")
    print("🎉" * 20)


if __name__ == "__main__":
    main()
