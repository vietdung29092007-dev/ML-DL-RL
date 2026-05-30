# -*- coding: utf-8 -*-
"""
=============================================================
  NSL-KDD — Reinforcement Learning IDS Pipeline
  Phát hiện xâm nhập mạng bằng DQN và PPO (Stable-Baselines3)
=============================================================
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# UTF-8 support cho Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings("ignore")

# ============================================================
# CẤU HÌNH
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "NSL-KDD")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ATTACK_MAP = {
    "normal": "Normal",
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS",
    "processtable": "DoS", "mailbomb": "DoS",
    "satan": "Probe", "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "mscan": "Probe", "saint": "Probe",
    "guess_passwd": "R2L", "ftp_write": "R2L", "imap": "R2L", "phf": "R2L",
    "multihop": "R2L", "warezmaster": "R2L", "warezclient": "R2L", "spy": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "snmpguess": "R2L", "snmpgetattack": "R2L",
    "httptunnel": "R2L", "sendmail": "R2L", "named": "R2L", "worm": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "rootkit": "U2R", "perl": "U2R",
    "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R",
}

# ============================================================
# 1. TẢI & TIỀN XỬ LÝ DỮ LIỆU
# ============================================================
def load_and_preprocess():
    print("\n" + "=" * 60)
    print("  📊  BƯỚC 1: TẢI & TIỀN XỬ LÝ DỮ LIỆU")
    print("=" * 60)

    df_train = pd.read_csv(os.path.join(DATA_DIR, "KDDTrain+.csv"))
    df_test = pd.read_csv(os.path.join(DATA_DIR, "KDDTest+.csv"))

    for df in (df_train, df_test):
        if "difficulty_level" in df.columns:
            df.drop("difficulty_level", axis=1, inplace=True)

    df_train["attack_cat"] = df_train["label"].map(ATTACK_MAP).fillna("Unknown")
    df_test["attack_cat"]  = df_test["label"].map(ATTACK_MAP).fillna("Unknown")

    # Binary: 0 = Normal, 1 = Attack
    df_train["binary"] = (df_train["attack_cat"] != "Normal").astype(int)
    df_test["binary"]  = (df_test["attack_cat"]  != "Normal").astype(int)

    cat_cols = ["protocol_type", "service", "flag"]
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([df_train[col], df_test[col]])
        le.fit(combined)
        df_train[col] = le.transform(df_train[col])
        df_test[col]  = le.transform(df_test[col])

    drop_cols = ["label", "attack_cat", "binary"]
    X_train = df_train.drop(columns=drop_cols).values.astype("float32")
    X_test  = df_test.drop(columns=drop_cols).values.astype("float32")
    y_train = df_train["binary"].values
    y_test  = df_test["binary"].values

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"  📁 Train: {X_train.shape[0]:,} mẫu")
    print(f"  📁 Test:  {X_test.shape[0]:,} mẫu")
    print(f"  ✅ Features: {X_train.shape[1]}")
    
    return X_train, y_train, X_test, y_test

# ============================================================
# 2. CUSTOM GYM ENVIRONMENT
# ============================================================
class NSLKDDEnv(gym.Env):
    """Môi trường RL giả lập phân loại kết nối mạng."""
    def __init__(self, X, y):
        super(NSLKDDEnv, self).__init__()
        self.X = X
        self.y = y
        self.n_samples = len(X)
        self.n_features = X.shape[1]
        
        # Action Space: 0 = Normal, 1 = Attack
        self.action_space = spaces.Discrete(2)
        
        # Observation Space: Đặc trưng kết nối (đã scale về [0, 1])
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_features,), dtype=np.float32
        )
        
        self.current_step = 0
        self.correct_predictions = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.correct_predictions = 0
        return self.X[self.current_step], {}

    def step(self, action):
        true_label = self.y[self.current_step]
        
        # Reward logic: +1 nếu đúng, -1 nếu sai
        if action == true_label:
            reward = 1.0
            self.correct_predictions += 1
        else:
            reward = -1.0
            
        self.current_step += 1
        
        terminated = self.current_step >= self.n_samples
        truncated = False
        
        # Nếu terminated, obs có thể là 0
        obs = self.X[self.current_step] if not terminated else np.zeros(self.n_features, dtype=np.float32)
        
        return obs, reward, terminated, truncated, {}

# Callback lưu lại thông số quá trình huấn luyện
class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.current_reward = 0.0

    def _on_step(self) -> bool:
        self.current_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_reward)
            self.current_reward = 0.0
        return True

# ============================================================
# 3. HUẤN LUYỆN & ĐÁNH GIÁ
# ============================================================
def train_and_evaluate_rl(algo_class, algo_name, X_train, y_train, X_test, y_test, total_timesteps=50000):
    print(f"\n{'=' * 60}")
    print(f"  🤖  HUẤN LUYỆN {algo_name}")
    print(f"{'=' * 60}")
    
    # Rút ngắn dataset cho RL trên CPU (huấn luyện 50k steps)
    env_train = NSLKDDEnv(X_train, y_train)
    
    model = algo_class("MlpPolicy", env_train, verbose=0)
    callback = RewardLoggerCallback()
    
    t0 = time.time()
    print(f"  🔹 Đang huấn luyện {algo_name} ({total_timesteps:,} steps)...")
    model.learn(total_timesteps=total_timesteps, callback=callback)
    train_time = time.time() - t0
    print(f"      ⏱ Thời gian: {train_time:.1f}s")
    
    # ── Đánh giá ──
    print(f"  🔹 Đang đánh giá {algo_name} trên Test Set...")
    env_test = NSLKDDEnv(X_test, y_test)
    obs, _ = env_test.reset()
    
    y_pred = []
    t_test0 = time.time()
    for _ in range(len(X_test)):
        action, _states = model.predict(obs, deterministic=True)
        y_pred.append(action)
        obs, reward, terminated, truncated, info = env_test.step(action)
        if terminated:
            break
            
    y_pred = np.array(y_pred)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "train_time": train_time,
        "y_pred": y_pred,
        "episode_rewards": callback.episode_rewards
    }
    
    print(f"      ── Test ──  Acc {metrics['accuracy']:.4f}  "
          f"Prec {metrics['precision']:.4f}  "
          f"Rec {metrics['recall']:.4f}  "
          f"F1 {metrics['f1']:.4f}")
          
    return metrics

# ============================================================
# 4. TRỰC QUAN HÓA
# ============================================================
def plot_rl_results(results_dict, y_test):
    print(f"\n{'=' * 60}")
    print(f"  📊  VẼ BIỂU ĐỒ RL")
    print(f"{'=' * 60}")
    
    # 1. Episode Rewards
    plt.figure(figsize=(10, 5))
    for name, res in results_dict.items():
        # Do RL môi trường này 1 episode = toàn bộ tập train
        # Tuy nhiên với total_timesteps nhỏ hơn n_samples, episode không kết thúc tự nhiên
        # Stable-Baselines có thể gọi done do TimeLimit (nếu có)
        rewards = res["episode_rewards"]
        if len(rewards) > 0:
            plt.plot(rewards, label=f"{name} Episode Rewards")
    plt.title("Reward qua các Episode (nếu có kết thúc)", fontweight="bold")
    plt.xlabel("Episode")
    plt.ylabel("Tổng Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    path1 = os.path.join(OUTPUT_DIR, "rl_01_rewards.png")
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 → {path1}")

    # 2. Confusion Matrices
    n = len(results_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1: axes = [axes]
    
    for ax, name in zip(axes, results_dict.keys()):
        cm = confusion_matrix(y_test, results_dict[name]["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                    xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
        ax.set_title(f"{name}\nAcc={results_dict[name]['accuracy']:.4f}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        
    plt.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, "rl_02_confusion_matrices.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 → {path2}")
    
    # 3. Model Comparison
    names = list(results_dict.keys())
    metrics_list = ["accuracy", "precision", "recall", "f1"]
    
    x = np.arange(len(names))
    w = 0.2
    
    plt.figure(figsize=(10, 6))
    for i, m in enumerate(metrics_list):
        vals = [results_dict[n][m] for n in names]
        plt.bar(x + i * w, vals, w, label=m.title())
        
    plt.xticks(x + w * 1.5, names)
    plt.ylim(0.5, 1.05)
    plt.ylabel("Score")
    plt.title("So Sánh PPO và DQN (Binary Classification)", fontweight="bold")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    path3 = os.path.join(OUTPUT_DIR, "rl_03_model_comparison.png")
    plt.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 → {path3}")


def main():
    total_start = time.time()
    
    print("\n" + "🕹️" * 30)
    print("  NSL-KDD — REINFORCEMENT LEARNING PIPELINE")
    print("  Stable-Baselines3 (DQN & PPO)")
    print("🕹️" * 30)
    
    X_train, y_train, X_test, y_test = load_and_preprocess()
    
    # total_timesteps = 50,000 để chạy nhanh (~1-2 phút/mô hình trên CPU)
    # NSL-KDD train set có ~125k mẫu, nên 50k steps nghĩa là model đi qua gần nửa tập train
    timesteps = 50000
    
    results = {}
    
    results["DQN"] = train_and_evaluate_rl(DQN, "DQN", X_train, y_train, X_test, y_test, timesteps)
    results["PPO"] = train_and_evaluate_rl(PPO, "PPO", X_train, y_train, X_test, y_test, timesteps)
    
    plot_rl_results(results, y_test)
    
    print(f"\n{'=' * 60}")
    print("  📋 CLASSIFICATION REPORT — PPO")
    print(f"{'=' * 60}")
    print(classification_report(y_test, results["PPO"]["y_pred"], target_names=["Normal", "Attack"]))
    
    print(f"\n{'🎉' * 20}")
    print(f"  HOÀN TẤT! Tổng thời gian: {time.time() - total_start:.1f}s")
    print(f"  📂 Kết quả: {OUTPUT_DIR}")
    print(f"{'🎉' * 20}")

if __name__ == "__main__":
    main()
