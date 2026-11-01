# -*- coding: utf-8 -*-
"""MLP_ROA.py

Model Prediksi Harga Saham Multi-Layer Perceptron (3 Hidden Layers: 128, 64, 32)
Teroptimasi Remora Optimization Algorithm (ROA) vs Backpropagation (Adam).
Diimplementasikan berdasarkan Jia et al. (2021).
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. REMORA OPTIMIZATION ALGORITHM (ROA) ENGINE
# ==============================================================================
class RemoraOptimizationAlgorithm:
    """
    Remora Optimization Algorithm (ROA) - Jia et al. (2021).
    Algoritma Metaheuristik Remora Optimization Algorithm (ROA) dengan parameter konsisten:
    N = 150, Max Iter = 200, Bounds = [-1.0, 1.0], Seed = 67.
    """
    def __init__(self, cost_function, dim, n_agents=150, max_iter=200,
                 lower_bound=-1.0, upper_bound=1.0, seed=67, C=0.1, patience=40):
        self.cost_function = cost_function
        self.dim = dim
        self.n_agents = n_agents
        self.max_iter = max_iter
        self.lb = lower_bound
        self.ub = upper_bound
        self.seed = seed
        self.C = C  # Remora Factor
        self.patience = patience
        self.rng = np.random.default_rng(seed)

    def optimize(self):
        population = self.rng.uniform(self.lb, self.ub, size=(self.n_agents, self.dim))
        fitness = np.zeros(self.n_agents)
        for i in range(self.n_agents):
            fitness[i] = self.cost_function(population[i])

        prev_population = population.copy()
        host = self.rng.integers(0, 2, size=self.n_agents)

        best_idx = np.argmin(fitness)
        global_best_cost = fitness[best_idx]
        global_best_pos = population[best_idx].copy()

        convergence_curve = np.zeros(self.max_iter)
        patience_counter = 0

        print(f"\n--- Memulai Remora Optimization Algorithm (ROA) ---")
        print(f"[*] Populasi (N): {self.n_agents} | Max Iterasi (T): {self.max_iter} | Dimensi (D): {self.dim}")

        for t in range(self.max_iter):
            B1 = 2.0 * (1.0 - t / self.max_iter)
            sailfish_count = int(np.sum(host == 0))
            whale_count = int(self.n_agents - sailfish_count)
            iter_improved = False

            for i in range(self.n_agents):
                # FASE 1: Pergerakan Inang (Sailfish / Whale)
                if host[i] == 0:
                    r_idx = self.rng.integers(0, self.n_agents)
                    while r_idx == i:
                        r_idx = self.rng.integers(0, self.n_agents)
                    x_rand = population[r_idx]
                    r1 = self.rng.uniform(0, 1, self.dim)
                    new_pos = global_best_pos - (r1 * ((global_best_pos + x_rand) / 2.0) - x_rand)
                else:
                    dist = np.abs(global_best_pos - population[i])
                    l = self.rng.uniform(-1, 1)
                    new_pos = dist * np.exp(l) * np.cos(2.0 * np.pi * l) + population[i]

                new_pos = np.clip(new_pos, self.lb, self.ub)
                new_fit = self.cost_function(new_pos)

                # FASE 2: Experience Attack & Host Switching
                if t > 0:
                    v = self.rng.standard_normal(self.dim)
                    att_pos = population[i] + (population[i] - prev_population[i]) * v
                    att_pos = np.clip(att_pos, self.lb, self.ub)
                    att_fit = self.cost_function(att_pos)
                else:
                    att_fit = float('inf')

                if att_fit < new_fit:
                    cand_pos = att_pos
                    cand_fit = att_fit
                    host[i] = 1 - host[i]
                else:
                    cand_pos = new_pos
                    cand_fit = new_fit

                # FASE 3: Host Feeding
                if cand_fit >= fitness[i]:
                    A = 2.0 * B1 * self.rng.uniform(0, 1) - B1
                    feed_pos = population[i] + A * (population[i] - self.C * global_best_pos)
                    feed_pos = np.clip(feed_pos, self.lb, self.ub)
                    feed_fit = self.cost_function(feed_pos)
                    if feed_fit < cand_fit:
                        cand_pos = feed_pos
                        cand_fit = feed_fit

                # FASE 4: Greedy Update
                if cand_fit < fitness[i]:
                    population[i] = cand_pos
                    fitness[i] = cand_fit
                    if cand_fit < global_best_cost:
                        global_best_cost = cand_fit
                        global_best_pos = cand_pos.copy()
                        iter_improved = True

            prev_population = population.copy()
            convergence_curve[t] = global_best_cost

            if iter_improved:
                patience_counter = 0
            else:
                patience_counter += 1

            if (t + 1) % 10 == 0 or t == 0 or (t + 1) == self.max_iter:
                print(f"Iterasi {t+1:03d}/{self.max_iter} | Inang: [Sailfish: {sailfish_count:02d}, Whale: {whale_count:02d}] | Best Fitness (RMSE): {global_best_cost:.6f}")

            if patience_counter >= self.patience and (t + 1) >= 50:
                print(f"[*] Early stopping pada iterasi {t+1} (patience = {self.patience}).")
                convergence_curve[t:] = global_best_cost
                break

        return global_best_pos, global_best_cost, convergence_curve

def run_remora_optimizer(n_agents, max_iter, dim, lower_bound, upper_bound, cost_function, seed=67, C=0.1, patience=40):
    roa = RemoraOptimizationAlgorithm(
        cost_function=cost_function,
        dim=dim,
        n_agents=n_agents,
        max_iter=max_iter,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        seed=seed,
        C=C,
        patience=patience
    )
    return roa.optimize()

# ==============================================================================
# 2. DATA ACQUISITION & PREPROCESSING
# ==============================================================================
def fetch_data(ticker, period="5y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].dropna()

def create_sliding_window(df, window_size=10):
    data = df.values
    close_series = df["Close"].values.ravel()
    target = np.log(close_series[1:] / close_series[:-1])

    X, y = [], []
    for i in range(window_size, len(data) - 1):
        X.append(data[i - window_size:i, :])
        y.append(target[i])

    return np.array(X), np.array(y).reshape(-1, 1)

def split_and_scale(X, y):
    n = len(X)
    tr_end = int(0.70 * n)
    val_end = int(0.85 * n)

    X_tr, y_tr = X[:tr_end], y[:tr_end]
    X_val, y_val = X[tr_end:val_end], y[tr_end:val_end]
    X_te, y_te = X[val_end:], y[val_end:]

    n_samples, window, n_feats = X.shape
    sx = MinMaxScaler().fit(X_tr.reshape(-1, n_feats))
    sy = MinMaxScaler().fit(y_tr)

    X_tr_scaled = sx.transform(X_tr.reshape(-1, n_feats)).reshape(X_tr.shape)
    X_val_scaled = sx.transform(X_val.reshape(-1, n_feats)).reshape(X_val.shape)
    X_te_scaled = sx.transform(X_te.reshape(-1, n_feats)).reshape(X_te.shape)

    return (X_tr_scaled, sy.transform(y_tr),
            X_val_scaled, sy.transform(y_val),
            X_te_scaled, sy.transform(y_te), sx, sy)

# ==============================================================================
# 3. MLP ARCHITECTURE & PARAMETER UNPACKING
# ==============================================================================
WINDOW_SIZE = 10
N_FEATURES = 6
FLATTEN_SIZE = WINDOW_SIZE * N_FEATURES

w1_params = (FLATTEN_SIZE * 128) + 128
w2_params = (128 * 64) + 64
w3_params = (64 * 32) + 32
w4_params = (32 * 1) + 1

D = w1_params + w2_params + w3_params + w4_params

def unpack(theta):
    i = 0
    W1 = theta[i:i + (FLATTEN_SIZE * 128)].reshape(FLATTEN_SIZE, 128); i += (FLATTEN_SIZE * 128)
    b1 = theta[i:i + 128]; i += 128
    W2 = theta[i:i + (128 * 64)].reshape(128, 64); i += (128 * 64)
    b2 = theta[i:i + 64]; i += 64
    W3 = theta[i:i + (64 * 32)].reshape(64, 32); i += (64 * 32)
    b3 = theta[i:i + 32]; i += 32
    W4 = theta[i:i + 32].reshape(32, 1); i += 32
    b4 = theta[i:i + 1]
    return W1, b1, W2, b2, W3, b3, W4, b4

def mlp_forward(theta, X):
    X_flat = X.reshape(X.shape[0], -1)
    W1, b1, W2, b2, W3, b3, W4, b4 = unpack(theta)

    h1 = np.tanh(X_flat @ W1 + b1)
    h2 = np.tanh(h1 @ W2 + b2)
    h3 = np.tanh(h2 @ W3 + b3)
    out = h3 @ W4 + b4
    return out

def fitness(theta, X, y):
    mse = np.mean((y - mlp_forward(theta, X)) ** 2)
    return np.sqrt(mse)

def create_roa_cost(X, y):
    def cost_function(theta):
        return fitness(theta, X, y)
    return cost_function

create_po_cost = create_roa_cost

def roa_optimize(X_train, y_train, X_val, y_val, N=150, T=200, bounds=(-1.0, 1.0), seed=67, patience=40):
    train_cost = create_roa_cost(X_train, y_train)
    best_theta, best_cost, convergence_curve = run_remora_optimizer(
        n_agents=N,
        max_iter=T,
        dim=D,
        lower_bound=bounds[0],
        upper_bound=bounds[1],
        cost_function=train_cost,
        seed=seed,
        C=0.1,
        patience=patience
    )
    best_val_cost = fitness(best_theta, X_val, y_val)
    return best_theta, best_val_cost, convergence_curve

po_optimize = roa_optimize

# ==============================================================================
# 4. BASELINE BACKPROPAGATION (Adam)
# ==============================================================================
def train_mlp_backprop(X_train, y_train, X_val, y_val, epochs=200, lr=0.01):
    X_tr_t = torch.FloatTensor(X_train.reshape(X_train.shape[0], -1))
    y_tr_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val.reshape(X_val.shape[0], -1))
    y_val_t = torch.FloatTensor(y_val)

    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(FLATTEN_SIZE, 128),
                nn.Tanh(),
                nn.Linear(128, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
            )
        def forward(self, x):
            return self.net(x)

    model = MLP()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history_mse = []
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_tr_t)
        loss = criterion(y_pred, y_tr_t)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()
                history_mse.append(val_loss)

    train_time = time.time() - start_time
    return model, history_mse, train_time

# ==============================================================================
# 5. PIPELINE UTAMA
# ==============================================================================
def run_comparative_pipeline(ticker):
    print(f"\n==================================================")
    print(f"         MEMPROSES SAHAM: {ticker}                ")
    print(f"==================================================")
    df = fetch_data(ticker)

    X, y = create_sliding_window(df, window_size=WINDOW_SIZE)
    print(f"[*] Dimensi tensor X: {X.shape}")

    close_prices = df["Close"].values[WINDOW_SIZE:].ravel()
    X_tr, y_tr, X_val, y_val, X_te, y_te, sx, sy = split_and_scale(X, y)

    # 1. ROA
    print("\n[1/2] Melatih dengan Remora Optimization Algorithm (ROA)...")
    start_roa = time.time()
    theta, f_best, roa_log = roa_optimize(X_tr, y_tr, X_val, y_val)
    time_roa = time.time() - start_roa

    # 2. Backprop
    print("\n[2/2] Melatih dengan Backpropagation (Adam)...")
    model_bp, bp_log, time_bp = train_mlp_backprop(X_tr, y_tr, X_val, y_val, epochs=200, lr=0.01)

    # 3. Rekonstruksi Harga
    n = len(X)
    val_end = int(0.85 * n)

    close_yesterday_test = close_prices[val_end : -1].ravel()
    close_actual_test = close_prices[val_end + 1 :].ravel()

    y_pred_roa_log_ret = sy.inverse_transform(mlp_forward(theta, X_te)).ravel()

    model_bp.eval()
    with torch.no_grad():
        X_te_flat_torch = torch.FloatTensor(X_te.reshape(X_te.shape[0], -1))
        y_pred_bp_log_ret = sy.inverse_transform(model_bp(X_te_flat_torch).numpy()).ravel()

    min_len = min(len(close_yesterday_test), len(y_pred_roa_log_ret), len(y_pred_bp_log_ret), len(close_actual_test))
    close_yesterday_test = close_yesterday_test[:min_len]
    y_pred_roa_price = (close_yesterday_test * np.exp(y_pred_roa_log_ret[:min_len])).ravel()
    y_pred_bp_price = (close_yesterday_test * np.exp(y_pred_bp_log_ret[:min_len])).ravel()
    y_true_price = close_actual_test[:min_len].ravel()

    def calc_metrics(y_true, y_pred):
        return {
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "MAPE": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
            "R2": float(r2_score(y_true, y_pred))
        }

    metrics_roa = calc_metrics(y_true_price, y_pred_roa_price)
    metrics_bp = calc_metrics(y_true_price, y_pred_bp_price)

    print(f"\n--- Rangkuman Hasil {ticker} ---")
    print(f"Waktu ROA : {time_roa:.2f} detik | Waktu BP: {time_bp:.2f} detik")
    print(f"Metrik ROA: {metrics_roa}")
    print(f"Metrik BP : {metrics_bp}")

    return {
        "roa": {"metrics": metrics_roa, "time": time_roa, "log": roa_log, "pred": y_pred_roa_price},
        "eefo": {"metrics": metrics_roa, "time": time_roa, "log": roa_log, "pred": y_pred_roa_price},
        "bp": {"metrics": metrics_bp, "time": time_bp, "log": bp_log, "pred": y_pred_bp_price},
        "y_true": y_true_price
    }

# ==============================================================================
# 6. VISUALIZATION & EXPORT
# ==============================================================================
def generate_comparative_plots(results_compare, tickers):
    """
    Menghasilkan 4 set grafik visualisasi komparatif beresolusi tinggi (200 DPI)
    yang dirancang khusus agar mudah dipresentasikan dalam sidang/seminar akademik:
    1. Executive Performance Dashboard (Bar Chart R2 & MAPE - Slide Ringkasan)
    2. Actual vs Predicted Overlaid Time-Series (Perbandingan Langsung dalam 1 Plot)
    3. Regression Scatter Parity Plot (Titik sebaran vs Garis Ideal 45 Derajat y = x)
    4. Dinamika Konvergensi Teranotasi Early Stopping (Iterasi Riil & Alasan Konvergensi)
    """
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'axes.facecolor': '#FAFAFA',
        'figure.facecolor': '#FFFFFF',
        'axes.grid': True,
        'grid.color': '#E5E7EB',
        'grid.linestyle': '--',
        'grid.alpha': 0.7,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': '#9CA3AF',
        'axes.linewidth': 1.1,
        'font.size': 10
    })

    COLOR_ACTUAL = '#0F172A'  # Hitam arang ground truth
    COLOR_ROA = '#2563EB'     # Biru royal ROA-MLP
    COLOR_BP = '#EA580C'      # Oranye/Amber Backprop-Adam

    # =========================================================================
    # 1. EXECUTIVE PERFORMANCE DASHBOARD (Slide 1: Ringkasan Eksekutif)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(len(tickers))
    width = 0.35

    r2_roa = [results_compare[t]["roa"]["metrics"]["R2"] for t in tickers]
    r2_bp = [results_compare[t]["bp"]["metrics"]["R2"] for t in tickers]

    rects1 = axes[0].bar(x - width/2, r2_roa, width, label='Remora Optimization (ROA-MLP)', color=COLOR_ROA, alpha=0.9, edgecolor='#1D4ED8')
    rects2 = axes[0].bar(x + width/2, r2_bp, width, label='Backpropagation (Adam-MLP)', color=COLOR_BP, alpha=0.85, edgecolor='#C2410C')

    axes[0].set_title(r"Akurasi Regresi ($R^2$ Score) — Semakin Tinggi Semakin Baik", fontsize=13, fontweight='bold', pad=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tickers, fontsize=11, fontweight='bold')
    axes[0].set_ylabel(r"Koefisien Determinasi ($R^2$)", fontsize=11)
    axes[0].axhline(0, color='#6B7280', linestyle='-', linewidth=0.8)
    axes[0].legend(loc='lower right', frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')

    for rect in rects1:
        h = rect.get_height()
        y_pos = h + (0.04 if h >= 0 else -0.2)
        axes[0].annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, y_pos),
                         ha='center', va='bottom' if h >= 0 else 'top', fontsize=9.5, fontweight='bold', color=COLOR_ROA)

    for rect in rects2:
        h = rect.get_height()
        y_pos = h + (0.04 if h >= 0 else -0.2)
        axes[0].annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, y_pos),
                         ha='center', va='bottom' if h >= 0 else 'top', fontsize=9.5, fontweight='bold', color=COLOR_BP)

    mape_roa = [results_compare[t]["roa"]["metrics"]["MAPE"] for t in tickers]
    mape_bp = [results_compare[t]["bp"]["metrics"]["MAPE"] for t in tickers]

    rects3 = axes[1].bar(x - width/2, mape_roa, width, label='Remora Optimization (ROA-MLP)', color=COLOR_ROA, alpha=0.9, edgecolor='#1D4ED8')
    rects4 = axes[1].bar(x + width/2, mape_bp, width, label='Backpropagation (Adam-MLP)', color=COLOR_BP, alpha=0.85, edgecolor='#C2410C')

    axes[1].set_title("Tingkat Kesalahan Prediksi (MAPE %) — Semakin Rendah Semakin Baik", fontsize=13, fontweight='bold', pad=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tickers, fontsize=11, fontweight='bold')
    axes[1].set_ylabel("Mean Absolute Percentage Error (%)", fontsize=11)
    axes[1].legend(loc='upper right', frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')

    for rect in rects3:
        h = rect.get_height()
        axes[1].annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width() / 2, h + 0.3),
                         ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=COLOR_ROA)

    for rect in rects4:
        h = rect.get_height()
        axes[1].annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width() / 2, h + 0.3),
                         ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=COLOR_BP)

    plt.suptitle("Executive Summary: Performa Prediksi ROA-MLP vs BP-Adam", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("executive_performance_dashboard.png", dpi=200, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 2. TIME-SERIES ACTUAL VS PREDICTED (Slide 2: Overlaid Tracking)
    # =========================================================================
    fig, axes = plt.subplots(3, 1, figsize=(16, 13), sharex=False)
    for i, ticker in enumerate(tickers):
        y_true = results_compare[ticker]["y_true"]
        y_roa = results_compare[ticker]["roa"]["pred"]
        y_bp = results_compare[ticker]["bp"]["pred"]

        r2_roa = results_compare[ticker]['roa']['metrics']['R2']
        mape_roa = results_compare[ticker]['roa']['metrics']['MAPE']
        rmse_roa = results_compare[ticker]['roa']['metrics']['RMSE']

        r2_bp = results_compare[ticker]['bp']['metrics']['R2']
        mape_bp = results_compare[ticker]['bp']['metrics']['MAPE']
        rmse_bp = results_compare[ticker]['bp']['metrics']['RMSE']

        winner_badge = "[WINNER: ROA Lebih Akurat]" if r2_roa > r2_bp else "[Adam Lebih Baik]"
        badge_border = "#15803D" if r2_roa > r2_bp else "#B45309"

        axes[i].plot(y_true, label="Harga Aktual (Ground Truth)", color=COLOR_ACTUAL, linewidth=2.4, zorder=3)
        axes[i].plot(y_roa, label=f"Remora Optimization (ROA)  [R²={r2_roa:.4f}, MAPE={mape_roa:.2f}%]", color=COLOR_ROA, linewidth=2.0, linestyle='-', zorder=4)
        axes[i].plot(y_bp, label=f"Backpropagation (Adam)     [R²={r2_bp:.4f}, MAPE={mape_bp:.2f}%]", color=COLOR_BP, linewidth=1.7, linestyle='--', zorder=2)

        axes[i].fill_between(range(len(y_true)), y_true, y_roa, alpha=0.10, color=COLOR_ROA, label="Area Deviasi ROA")

        currency = "IDR" if "JK" in ticker else "USD"
        axes[i].set_title(f"Saham: {ticker} — Perbandingan Prediksi Tren Harga Out-of-Sample", fontsize=13, fontweight='bold', loc='left', pad=10)
        axes[i].set_xlabel("Sampel Hari Uji (Test Index)", fontsize=10)
        axes[i].set_ylabel(f"Harga ({currency})", fontsize=10)
        axes[i].legend(loc='upper left', frameon=True, facecolor='#FFFFFF', edgecolor='#D1D5DB', fontsize=9.5)

        stat_box = (
            f"Evaluasi Metrik {ticker}:\n"
            f"• ROA-MLP : RMSE={rmse_roa:.2f} | R²={r2_roa:.4f} | MAPE={mape_roa:.2f}%\n"
            f"• BP-Adam : RMSE={rmse_bp:.2f} | R²={r2_bp:.4f} | MAPE={mape_bp:.2f}%\n"
            f"Status: {winner_badge}"
        )
        axes[i].text(0.98, 0.08, stat_box, transform=axes[i].transAxes, fontsize=9.5,
                     verticalalignment='bottom', horizontalalignment='right',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8FAFC', edgecolor=badge_border, linewidth=1.3))

    plt.suptitle("Perbandingan Prediksi Harga Saham pada Test Set: ROA-MLP vs BP-Adam", fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig("actual_vs_predicted_time_series.png", dpi=200, bbox_inches='tight')
    plt.savefig("actual_vs_predicted_roa_3assets.png", dpi=200, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 3. REGRESSION SCATTER / PARITY PLOT (Slide 3: Garis Diagonal 45 Derajat)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for i, ticker in enumerate(tickers):
        y_true = results_compare[ticker]["y_true"]
        y_roa = results_compare[ticker]["roa"]["pred"]
        y_bp = results_compare[ticker]["bp"]["pred"]
        r2_roa = results_compare[ticker]['roa']['metrics']['R2']
        r2_bp = results_compare[ticker]['bp']['metrics']['R2']

        axes[i].scatter(y_true, y_roa, color=COLOR_ROA, alpha=0.6, s=28, label=f"ROA-MLP (R² = {r2_roa:.3f})", edgecolor='none')
        axes[i].scatter(y_true, y_bp, color=COLOR_BP, alpha=0.45, s=24, marker='^', label=f"BP-Adam (R² = {r2_bp:.3f})", edgecolor='none')

        min_val = min(np.min(y_true), np.min(y_roa), np.min(y_bp))
        max_val = max(np.max(y_true), np.max(y_roa), np.max(y_bp))
        axes[i].plot([min_val, max_val], [min_val, max_val], color='#DC2626', linestyle='--', linewidth=1.8, label="Garis Ideal Paritas (y = x)")

        axes[i].set_title(rf"{ticker}: Parity Scatter ($y_{{actual}}$ vs $\hat{{y}}_{{pred}}$)", fontsize=12, fontweight='bold', pad=10)
        axes[i].set_xlabel(r"Harga Aktual ($y_{true}$)", fontsize=10)
        axes[i].set_ylabel(r"Harga Prediksi ($\hat{y}$)", fontsize=10)
        axes[i].legend(loc='upper left', frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB', fontsize=9)

    plt.suptitle("Analisis Paritas Regresi (Titik Merapat ke Garis 45 Derajat Menandakan Prediksi Sangat Akurat)", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("regression_scatter_parity.png", dpi=200, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 4. CONVERGENCE DYNAMICS WITH EARLY STOPPING ANNOTATION (Slide 4)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for i, ticker in enumerate(tickers):
        roa_curve = results_compare[ticker]["roa"]["log"]
        bp_curve = results_compare[ticker]["bp"]["log"]

        iters_roa = np.arange(1, len(roa_curve) + 1)
        iters_bp = np.arange(1, len(bp_curve) + 1)

        axes[i].plot(iters_roa, roa_curve, label=f"Remora Optimization (ROA) [{len(roa_curve)} iterasi]", color=COLOR_ROA, linewidth=2.2)
        axes[i].plot(iters_bp, bp_curve, label=f"Backpropagation (Adam) [{len(bp_curve)} iterasi]", color=COLOR_BP, linestyle='--', linewidth=1.8)

        stop_iter = len(roa_curve)
        best_loss = min(roa_curve)
        if stop_iter < 200:
            axes[i].axvline(x=stop_iter, color='#DC2626', linestyle=':', linewidth=1.5, alpha=0.85)
            axes[i].scatter([stop_iter], [roa_curve[-1]], color='#DC2626', s=40, zorder=5, label=f"Early Stop (Iter {stop_iter})")
            axes[i].annotate(
                f"Early Stopping Terpicu\nIterasi {stop_iter} | Loss: {best_loss:.4e}",
                xy=(stop_iter, roa_curve[-1]),
                xytext=(-60, 25),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color='#DC2626', lw=1.2, shrinkB=4),
                fontsize=8.5,
                fontweight='bold',
                color='#991B1B',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FEF2F2', edgecolor='#FCA5A5')
            )

        axes[i].set_title(f"{ticker}: Dinamika Konvergensi Fitness Loss", fontsize=12, fontweight='bold', pad=10)
        axes[i].set_xlabel("Iterasi Riil (Actual Iteration / Epoch)", fontsize=10)
        axes[i].set_ylabel("Fitness Loss (MSE Skala Log)", fontsize=10)
        axes[i].set_yscale("log")
        axes[i].legend(loc='upper right', frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB', fontsize=9)

    plt.suptitle("Kurva Konvergensi Pelatihan: ROA-MLP vs Backpropagation (Adam)", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("roa_vs_backprop_convergence_clarified.png", dpi=200, bbox_inches='tight')
    plt.savefig("roa_vs_backprop_convergence_3assets.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("[+] Seluruh visualisasi komparatif presentasi berhasil disimpan.")

def export_research_data_to_csv(results_compare, tickers):
    print("\n[*] Mengekspor seluruh data metrik dan prediksi ke CSV...")

    metrics_data = []
    for ticker in tickers:
        for method in ["roa", "bp"]:
            metrics = results_compare[ticker][method]["metrics"]
            metrics_data.append({
                "Ticker": ticker,
                "Method": "Remora Optimization (ROA)" if method == "roa" else "Backprop (Adam)",
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "MAPE (%)": metrics["MAPE"],
                "R2_Score": metrics["R2"],
                "Training_Time_Sec": results_compare[ticker][method]["time"]
            })
    pd.DataFrame(metrics_data).to_csv("evaluation_metrics_summary_roa.csv", index=False)

    forecast_data = []
    for ticker in tickers:
        y_true = results_compare[ticker]["y_true"].ravel()
        y_roa = results_compare[ticker]["roa"]["pred"].ravel()
        y_bp = results_compare[ticker]["bp"]["pred"].ravel()

        df_ticker = pd.DataFrame({
            "Ticker": ticker,
            "Test_Sample_Idx": range(1, len(y_true) + 1),
            "Actual_Price": y_true,
            "ROA_Predicted_Price": y_roa,
            "Backprop_Predicted_Price": y_bp
        })
        forecast_data.append(df_ticker)
    pd.concat(forecast_data, ignore_index=True).to_csv("actual_vs_predicted_forecasts_roa.csv", index=False)

    convergence_data = []
    for ticker in tickers:
        roa_log = results_compare[ticker]["roa"]["log"]
        convergence_data.append(pd.DataFrame({
            "Ticker": ticker, "Method": "Remora Optimization (ROA)",
            "Iteration": range(1, len(roa_log) + 1), "Loss": roa_log
        }))

        bp_log = results_compare[ticker]["bp"]["log"]
        convergence_data.append(pd.DataFrame({
            "Ticker": ticker, "Method": "Backprop (Adam)",
            "Logged_Step": range(1, len(bp_log) + 1), "Loss": bp_log
        }))
    pd.concat(convergence_data, ignore_index=True).to_csv("training_convergence_history_roa.csv", index=False)
    print("[+] Seluruh berkas CSV berhasil diekspor.")

if __name__ == "__main__":
    tickers_to_test = ["NVDA", "BBCA.JK", "TSLA"]
    results_compare = {}

    for ticker in tickers_to_test:
        results_compare[ticker] = run_comparative_pipeline(ticker)

    generate_comparative_plots(results_compare, tickers_to_test)
    export_research_data_to_csv(results_compare, tickers_to_test)
