#!/usr/bin/env python3

import argparse
import os
import json
import yaml
import h5py
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    confusion_matrix,
    precision_score,
    recall_score,
)

# ============================================================
# 🔒 HARD-CODED FEATURE ORDER (from feature contract v1)
# NOTE: Must match training pipeline exactly
# ============================================================

FEATURE_NAMES = [
    "px","py","pz","p","pt","theta","phi",
    "vx","vy","vz","vr","v3",
    "n_hits","sum_adc","path","dEdx","dedx_recomputed",
    "p_drift","sum_residuals","residual_per_hit","adc_per_hit",
    "tof_time","pathlength","cluster_x","cluster_y","cluster_z",
    "cluster_energy","n_bar","n_wedge",
    "beta","m2",
    "log_p","log_pt","log_sum_adc","log_path",
    "log_dEdx","log_dedx_recomputed","log_cluster_energy"
]

CLASS_NAMES = ["proton", "deuteron", "triton", "helium3", "helium4"]

# ============================================================
# 📥 DATA LOADING
# ============================================================

def load_h5(path):
    with h5py.File(path, "r") as f:
        X = f["features/values"][:]
        mask = f["features/masks"][:]
        y = f["labels/class_index"][:]
    return X, mask, y


def filter_valid(X, mask, y):
    valid = mask.all(axis=1)
    return X[valid], y[valid]


# ============================================================
# 📊 METRICS
# ============================================================

def compute_metrics(y_true, y_pred, y_prob):
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["log_loss"] = log_loss(y_true, y_prob)

    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)

    metrics["precision_per_class"] = dict(zip(CLASS_NAMES, precision))
    metrics["recall_per_class"] = dict(zip(CLASS_NAMES, recall))

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    return metrics, cm, cm_norm


# ============================================================
# 🎯 HARD REGION ANALYSIS (d vs He4)
# ============================================================

def hard_region_analysis(y_true, y_pred):
    mask = np.isin(y_true, [1, 4])  # deuteron, helium4
    y_true_sub = y_true[mask]
    y_pred_sub = y_pred[mask]

    cm = confusion_matrix(y_true_sub, y_pred_sub, labels=[1,4])

    recall_d = cm[0,0] / cm[0].sum() if cm[0].sum() > 0 else 0
    recall_he4 = cm[1,1] / cm[1].sum() if cm[1].sum() > 0 else 0

    misclass = (cm[0,1] + cm[1,0]) / cm.sum() if cm.sum() > 0 else 0

    return {
        "confusion_matrix": cm.tolist(),
        "recall_deuteron": recall_d,
        "recall_helium4": recall_he4,
        "misclassification_rate": misclass
    }


# ============================================================
# 📈 PERFORMANCE VS VARIABLE
# ============================================================

def performance_vs_variable(X, y_true, y_pred, feature_idx, name, output_dir, bins=10):
    values = X[:, feature_idx]
    edges = np.linspace(values.min(), values.max(), bins + 1)

    accs = []
    centers = []

    for i in range(bins):
        mask = (values >= edges[i]) & (values < edges[i+1])
        if mask.sum() == 0:
            continue
        acc = accuracy_score(y_true[mask], y_pred[mask])
        accs.append(acc)
        centers.append((edges[i] + edges[i+1]) / 2)

    plt.figure()
    plt.plot(centers, accs, marker="o")
    plt.xlabel(name)
    plt.ylabel("Accuracy")
    plt.title(f"Performance vs {name}")
    plt.savefig(os.path.join(output_dir, f"perf_vs_{name}.png"))
    plt.close()


# ============================================================
# 📊 COEFFICIENT ANALYSIS
# ============================================================

def analyze_coefficients(model, output_dir):
    coef = model.coef_
    df = pd.DataFrame(coef, columns=FEATURE_NAMES, index=CLASS_NAMES)
    df.to_csv(os.path.join(output_dir, "coefficients.csv"))

    summary_lines = []

    for i, cls in enumerate(CLASS_NAMES):
        weights = coef[i]
        sorted_idx = np.argsort(weights)

        top_pos = sorted_idx[-10:]
        top_neg = sorted_idx[:10]

        summary_lines.append(f"\n=== {cls} ===\n")
        summary_lines.append("Top positive:\n")
        for idx in reversed(top_pos):
            summary_lines.append(f"{FEATURE_NAMES[idx]}: {weights[idx]:.4f}\n")

        summary_lines.append("Top negative:\n")
        for idx in top_neg:
            summary_lines.append(f"{FEATURE_NAMES[idx]}: {weights[idx]:.4f}\n")

    with open(os.path.join(output_dir, "coefficient_summary.txt"), "w") as f:
        f.writelines(summary_lines)


# ============================================================
# 📊 PROBABILITY DIAGNOSTICS
# ============================================================

def probability_diagnostics(y_true, y_pred, y_prob, output_dir):
    confidence = y_prob.max(axis=1)
    correct = (y_true == y_pred)

    plt.figure()
    plt.hist(confidence, bins=50)
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.title("Confidence Histogram")
    plt.savefig(os.path.join(output_dir, "confidence_hist.png"))
    plt.close()

    plt.figure()
    plt.scatter(confidence, correct.astype(int), alpha=0.3)
    plt.xlabel("Confidence")
    plt.ylabel("Correct")
    plt.title("Confidence vs Correctness")
    plt.savefig(os.path.join(output_dir, "confidence_vs_correct.png"))
    plt.close()


# ============================================================
# 🚀 MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load
    X_train, m_train, y_train = load_h5(args.train)
    X_val, m_val, y_val = load_h5(args.val)
    X_test, m_test, y_test = load_h5(args.test)

    # Filter masked rows
    X_train, y_train = filter_valid(X_train, m_train, y_train)
    X_val, y_val = filter_valid(X_val, m_val, y_val)
    X_test, y_test = filter_valid(X_test, m_test, y_test)

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    joblib.dump(scaler, os.path.join(args.output_dir, "scaler.joblib"))

    # Hyperparameter search (simple grid on val)
    best_model = None
    best_score = -np.inf

    for C in config["C_values"]:
        model = LogisticRegression(
            C=C,
            max_iter=config["max_iter"],
            solver=config["solver"],
            class_weight=config["class_weight"],
            random_state=config["seed"],
        )

        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)

        if val_acc > best_score:
            best_score = val_acc
            best_model = model

    # Final evaluation
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)

    metrics, cm, cm_norm = compute_metrics(y_test, y_pred, y_prob)

    # Save metrics
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    np.savetxt(os.path.join(args.output_dir, "confusion_matrix.txt"), cm, fmt="%d")
    np.savetxt(os.path.join(args.output_dir, "confusion_matrix_norm.txt"), cm_norm, fmt="%.4f")

    # Hard region
    hard = hard_region_analysis(y_test, y_pred)
    with open(os.path.join(args.output_dir, "hard_region.json"), "w") as f:
        json.dump(hard, f, indent=2)

    # Performance vs variables
    for name in ["p", "n_hits", "tof_time", "cluster_energy"]:
        idx = FEATURE_NAMES.index(name)
        performance_vs_variable(X_test, y_test, y_pred, idx, name, args.output_dir)

    # Coefficients
    analyze_coefficients(best_model, args.output_dir)

    # Prob diagnostics
    probability_diagnostics(y_test, y_pred, y_prob, args.output_dir)

    # Save model
    joblib.dump(best_model, os.path.join(args.output_dir, "model.joblib"))

    # Summary
    with open(os.path.join(args.output_dir, "summary.md"), "w") as f:
        f.write("# Phase E Logistic Regression Summary\n\n")
        f.write(f"Train size: {len(y_train)}\n")
        f.write(f"Val size: {len(y_val)}\n")
        f.write(f"Test size: {len(y_test)}\n\n")
        f.write("Masking rule: rows with any invalid feature removed\n\n")
        f.write("Model: multinomial logistic regression\n\n")
        f.write(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()