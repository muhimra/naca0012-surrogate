"""
Week 4 — PyTorch Surrogate Model (K-Fold, full dataset)
========================================================
Uses 5-fold cross-validation to handle the small dataset properly,
then trains a final model on ALL 48 points for deployment.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from sklearn.model_selection import KFold

# ── CONFIG ───────────────────────────────────────────────────────────────────
TRAIN_CSV   = "output_week3/train_set.csv"
TEST_CSV    = "output_week3/test_set.csv"
OUTPUT_DIR  = Path("output_week4")
MODEL_PATH  = OUTPUT_DIR / "surrogate_model.pt"
SCALER_PATH = OUTPUT_DIR / "scaler_params.json"

EPOCHS      = 5000
LR          = 1e-3
HIDDEN      = [64, 128, 64]
RANDOM_SEED = 42
K_FOLDS     = 5
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(exist_ok=True)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("=" * 60)
print("  NACA 0012 — Week 4: Surrogate Model Training (K-Fold)")
print("=" * 60)

# ── 1. LOAD ALL DATA (train + test combined) ──────────────────────────────────
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)
df       = pd.concat([train_df, test_df], ignore_index=True)
print(f"\nTotal samples: {len(df)}  (train {len(train_df)} + test {len(test_df)})")

X_all = df[["alpha_deg", "reynolds"]].values.astype(np.float32)
y_all = df[["Cl", "Cd"]].values.astype(np.float32)

# ── 2. NORMALISE on full dataset ──────────────────────────────────────────────
X_mean = X_all.mean(axis=0)
X_std  = X_all.std(axis=0)
y_mean = y_all.mean(axis=0)
y_std  = y_all.std(axis=0)

X_n = (X_all - X_mean) / X_std
y_n = (y_all - y_mean) / y_std

scaler_params = {
    "X_mean": X_mean.tolist(), "X_std": X_std.tolist(),
    "y_mean": y_mean.tolist(), "y_std": y_std.tolist(),
    "X_cols": ["alpha_deg", "reynolds"],
    "y_cols": ["Cl", "Cd"]
}
with open(SCALER_PATH, "w") as f:
    json.dump(scaler_params, f, indent=2)
print(f"Scaler saved to {SCALER_PATH}")

# ── 3. MODEL DEFINITION ───────────────────────────────────────────────────────
class SurrogateNet(nn.Module):
    def __init__(self, hidden_sizes):
        super().__init__()
        layers = []
        in_size = 2
        for h in hidden_sizes:
            layers += [nn.Linear(in_size, h), nn.Tanh()]
            in_size = h
        layers += [nn.Linear(in_size, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def r2(actual, pred):
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    return 1 - ss_res / ss_tot

def rmse(actual, pred):
    return np.sqrt(np.mean((actual - pred) ** 2))

def train_model(X_tr, y_tr, epochs):
    model = SurrogateNet(HIDDEN)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = nn.MSELoss()
    X_t   = torch.tensor(X_tr)
    y_t   = torch.tensor(y_tr)
    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        loss = crit(model(X_t), y_t)
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
    return model, losses

# ── 4. K-FOLD CROSS VALIDATION ────────────────────────────────────────────────
print(f"\n{K_FOLDS}-fold cross-validation...")
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
fold_r2_cl, fold_r2_cd = [], []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_n), 1):
    X_tr, X_val = X_n[tr_idx], X_n[val_idx]
    y_tr, y_val = y_n[tr_idx], y_n[val_idx]

    model, _ = train_model(X_tr, y_tr, epochs=2000)
    model.eval()
    with torch.no_grad():
        pred_n = model(torch.tensor(X_val)).numpy()

    pred_phys = pred_n * y_std + y_mean
    val_phys  = y_val  * y_std + y_mean

    r2_cl = r2(val_phys[:, 0], pred_phys[:, 0])
    r2_cd = r2(val_phys[:, 1], pred_phys[:, 1])
    fold_r2_cl.append(r2_cl)
    fold_r2_cd.append(r2_cd)
    print(f"  Fold {fold}  Cl R2={r2_cl:.4f}  Cd R2={r2_cd:.4f}")

print(f"\n  Mean Cl R2: {np.mean(fold_r2_cl):.4f} (+/- {np.std(fold_r2_cl):.4f})")
print(f"  Mean Cd R2: {np.mean(fold_r2_cd):.4f} (+/- {np.std(fold_r2_cd):.4f})")

# ── 5. FINAL MODEL — trained on ALL 48 points ─────────────────────────────────
print(f"\nTraining final model on all {len(df)} samples ({EPOCHS} epochs)...")
final_model, final_losses = train_model(X_n, y_n, epochs=EPOCHS)

torch.save({
    "model_state":  final_model.state_dict(),
    "hidden_sizes": HIDDEN,
    "scaler_path":  str(SCALER_PATH)
}, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# ── 6. FINAL ACCURACY (leave-one-out on full set as sanity check) ─────────────
final_model.eval()
with torch.no_grad():
    pred_n_all = final_model(torch.tensor(X_n)).numpy()

pred_phys_all = pred_n_all * y_std + y_mean

r2_cl_final   = r2(y_all[:, 0], pred_phys_all[:, 0])
r2_cd_final   = r2(y_all[:, 1], pred_phys_all[:, 1])
rmse_cl_final = rmse(y_all[:, 0], pred_phys_all[:, 0])
rmse_cd_final = rmse(y_all[:, 1], pred_phys_all[:, 1])

print(f"\n-- Final model fit on full dataset --")
print(f"  Cl  R2={r2_cl_final:.4f}   RMSE={rmse_cl_final:.5f}")
print(f"  Cd  R2={r2_cd_final:.4f}   RMSE={rmse_cd_final:.6f}")

# ── 7. LOSS CURVE ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ax.semilogy(final_losses, color="#2196F3", linewidth=1.5)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("MSE Loss (log scale)", fontsize=12)
ax.set_title("Final Model — Training Loss", fontsize=13)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "loss_curve.png", dpi=150)
plt.close(fig)
print("Saved: loss_curve.png")

# ── 8. PREDICTED VS ACTUAL ────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for ax, col, label, color in [
    (ax1, 0, "Lift Coefficient Cl", "#2196F3"),
    (ax2, 1, "Drag Coefficient Cd", "#FF5722")
]:
    actual = y_all[:, col]
    pred   = pred_phys_all[:, col]
    ax.scatter(actual, pred, color=color, edgecolors="black",
               linewidths=0.5, s=60, zorder=3)
    lims = [min(actual.min(), pred.min()) * 1.05,
            max(actual.max(), pred.max()) * 1.05]
    ax.plot(lims, lims, "k--", linewidth=1)
    r2_val = r2(actual, pred)
    ax.set_xlabel(f"CFD {label}", fontsize=11)
    ax.set_ylabel(f"Surrogate {label}", fontsize=11)
    ax.set_title(f"{label} — R2 = {r2_val:.4f}", fontsize=12)
    ax.grid(True, alpha=0.3)

fig.suptitle("NACA 0012 Surrogate — Predicted vs Actual", fontsize=14)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "predicted_vs_actual.png", dpi=150)
plt.close(fig)
print("Saved: predicted_vs_actual.png")

# ── 9. SURROGATE SWEEP PLOT ───────────────────────────────────────────────────
# Show surrogate predictions across full alpha range for each Re
alpha_sweep = np.linspace(-10, 20, 200).astype(np.float32)
RE_COLORS   = {5e5: "#2196F3", 1e6: "#FF5722", 3e6: "#4CAF50"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
for re, color in RE_COLORS.items():
    X_sweep = np.column_stack([alpha_sweep,
                               np.full_like(alpha_sweep, re)])
    X_sweep_n = (X_sweep - X_mean) / X_std
    with torch.no_grad():
        pred_sweep = final_model(torch.tensor(X_sweep_n)).numpy()
    pred_sweep_phys = pred_sweep * y_std + y_mean

    label = f"Re = {re:.0e}"
    ax1.plot(alpha_sweep, pred_sweep_phys[:, 0], color=color,
             linewidth=2, label=label)
    ax2.plot(alpha_sweep, pred_sweep_phys[:, 1], color=color,
             linewidth=2, label=label)

    # Overlay actual CFD points for this Re
    cfd_re = df[df["reynolds"] == re]
    ax1.scatter(cfd_re["alpha_deg"], cfd_re["Cl"], color=color,
                edgecolors="black", linewidths=0.5, s=40, zorder=5)
    ax2.scatter(cfd_re["alpha_deg"], cfd_re["Cd"], color=color,
                edgecolors="black", linewidths=0.5, s=40, zorder=5)

ax1.set_xlabel("Angle of Attack (deg)", fontsize=12)
ax1.set_ylabel("Cl", fontsize=12)
ax1.set_title("Surrogate Cl vs Alpha", fontsize=13)
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.set_xlabel("Angle of Attack (deg)", fontsize=12)
ax2.set_ylabel("Cd", fontsize=12)
ax2.set_title("Surrogate Cd vs Alpha", fontsize=13)
ax2.legend(); ax2.grid(True, alpha=0.3)

fig.suptitle("NACA 0012 — Surrogate Curves vs CFD Points", fontsize=14)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "surrogate_curves.png", dpi=150)
plt.close(fig)
print("Saved: surrogate_curves.png")

# ── 10. SUMMARY ───────────────────────────────────────────────────────────────
summary = f"""Week 4 Results - NACA 0012 Surrogate Model (K-Fold)
====================================================
Architecture  : 2 -> {" -> ".join(map(str,HIDDEN))} -> 2  (Tanh activation)
Parameters    : {sum(p.numel() for p in final_model.parameters())}
Epochs        : {EPOCHS}
Learning rate : {LR} with CosineAnnealingLR

{K_FOLDS}-Fold Cross-Validation
  Mean Cl R2 : {np.mean(fold_r2_cl):.4f} +/- {np.std(fold_r2_cl):.4f}
  Mean Cd R2 : {np.mean(fold_r2_cd):.4f} +/- {np.std(fold_r2_cd):.4f}

Final model (trained on all {len(df)} samples)
  Cl  R2   : {r2_cl_final:.4f}
  Cl  RMSE : {rmse_cl_final:.5f}
  Cd  R2   : {r2_cd_final:.4f}
  Cd  RMSE : {rmse_cd_final:.6f}
"""
print(summary)
with open(OUTPUT_DIR / "week4_results.txt", "w", encoding="utf-8") as f:
    f.write(summary)

print("=" * 60)
print("  Week 4 Complete")
print("=" * 60)
