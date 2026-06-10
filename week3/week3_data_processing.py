"""
Week 3 — Data Processing, EDA, and Train/Test Split
=====================================================
Run this after your STAR-CCM+ sweep has produced naca0012_sweep.csv.

Usage:
    python week3_data_processing.py

Outputs:
    - Console: data summary, outlier report, NASA comparison
    - Figures:  Cl_vs_alpha.png, Cd_vs_alpha.png, ClCd_vs_alpha.png,
                Cl_predicted_vs_actual_baseline.png
    - Data:     train_set.csv, test_set.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────────────────────
CSV_PATH      = "naca0012_sweep.csv"     # path to your STAR-CCM+ output
OUTPUT_DIR    = Path("output_week3")
RANDOM_SEED   = 42
TEST_FRACTION = 0.20
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(exist_ok=True)

# ── NASA NACA 0012 EXPERIMENTAL REFERENCE DATA ───────────────────────────────
# Source: Ladson et al. (1988), NASA TM-4071
# Re = 3×10⁶, M ≈ 0.15 (effectively incompressible)
# These are the key points from Table 1 of that report.
NASA_CL = {
    -10: -1.025, -8: -0.862, -6: -0.650, -4: -0.431,
     -2: -0.214,   0:  0.000,  2:  0.221,  4:  0.441,
      6:  0.660,   8:  0.867, 10:  1.050, 12:  1.212,
     14:  1.310, 16:  1.050
}
NASA_CD = {
     0: 0.00605,  2: 0.00617,  4: 0.00658,  6: 0.00780,
     8: 0.01040, 10: 0.01380, 12: 0.01820, 14: 0.02560, 16: 0.06820
}

# ── 1. LOAD DATA ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  NACA 0012 — Week 3: Data Processing and EDA")
print("=" * 60)

try:
    df = pd.read_csv(CSV_PATH)
    print(f"\n✅ Loaded: {CSV_PATH}")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   Columns: {list(df.columns)}")
except FileNotFoundError:
    print(f"\n⚠️  CSV not found at '{CSV_PATH}'.")
    print("   Generating synthetic placeholder data for pipeline testing...\n")
    # Synthetic data so the script can be tested before CFD runs complete
    alpha_vals = np.arange(-10, 22, 2, dtype=float)
    re_vals    = [5e5, 1e6, 3e6]
    rows = []
    for re in re_vals:
        for a in alpha_vals:
            ar = np.radians(a)
            # Thin-aerofoil approximation + stall model
            cl_linear = 2 * np.pi * np.sin(ar) * (1 - max(0, (abs(a) - 12) / 8))
            cd_base   = 0.006 + 0.01 * ar**2
            noise     = np.random.normal(0, 0.005)
            cl = cl_linear + noise
            cd = max(0.005, cd_base + abs(noise) * 0.1)
            rows.append({
                "alpha_deg": a, "reynolds": re,
                "velocity_ms": re * 1.5e-5 / 1.0,
                "Cl": round(cl, 6), "Cd": round(cd, 6),
                "ClCd": round(cl / cd, 4) if cd > 0 else 0
            })
    df = pd.DataFrame(rows)
    print("   ⚠️  Using synthetic data — replace with your real CSV.\n")

# Normalise column names defensively
df.columns = [c.strip() for c in df.columns]

# ── 2. CLEAN & INSPECT ───────────────────────────────────────────────────────
print("\n── Raw data summary ─────────────────────────────────────────")
print(df.describe().round(4))

# Check for NaN / Inf
n_nan = df.isnull().sum().sum()
n_inf = np.isinf(df.select_dtypes(include=np.number)).sum().sum()
print(f"\nNaN values:  {n_nan}")
print(f"Inf values:  {n_inf}")
if n_nan or n_inf:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    print(f"→ Cleaned. Remaining rows: {len(df)}")

# Sanity check on Reynolds numbers
re_found = sorted(df["reynolds"].unique())
print(f"\nReynolds numbers in dataset: {re_found}")
alpha_found = sorted(df["alpha_deg"].unique())
print(f"Angles of attack (°):        {alpha_found}")

# ── 3. OUTLIER DETECTION ────────────────────────────────────────────────────
print("\n── Outlier check (IQR method) ───────────────────────────────")
outlier_flags = pd.DataFrame(index=df.index)
for col in ["Cl", "Cd"]:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR     = Q3 - Q1
    lo, hi  = Q1 - 3.0 * IQR, Q3 + 3.0 * IQR
    mask    = (df[col] < lo) | (df[col] > hi)
    outlier_flags[col] = mask
    if mask.any():
        print(f"  {col}: {mask.sum()} outliers outside [{lo:.4f}, {hi:.4f}]")
        print(df[mask][["alpha_deg", "reynolds", col]])
    else:
        print(f"  {col}: no outliers detected ✓")

# Physical sanity: Cd must be positive
neg_cd = df["Cd"] <= 0
if neg_cd.any():
    print(f"\n  ⚠️  {neg_cd.sum()} rows with Cd ≤ 0 — removing:")
    print(df[neg_cd])
    df = df[~neg_cd].copy()

# ── 4. DERIVED QUANTITIES ────────────────────────────────────────────────────
df["ClCd"] = df["Cl"] / df["Cd"]

# ── 5. PLOTTING ──────────────────────────────────────────────────────────────
RE_LABELS = {
    5e5:  "Re = 5×10⁵",
    1e6:  "Re = 1×10⁶",
    3e6:  "Re = 3×10⁶",
    6e6:  "Re = 6×10⁶",
}
RE_COLORS = {5e5: "#2196F3", 1e6: "#FF5722", 3e6: "#4CAF50", 6e6: "#9C27B0"}
MARKERS   = {5e5: "o", 1e6: "s", 3e6: "^", 6e6: "D"}

def re_label(re):
    return RE_LABELS.get(re, f"Re = {re:.1e}")

def re_color(re):
    return RE_COLORS.get(re, "#666666")

# ─── Fig 1: Cl vs α ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
for re in sorted(df["reynolds"].unique()):
    sub = df[df["reynolds"] == re].sort_values("alpha_deg")
    ax.plot(sub["alpha_deg"], sub["Cl"],
            color=re_color(re), marker=MARKERS.get(re, "o"),
            label=f"CFD — {re_label(re)}", linewidth=2, markersize=5)

# NASA reference (Re = 3e6)
nasa_alpha = sorted(NASA_CL.keys())
nasa_cl    = [NASA_CL[a] for a in nasa_alpha]
ax.plot(nasa_alpha, nasa_cl, "k--", linewidth=1.5, label="NASA Exp. (Re=3×10⁶)")

ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
ax.axvline(0, color="grey", linewidth=0.5, linestyle=":")
ax.set_xlabel("Angle of Attack α (°)", fontsize=12)
ax.set_ylabel("Lift Coefficient Cl", fontsize=12)
ax.set_title("NACA 0012 — Cl vs α\nSTAR-CCM+ CFD vs NASA Experimental", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "Cl_vs_alpha.png", dpi=150)
plt.close(fig)
print("\n✅ Saved: Cl_vs_alpha.png")

# ─── Fig 2: Cd vs α ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
for re in sorted(df["reynolds"].unique()):
    sub = df[df["reynolds"] == re].sort_values("alpha_deg")
    ax.plot(sub["alpha_deg"], sub["Cd"],
            color=re_color(re), marker=MARKERS.get(re, "o"),
            label=f"CFD — {re_label(re)}", linewidth=2, markersize=5)

# NASA reference
nasa_alpha_cd = sorted(NASA_CD.keys())
nasa_cd       = [NASA_CD[a] for a in nasa_alpha_cd]
ax.plot(nasa_alpha_cd, nasa_cd, "k--", linewidth=1.5, label="NASA Exp. (Re=3×10⁶)")

ax.set_xlabel("Angle of Attack α (°)", fontsize=12)
ax.set_ylabel("Drag Coefficient Cd", fontsize=12)
ax.set_title("NACA 0012 — Cd vs α\nSTAR-CCM+ CFD vs NASA Experimental", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "Cd_vs_alpha.png", dpi=150)
plt.close(fig)
print("✅ Saved: Cd_vs_alpha.png")

# ─── Fig 3: Cl/Cd vs α ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
for re in sorted(df["reynolds"].unique()):
    sub = df[df["reynolds"] == re].sort_values("alpha_deg")
    ax.plot(sub["alpha_deg"], sub["ClCd"],
            color=re_color(re), marker=MARKERS.get(re, "o"),
            label=f"CFD — {re_label(re)}", linewidth=2, markersize=5)

ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
ax.set_xlabel("Angle of Attack α (°)", fontsize=12)
ax.set_ylabel("Lift-to-Drag Ratio Cl/Cd", fontsize=12)
ax.set_title("NACA 0012 — Aerodynamic Efficiency (Cl/Cd) vs α", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "ClCd_vs_alpha.png", dpi=150)
plt.close(fig)
print("✅ Saved: ClCd_vs_alpha.png")

# ─── Fig 4: Cl/Cd Polar (Cl vs Cd) ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 8))
for re in sorted(df["reynolds"].unique()):
    sub = df[df["reynolds"] == re].sort_values("alpha_deg")
    sc  = ax.scatter(sub["Cd"], sub["Cl"], c=sub["alpha_deg"],
                     cmap="coolwarm", marker=MARKERS.get(re, "o"),
                     label=re_label(re), s=50, edgecolors="grey", linewidths=0.3)
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("α (°)", fontsize=10)
ax.set_xlabel("Drag Coefficient Cd", fontsize=12)
ax.set_ylabel("Lift Coefficient Cl", fontsize=12)
ax.set_title("NACA 0012 — Aerodynamic Polar\n(Cl vs Cd)", fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "polar_ClvsCd.png", dpi=150)
plt.close(fig)
print("✅ Saved: polar_ClvsCd.png")

# ── 6. NASA VALIDATION COMPARISON ────────────────────────────────────────────
print("\n── NASA validation (Re = 3×10⁶) ────────────────────────────")
re3_data = df[df["reynolds"] == 3e6].copy()
if len(re3_data) == 0:
    # Fallback: use nearest available Re
    nearest_re = min(df["reynolds"].unique(), key=lambda r: abs(r - 3e6))
    re3_data   = df[df["reynolds"] == nearest_re].copy()
    print(f"   (No Re=3e6 data — using Re={nearest_re:.0e} as proxy)")

errors = []
print(f"\n  {'α (°)':>6}  {'CFD Cl':>8}  {'NASA Cl':>8}  {'Error %':>8}")
print(f"  {'------':>6}  {'------':>8}  {'-------':>8}  {'-------':>8}")
for alpha, nasa_cl_val in NASA_CL.items():
    row = re3_data[re3_data["alpha_deg"] == alpha]
    if not row.empty:
        cfd_cl = row["Cl"].values[0]
        err    = abs(cfd_cl - nasa_cl_val) / max(abs(nasa_cl_val), 0.01) * 100
        errors.append(err)
        flag = " ⚠️ " if err > 10 else "  ✓ "
        print(f"  {alpha:>6.0f}  {cfd_cl:>8.4f}  {nasa_cl_val:>8.4f}  {err:>7.1f}%{flag}")

if errors:
    print(f"\n  Mean error: {np.mean(errors):.1f}%")
    print(f"  Max  error: {np.max(errors):.1f}%")
    if np.mean(errors) < 10:
        print("  → Validation PASSED ✅")
    else:
        print("  → Mean error > 10% — check mesh refinement or turbulence settings")

# ── 7. TRAIN / TEST SPLIT ─────────────────────────────────────────────────
print("\n── Train / test split ───────────────────────────────────────")

# Stratified by Reynolds number so each Re has 80/20 split
from sklearn.model_selection import train_test_split

# If sklearn not installed: pip install scikit-learn
df_train, df_test = train_test_split(
    df,
    test_size=TEST_FRACTION,
    random_state=RANDOM_SEED,
    stratify=df["reynolds"].astype(str)   # ensure each Re is represented
)

print(f"  Total samples : {len(df)}")
print(f"  Training set  : {len(df_train)} ({100*(1-TEST_FRACTION):.0f}%)")
print(f"  Test set      : {len(df_test)}  ({100*TEST_FRACTION:.0f}%)")

# Verify Re distribution in each split
for name, split in [("Train", df_train), ("Test", df_test)]:
    print(f"\n  {name} set Re distribution:")
    print(split["reynolds"].value_counts().sort_index().to_string())

df_train.to_csv(OUTPUT_DIR / "train_set.csv", index=False)
df_test.to_csv(OUTPUT_DIR / "test_set.csv",   index=False)
print(f"\n✅ Saved: train_set.csv  ({len(df_train)} rows)")
print(f"✅ Saved: test_set.csv   ({len(df_test)} rows)")

# ── 8. SUMMARY ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Week 3 Complete — Summary")
print("=" * 60)
print(f"  Simulations loaded   : {len(df)}")
print(f"  Reynolds numbers     : {len(df['reynolds'].unique())}")
print(f"  α range              : {df['alpha_deg'].min():.0f}° to {df['alpha_deg'].max():.0f}°")
print(f"  Cl range             : {df['Cl'].min():.3f} to {df['Cl'].max():.3f}")
print(f"  Cd range             : {df['Cd'].min():.5f} to {df['Cd'].max():.5f}")
print(f"  Max Cl/Cd            : {df['ClCd'].max():.1f} at α={df.loc[df['ClCd'].idxmax(),'alpha_deg']:.0f}°")
print(f"  Plots saved to       : {OUTPUT_DIR}/")
print(f"  Ready for Week 4     : train_set.csv + test_set.csv")
print("=" * 60)
