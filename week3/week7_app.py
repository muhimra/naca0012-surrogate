"""
Week 7 - Streamlit Dashboard
=============================
Interactive web app combining:
  - Real-time surrogate predictions
  - Aerodynamic polar plots
  - Cl/Cd vs alpha curves
  - Claude API design assistant chat

Usage:
    streamlit run week7_app.py
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import json
import os
from anthropic import Anthropic

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH  = "output_week4/surrogate_model.pt"
SCALER_PATH = "output_week4/scaler_params.json"
ALL_CSV     = ["output_week3/train_set.csv", "output_week3/test_set.csv"]

NASA_CL = {-10:-1.025,-8:-0.862,-6:-0.650,-4:-0.431,-2:-0.214,
            0:0.000,2:0.221,4:0.441,6:0.660,8:0.867,
           10:1.050,12:1.212,14:1.310,16:1.050}
NASA_CD = {0:0.00605,2:0.00617,4:0.00658,6:0.00780,
           8:0.01040,10:0.01380,12:0.01820,14:0.02560,16:0.06820}

RE_OPTIONS = {"500,000 (Small UAV)": 5e5,
              "1,000,000 (Light Aircraft)": 1e6,
              "3,000,000 (General Aviation)": 3e6}
RE_COLORS  = {5e5: "#2196F3", 1e6: "#FF5722", 3e6: "#4CAF50"}

# ── SURROGATE MODEL ───────────────────────────────────────────────────────────
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

@st.cache_resource
def load_surrogate():
    checkpoint = torch.load(MODEL_PATH, weights_only=True)
    model = SurrogateNet(checkpoint["hidden_sizes"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with open(SCALER_PATH) as f:
        sc = json.load(f)
    X_mean = np.array(sc["X_mean"], dtype=np.float32)
    X_std  = np.array(sc["X_std"],  dtype=np.float32)
    y_mean = np.array(sc["y_mean"], dtype=np.float32)
    y_std  = np.array(sc["y_std"],  dtype=np.float32)
    return model, X_mean, X_std, y_mean, y_std

@st.cache_resource
def load_cfd_data():
    dfs = [pd.read_csv(p) for p in ALL_CSV if os.path.exists(p)]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def predict(model, X_mean, X_std, y_mean, y_std, alpha_deg, re):
    X = np.array([[alpha_deg, re]], dtype=np.float32)
    X_n = (X - X_mean) / X_std
    with torch.no_grad():
        y_n = model(torch.tensor(X_n)).numpy()
    result = (y_n * y_std + y_mean)[0]
    return float(result[0]), float(result[1])

def sweep(model, X_mean, X_std, y_mean, y_std, re, n=200):
    alphas = np.linspace(-10, 20, n)
    cls, cds = [], []
    for a in alphas:
        cl, cd = predict(model, X_mean, X_std, y_mean, y_std, a, re)
        cls.append(cl); cds.append(cd)
    return alphas, np.array(cls), np.array(cds)

# ── CLAUDE TOOLS ──────────────────────────────────────────────────────────────
tools = [
    {
        "name": "query_surrogate",
        "description": "Query the NACA 0012 surrogate at a specific alpha and Reynolds number. Returns Cl, Cd, Cl/Cd.",
        "input_schema": {
            "type": "object",
            "properties": {
                "alpha_deg": {"type": "number", "description": "Angle of attack in degrees (-10 to 20)."},
                "reynolds":  {"type": "number", "description": "Reynolds number (5e5 to 3e6)."}
            },
            "required": ["alpha_deg", "reynolds"]
        }
    },
    {
        "name": "optimise_for_objective",
        "description": "Sweep alpha at a given Re and return top 5 candidates for max_ClCd, max_Cl, or min_Cd.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reynolds":  {"type": "number"},
                "objective": {"type": "string", "enum": ["max_ClCd", "max_Cl", "min_Cd"]},
                "alpha_min": {"type": "number"},
                "alpha_max": {"type": "number"}
            },
            "required": ["reynolds", "objective"]
        }
    }
]

SYSTEM_PROMPT = """You are an aerodynamic design assistant for NACA 0012 aerofoil analysis.
You have access to a trained neural network surrogate model predicting Cl and Cd for any
angle of attack (-10 to 20 deg) and Reynolds number (5e5 to 3e6).

Reynolds number guide:
- Small UAV / drone: Re ~ 5e5
- Light aircraft / glider: Re ~ 1e6
- General aviation: Re ~ 3e6

When given a flight requirement, identify the Re and objective, call the appropriate tool,
then give a concise engineering recommendation with the optimal alpha, Cl, Cd, and Cl/Cd.
Be brief and engineering-focused. Quote numbers to 3 significant figures."""

def run_tool(name, inp, model, X_mean, X_std, y_mean, y_std):
    if name == "query_surrogate":
        cl, cd = predict(model, X_mean, X_std, y_mean, y_std,
                         inp["alpha_deg"], inp["reynolds"])
        clcd = cl / cd if cd > 1e-6 else 0.0
        return {"alpha_deg": inp["alpha_deg"], "reynolds": inp["reynolds"],
                "Cl": round(cl,4), "Cd": round(cd,5), "ClCd": round(clcd,2)}
    elif name == "optimise_for_objective":
        re   = inp["reynolds"]
        obj  = inp["objective"]
        amin = inp.get("alpha_min", -10)
        amax = inp.get("alpha_max", 16)
        alphas = np.linspace(amin, amax, 61)
        rows = []
        for a in alphas:
            cl, cd = predict(model, X_mean, X_std, y_mean, y_std, a, re)
            clcd = cl/cd if cd > 1e-6 else 0.0
            rows.append({"alpha": round(float(a),2), "Cl": round(cl,4),
                         "Cd": round(cd,5), "ClCd": round(clcd,2)})
        key = {"max_ClCd":"ClCd","max_Cl":"Cl","min_Cd":"Cd"}[obj]
        ranked = sorted(rows, key=lambda r: r[key], reverse=(obj != "min_Cd"))
        return {"objective": obj, "reynolds": re, "top_5": ranked[:5]}

# ── PAGE SETUP ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NACA 0012 Surrogate",
    page_icon="✈",
    layout="wide"
)

st.title("NACA 0012 Aerodynamic Surrogate Model")
st.caption("Neural network trained on STAR-CCM+ CFD data · Validated against NASA TM-4071")

model, X_mean, X_std, y_mean, y_std = load_surrogate()
cfd_df = load_cfd_data()

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "Predictor",
    "Polar Curves",
    "Design Assistant"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — REAL-TIME PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Real-Time Aerodynamic Predictor")
    st.write("Adjust the sliders to get instant surrogate predictions.")

    col1, col2 = st.columns([1, 2])

    with col1:
        alpha_input = st.slider("Angle of Attack (deg)", -10.0, 20.0, 5.0, 0.5)
        re_label    = st.selectbox("Reynolds Number", list(RE_OPTIONS.keys()))
        re_input    = RE_OPTIONS[re_label]

        cl, cd = predict(model, X_mean, X_std, y_mean, y_std, alpha_input, re_input)
        clcd   = cl / cd if cd > 1e-6 else 0.0

        st.markdown("---")
        st.metric("Lift Coefficient Cl",       f"{cl:.4f}")
        st.metric("Drag Coefficient Cd",        f"{cd:.5f}")
        st.metric("Lift-to-Drag Ratio (Cl/Cd)", f"{clcd:.1f}")

        # Stall warning
        if alpha_input > 14:
            st.warning("Approaching stall region (>14 deg). Results may be unreliable for steady flight.")

    with col2:
        # Mini Cl vs alpha plot with current point highlighted
        alphas, cls, cds = sweep(model, X_mean, X_std, y_mean, y_std, re_input)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
        color = RE_COLORS.get(re_input, "#666")

        ax1.plot(alphas, cls, color=color, linewidth=2)
        ax1.axvline(alpha_input, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax1.scatter([alpha_input], [cl], color="red", s=80, zorder=5)
        ax1.set_ylabel("Cl", fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f"Surrogate Predictions — {re_label}", fontsize=11)

        ax2.plot(alphas, cds, color=color, linewidth=2)
        ax2.axvline(alpha_input, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax2.scatter([alpha_input], [cd], color="red", s=80, zorder=5)
        ax2.set_xlabel("Angle of Attack (deg)", fontsize=11)
        ax2.set_ylabel("Cd", fontsize=11)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — POLAR CURVES
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Aerodynamic Polar Curves")
    st.write("Surrogate predictions vs CFD data vs NASA TM-4071 experimental results.")

    show_nasa = st.checkbox("Show NASA experimental data", value=True)
    show_cfd  = st.checkbox("Show CFD data points", value=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax_cl, ax_cd, ax_polar = axes

    for re_lbl, re_val in RE_OPTIONS.items():
        color = RE_COLORS[re_val]
        alphas, cls, cds = sweep(model, X_mean, X_std, y_mean, y_std, re_val)
        clcds = cls / np.where(cds > 1e-6, cds, 1e-6)
        short_lbl = re_lbl.split(" ")[0].replace(",","")
        label = f"Surrogate Re={float(short_lbl):.0e}"

        ax_cl.plot(alphas, cls, color=color, linewidth=2, label=label)
        ax_cd.plot(alphas, cds, color=color, linewidth=2, label=label)
        ax_polar.plot(cds, cls, color=color, linewidth=2, label=label)

        if show_cfd and not cfd_df.empty:
            sub = cfd_df[cfd_df["reynolds"] == re_val]
            ax_cl.scatter(sub["alpha_deg"], sub["Cl"], color=color,
                          edgecolors="black", linewidths=0.5, s=25, zorder=4)
            ax_cd.scatter(sub["alpha_deg"], sub["Cd"], color=color,
                          edgecolors="black", linewidths=0.5, s=25, zorder=4)
            ax_polar.scatter(sub["Cd"], sub["Cl"], color=color,
                             edgecolors="black", linewidths=0.5, s=25, zorder=4)

    if show_nasa:
        nasa_a  = sorted(NASA_CL.keys())
        nasa_cl = [NASA_CL[a] for a in nasa_a]
        ax_cl.scatter(nasa_a, nasa_cl, color="black", marker="D",
                      s=50, zorder=5, label="NASA TM-4071")
        nasa_a_cd  = sorted(NASA_CD.keys())
        nasa_cd    = [NASA_CD[a] for a in nasa_a_cd]
        nasa_cl_cd = [NASA_CL[a] for a in nasa_a_cd]
        ax_cd.scatter(nasa_a_cd, nasa_cd, color="black", marker="D",
                      s=50, zorder=5, label="NASA TM-4071")
        ax_polar.scatter(nasa_cd, nasa_cl_cd, color="black", marker="D",
                         s=50, zorder=5, label="NASA TM-4071")

    ax_cl.set_xlabel("Alpha (deg)"); ax_cl.set_ylabel("Cl")
    ax_cl.set_title("Lift Coefficient"); ax_cl.legend(fontsize=8); ax_cl.grid(True, alpha=0.3)

    ax_cd.set_xlabel("Alpha (deg)"); ax_cd.set_ylabel("Cd")
    ax_cd.set_title("Drag Coefficient"); ax_cd.legend(fontsize=8); ax_cd.grid(True, alpha=0.3)

    ax_polar.set_xlabel("Cd"); ax_polar.set_ylabel("Cl")
    ax_polar.set_title("Aerodynamic Polar"); ax_polar.legend(fontsize=8); ax_polar.grid(True, alpha=0.3)

    fig.suptitle("NACA 0012 — Surrogate vs CFD vs NASA", fontsize=13, y=1.01)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Accuracy summary
    st.markdown("---")
    st.markdown("**Model accuracy (trained on STAR-CCM+ CFD data)**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cl R²",        "0.9999")
    col2.metric("Cd R²",        "0.9998")
    col3.metric("Cl vs NASA",   "4.9% mean error")
    col4.metric("Speedup vs CFD", "~3.5M x")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CLAUDE DESIGN ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Aerodynamic Design Assistant")
    st.write("Describe your flight requirement in plain English and get an optimal aerofoil configuration.")

    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Get a key at console.anthropic.com"
    )

    # Chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:
        st.session_state.history = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    user_input = st.chat_input("e.g. Best cruise efficiency for a small UAV")

    if user_input:
        if not api_key:
            st.error("Please enter your Anthropic API key above.")
        else:
            # Show user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Add to Claude conversation history
            st.session_state.history.append({"role": "user", "content": user_input})

            client = Anthropic(api_key=api_key)

            with st.chat_message("assistant"):
                with st.spinner("Querying surrogate model..."):
                    # Agentic tool loop
                    while True:
                        response = client.messages.create(
                            model="claude-haiku-4-5-20251001",
                            max_tokens=1024,
                            system=SYSTEM_PROMPT,
                            tools=tools,
                            messages=st.session_state.history
                        )

                        if response.stop_reason == "tool_use":
                            tool_results = []
                            for block in response.content:
                                if block.type == "tool_use":
                                    result = run_tool(block.name, block.input,
                                                      model, X_mean, X_std,
                                                      y_mean, y_std)
                                    tool_results.append({
                                        "type": "tool_result",
                                        "tool_use_id": block.id,
                                        "content": json.dumps(result)
                                    })
                            st.session_state.history.append(
                                {"role": "assistant", "content": response.content})
                            st.session_state.history.append(
                                {"role": "user", "content": tool_results})
                        else:
                            final = "".join(
                                b.text for b in response.content if b.type == "text")
                            st.markdown(final)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": final})
                            st.session_state.history.append(
                                {"role": "assistant", "content": final})
                            break

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.history  = []
        st.rerun()
