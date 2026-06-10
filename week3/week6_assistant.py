"""
Week 6 - Claude API Design Assistant
=====================================
Accepts plain-English flight requirements and returns optimal
aerofoil configuration recommendations using the surrogate model.

Usage:
    python week6_assistant.py

Requirements:
    pip install anthropic
    Set environment variable: ANTHROPIC_API_KEY=your_key_here
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from anthropic import Anthropic

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH  = "output_week4/surrogate_model.pt"
SCALER_PATH = "output_week4/scaler_params.json"

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

def predict(model, X_mean, X_std, y_mean, y_std, alpha_deg, re):
    X = np.array([[alpha_deg, re]], dtype=np.float32)
    X_n = (X - X_mean) / X_std
    with torch.no_grad():
        y_n = model(torch.tensor(X_n)).numpy()
    result = (y_n * y_std + y_mean)[0]
    return float(result[0]), float(result[1])  # Cl, Cd

def sweep_surrogate(model, X_mean, X_std, y_mean, y_std, re,
                    alpha_min=-10, alpha_max=20, n_points=61):
    """Sweep alpha at a given Re, return list of dicts."""
    alphas = np.linspace(alpha_min, alpha_max, n_points)
    results = []
    for a in alphas:
        cl, cd = predict(model, X_mean, X_std, y_mean, y_std, a, re)
        clcd = cl / cd if cd > 1e-6 else 0.0
        results.append({"alpha": round(float(a), 2), "Cl": round(cl, 4),
                         "Cd": round(cd, 5), "ClCd": round(clcd, 2)})
    return results

# ── CLAUDE TOOL DEFINITIONS ──────────────────────────────────────────────────
tools = [
    {
        "name": "query_surrogate",
        "description": (
            "Query the NACA 0012 aerodynamic surrogate model. "
            "Given an angle of attack (alpha_deg) and Reynolds number (reynolds), "
            "returns lift coefficient (Cl), drag coefficient (Cd), and lift-to-drag ratio (Cl/Cd). "
            "Use this to evaluate specific operating points."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "alpha_deg": {
                    "type": "number",
                    "description": "Angle of attack in degrees. Valid range: -10 to 20."
                },
                "reynolds": {
                    "type": "number",
                    "description": "Reynolds number. Typical range: 5e5 to 3e6."
                }
            },
            "required": ["alpha_deg", "reynolds"]
        }
    },
    {
        "name": "optimise_for_objective",
        "description": (
            "Sweep the surrogate across a range of angles of attack at a given Reynolds number "
            "and return the top candidates ranked by a specified objective: "
            "'max_ClCd' (best efficiency), 'max_Cl' (maximum lift), or 'min_Cd' (minimum drag). "
            "Returns the top 5 candidates with their aerodynamic coefficients."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reynolds": {
                    "type": "number",
                    "description": "Reynolds number for the sweep."
                },
                "objective": {
                    "type": "string",
                    "enum": ["max_ClCd", "max_Cl", "min_Cd"],
                    "description": "Optimisation objective."
                },
                "alpha_min": {
                    "type": "number",
                    "description": "Minimum alpha to sweep (default -10)."
                },
                "alpha_max": {
                    "type": "number",
                    "description": "Maximum alpha to sweep (default 16 to stay below stall)."
                }
            },
            "required": ["reynolds", "objective"]
        }
    }
]

# ── TOOL EXECUTION ────────────────────────────────────────────────────────────
def run_tool(tool_name, tool_input, model, X_mean, X_std, y_mean, y_std):
    if tool_name == "query_surrogate":
        alpha = tool_input["alpha_deg"]
        re    = tool_input["reynolds"]
        cl, cd = predict(model, X_mean, X_std, y_mean, y_std, alpha, re)
        clcd = cl / cd if cd > 1e-6 else 0.0
        return {
            "alpha_deg": alpha, "reynolds": re,
            "Cl": round(cl, 4), "Cd": round(cd, 5),
            "ClCd": round(clcd, 2)
        }

    elif tool_name == "optimise_for_objective":
        re        = tool_input["reynolds"]
        objective = tool_input["objective"]
        a_min     = tool_input.get("alpha_min", -10)
        a_max     = tool_input.get("alpha_max", 16)
        results   = sweep_surrogate(model, X_mean, X_std, y_mean, y_std,
                                    re, a_min, a_max)
        key_map   = {"max_ClCd": "ClCd", "max_Cl": "Cl", "min_Cd": "Cd"}
        key       = key_map[objective]
        reverse   = objective != "min_Cd"
        ranked    = sorted(results, key=lambda r: r[key], reverse=reverse)
        return {"objective": objective, "reynolds": re, "top_5": ranked[:5]}

# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an aerodynamic design assistant for NACA 0012 aerofoil analysis.

You have access to a trained neural network surrogate model that predicts lift (Cl) and drag (Cd) 
coefficients for any angle of attack (-10 to 20 deg) and Reynolds number (5e5 to 3e6).

When a user describes a flight requirement in plain English, you:
1. Identify the relevant Reynolds number from the application context:
   - Small UAV / drone: Re ~ 5e5
   - Light aircraft / glider: Re ~ 1e6
   - General aviation: Re ~ 3e6
2. Identify the optimisation objective:
   - Cruise efficiency / endurance / gliding -> maximise Cl/Cd
   - Short takeoff / maximum lift -> maximise Cl
   - Low drag / high speed -> minimise Cd
3. Query the surrogate using the appropriate tool
4. Return a clear recommendation with the optimal angle of attack, 
   expected Cl and Cd values, and brief physical reasoning.

Always be concise and engineering-focused. Quote numbers to 3 significant figures.
If the requirement is ambiguous, state your assumptions clearly."""

# ── MAIN CHAT LOOP ────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  NACA 0012 Aerodynamic Design Assistant")
    print("  Powered by surrogate model + Claude API")
    print("=" * 60)
    print("  Type your flight requirement in plain English.")
    print("  Examples:")
    print("    'I need maximum lift for a short takeoff'")
    print("    'Best cruise efficiency for a small UAV'")
    print("    'Minimise drag for a high-speed glider'")
    print("  Type 'quit' to exit.\n")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        print("Set it with:  set ANTHROPIC_API_KEY=your_key_here  (Windows)")
        return

    client = Anthropic(api_key=api_key)
    surr_model, X_mean, X_std, y_mean, y_std = load_surrogate()
    print("Surrogate model loaded.\n")

    conversation_history = []

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        conversation_history.append({"role": "user", "content": user_input})

        # Agentic loop — Claude may call tools multiple times
        while True:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=tools,
                messages=conversation_history
            )

            # Collect any text to display
            text_parts = [b.text for b in response.content if b.type == "text"]

            if response.stop_reason == "tool_use":
                # Execute all tool calls in this response
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        print(f"  [tool] {block.name}({json.dumps(block.input)})")
                        result = run_tool(block.name, block.input,
                                          surr_model, X_mean, X_std, y_mean, y_std)
                        print(f"  [result] {json.dumps(result)}")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result)
                        })

                # Add assistant turn + tool results to history
                conversation_history.append(
                    {"role": "assistant", "content": response.content})
                conversation_history.append(
                    {"role": "user", "content": tool_results})

            else:
                # Final response — print and break inner loop
                final_text = "\n".join(text_parts)
                print(f"\nAssistant: {final_text}\n")
                conversation_history.append(
                    {"role": "assistant", "content": final_text})
                break

if __name__ == "__main__":
    main()
