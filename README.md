# NACA 0012 Aerodynamic Surrogate Model

A neural network trained on STAR-CCM+ CFD simulations that predicts aerofoil lift and drag
coefficients in milliseconds — with a Claude-powered design assistant that recommends optimal
configurations for given flight conditions.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red)

## Results

| Metric | Value |
|---|---|
| Cl R² (surrogate vs CFD) | 0.9999 |
| Cd R² (surrogate vs CFD) | 0.9998 |
| Cl error vs NASA TM-4071 | 4.9% mean |
| Surrogate prediction time | 0.17 ms |
| CFD run time | ~10 min |
| **Speedup factor** | **~3.5 million x** |

## What it does

1. **CFD data generation** — parametric sweep of 48 NACA 0012 simulations in STAR-CCM+
   across angles of attack (-10 to 20 deg) and Reynolds numbers (5e5, 1e6, 3e6)
2. **Surrogate training** — feedforward PyTorch network (2 -> 64 -> 128 -> 64 -> 2)
   trained to predict Cl and Cd from alpha and Re
3. **Validation** — compared against held-out CFD data and NASA TM-4071 wind tunnel results
4. **Design assistant** — Claude API integration accepts plain-English flight requirements
   and returns optimal aerofoil configurations
5. **Streamlit dashboard** — interactive app with real-time predictions and polar curves

## Installation

```bash
git clone https://github.com/muhimra/naca0012-surrogate
cd naca0012-surrogate
pip install -r requirements.txt
```

## Usage

**Run the dashboard:**
```bash
streamlit run week7_app.py
```

**Train the model from scratch:**
```bash
python week3_data_processing.py
python week4_train_surrogate.py
python week5_validation.py
```

**Command-line design assistant:**
```bash
set ANTHROPIC_API_KEY=your_key_here   # Windows
python week6_assistant.py
```

## Tech Stack

| Layer | Tool |
|---|---|
| CFD simulations | STAR-CCM+ 2502 (university licence) |
| Data processing | Python, pandas, numpy |
| Neural network | PyTorch |
| Validation data | NASA TM-4071 (Ladson et al. 1988) |
| Design assistant | Claude API (Haiku) |
| Dashboard | Streamlit |

## Key Findings

- **Surrogate accuracy:** R² > 0.999 for both Cl and Cd across the full parameter space
- **Speed:** 0.17 ms per prediction vs ~10 minutes per CFD run (~3.5 million x speedup)
- **Cl validation:** 4.9% mean error vs NASA experimental data for attached flow (alpha < 14 deg)
- **Cd deviation:** 31.8% vs NASA — attributable to k-omega SST turbulence model
  overprediction of skin friction drag, a known characteristic at low AoA
- **Stall:** Steady RANS diverges from experiment above ~14 deg as expected


## Project Structure

```
naca0012_sweep.java          STAR-CCM+ parametric sweep macro
week3_data_processing.py     Data cleaning, EDA, train/test split
week4_train_surrogate.py     PyTorch model training (k-fold CV)
week5_validation.py          Accuracy analysis vs CFD and NASA
week6_assistant.py           Claude API command-line assistant
week7_app.py                 Streamlit dashboard
output_week3/                Processed data and EDA plots
output_week4/                Trained model weights and scaler
output_week5/                Validation plots and report
```

## Background


Surrogate modelling is an industrial technique used by Airbus, Boeing, and Formula 1 teams to
replace expensive simulation runs with fast neural network approximations during design iteration.

## Reference

Ladson, C. L., et al. (1988). *Effects of Independent Variation of Mach and Reynolds Numbers
on the Low-Speed Aerodynamic Characteristics of the NACA 0012 Airfoil Section.*
NASA TM-4071.
