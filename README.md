# Gaussian Process Modeling of Heat Generation in Lithium-Ion Batteries

This repository contains the modular implementation of my Master’s thesis project:  
**"Electrochemical–Thermal Modeling and Gaussian Process Sensitivity Analysis of Li-ion Batteries"**.  

The goal is to bridge **physics-based battery models** with **machine learning methods** (Gaussian Processes, kernel design, global sensitivity) to better understand and predict the mechanisms of heat generation during charge and discharge.

---

## 🎯 Motivation & Proposal

Lithium-ion batteries are complex electrochemical–thermal systems where heat generation impacts safety, lifetime, and efficiency. Traditional models (e.g. Doyle–Fuller–Newman or its reduced versions) provide detailed physics but are computationally heavy.  

My thesis proposes a **hybrid approach**:

- **Simulate** batteries under controlled charge/discharge conditions using **PyBaMM** (Python Battery Mathematical Modeling).
- **Extract** key thermal and electrochemical quantities (overpotentials, voltage dynamics, temperature derivatives).
- **Train Gaussian Process (GP) models** with advanced kernel structures to map from these signals to **temperature change and heat generation**.
- **Evaluate sensitivity** to both **inputs** (features) and **hyperparameters** via **Sobol global variance analysis**.
- **Compare kernel structures** (multiplicative, additive, hybrid) to assess robustness and physical interpretability.

This framework produces a **data-driven yet physics-aware surrogate model** for heat generation — valuable for control, diagnosis, and design of Li-ion systems.

---

## 🛠️ Implementation

The code is designed as a **modular pipeline**, with each step tracked and reproducible via [DVC](https://dvc.org/).  

### 🔧 Simulation
- **Tool:** [PyBaMM](https://pybamm.org/) (TSPMe and DFN-based thermal models).
- **Experiments:** grids of charge/discharge C-rates and ambient temperatures.
- **Outputs:** CSV datasets with electrochemical states, overpotentials, and temperature derivatives.

### 📊 Gaussian Process Modeling
- **Library:** [GPy](https://sheffieldml.github.io/GPy/).
- **Targets:**  
  - \( \dot{T} \) (temperature time derivative)  
  - Derived heat generation terms (\( Q_{\text{rev}}, Q_{\text{ohm}}, Q_{\text{irrev}} \))
- **Features (ARD kernels):**
  - \( \dot{V} \) – terminal voltage derivative
  - \( \dot{\eta}_{rxn} \) – reaction overpotential derivative
  - \( \dot{\eta}_{s-cnc} \) – solid concentration overpotential derivative
  - \( \dot{\eta}_{l-cnc} \) – electrolyte concentration overpotential derivative
  - \( \dot{\eta}_{l-ohm} \) – electrolyte ohmic overpotential derivative
- **Kernels implemented:**
  - **Multiplicative** (baseline physics-informed: \( k_V \times k_{η} \))
  - **Additive** (\( k_V + k_{η} \))
  - **Hybrid/structured** (\( k_V + k_V \times k_{η} \)) with different priors.

### 📈 Evaluation & Tracking
- **Metrics:** RMSE, MAE, negative log-likelihood (NLL).
- **Visualizations:**  
  - \( \dot{T} \) predictions vs model ground-truth  
  - Decomposition of reversible/irreversible/ohmic heat contributions  
  - Posterior predictive intervals  
- **Tracking:** [MLflow](https://mlflow.org/) for experiments, DVC for reproducibility.

### 🌐 Sensitivity Analysis
- **Framework:** [SALib](https://salib.readthedocs.io/) with Saltelli sampling.
- **Sobol indices computed for:**
  - **Features:** which electrochemical signals dominate variance in \( \dot{T} \).
  - **Hyperparameters:** lengthscales (\( \ell \)), variances (\( \sigma_f \)), grouped by physical category (voltage vs overpotentials).
- **Robustness checks:** kernel structure, priors on hyperparameters, dataset ranges.
- **Outputs:** heatmaps of S1 and ST indices, grouped barplots per feature group.

---

## 📂 Project Structure

gpheat/
├── config/ # Configuration (YAML)
├── src/gpheat/ # Core package
│ ├── data/ # Data loading, slicing, preprocessing
│ ├── simulate/ # PyBaMM model wrappers
│ ├── models/ # GP models, kernels, bounds
│ ├── evaluation/ # Metrics, plotting
│ ├── sensitivity/ # Sobol analysis (features & hyperparams)
│ ├── experiments/ # Run orchestration
│ └── cli.py # Command-line interface
├── reports/ # Figures, metrics, summaries
├── artifacts/ # Datasets, trained models, test sets
├── dvc.yaml # DVC pipeline definition
└── requirements.txt # Python dependencies


---

## 🚀 Reproducible Pipeline (DVC)

1. **Generate datasets:**
   ```bash
   dvc repro data
2. **Train GP models:**
dvc repro train_charge
dvc repro train_discharge
3. **Evaluate models:**
dvc repro eval_charge
dvc repro eval_discharge

4. **Run Sobol sensitivity:**
python -m gpheat.cli sobol-x --mode charge
python -m gpheat.cli sobol-hp --mode discharge --kernel mult
