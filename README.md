Gaussian Process Modeling of Heat Generation in Lithium-Ion Batteries

This repository contains the modular implementation of my Master’s thesis project:
"Electrochemical–Thermal Modeling and Gaussian Process Sensitivity Analysis of Li-ion Batteries".

The goal is to bridge physics-based battery models with machine learning methods (Gaussian Processes, kernel design, global sensitivity) to better understand and predict the mechanisms of heat generation during charge and discharge.

🎯 Motivation & Proposal

Lithium-ion batteries are complex electrochemical–thermal systems where heat generation impacts safety, lifetime, and efficiency. Traditional models (e.g. Doyle–Fuller–Newman or its reduced versions) provide detailed physics but are computationally heavy.

My thesis proposes a hybrid approach:

Simulate batteries under controlled charge/discharge conditions using PyBaMM (Python Battery Mathematical Modeling).

Extract key thermal and electrochemical quantities (overpotentials, voltage dynamics, temperature derivatives).

Train Gaussian Process (GP) models with advanced kernel structures to map from these signals to temperature change and heat generation.

Evaluate sensitivity to both inputs (features) and hyperparameters via Sobol global variance analysis.

Compare kernel structures (multiplicative, additive, hybrid) to assess robustness and physical interpretability.

This framework produces a data-driven yet physics-aware surrogate model for heat generation — valuable for control, diagnosis, and design of Li-ion systems.

🛠️ Implementation

The code is designed as a modular pipeline, with each step tracked and reproducible via DVC
.

🔧 Simulation

Tool: PyBaMM
 (TSPMe and DFN-based thermal models).

Experiments: grids of charge/discharge C-rates and ambient temperatures.

Outputs: CSV datasets with electrochemical states, overpotentials, and temperature derivatives.

📊 Gaussian Process Modeling

Library: GPy
.

Targets:

𝑇
˙
T
˙
 (temperature time derivative)

Derived heat generation terms (
𝑄
rev
,
𝑄
ohm
,
𝑄
irrev
Q
rev
	​

,Q
ohm
	​

,Q
irrev
	​

)

Features (ARD kernels):

𝑉
˙
V
˙
 – terminal voltage derivative

𝜂
˙
𝑟
𝑥
𝑛
η
˙
	​

rxn
	​

 – reaction overpotential derivative

𝜂
˙
𝑠
−
𝑐
𝑛
𝑐
η
˙
	​

s−cnc
	​

 – solid concentration overpotential derivative

𝜂
˙
𝑙
−
𝑐
𝑛
𝑐
η
˙
	​

l−cnc
	​

 – electrolyte concentration overpotential derivative

𝜂
˙
𝑙
−
𝑜
ℎ
𝑚
η
˙
	​

l−ohm
	​

 – electrolyte ohmic overpotential derivative

Kernels implemented:

Multiplicative (baseline physics-informed: 
𝑘
𝑉
×
𝑘
𝜂
k
V
	​

×k
η
	​

)

Additive (
𝑘
𝑉
+
𝑘
𝜂
k
V
	​

+k
η
	​

)

Hybrid/structured (
𝑘
𝑉
+
𝑘
𝑉
×
𝑘
𝜂
k
V
	​

+k
V
	​

×k
η
	​

) with different priors.

📈 Evaluation & Tracking

Metrics: RMSE, MAE, negative log-likelihood (NLL).

Visualizations:

𝑇
˙
T
˙
 predictions vs model ground-truth

Decomposition of reversible/irreversible/ohmic heat contributions

Posterior predictive intervals

Tracking: MLflow
 for experiments, DVC for reproducibility.

🌐 Sensitivity Analysis

Framework: SALib
 with Saltelli sampling.

Sobol indices computed for:

Features: which electrochemical signals dominate variance in 
𝑇
˙
T
˙
.

Hyperparameters: lengthscales (
ℓ
ℓ), variances (
𝜎
𝑓
σ
f
	​

), grouped by physical category (voltage vs overpotentials).

Robustness checks: kernel structure, priors on hyperparameters, dataset ranges.

Outputs: heatmaps of S1 and ST indices, grouped barplots per feature group.

📂 Project Structure
gpheat/
├── config/                  # Configuration (YAML)
├── src/gpheat/              # Core package
│   ├── data/                # Data loading, slicing, preprocessing
│   ├── simulate/            # PyBaMM model wrappers
│   ├── models/              # GP models, kernels, bounds
│   ├── evaluation/          # Metrics, plotting
│   ├── sensitivity/         # Sobol analysis (features & hyperparams)
│   ├── experiments/         # Run orchestration
│   └── cli.py               # Command-line interface
├── reports/                 # Figures, metrics, summaries
├── artifacts/               # Datasets, trained models, test sets
├── dvc.yaml                 # DVC pipeline definition
└── requirements.txt         # Python dependencies

🚀 Reproducible Pipeline (DVC)

Generate datasets:

dvc repro data


Train GP models:

dvc repro train_charge
dvc repro train_discharge


Evaluate models:

dvc repro eval_charge
dvc repro eval_discharge


Run Sobol sensitivity:

python -m gpheat.cli sobol-x --mode charge
python -m gpheat.cli sobol-hp --mode discharge --kernel mult

📑 Main Contributions

Pipeline Design: End-to-end modular ML–physics integration with reproducibility (DVC, MLflow).

Physics-Informed Kernels: Structured GP kernels matching electrochemical mechanisms.

Sensitivity Analysis: First application (to my knowledge) of Sobol global sensitivity to GP hyperparameters and Li-ion battery thermal features.

Robustness Studies: Evaluation of kernel structure and prior distributions on GP performance.

Open Science Practices: Code modularization, experiment tracking, and data versioning for reproducibility.

📌 Next Steps

Expand to larger-scale DFN simulations and experimental datasets.

Integrate control-oriented outputs (thermal runaway risk indices).

Explore alternative surrogate models (Bayesian Neural Nets, Deep GPs).

Publish results as a framework for physics-informed GP modeling in energy systems.

👤 Author

Samuel Alejandro Durán Enríquez
MSc Applied Mathematics – CIMAT (Centro de Investigación en Matemáticas, México)

Focus: inverse problems, electrochemical–thermal modeling, Gaussian Processes, sensitivity analysis, and data-driven approaches for energy systems.