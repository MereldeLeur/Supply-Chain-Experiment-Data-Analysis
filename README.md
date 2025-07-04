
# Experimental Data Analysis – Multivariate & Non-Parametric Testing

This repository contains the full Python code used for the analysis of the behavioural experiment in the MSc thesis _"Timeliness Criticality in Supply Chains"_ by Merel de Leur (2025). It combines multivariate regression modeling with non-parametric group-level testing to analyze market activity, learning, and system fragility.

---

## Features

- Preprocess raw experimental event logs and survey data
- Compute group-level and individual-level metrics (e.g. click frequency, inverse flow)
- Estimate fixed-effects OLS regression models with clustered standard errors
- Conduct Wilcoxon, Mann–Whitney, Kruskal–Wallis, and Spearman tests
- Generate all final figures and tables used in the thesis (e.g. KDE plots, boxplots, regression tables)

---

## Structure

```
Thesis Experimental Data Analysis.py   # Main analysis script
data/                                  # Raw input data (event logs, survey responses)
results/                               # All output files (figures, test results, tables)
├── figures/                           # Final plots (PNG and PDF)
├── tables/                            # LaTeX/CSV regression tables
├── metrics/                           # Intermediate computation outputs
└── logs/                              # Notes on runtime or warnings
```

---

## Requirements

Install required packages using:

```bash
pip install pandas numpy matplotlib seaborn statsmodels scipy
```

> Tested with Python 3.10+

---

## Running the Code

To reproduce the full analysis pipeline, run:

```bash
python "Thesis Experimental Data Analysis.py"
```

You can modify which hypotheses or figures are executed by toggling the `if run_Hx:` flags in the script.

---

## Key Outputs

- **Regression models**: OLS with group fixed effects, session-clustered errors
- **Clicking behavior metrics**: Frequency per second, round, minute
- **Fragility indicators**: Mean and std. deviation of inverse flow (click-space)
- **Non-parametric tests**: Wilcoxon, Mann–Whitney, Kruskal–Wallis
- **Thesis-ready plots**:
  - Boxplots and KDEs by treatment
  - Inverse flow panels
  - Within-/Across-round Spearman trends

---

## Analysis Workflow

### 1. Preprocessing
- Clean and merge event-level logs with group metadata and survey results
- Generate identifiers for round, agent, treatment, group, and session

### 2. Market Activity (H1–H4)
- OLS regressions predicting final balance and click frequency
- Control for treatment, round, age, gender, education, risk, task understanding
- Group-level medians used for non-parametric comparisons
- Key Figures:
  - Activity vs. Balance (H1)
  - Inventory/Asymmetry/Transparency effects (H2–H4)

### 3. Fragility Measures (H5–H7)
- Compute inverse flow: time to collect 5 items per agent
- Analyze mean and std. dev. by inventory distribution and condition
- Compare symmetric vs asymmetric, transparent vs non-transparent and different total inventory setups


### 4. Learning (H8–H9)
- Divide rounds into 3 one-minute intervals
- Estimate regressions and correlations per time slice
- Compute Spearman rho across time (within rounds) and across rounds
- Group-level rho visualizations by treatment and session

---

## Reproducibility

All analysis outputs are saved and figures can be regenerated by re-running the script. Figures used in the thesis are located in:

```
results/figures/
```

Tables for the appendix and statistical summaries are located in:

```
results/tables/
```

---

## Citation

If you use this code, please cite:

> de Leur, M. J. M. (2025). _Timeliness Criticality in Supply Chains_. MSc Finance Thesis. Vrije Universiteit Amsterdam.

---

## Contact

Merel de Leur  
mjm.deleur@gmail.com
