# Credit Risk Default

![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Build](https://img.shields.io/badge/Build-Passing-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-Credit%20risk%20data%20-%20TBD-orange)
![LLM](https://img.shields.io/badge/LLM-Gemini%20API-purple)

Build an explainable machine learning credit risk platform for NorthBay Bank that predicts credit card account defaults using behavioural repayment patterns, assists credit officers through an interactive review dashboard, and enables early intervention to reduce financial losses while minimizing unnecessary customer flags.

Built by the BuildersLab team for NorthBay Bank TBD.

---

## What this project does

<!-- Fill in after Week 1. Describe the problem, the system, and the output in 3 to 5 bullet points. -->

- TODO
- TODO
- TODO

## Live demo

[Credit-Risk-Default.replit.app](#) — public, no login required

---

## Team

| Name | Role |
|---|---|
| Nafisat Ibrahim, Marienne Dosso | Project Lead: delivery, stakeholder framing, demo |
| Bintou Ba, Marienne Dosso, Lynda Allepo | Data Science: EDA, feature engineering, modeling, explainability, documentation |
| Divyanshi kashyap | ML Engineer: pipeline, API integration, Replit app |

---

## Quickstart

```bash
git clone https://github.com/builderslab/Credit-Risk-Default.git
cd Credit-Risk-Default
make setup
```

Download the dataset from URL - TBD and place the files in `data/raw/`. See [docs/setup_guide.md](docs/setup_guide.md) for full instructions.

```bash
make data      # build features
make train     # train champion model
make evaluate  # print metrics report
make app       # launch dashboard
```

---

## Repo structure

```
Credit-Risk-Default/
├── data/
│   ├── raw/                  # original files, never edited, gitignored
│   ├── processed/            # pipeline outputs, gitignored
│   └── DATA_CARD.md
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_explainability.ipynb
├── src/
│   ├── data_pipeline.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explain.py
│   ├── predict.py
│   ├── gemini.py             # optional: remove if not using Gemini
│   └── prompts.py            # optional: remove if not using Gemini
├── models/
│   └── MODEL_CARD.md
├── app/
│   ├── main.py
│   ├── pages/
│   ├── components/
│   └── requirements.txt
├── tests/
├── docs/
├── .github/
├── .env.example
├── Makefile
└── pyproject.toml
```

---

## Key results

> Populated after Week 8 modeling milestone

| Metric | Value |
|---|---|
| PR-AUC | TBD |
| Recall at threshold | TBD |
| Business impact | TBD |
| False positive rate | TBD |

---

## Documentation

- [Data Card](data/DATA_CARD.md)
- [Model Card](models/MODEL_CARD.md)
- [Setup Guide](docs/setup_guide.md)
- [Workflow and PR conventions](docs/workflow.md)
- [Decision log](docs/decisions.md)
- [Notion workspace](https://www.notion.so/builderslab/Credit-Risk-Scoring-System-360663a5d5d180b18f73d93d27db9c42)
