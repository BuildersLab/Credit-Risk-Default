# Model Card: {{PROJECT_NAME}} Champion Model

> Fill in TBD fields after the Week 8 modeling milestone.

## Model details

| Property | Value |
|---|---|
| Model type | XGBoost binary classifier (update if different) |
| Version | v0.1 |
| Trained by | {{TEAM_DS}} |
| Date trained | TBD |
| Framework | XGBoost 2.0 + scikit-learn 1.4 |

## Intended use

**Primary use:** TODO — describe what the model predicts and who uses it at {{COMPANY_NAME}}.
**Primary users:** {{PERSONA}}
**Out-of-scope uses:** TODO — list uses this model should not be applied to.

## Training data

See `data/DATA_CARD.md`.

- Dataset: {{DATASET_NAME}}
- Train period / split: TODO
- Features: TODO (number of engineered features)

## Performance metrics

| Metric | Value |
|---|---|
| {{PRIMARY_METRIC}} | TBD |
| Recall at threshold | TBD |
| Precision at threshold | TBD |
| False positives per 1,000 negatives | TBD |
| Business impact | TBD |

## Operating threshold

**Chosen threshold:** TBD
**Rationale:** TBD

## Explainability

SHAP values are computed for every prediction. See `notebooks/04_explainability.ipynb`.
The Gemini API translates SHAP output into plain English for {{PERSONA}}.
Gemini output is advisory only. The model score is the authoritative decision.

## Bias evaluation

TODO: document bias audit results across demographic and geographic groups.

## Limitations

TODO: what are the known limitations of this model?

## How to reproduce

```bash
make data
make train
make evaluate
```
