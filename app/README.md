# NorthBay Loan Default Prediction

A polished Streamlit decision-support prototype for NorthBay Bank. The app loads the trained XGBoost model from the public BuildersLab Hugging Face repository.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Application flow

1. **Home** — project context, model profile, evaluation-metric placeholders, and decision workflow.
2. **Model results** — reserved panels for validated charts and evaluation results.
3. **Score a loan** — interactive origination-time inputs scored by the deployed model.
4. **Loan summary** — probability, tier, thresholds, global drivers, interpretation, review effect, and decision audit trail.

## Model source

- Repository: `BuildersLab/credit-risk-default-model`
- Files: `final_model_bundle.pkl` and `feature_defaults.json`
- The model is downloaded automatically when a scoring page first needs it.

The optional `.streamlit/config.toml` adds matching Streamlit theme settings.
