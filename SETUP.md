# BuildersLab Project Template: Setup Guide

This is the official BuildersLab project template. Every new project starts here.

---

## Option A: Automated setup (recommended)

Run the init script once after cloning. It will ask you a series of questions and replace all placeholder text across every file automatically.

```bash
git clone https://github.com/builderslab/builderslab-template.git {{PROJECT_SLUG}}
cd {{PROJECT_SLUG}}
python init_project.py
```

If you are setting up the project in Windows PowerShell, create and activate a virtual environment first, then install the app dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r app/requirements.txt
```

Next, create your local environment file and fill in any required values:

```bash
cp .env.example .env
# then open .env in an editor and fill any values (e.g. GEMINI_API_KEY if you use Gemini)
```

Then delete the script and commit:

```bash
rm init_project.py
rm SETUP.md
git add -A
git commit -m "initialise project from builderslab template"
git remote set-url origin https://github.com/builderslab/{{PROJECT_SLUG}}.git
git push origin main
```

---

## Option B: Manual setup

If you prefer to customise manually, find and replace the following placeholders across all files:

| Placeholder | Replace with | Example |
|---|---|---|
| `{{PROJECT_NAME}}` | Full project name | `CardGuard` |
| `{{PROJECT_SLUG}}` | Lowercase repo name | `cardguard` |
| `{{PROJECT_DESCRIPTION}}` | One-sentence description | `Real-time credit card fraud detection for NorthBay Bank` |
| `{{COMPANY_NAME}}` | Fictional company name | `NorthBay Bank` |
| `{{PERSONA}}` | Primary user persona | `Fraud analyst` |
| `{{DATASET_NAME}}` | Dataset name | `Sparkov Credit Card Transactions` |
| `{{DATASET_URL}}` | Dataset URL | `https://kaggle.com/datasets/...` |
| `{{DURATION}}` | Project duration | `12 weeks` |
| `{{PRIMARY_METRIC}}` | Primary ML metric | `PR-AUC` |
| `{{TEAM_LEAD}}` | Project lead name or role | `Project Lead` |
| `{{TEAM_DS}}` | Data Science name or role | `Data Science` |
| `{{TEAM_MLE}}` | ML Engineer name or role | `ML Engineer` |

---

## After setup: files to customise per project

These files have the right structure but need project-specific content:

| File | What to add |
|---|---|
| `data/DATA_CARD.md` | Dataset schema, fraud/target rate, missing values, split strategy |
| `models/MODEL_CARD.md` | Model type, training data, metrics (fill in after Week 8) |
| `docs/decisions.md` | Dataset choice, metric choice, architecture decisions |
| `src/features.py` | All engineered features with domain rationale |
| `src/prompts.py` | Gemini prompts tailored to the project domain and persona |
| `notebooks/01_eda.ipynb` | EDA starter cells |
| `app/pages/` | Page content and UI copy |

---

## Removing Gemini (optional)

If this project does not use the Gemini API:

```bash
rm src/gemini.py
rm src/prompts.py
rm tests/test_gemini.py
```

Then remove the `google-generativeai` line from `pyproject.toml` and `app/requirements.txt`, and remove the Gemini import and feature sections from `app/pages/case_review.py` and `app/pages/summary.py`.

---

## Template maintenance

If you improve the template during a project (better CI, better PR template, new component), open a PR against `builderslab/builderslab-template` so all future projects benefit.
