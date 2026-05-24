# Setup Guide: {{PROJECT_NAME}}

Follow these steps to go from a fresh clone to a running local environment.

## 1. Clone and install

```bash
git clone https://github.com/builderslab/{{PROJECT_SLUG}}.git
cd {{PROJECT_SLUG}}
make setup
```

If you want to set up the environment manually on Windows PowerShell, use:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r app/requirements.txt
```

## 2. Configure environment

```bash
cp .env.example .env
```

Open `.env` and fill in:
- `GEMINI_API_KEY`: get a free key from [aistudio.google.com](https://aistudio.google.com)
- `DATA_DIR`: path to the folder where you placed the raw data files

## 3. Download the dataset

Download {{DATASET_NAME}} from {{DATASET_URL}} and place files in `data/raw/`.

See `data/DATA_CARD.md` for the expected file names and schema.

## 4. Run the pipeline

```bash
make data      # process raw data into features
make train     # train the champion model
make evaluate  # print metrics report
```

## 5. Launch the app

```bash
make app
```

Open [localhost:8501](http://localhost:8501).

## 6. Run tests

```bash
make test
```

All tests should pass before opening a PR.

## Troubleshooting

- If `make setup` fails, ensure you are using Python 3.11 or higher
- If the model is not found, run `make train` before `make app`
- If Gemini features show errors, check your `GEMINI_API_KEY` in `.env`
