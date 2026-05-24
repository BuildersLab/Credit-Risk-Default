# Workflow

Identical across all BuildersLab projects.

## Branching

Commit directly to `main`. Use a branch only if two people are working on conflicting changes at the same time.

## Commit messages

Format: `<verb>: <description>` — lowercase, present tense, imperative, one line, under 72 characters, no full stop at the end.

| Verb | Use for | Example |
|---|---|---|
| `add` | New file, feature, or capability that did not exist before | `add: add SHAP waterfall chart component` |
| `update` | Change to something that already exists | `update: update threshold from 0.5 to 0.6` |
| `fix` | Correcting a bug or broken behaviour | `fix: fix encoding leak in test set` |
| `remove` | Deleting a file, function, or capability | `remove: remove gemini integration` |
| `rename` | Renaming a file or identifier without logic changes | `rename: rename risk_badge to score_badge` |
| `refactor` | Restructuring code without changing behaviour | `refactor: refactor feature pipeline into separate functions` |
| `docs` | Documentation only, no code changes | `docs: add commit message conventions` |

## Data and model files

- Never commit data files (gitignored)
- Never commit model `.pkl` files (gitignored)
- Always commit `DATA_CARD.md` and `MODEL_CARD.md` when underlying data or model changes
