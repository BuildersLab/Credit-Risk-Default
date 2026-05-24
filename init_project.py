"""
BuildersLab project initialisation script.

Run once after cloning the template to personalise all placeholder text.
Replaces {{PLACEHOLDERS}} across every file in the repo.

Usage:
    python init_project.py

Then delete this file and commit the result.
"""

import os
import re
from pathlib import Path

PLACEHOLDERS = [
    "{{PROJECT_NAME}}",
    "{{PROJECT_SLUG}}",
    "{{PROJECT_DESCRIPTION}}",
    "{{COMPANY_NAME}}",
    "{{PERSONA}}",
    "{{DATASET_NAME}}",
    "{{DATASET_URL}}",
    "{{TEAM_LEAD}}",
    "{{TEAM_DS}}",
    "{{TEAM_MLE}}",
    "{{DURATION}}",
    "{{PRIMARY_METRIC}}",
]

SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", "node_modules"}
SKIP_FILES = {"init_project.py"}
TEXT_EXTENSIONS = {
    ".py", ".md", ".txt", ".yaml", ".yml", ".toml",
    ".env", ".gitignore", ".cfg", ".ini", ".json", ".ipynb",
}


def prompt(label: str, default: str = "") -> str:
    hint = f" [{default}]" if default else ""
    val = input(f"{label}{hint}: ").strip()
    return val if val else default


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def collect_inputs() -> dict:
    print("\n" + "=" * 55)
    print("  BuildersLab Project Setup")
    print("=" * 55)
    print("Press Enter to accept the default shown in brackets.\n")

    project_name = prompt("Project name (e.g. CardGuard)")
    project_slug = prompt("Repo slug", slugify(project_name))
    project_description = prompt("One-sentence description")
    company_name = prompt("Fictional company name (e.g. NorthBay Bank)", "BuildersLab Co.")
    persona = prompt("Primary user persona (e.g. Fraud analyst)", "Data analyst")
    dataset_name = prompt("Dataset name")
    dataset_url = prompt("Dataset URL")
    duration = prompt("Project duration", "12 weeks")
    primary_metric = prompt("Primary ML metric", "PR-AUC")

    print("\nTeam members (role names or real names):")
    team_lead = prompt("  Project Lead")
    team_ds = prompt("  Data Science")
    team_mle = prompt("  ML Engineer")

    return {
        "{{PROJECT_NAME}}": project_name,
        "{{PROJECT_SLUG}}": project_slug,
        "{{PROJECT_DESCRIPTION}}": project_description,
        "{{COMPANY_NAME}}": company_name,
        "{{PERSONA}}": persona,
        "{{DATASET_NAME}}": dataset_name,
        "{{DATASET_URL}}": dataset_url,
        "{{DURATION}}": duration,
        "{{PRIMARY_METRIC}}": primary_metric,
        "{{TEAM_LEAD}}": team_lead,
        "{{TEAM_DS}}": team_ds,
        "{{TEAM_MLE}}": team_mle,
    }


def replace_in_file(path: Path, replacements: dict) -> bool:
    try:
        content = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, PermissionError):
        return False

    new_content = content
    for placeholder, value in replacements.items():
        new_content = new_content.replace(placeholder, value)

    if new_content != content:
        path.write_text(new_content, encoding="utf-8")
        return True
    return False


def rename_files(root: Path, replacements: dict) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        new_name = path.name
        for placeholder, value in replacements.items():
            new_name = new_name.replace(placeholder, value)
        if new_name != path.name:
            path.rename(path.parent / new_name)


def run():
    replacements = collect_inputs()
    root = Path(__file__).parent

    print("\nReplacing placeholders...")
    changed = 0
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name in SKIP_FILES:
            continue
        if path.suffix in TEXT_EXTENSIONS or path.name.startswith("."):
            if replace_in_file(path, replacements):
                print(f"  updated {path.relative_to(root)}")
                changed += 1

    rename_files(root, replacements)

    print(f"\n{changed} files updated.")
    print("\nNext steps:")
    print("  1. Review the changes with: git diff")
    print("  2. Delete this file: rm init_project.py")
    print("  3. Commit: git add -A && git commit -m 'initialise project from builderslab template'")
    print(f"  4. Push: git push origin main")
    print(f"\nYour project '{replacements['{{PROJECT_NAME}}']}' is ready.\n")


if __name__ == "__main__":
    run()
