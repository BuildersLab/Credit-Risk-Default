from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.request import urlretrieve

import joblib
import pandas as pd
import streamlit as st

MODEL_REPO_ID = "BuildersLab/credit-risk-default-model"
LOCAL_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "final_model_bundle.pkl"
LOCAL_DEFAULTS_PATH = Path(__file__).resolve().parent.parent / "models" / "feature_defaults.json"


def download_model_file(filename: str) -> Path:
    cache_dir = Path("/tmp/northbay-model-cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / filename
    if not destination.exists():
        url = f"https://huggingface.co/{MODEL_REPO_ID}/resolve/main/{filename}"
        urlretrieve(url, destination)
    return destination


@st.cache_resource
def load_model_bundle() -> dict:
    path = LOCAL_MODEL_PATH if LOCAL_MODEL_PATH.exists() else Path(
        download_model_file("final_model_bundle.pkl")
    )
    return joblib.load(path)


@st.cache_resource
def load_feature_defaults() -> dict:
    path = LOCAL_DEFAULTS_PATH if LOCAL_DEFAULTS_PATH.exists() else Path(
        download_model_file("feature_defaults.json")
    )
    with open(path) as f:
        return json.load(f)


def score_loan(bundle: dict, raw_inputs: dict) -> float:
    """Returns the predicted default probability as a percentage (0-100)."""
    row = pd.DataFrame([raw_inputs])[bundle["raw_feature_columns"]]
    encoded = bundle["preprocessor"].transform(row)
    proba = float(bundle["model"].predict_proba(encoded)[:, 1][0])
    return proba * 100


FIELD_LABELS: dict[str, str] = {
    "int_rate": "Interest Rate (%)",
    "term_months": "Loan Term (months)",
    "dti": "Debt-to-Income Ratio (%)",
    "acc_open_past_24mths": "Accounts Opened in Last 24 Months",
    "fico_score": "FICO Credit Score",
    "loan_to_income": "Loan Amount ÷ Annual Income",
    "mo_sin_old_rev_tl_op": "Age of Oldest Revolving Account (months)",
    "installment_to_income": "Monthly Payment ÷ Annual Income",
    "total_bc_limit": "Total Bankcard Credit Limit ($)",
    "loan_amnt": "Loan Amount ($)",
    "emp_length_missing": "Employment Length Not Reported",
    "mort_acc": "Number of Mortgage Accounts",
    "mths_since_recent_inq": "Months Since Last Credit Inquiry",
    "mths_since_recent_bc": "Months Since Newest Bankcard Account",
    "bc_util": "Bankcard Utilization (%)",
    "percent_bc_gt_75": "Bankcards Over 75% Utilized (%)",
    "num_actv_rev_tl": "Active Revolving Accounts",
    "all_util_missing": "Credit Utilization Data Not Reported",
    "total_il_high_credit_limit": "Total Installment Credit Limit ($)",
    "total_cu_tl_missing": "Joint Credit Data Not Reported",
    "credit_history_months": "Length of Credit History (months)",
    "annual_inc": "Annual Income ($)",
    "revol_util": "Revolving Credit Utilization (%)",
    "tot_hi_cred_lim": "Total Credit Limit, All Accounts ($)",
    "avg_cur_bal": "Average Balance Across Accounts ($)",
    "total_rev_hi_lim": "Total Revolving Credit Limit ($)",
    "num_il_tl": "Number of Installment Accounts",
    "total_acc": "Total Credit Accounts Ever Opened",
    "revol_bal": "Revolving Credit Balance ($)",
    "mths_since_last_delinq": "Months Since Last Delinquency",
    "pct_tl_nvr_dlq": "Accounts Never Delinquent (%)",
    "mo_sin_rcnt_rev_tl_op": "Months Since Newest Revolving Account",
    "mo_sin_rcnt_tl": "Months Since Most Recent Account Opened",
    "inq_last_6mths": "Credit Inquiries in Last 6 Months",
    "num_tl_op_past_12m": "Accounts Opened in Last 12 Months",
    "num_rev_tl_bal_gt_0": "Revolving Accounts With a Balance",
    "num_rev_accts": "Number of Revolving Accounts",
    "emp_length_years": "Employment Length (years)",
    "bc_open_to_buy": "Available Bankcard Credit ($)",
    "mo_sin_old_il_acct": "Age of Oldest Installment Account (months)",
    "num_tl_120dpd_2m_missing": "Severe Delinquency Data Not Reported",
    "installment": "Monthly Payment ($)",
    "open_rv_24m": "Revolving Accounts Opened in Last 24 Months",
    "num_actv_bc_tl": "Active Bankcard Accounts",
    "tot_cur_bal": "Total Current Balance, All Accounts ($)",
    "addr_state": "State",
    "home_ownership": "Home Ownership",
    "purpose": "Loan Purpose",
    "verification_status": "Income Verification Status",
}

# Fields that are really yes/no flags, not numbers.
BOOLEAN_FIELDS = {
    "emp_length_missing",
    "all_util_missing",
    "total_cu_tl_missing",
    "num_tl_120dpd_2m_missing",
}

# Fields that only make sense as whole numbers (counts, months).
INTEGER_FIELDS = {
    "acc_open_past_24mths", "mort_acc", "num_actv_rev_tl", "num_il_tl",
    "num_rev_tl_bal_gt_0", "num_rev_accts", "mo_sin_old_rev_tl_op",
    "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mo_sin_old_il_acct",
    "mths_since_recent_inq", "mths_since_recent_bc", "mths_since_last_delinq",
    "inq_last_6mths", "num_tl_op_past_12m", "num_actv_bc_tl", "total_acc",
    "emp_length_years", "open_rv_24m", "credit_history_months", "fico_score",
}

PERCENT_FIELDS = {"int_rate", "dti", "bc_util", "percent_bc_gt_75", "revol_util", "pct_tl_nvr_dlq"}

# The only two loan terms LendingClub actually offers, not a free continuous value.
TERM_FIELD = "term_months"
TERM_CHOICES = [36, 60]

# Commonly-known, commonly-changed fields shown up front. Everything else in
# numeric_columns is tucked into an "Additional details" expander, pre-filled
# with the training data's typical (median) value.
PRIMARY_NUMERIC_FIELDS = [
    "loan_amnt", "term_months", "int_rate", "annual_inc", "dti", "fico_score",
    "emp_length_years", "installment", "mort_acc", "acc_open_past_24mths",
]


def render_numeric_field(col: str, meta: dict, container) -> float:
    label = FIELD_LABELS.get(col, col.replace("_", " ").title())
    help_text = meta.get("description", "")
    default = meta.get("default", 0.0)
    lo = meta.get("min", 0.0)
    hi = meta.get("max", max(default, lo + 1))

    if col in BOOLEAN_FIELDS:
        value = container.checkbox(label, value=bool(default), help=help_text, key=f"num_{col}")
        return float(value)

    if col == TERM_FIELD:
        idx = TERM_CHOICES.index(int(default)) if int(default) in TERM_CHOICES else 0
        value = container.selectbox(label, TERM_CHOICES, index=idx, help=help_text, key=f"num_{col}")
        return float(value)

    if col in INTEGER_FIELDS:
        value = container.number_input(
            label,
            min_value=math.floor(lo),
            max_value=math.ceil(hi),
            value=int(round(default)),
            step=1,
            help=help_text,
            key=f"num_{col}",
        )
        return float(value)

    if col in PERCENT_FIELDS:
        value = container.number_input(
            label,
            min_value=float(lo),
            max_value=float(hi),
            value=float(default),
            step=0.1,
            format="%.1f",
            help=help_text,
            key=f"num_{col}",
        )
        return float(value)

    value = container.number_input(
        label,
        min_value=float(lo),
        max_value=float(hi),
        value=float(default),
        help=help_text,
        key=f"num_{col}",
    )
    return float(value)

@dataclass(frozen=True)
class Loan:
    loan_id: str
    borrower: str
    grade: str
    purpose: str
    balance: int
    amount: int
    pd: float
    tier: str
    top_driver: str
    status: str
    annual_income: int
    dti: float
    credit_score: int
    employment_years: int


LOANS = [
    Loan("LC-882041", "Marcus Chen", "B", "Debt consolidation", 18_420, 45_000, 34.7, "High", "Recent missed payments", "Queued", 92_000, 31.8, 687, 6),
    Loan("LC-881772", "Alina Patel", "C", "Home improvement", 31_850, 36_000, 19.2, "Medium", "Debt-to-income pressure", "In review", 78_500, 28.4, 712, 4),
    Loan("LC-879334", "Jordan Williams", "A", "Major purchase", 12_600, 15_000, 7.4, "Low", "Stable employment", "Monitored", 118_000, 18.6, 756, 9),
    Loan("LC-878921", "Sofia Martinez", "C", "Medical expenses", 27_940, 32_000, 28.6, "High", "Credit utilization", "Escalated", 64_000, 35.2, 661, 2),
    Loan("LC-877614", "Ethan Brooks", "B", "Small business", 41_200, 50_000, 22.4, "Medium", "Short credit history", "Queued", 105_000, 26.9, 701, 5),
    Loan("LC-876309", "Maya Thompson", "A", "Home improvement", 9_875, 12_000, 5.1, "Low", "Low revolving utilization", "Monitored", 134_000, 14.1, 781, 11),
    Loan("LC-875288", "Noah Kim", "D", "Debt consolidation", 38_560, 42_000, 41.3, "High", "Payment history", "Queued", 71_000, 39.6, 638, 1),
    Loan("LC-874103", "Avery Johnson", "B", "Auto", 21_400, 28_000, 14.8, "Medium", "Loan-to-income ratio", "In review", 84_000, 24.7, 719, 7),
]


TEAM = [
    ("Nafisat Ibrahim", "Data Scientist & Project Lead"),
    ("Marienne Dosso", "Data Scientist"),
    ("Bintou Ba", "Data Scientist · Secondary Project Lead"),
]


st.set_page_config(
    page_title="NorthBay Portfolio Risk",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAV_PAGES = ["Home", "Model results", "Score a loan", "Loan summary"]


def apply_visual_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --nb-primary: #F05A24;
            --nb-primary-glow: rgba(240, 90, 36, 0.35);
            --nb-surface: #101217;
            --nb-surface-raised: #1A1D24;
            --nb-border: rgba(255, 255, 255, 0.08);
            --nb-text: #E4E6EB;
            --nb-text-muted: #A0A5B1;
        }

        .stApp {
            background-image: 
                radial-gradient(ellipse at 80% 0%, rgba(240, 90, 36, 0.08), transparent 40%),
                radial-gradient(ellipse at 20% 100%, rgba(240, 90, 36, 0.05), transparent 40%);
        }

        [data-testid="stHeader"] {
            display: none;
        }
        
        .block-container {
            padding-top: 3rem;
            max-width: 72rem;
        }

        h1, h2, h3, h4 {
            color: #FFFFFF !important;
            letter-spacing: -0.02em;
        }
        
        h1 {
            font-size: clamp(2rem, 4vw, 3rem) !important;
            font-weight: 700 !important;
            margin-bottom: 1.5rem !important;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid var(--nb-border);
        }
        [data-testid="stSidebar"] hr {
            border-color: var(--nb-border);
        }
        [data-testid="stSidebar"] [role="radiogroup"] label {
            padding: 0.5rem 0.75rem;
            border-radius: 6px;
            transition: background 0.2s ease, transform 0.2s ease;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: rgba(255, 255, 255, 0.03);
            transform: translateX(2px);
        }

        .nb-hero {
            position: relative;
            overflow: hidden;
            padding: clamp(2rem, 5vw, 4rem);
            margin: 0.5rem 0 2rem;
            border-radius: 12px;
            background: linear-gradient(135deg, #13151A 0%, #1A1210 100%);
            border: 1px solid var(--nb-border);
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }

        .nb-hero::after {
            content: "";
            position: absolute;
            top: 0; right: 0; bottom: 0; left: 0;
            background: radial-gradient(circle at 100% 100%, rgba(240, 90, 36, 0.15), transparent 60%);
            pointer-events: none;
        }

        .nb-kicker {
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.15em;
            color: var(--nb-primary);
            text-transform: uppercase;
            margin-bottom: 0.5rem;
        }

        .nb-hero-title {
            position: relative;
            z-index: 1;
            font-size: clamp(2.2rem, 4vw, 3.5rem);
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 1rem;
            color: #FFFFFF;
            letter-spacing: -0.03em;
        }

        .nb-hero-copy {
            position: relative;
            z-index: 1;
            max-width: 48rem;
            color: var(--nb-text-muted);
            font-size: 1.1rem;
            line-height: 1.6;
        }

        .nb-metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0 2rem;
        }

        .nb-metric-card {
            padding: 1.25rem;
            background: var(--nb-surface);
            border: 1px solid var(--nb-border);
            border-radius: 8px;
            position: relative;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            container-type: inline-size;
        }

        .nb-metric-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; width: 100%; height: 2px;
            background: var(--nb-primary);
            opacity: 0.8;
        }

        .nb-metric-label {
            color: var(--nb-text-muted);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
            font-weight: 600;
        }

        .nb-metric-value {
            color: #FFFFFF;
            font-size: clamp(1.5rem, 15cqi, 2rem);
            font-weight: 700;
            letter-spacing: -0.02em;
            white-space: normal !important;
            overflow-wrap: anywhere;
            word-break: normal;
        }

        [data-testid="stButton"] button {
            border-radius: 6px;
            transition: all 0.2s ease;
            font-weight: 500;
            min-height: 2.75rem;
            height: auto;
            white-space: normal;
        }

        [data-testid="stButton"] button p {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            line-height: 1.25;
        }
        
        [data-testid="stButton"] button[kind="primary"] {
            box-shadow: 0 4px 12px var(--nb-primary-glow);
            font-weight: 600;
        }
        
        [data-testid="stButton"] button[kind="primary"]:hover {
            box-shadow: 0 6px 16px var(--nb-primary-glow);
            transform: translateY(-1px);
        }

        .nb-section-label {
            display: inline-block;
            margin: 1.5rem 0 0.5rem;
            padding: 0.25rem 0.75rem;
            color: #FFFFFF;
            background: rgba(240, 90, 36, 0.15);
            border: 1px solid rgba(240, 90, 36, 0.3);
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }

        .nb-app-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: .75rem 1.5rem;
            flex-wrap: wrap;
            margin-bottom: 1rem;
            color: var(--nb-text-muted);
            font-size: .72rem;
            font-weight: 650;
            letter-spacing: .075em;
            text-transform: uppercase;
        }

        .nb-app-header span:last-child {
            color: rgba(240, 90, 36, .82);
            text-align: right;
        }

        /* Native metric styling */
        [data-testid="stMetric"] {
            background: var(--nb-surface);
            border: 1px solid var(--nb-border);
            padding: 1rem 1.25rem;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            border-top: 2px solid var(--nb-primary);
            min-width: 0;
        }
        [data-testid="stMetricLabel"] {
            color: var(--nb-text-muted) !important;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-size: 0.75rem;
            white-space: normal !important;
            overflow-wrap: anywhere;
        }
        [data-testid="stMetricValue"] {
            color: #FFFFFF !important;
            font-weight: 700;
            letter-spacing: -0.02em;
            white-space: normal !important;
            overflow-wrap: anywhere;
            word-break: normal;
        }

        [data-testid="stProgress"] > div > div > div {
            background: linear-gradient(90deg, #c83b0d, #F05A24, #FFB067);
        }
        
        /* Tier Badges */
        .tier-badge {
            display: inline-block;
            padding: 0.35rem 0.85rem;
            border-radius: 6px;
            font-weight: 700;
            font-size: 0.85rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
        .tier-low {
            background-color: rgba(46, 160, 67, 0.15);
            color: #4ade80;
            border: 1px solid rgba(46, 160, 67, 0.3);
        }
        .tier-medium {
            background-color: rgba(240, 150, 36, 0.15);
            color: #fbbf24;
            border: 1px solid rgba(240, 150, 36, 0.3);
        }
        .tier-high {
            background-color: rgba(220, 53, 69, 0.15);
            color: #f87171;
            border: 1px solid rgba(220, 53, 69, 0.3);
        }

        /* Clean Alerts */
        [data-testid="stAlert"] {
            border-radius: 6px;
            border: 1px solid var(--nb-border);
            background-color: var(--nb-surface);
            color: var(--nb-text);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        
        [data-testid="stAlert"] [data-testid="stMarkdownContainer"] {
            color: var(--nb-text);
        }

        .nb-placeholder-pulse {
            height: 120px;
            border-radius: 6px;
            background: linear-gradient(90deg, var(--nb-surface) 0%, rgba(255, 255, 255, 0.03) 50%, var(--nb-surface) 100%);
            background-size: 200% 100%;
            animation: pulse 2s infinite ease-in-out;
            margin: 1rem 0;
            border: 1px dashed var(--nb-border);
        }

        @keyframes pulse {
            0% { background-position: 100% 0; }
            100% { background-position: -100% 0; }
        }

        [data-testid="stExpander"],
        [data-testid="stForm"],
        div[data-testid="stVerticalBlock"] > div > div > div[style*="border"] {
            border: 1px solid var(--nb-border) !important;
            border-radius: 8px !important;
            background-color: var(--nb-surface) !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    defaults = {
        "page": "Home",
        "selected_loan_id": LOANS[0].loan_id,
        "low_threshold": 10,
        "high_threshold": 25,
        "actions": [],
        "last_scored": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state.page not in NAV_PAGES:
        st.session_state.page = "Home"


def navigate(page: str, loan_id: str | None = None) -> None:
    st.session_state.page = page
    if loan_id:
        st.session_state.selected_loan_id = loan_id
    st.rerun()


def app_header(section: str) -> None:
    st.markdown(
        (
            '<div class="nb-app-header">'
            f"<span>NORTHBAY BANK / {section.upper()}</span>"
            "<span>BUILDERSLAB TEAM 03 · TRAINING USE</span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_metric_cards(items: list[tuple[str, str]]) -> None:
    cards = "".join(
        (
            '<div class="nb-metric-card">'
            f'<div class="nb-metric-label">{label}</div>'
            f'<div class="nb-metric-value">{value}</div>'
            "</div>"
        )
        for value, label in items
    )
    st.markdown(
        f'<div class="nb-metric-grid">{cards}</div>',
        unsafe_allow_html=True,
    )


def risk_tier(probability: float) -> str:
    if probability < st.session_state.low_threshold:
        return "Low"
    if probability < st.session_state.high_threshold:
        return "Medium"
    return "High"


def tier_icon(tier: str) -> str:
    return {"Low": "Low Risk", "Medium": "Medium Risk", "High": "High Risk"}[tier]


def _tier_cell_style(value: str) -> str:
    if "High" in value:
        return "background-color: rgba(220,53,69,.15); color:#f87171; font-weight:700; border-radius:4px;"
    if "Medium" in value:
        return "background-color: rgba(240,150,36,.15); color:#fbbf24; font-weight:700; border-radius:4px;"
    return "background-color: rgba(46,160,67,.15); color:#4ade80; font-weight:700; border-radius:4px;"


def find_loan(loan_id: str) -> Loan:
    return next((loan for loan in LOANS if loan.loan_id == loan_id), LOANS[0])


def most_recent_quarter_end() -> str:
    today = datetime.now(timezone.utc).date()
    quarter = (today.month - 1) // 3
    if quarter == 0:
        year, end_month = today.year - 1, 12
    else:
        year, end_month = today.year, quarter * 3
    last_day = (
        date(year, 12, 31)
        if end_month == 12
        else date(year, end_month + 1, 1) - timedelta(days=1)
    )
    return last_day.strftime("%d %b %Y")


def render_sidebar() -> None:
    with st.sidebar:
        st.title("NorthBay Bank")
        st.caption("PORTFOLIO RISK DESK")
        pages = NAV_PAGES
        selected = st.radio(
            "Navigate",
            pages,
            index=pages.index(st.session_state.page),
        )
        if selected != st.session_state.page:
            st.session_state.page = selected
            st.rerun()



def render_home() -> None:
    app_header("Client project")
    st.markdown(
        """
        <section class="nb-hero">
            <div class="nb-kicker">NORTHBAY BANK · PORTFOLIO INTELLIGENCE</div>
            <div class="nb-hero-title">Loan Default Prediction.</div>
            <p class="nb-hero-copy">
                A model-connected decision-support prototype that estimates the
                probability of loan default from information available at origination.
                It helps analysts understand risk; it does not automate approval.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    primary, secondary = st.columns(2)
    with primary:
        if st.button("View model results", type="primary"):
            navigate("Model results")
    with secondary:
        if st.button("Score an individual loan"):
            navigate("Score a loan")

    st.divider()
    st.markdown('<span class="nb-section-label">MODEL PROFILE</span>', unsafe_allow_html=True)
    st.header("What is running behind the interface")
    facts = [
        ("XGBoost", "Model family"),
        ("49", "Selected grouped features"),
        ("0–100%", "Probability output"),
        ("HF Hub", "Model deployment"),
    ]
    render_metric_cards(facts)

    st.divider()
    st.markdown('<span class="nb-section-label">PROJECT CONTEXT</span>', unsafe_allow_html=True)
    st.header("Why this project exists")
    first, second = st.columns(2)
    with first:
        st.subheader("The business problem")
        st.write(
            "Traditional grades compress risk into broad categories. Loans in the "
            "same grade can still have materially different estimated default "
            "probabilities, making prioritization and portfolio review harder."
        )
    with second:
        st.subheader("The model output")
        st.write(
            "The trained model returns a continuous probability of default. "
            "Configurable business thresholds translate that probability into "
            "Low, Medium, or High monitoring tiers for human review."
        )

    st.markdown('<span class="nb-section-label">MODEL EVALUATION</span>', unsafe_allow_html=True)
    st.header("Performance metrics")
    st.caption("Values will be added when the final evaluation report is uploaded. No placeholder values are presented as model facts.")
    render_metric_cards(
        [
            ("Pending", "ROC-AUC"),
            ("Pending", "PR-AUC"),
            ("Pending", "Recall at threshold"),
            ("Pending", "Brier score"),
        ]
    )

    st.markdown('<span class="nb-section-label">DECISION FLOW</span>', unsafe_allow_html=True)
    st.header("How a loan moves through the prototype")
    flow = st.columns(4)
    steps = [
        ("01 · Enter", "Provide origination-time borrower and loan information."),
        ("02 · Score", "The deployed XGBoost model estimates default probability."),
        ("03 · Interpret", "Review the tier, thresholds, inputs, and model drivers."),
        ("04 · Decide", "A human records the appropriate monitoring response."),
    ]
    for column, (title, body) in zip(flow, steps):
        with column:
            with st.container(border=True):
                st.subheader(title)
                st.write(body)
    st.warning(
        "The deterministic ML model is the auditable system of record. "
        "Any future language-model explanation will be assistive only."
    )

    st.markdown('<span class="nb-section-label">THE TEAM</span>', unsafe_allow_html=True)
    st.header("BuildersLab Team 03")
    team_columns = st.columns(3)
    for column, (name, role) in zip(team_columns, TEAM):
        with column:
            with st.container(border=True):
                st.subheader(name)
                st.caption(role.upper())

    st.header("Project contact")
    contact_columns = st.columns(3)
    contact_columns[0].link_button(
        "GitHub repository", "https://github.com/BuildersLab/Credit-Risk-Default"
    )
    contact_columns[1].link_button(
        "LinkedIn", "https://www.linkedin.com/company/builderslabdev"
    )
    contact_columns[2].link_button(
        "Email the team", "mailto:contact@builderslab.dev"
    )
    st.caption(
        "NorthBay Bank, its portfolio, people, and events are fictional and "
        "were created for a BuildersLab training exercise."
    )


def render_threshold_settings() -> None:
    with st.expander("Draft risk-tier settings", expanded=False):
        st.caption(
            "These are configurable business boundaries, not objective model facts."
        )
        low, high = st.columns(2)
        low_value = low.slider(
            "Low to Medium boundary",
            min_value=1,
            max_value=30,
            value=st.session_state.low_threshold,
            format="%d%%",
        )
        high_value = high.slider(
            "Medium to High boundary",
            min_value=10,
            max_value=60,
            value=st.session_state.high_threshold,
            format="%d%%",
        )
        if low_value >= high_value:
            st.error("The High-risk boundary must be above the Low-risk boundary.")
        elif st.button("Apply draft boundaries"):
            st.session_state.low_threshold = low_value
            st.session_state.high_threshold = high_value
            st.success("Draft portfolio tiers updated.")
            st.rerun()


def render_portfolio() -> None:
    app_header("Model results / 01")
    st.title("Model evaluation results.")
    st.write(
        "This page is reserved for the final charts, tables, and interpretation "
        "from the model evaluation package you will upload."
    )
    st.info(
        "Placeholders are intentional. They avoid presenting invented values as "
        "validated model performance."
    )

    first, second = st.columns(2)
    placeholders = [
        (first, "Discrimination", "ROC curve and ROC-AUC", "How well the model ranks defaults above non-defaults."),
        (second, "Precision and recall", "PR curve and PR-AUC", "Performance when the default class is less common."),
        (first, "Threshold performance", "Confusion matrix", "Precision, recall, specificity, and error counts at the chosen threshold."),
        (second, "Probability quality", "Calibration curve", "Whether predicted probabilities align with observed default frequency."),
        (first, "Model understanding", "Feature importance", "Global SHAP importance and grouped feature contributions."),
        (second, "Robustness", "Segment results", "Performance by grade, purpose, amount band, or other approved segments."),
    ]
    for column, eyebrow, title, description in placeholders:
        with column:
            with st.container(border=True):
                st.caption(eyebrow.upper())
                st.subheader(title)
                st.write(description)
                st.markdown('<div class="nb-placeholder-pulse"></div>', unsafe_allow_html=True)
                st.caption("AWAITING UPLOADED RESULT")

    st.divider()
    st.subheader("What to upload")
    st.write(
        "Recommended: evaluation metric table, ROC and precision-recall curves, "
        "confusion matrix, calibration plot, SHAP summary, and any approved "
        "segment-level validation."
    )
    if st.button("Continue to loan scoring", type="primary"):
        navigate("Score a loan")


def render_score() -> None:
    app_header("Origination scoring / 02")
    st.title("Score an individual loan.")
    st.write(
        "Use borrower and loan characteristics available at issuance. This "
        "runs the actual trained model, XGBoost, calibrated, 49 selected features."
    )
    st.info(
        "Origination-only policy: no payment performance, collections activity, "
        "or other post-issuance information is used."
    )

    try:
        bundle = load_model_bundle()
        defaults = load_feature_defaults()
    except Exception as exc:
        st.error(
            f"The trained model isn't available right now ({exc}). "
            "Try again shortly, or contact the team if this persists."
        )
        st.stop()

    raw_inputs: dict = {}

    with st.form("origination_scoring"):
        name = st.text_input("Borrower name", value="New applicant")

        st.subheader("Loan and borrower details")
        st.caption("The details you'd typically know or decide on for this loan.")
        primary_widgets = st.columns(3)
        for i, col in enumerate(PRIMARY_NUMERIC_FIELDS):
            meta = defaults.get(col, {})
            raw_inputs[col] = render_numeric_field(col, meta, primary_widgets[i % 3])

        cat_transformer = bundle["preprocessor"].named_transformers_["cat"]
        cat_options = dict(zip(bundle["categorical_columns"], cat_transformer.categories_))
        cat_widgets = st.columns(3)
        for i, col in enumerate(bundle["categorical_columns"]):
            meta = defaults.get(col, {})
            options = list(cat_options[col])
            label = FIELD_LABELS.get(col, col.replace("_", " ").title())
            default_value = meta.get("default")
            default_index = options.index(default_value) if default_value in options else 0
            with cat_widgets[i % 3]:
                raw_inputs[col] = st.selectbox(
                    label,
                    options,
                    index=default_index,
                    help=meta.get("description", ""),
                    key=f"cat_{col}",
                )

        secondary_cols = [c for c in bundle["numeric_columns"] if c not in PRIMARY_NUMERIC_FIELDS]
        with st.expander(f"Additional credit history details ({len(secondary_cols)} fields, defaults shown)"):
            st.caption("Pre-filled with typical (median) values from the training data. Edit only what you know.")
            secondary_widgets = st.columns(3)
            for i, col in enumerate(secondary_cols):
                meta = defaults.get(col, {})
                raw_inputs[col] = render_numeric_field(col, meta, secondary_widgets[i % 3])

        submitted = st.form_submit_button("Run default prediction", type="primary")

    if submitted and not name.strip():
        st.error("Enter a borrower name before scoring.")
    elif submitted:
        probability_pct = score_loan(bundle, raw_inputs)
        st.session_state.last_scored = {
            "borrower": name,
            "amount": int(raw_inputs.get("loan_amnt", 0)),
            "purpose": raw_inputs.get("purpose", ""),
            "pd": round(probability_pct, 1),
        }
        st.session_state.selected_loan_id = LOANS[0].loan_id

    if st.session_state.last_scored:
        result = st.session_state.last_scored
        st.success(
            f"Scored: {result['pd']:.1f}% PD · "
            f"{risk_tier(result['pd'])} monitoring tier."
        )
        if st.button("Review scored result", type="primary"):
            navigate("Loan summary")


def render_shap_drivers(bundle: dict) -> None:
    st.subheader("Top global risk drivers")
    st.caption("MODEL-WIDE SHAP IMPORTANCE · not specific to this loan")
    
    top_features = bundle["feature_list"][:5]
    if not top_features:
        return
        
    max_shap = max(f['mean_abs_shap'] for f in top_features)
    
    for feature in top_features:
        left, right = st.columns([4, 1])
        with left:
            st.write(f"**{feature['Variable']}**")
            st.caption(feature.get("Description", ""))
        with right:
            shap_value = feature['mean_abs_shap']
            pct = min((shap_value / max_shap) * 100, 100) if max_shap > 0 else 0
            st.markdown(
                f'<div style="text-align: right;">'
                f'<div style="font-weight: 700; font-size: 1.25rem; color: #FFFFFF; letter-spacing: -0.02em;">{shap_value:.3f}</div>'
                f'<div style="height: 4px; background: rgba(255,255,255,0.1); border-radius: 2px; margin-top: 6px; overflow: hidden; display: flex; justify-content: flex-end;">'
                f'<div style="width: {pct}%; height: 100%; background: var(--nb-primary); border-radius: 2px;"></div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        st.divider()


def render_loan_review() -> None:
    simulation = st.session_state.last_scored
    if not simulation:
        app_header("Loan summary / 03")
        st.title("Awaiting loan selection.")
        with st.container(border=True):
            st.markdown("### No loan has been scored in this session")
            st.write("Run a loan profile through the model to generate a risk assessment, decision explanation, and monitoring actions.")
            st.write("")
            if st.button("Open scoring interface", type="primary"):
                navigate("Score a loan")
        return

    borrower = simulation["borrower"]
    amount = simulation["amount"]
    purpose = simulation["purpose"]
    probability = simulation["pd"]
    tier = risk_tier(probability)

    app_header("Loan summary / 03")
    st.title(f"Decision summary · {borrower}")
    st.caption(
        f"${amount:,.0f} requested · {purpose or 'Purpose not supplied'} · "
        "scored from origination-time information"
    )
    if st.button("Score another loan"):
        st.session_state.last_scored = None
        navigate("Score a loan")

    try:
        bundle = load_model_bundle()
    except Exception as exc:
        bundle = None
        st.warning(f"Global driver data unavailable right now ({exc}).")

    render_threshold_settings()
    score_col, driver_col = st.columns([1, 1.35])
    with score_col:
        st.subheader("Probability and tier")
        st.metric("Default probability", f"{probability:.1f}%")
        st.progress(min(probability / 100, 1.0), text=f"{probability:.1f}% estimated probability")
        
        tier_class = tier.lower()
        st.markdown(f'<div style="margin: 1rem 0;"><span class="tier-badge tier-{tier_class}">{tier.upper()} RISK</span></div>', unsafe_allow_html=True)
        
        st.write(
            f"**Draft business boundary:** High ≥ "
            f"{st.session_state.high_threshold:.1f}%"
        )
        st.caption(
            f"Low: below {st.session_state.low_threshold}% · Medium: "
            f"{st.session_state.low_threshold}% to below {st.session_state.high_threshold}% · "
            f"High: {st.session_state.high_threshold}% or above."
        )
        st.write("**Model:** XGBoost · 49 selected grouped features")
        st.write("**Source:** BuildersLab model deployed on Hugging Face")
        st.write(
            f"**Scored:** {datetime.now(timezone.utc).strftime('%d %b %Y · %H:%M UTC')}"
        )
    with driver_col:
        if bundle is not None:
            render_shap_drivers(bundle)

    st.divider()
    st.markdown('<span class="nb-section-label">INTERPRETATION</span>', unsafe_allow_html=True)
    with st.container(border=True):
        meaning, effect = st.columns(2)
        with meaning:
            st.subheader("What the probability means")
            st.write(
                f"The model estimates a **{probability:.1f}% probability of default** "
                "for applications with this submitted feature profile, according to "
                "the outcome definition and time window used during model training."
            )
            st.caption(
                "This is a model estimate, not a certainty and not a causal statement."
            )
        with effect:
            st.subheader("How it affects review")
            if tier == "High":
                st.warning("Prioritize human review and consider enhanced monitoring.")
            elif tier == "Medium":
                st.info("Apply standard review with additional attention to risk factors.")
            else:
                st.success("Continue routine review; low risk does not mean zero risk.")
            st.caption(
                "The tier supports monitoring and prioritization. It does not approve "
                "or deny the loan."
            )

    st.markdown('<span class="nb-section-label">MONITORING</span>', unsafe_allow_html=True)
    explanation, actions = st.columns([1.25, 1])
    with explanation:
        st.subheader("Decision explanation")
        st.info("Loan-specific SHAP values are not connected yet.")
        st.write(
            "The factors shown above are global model drivers. A future "
            "loan-specific explanation should show which submitted features "
            "pushed this prediction higher or lower, without changing the score."
        )
        st.caption(
            "The deterministic model remains the system of record. Gemini is "
            "assistive only and will not make or alter the risk decision."
        )

    with actions:
        st.subheader("Record a monitoring decision")
        choices = [
            "Prioritize review",
            "Adjust provisioning",
            "Closer monitoring",
            "Escalate to Head of Risk",
        ]
        for choice in choices:
            if st.button(choice, key=f"action-{choice}", width="stretch"):
                st.session_state.actions.insert(
                    0,
                    {
                        "Time": datetime.now(timezone.utc).strftime("%H:%M UTC"),
                        "Loan": borrower,
                        "Action": choice,
                    },
                )
                st.success(f"Logged: {choice}")

    st.subheader("Audit trail")
    if st.session_state.actions:
        st.dataframe(
            st.session_state.actions,
            width="stretch",
            hide_index=True,
        )
    else:
        st.caption("No monitoring actions have been recorded in this session.")


initialize_state()
apply_visual_theme()
render_sidebar()

if st.session_state.page == "Home":
    render_home()
elif st.session_state.page == "Model results":
    render_portfolio()
elif st.session_state.page == "Score a loan":
    render_score()
else:
    render_loan_review()