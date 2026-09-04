from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import streamlit as st

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


SHAP_DRIVERS = [
    ("Recent missed payments", 12.4, "Increases PD"),
    ("Debt-to-income ratio", 8.1, "Increases PD"),
    ("Credit utilization", 5.7, "Increases PD"),
    ("Employment tenure", -4.2, "Reduces PD"),
    ("Credit score", -2.6, "Reduces PD"),
]


st.set_page_config(
    page_title="NorthBay Portfolio Risk",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAV_ICONS = {
    "Home": "🏠",
    "Portfolio": "📊",
    "Score a loan": "🧮",
    "Loan review": "🔎",
}


def apply_visual_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --nb-orange: #f05a24;
            --nb-orange-dark: #c83b0d;
            --nb-orange-soft: #ffb067;
            --nb-amber: #ffc857;
            --nb-ink: #15171c;
            --nb-slate: #5c6472;
        }

        .stApp {
            background:
                radial-gradient(circle at 88% 4%, rgba(255, 176, 103, .22), transparent 24rem),
                radial-gradient(circle at 12% 96%, rgba(240, 90, 36, .10), transparent 28rem),
                linear-gradient(135deg, #fffdf9 0%, #f7f8fb 48%, #fff6ed 100%);
        }

        [data-testid="stSidebar"] {
            background:
                radial-gradient(circle at 30% 5%, rgba(240, 90, 36, .42), transparent 15rem),
                linear-gradient(165deg, #242832 0%, #15171c 58%, #31180f 100%);
            border-right: 1px solid rgba(255, 176, 103, .28);
        }

        [data-testid="stSidebar"] * {
            color: #f8f4ef;
        }

        [data-testid="stSidebar"] hr {
            border-color: rgba(255, 255, 255, .16);
        }

        [data-testid="stSidebar"] [data-testid="stAlert"] {
            background: linear-gradient(135deg, rgba(240, 90, 36, .23), rgba(255, 200, 87, .12));
            border: 1px solid rgba(255, 176, 103, .38);
        }

        [data-testid="stSidebar"] [role="radiogroup"] label {
            padding: .38rem .5rem;
            border-radius: .55rem;
            transition: background .2s ease, transform .2s ease;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: rgba(255, 255, 255, .08);
            transform: translateX(3px);
        }

        h1 {
            letter-spacing: -.045em;
            background: linear-gradient(100deg, #15171c 8%, #d44311 58%, #ff8b3d 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        h2, h3 {
            letter-spacing: -.025em;
        }

        [data-testid="stMetric"] {
            min-height: 8rem;
            padding: 1.25rem 1.3rem;
            border: 1px solid rgba(240, 90, 36, .22);
            border-top: 4px solid var(--nb-orange);
            border-radius: 1rem;
            background:
                linear-gradient(145deg, rgba(255,255,255,.98), rgba(255,244,233,.88));
            box-shadow: 0 14px 34px rgba(83, 47, 25, .09);
        }

        [data-testid="stMetricValue"] {
            color: var(--nb-orange-dark);
            font-weight: 760;
        }

        [data-testid="stButton"] button,
        [data-testid="stLinkButton"] a {
            border-radius: .72rem;
            border-color: rgba(240, 90, 36, .38);
            font-weight: 650;
            transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
        }

        [data-testid="stButton"] button:hover,
        [data-testid="stLinkButton"] a:hover {
            transform: translateY(-2px);
            border-color: var(--nb-orange);
            box-shadow: 0 10px 24px rgba(240, 90, 36, .18);
        }

        [data-testid="stButton"] button[kind="primary"] {
            color: white;
            border: 0;
            background: linear-gradient(115deg, #c83b0d 0%, #f05a24 52%, #ff9a45 100%);
            box-shadow: 0 10px 24px rgba(240, 90, 36, .24);
        }

        [data-testid="stExpander"],
        [data-testid="stDataFrame"],
        [data-testid="stForm"] {
            border-radius: .9rem;
            overflow: hidden;
            box-shadow: 0 10px 28px rgba(36, 40, 50, .07);
        }

        [data-testid="stAlert"] {
            border-radius: .8rem;
            border-left-width: 5px;
        }

        [data-testid="stProgress"] > div > div > div {
            background: linear-gradient(90deg, #c83b0d, #f05a24, #ffc857);
        }

        .nb-hero {
            position: relative;
            overflow: hidden;
            padding: clamp(2rem, 5vw, 4.5rem);
            margin: .4rem 0 2rem;
            border-radius: 1.4rem;
            color: white;
            background:
                radial-gradient(circle at 86% 18%, rgba(255, 200, 87, .42), transparent 15rem),
                linear-gradient(125deg, #15171c 0%, #5f210f 45%, #d44311 75%, #ff8b3d 100%);
            box-shadow: 0 24px 60px rgba(104, 43, 17, .22);
        }

        .nb-hero::after {
            content: "";
            position: absolute;
            width: 22rem;
            height: 22rem;
            right: -8rem;
            bottom: -12rem;
            border: 1px solid rgba(255,255,255,.25);
            border-radius: 50%;
            box-shadow: 0 0 0 3rem rgba(255,255,255,.035), 0 0 0 7rem rgba(255,255,255,.025);
        }

        .nb-kicker {
            font-size: .75rem;
            font-weight: 750;
            letter-spacing: .2em;
            color: #ffd8b7;
        }

        .nb-hero-title {
            position: relative;
            z-index: 1;
            max-width: 58rem;
            margin: .65rem 0 .8rem;
            font-size: clamp(2.5rem, 5vw, 5.6rem);
            line-height: .98;
            font-weight: 820;
            letter-spacing: -.055em;
        }

        .nb-hero-copy {
            position: relative;
            z-index: 1;
            max-width: 48rem;
            margin: 0;
            color: rgba(255,255,255,.86);
            font-size: 1.08rem;
            line-height: 1.65;
        }

        .nb-section-label {
            display: inline-block;
            margin: 1rem 0 .55rem;
            padding: .28rem .65rem;
            color: #a52f08;
            background: linear-gradient(90deg, #ffe2ca, #fff2df);
            border: 1px solid #ffc99e;
            border-radius: 999px;
            font-size: .72rem;
            font-weight: 800;
            letter-spacing: .14em;
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
        "officer_name": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def navigate(page: str, loan_id: str | None = None) -> None:
    st.session_state.page = page
    if loan_id:
        st.session_state.selected_loan_id = loan_id
    st.rerun()


def app_header(section: str) -> None:
    left, right = st.columns([3, 1])
    with left:
        st.caption(f"NORTHBAY BANK  /  {section.upper()}")
    with right:
        st.caption("BUILDERSLAB TEAM 03 · TRAINING USE")


def risk_tier(probability: float) -> str:
    if probability < st.session_state.low_threshold:
        return "Low"
    if probability < st.session_state.high_threshold:
        return "Medium"
    return "High"


def tier_icon(tier: str) -> str:
    return {"Low": "🟢 Low", "Medium": "🟠 Medium", "High": "🔴 High"}[tier]


def _tier_cell_style(value: str) -> str:
    if "High" in value:
        return "background-color: rgba(226,75,74,.14); color:#c22a1d; font-weight:700; border-radius:6px;"
    if "Medium" in value:
        return "background-color: rgba(255,200,87,.24); color:#a66300; font-weight:700; border-radius:6px;"
    return "background-color: rgba(70,180,120,.16); color:#1e7d43; font-weight:700; border-radius:6px;"


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
        st.title("🏦 NorthBay Bank")
        st.caption("PORTFOLIO RISK DESK")
        pages = list(NAV_ICONS)
        selected = st.radio(
            "Navigate",
            pages,
            index=pages.index(st.session_state.page),
            format_func=lambda page: f"{NAV_ICONS[page]}  {page}",
        )
        if selected != st.session_state.page:
            st.session_state.page = selected
            st.rerun()

        st.divider()
        st.caption("MODEL")
        st.write("NS-PD-3.4 (simulated)")
        st.caption("DATA AS OF")
        st.write(most_recent_quarter_end())
        st.text_input(
            "Signed in as",
            key="officer_name",
            placeholder="Your name",
            help="Used to attribute monitoring actions you log during loan review.",
        )
        st.info("Fictional case study. Training use only.")


def render_home() -> None:
    app_header("Client project")
    st.markdown(
        """
        <section class="nb-hero">
            <div class="nb-kicker">NORTHBAY BANK · PORTFOLIO INTELLIGENCE</div>
            <div class="nb-hero-title">Loan Default Prediction.</div>
            <p class="nb-hero-copy">
                Continuous probability of default for clearer portfolio decisions.
                See beyond the letter grade, understand the drivers, and focus review
                where exposure is highest.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    primary, secondary, _ = st.columns([1.1, 1.1, 3])
    with primary:
        if st.button("Open portfolio dashboard", type="primary", width="stretch"):
            navigate("Portfolio")
    with secondary:
        if st.button("Score an individual loan", width="stretch"):
            navigate("Score a loan")

    st.divider()
    stats = st.columns(4)
    facts = [
        ("800K", "Originated loans"),
        ("5 yrs", "Portfolio history"),
        ("150+", "Features available"),
        ("22%", "Historical default rate"),
    ]
    for column, (value, label) in zip(stats, facts):
        column.metric(label, value)

    st.divider()
    st.markdown('<span class="nb-section-label">THE OPPORTUNITY</span>', unsafe_allow_html=True)
    st.header("Why probability of default")
    first, second = st.columns(2)
    with first:
        st.subheader("Today: one letter grade")
        st.write(
            "Loans within a single LendingClub grade can have very different "
            "underlying risk. A static grade cannot show how strongly each "
            "borrower and loan characteristic changes expected default risk."
        )
    with second:
        st.subheader("Target: a continuous risk signal")
        st.write(
            "A deterministic model returns a probability from 0 to 1, a "
            "configurable monitoring tier, and the five strongest SHAP drivers. "
            "The score supports portfolio review; it does not automate denial."
        )

    st.markdown('<span class="nb-section-label">THE SYSTEM</span>', unsafe_allow_html=True)
    st.header("How one account reaches the credit officer")
    flow = st.columns(4)
    steps = [
        ("01 · Loan in", "Origination features only."),
        ("02 · ML scores", "Default probability, tier, and SHAP."),
        ("03 · Explain", "Future plain-English assistive summary."),
        ("04 · Officer acts", "Monitor, provision, review, or escalate."),
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
    app_header("Portfolio overview / 01")
    st.title("Risk, in portfolio context.")
    st.caption(
        "800,000 originated personal loans · LendingClub marketplace · illustrative training data"
    )

    render_threshold_settings()
    metrics = st.columns(4)
    metrics[0].metric("Outstanding balance", "$4.82B")
    metrics[1].metric("Historical default prevalence", "22.0%")
    metrics[2].metric("Weighted average PD", "18.7%")
    metrics[3].metric("High-risk exposure", "$612.4M", "12.7% of balance")

    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("Balance by continuous PD tier")
        distribution = pd.DataFrame(
            {"Portfolio share": [55, 32, 13]},
            index=["Low", "Medium", "High"],
        )
        st.bar_chart(distribution, color="#F05A24")
        st.caption("Low 440K loans · Medium 256K · High 104K")
    with right:
        st.subheader("Same grade, different risk")
        grade_data = pd.DataFrame(
            {
                "Minimum PD": [5.1, 7.4, 12.1],
                "Maximum PD": [12.8, 22.4, 38.0],
            },
            index=["Grade A", "Grade B", "Grade C"],
        )
        st.dataframe(grade_data, width="stretch")
        st.info(
            "Grades hide within-band variation. Continuous PD shows where "
            "portfolio officers should spend review time."
        )

    st.success(
        "At our chosen risk tier boundaries, the model separates loans into "
        "Low, Medium, and High default risk, with actual default rates of "
        "6.2%, 18.7%, and 41.3% respectively, giving portfolio managers a "
        "clearer view of exposure than grade alone."
    )

    st.header("Loans needing a closer look")
    search_col, tier_col = st.columns([3, 1])
    search = search_col.text_input(
        "Search the review queue",
        placeholder="Loan ID, borrower, purpose, or driver",
    )
    tier_filter = tier_col.multiselect(
        "Risk tier", ["Low", "Medium", "High"], default=["Medium", "High"]
    )

    filtered = []
    for loan in LOANS:
        current_tier = risk_tier(loan.pd)
        haystack = (
            f"{loan.loan_id} {loan.borrower} {loan.purpose} {loan.top_driver}"
        ).lower()
        if current_tier in tier_filter and search.lower() in haystack:
            filtered.append(loan)

    if not filtered:
        st.info("No loans match the current filters.")
        return

    rows = [
        {
            "Loan ID": loan.loan_id,
            "Borrower": loan.borrower,
            "Grade": loan.grade,
            "Purpose": loan.purpose,
            "Outstanding": f"${loan.balance:,.0f}",
            "PD": f"{loan.pd:.1f}%",
            "Tier": tier_icon(risk_tier(loan.pd)),
            "Top driver": loan.top_driver,
            "Status": loan.status,
        }
        for loan in filtered
    ]
    queue_df = pd.DataFrame(rows)
    st.dataframe(
        queue_df.style.map(_tier_cell_style, subset=["Tier"]),
        width="stretch",
        hide_index=True,
    )

    chosen_id = st.selectbox(
        "Select a loan to review",
        options=[loan.loan_id for loan in filtered],
        format_func=lambda loan_id: (
            f"{loan_id} · {find_loan(loan_id).borrower} · "
            f"{find_loan(loan_id).pd:.1f}% PD"
        ),
    )
    if st.button("Open individual risk review", type="primary"):
        navigate("Loan review", chosen_id)


def render_score() -> None:
    app_header("Origination scoring / 02")
    st.title("Score an individual loan.")
    st.write(
        "Use borrower and loan characteristics available at issuance. "
        "The simulated score will be added to the portfolio review workflow."
    )
    st.info(
        "Origination-only policy: no payment performance, collections activity, "
        "or other post-issuance information is used."
    )

    with st.form("origination_scoring"):
        st.subheader("Borrower profile")
        a, b, c = st.columns(3)
        name = a.text_input("Borrower name", value="Marcus Chen")
        income = b.number_input(
            "Annual income (USD)", min_value=0, value=92_000, step=1_000
        )
        employment = c.number_input(
            "Employment tenure (years)", min_value=0, max_value=50, value=6
        )

        st.subheader("Loan request")
        d, e, f = st.columns(3)
        amount = d.number_input(
            "Requested amount (USD)", min_value=1_000, value=45_000, step=500
        )
        purpose = e.selectbox(
            "Purpose",
            [
                "Debt consolidation",
                "Home improvement",
                "Medical expenses",
                "Major purchase",
                "Small business",
                "Auto",
            ],
        )
        term = f.selectbox("Term", ["36 months", "60 months"], index=1)

        st.subheader("Credit profile")
        g, h, i = st.columns(3)
        score = g.slider("Credit score", 300, 850, 687)
        dti = h.number_input(
            "Debt-to-income ratio (%)",
            min_value=0.0,
            max_value=100.0,
            value=31.8,
            step=0.1,
        )
        utilization = i.slider("Revolving utilization (%)", 0, 100, 67)

        submitted = st.form_submit_button(
            "Run simulated default prediction", type="primary"
        )

    if submitted and not name.strip():
        st.error("Enter a borrower name before scoring.")
    elif submitted and income <= 0:
        st.error("Annual income must be greater than $0 before scoring.")
    elif submitted:
        simulated_pd = max(
            2.0,
            min(
                58.0,
                15
                + (700 - score) * 0.08
                + max(dti - 25, 0) * 0.45
                + max(utilization - 50, 0) * 0.18
                + (4 if employment < 2 else 0),
            ),
        )
        st.session_state.last_scored = {
            "borrower": name,
            "amount": amount,
            "purpose": purpose,
            "pd": round(simulated_pd, 1),
        }
        st.session_state.selected_loan_id = LOANS[0].loan_id

    if st.session_state.last_scored:
        result = st.session_state.last_scored
        st.success(
            f"Simulation complete: {result['pd']:.1f}% PD · "
            f"{risk_tier(result['pd'])} monitoring tier."
        )
        if st.button("Review simulated result", type="primary"):
            navigate("Loan review")


def render_shap_drivers() -> None:
    st.subheader("What moved the score")
    st.caption("TOP FIVE SHAP DRIVERS")
    for label, contribution, direction in SHAP_DRIVERS:
        left, right = st.columns([4, 1])
        with left:
            st.write(f"**{label}**")
            st.caption(direction)
        with right:
            sign = "+" if contribution > 0 else ""
            st.metric("PD contribution", f"{sign}{contribution:.1f} pp")
        st.divider()


def render_loan_review() -> None:
    loan = find_loan(st.session_state.selected_loan_id)
    simulation = st.session_state.last_scored
    borrower = simulation["borrower"] if simulation else loan.borrower
    amount = simulation["amount"] if simulation else loan.amount
    purpose = simulation["purpose"] if simulation else loan.purpose
    probability = simulation["pd"] if simulation else loan.pd
    tier = risk_tier(probability)

    app_header("Individual loan risk review / 03")
    st.title(f"{borrower} · {purpose}")
    st.caption(
        f"Loan {loan.loan_id} · ${amount:,.0f} requested · "
        f"Grade {loan.grade} · scored before portfolio entry"
    )
    if st.button("Back to portfolio queue"):
        navigate("Portfolio")

    score_col, driver_col = st.columns([1, 1.45])
    with score_col:
        st.subheader("System of record")
        st.metric("Default probability", f"{probability:.1f}%")
        if tier == "High":
            st.error(f"{tier.upper()} RISK")
        elif tier == "Medium":
            st.warning(f"{tier.upper()} RISK")
        else:
            st.success(f"{tier.upper()} RISK")
        st.write(
            f"**Draft business boundary:** High ≥ "
            f"{st.session_state.high_threshold:.1f}%"
        )
        st.caption(
            "Thresholds are configurable business settings and are not "
            "objective model facts."
        )
        st.write("**Model:** NS-PD-3.4")
        st.write(
            f"**Scored:** {datetime.now(timezone.utc).strftime('%d %b %Y · %H:%M UTC')}"
        )
    with driver_col:
        render_shap_drivers()

    explanation, actions = st.columns([1.25, 1])
    with explanation:
        st.subheader("Future explainability capability")
        st.info("Plain-English summary is not connected yet.")
        st.write(
            "A summary generated from SHAP factors and portfolio context will "
            "appear here after the explanation service is connected."
        )
        st.caption(
            "The deterministic model remains the system of record. Gemini is "
            "assistive only and will not make or alter the risk decision."
        )

    with actions:
        st.subheader("Record a monitoring decision")
        officer = st.session_state.officer_name.strip()
        if not officer:
            st.caption("⚠️ Enter your name in the sidebar to attribute logged actions.")
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
                        "Loan": loan.loan_id,
                        "Action": choice,
                        "Officer": officer or "Unspecified officer",
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
elif st.session_state.page == "Portfolio":
    render_portfolio()
elif st.session_state.page == "Score a loan":
    render_score()
else:
    render_loan_review()