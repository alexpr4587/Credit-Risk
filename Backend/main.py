from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import numpy as np
import joblib
import json
import os
import uvicorn

# ── Register pipeline classes so joblib can deserialize the PKLs ─────────────
import credit_risk_pipeline
import __main__
for cls_name in [
    "ColumnNameCleaner", "CategoricalCleaner", "EmpLengthTransformer",
    "TermTransformer", "LoanStatusCleaner", "GradeTransformer",
    "IssueDateTransformer", "Winsorizer", "WoEEncoder", "FeatureDropper",
    "LoanFeatureEngineer", "PDPreprocessor", "PDModel", "ApprovalModel",
]:
    setattr(__main__, cls_name, getattr(credit_risk_pipeline, cls_name))

from credit_risk_pipeline import (
    GradeTransformer, ApprovalModel,
    calculate_el, calculate_el_ratio, calculate_income,
    calculate_ep, calculate_ep_ratio,
)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Credit Risk API",
    description="End-to-end credit risk scoring: PD · LGD · EAD · EL · EP · Approval",
    version="1.0.0",
)

# ── Globals ───────────────────────────────────────────────────────────────────
DATA_DIR   = os.getenv("DATA_DIR", "/app/data")
MODELS_DIR = os.getenv("MODELS_DIR", "/app/data")

pd_preprocessor = lgd_preprocessor = ead_preprocessor = None
pd_model_obj    = lgd_model_obj    = ead_model_obj    = None
approval_model  = None
mean_rates_by_grade: dict = {}          # for int_rate_residual feature
_portfolio_df: pd.DataFrame = None      # lazy-loaded ep_results.parquet


def _path(filename: str) -> str:
    return os.path.join(MODELS_DIR, filename)


@app.on_event("startup")
def load_models():
    global pd_preprocessor, lgd_preprocessor, ead_preprocessor
    global pd_model_obj, lgd_model_obj, ead_model_obj
    global approval_model, mean_rates_by_grade

    pkls = {
        "pd_preprocessor":  "pd_preprocessor.pkl",
        "pd_model":         "pd_model.pkl",
        "lgd_preprocessor": "lgd_preprocessor.pkl",
        "lgd_model":        "lgd_model.pkl",
        "ead_preprocessor": "ead_preprocessor.pkl",
        "ead_model":        "ead_model.pkl",
        "approval_model":   "approval_model.pkl",
    }
    loaded = {}
    for key, fname in pkls.items():
        path = _path(fname)
        if os.path.exists(path):
            loaded[key] = joblib.load(path)
            print(f"  ✓ {fname}")
        else:
            print(f"  ✗ {fname} NOT FOUND")

    pd_preprocessor  = loaded.get("pd_preprocessor")
    pd_model_obj     = loaded.get("pd_model")
    lgd_preprocessor = loaded.get("lgd_preprocessor")
    lgd_model_obj    = loaded.get("lgd_model")
    ead_preprocessor = loaded.get("ead_preprocessor")
    ead_model_obj    = loaded.get("ead_model")
    approval_model   = loaded.get("approval_model")

    # Optional: mean int_rate by sub_grade for int_rate_residual feature
    rates_path = _path("mean_rates_by_grade.json")
    if os.path.exists(rates_path):
        with open(rates_path) as f:
            mean_rates_by_grade = {int(k): v for k, v in json.load(f).items()}
        print(f"  ✓ mean_rates_by_grade.json ({len(mean_rates_by_grade)} grades)")
    else:
        print("  ⚠  mean_rates_by_grade.json not found — int_rate_residual will be 0")

    print("Startup complete.")


def _load_portfolio() -> pd.DataFrame:
    """Lazy-load ep_results.parquet once, cache in memory."""
    global _portfolio_df
    if _portfolio_df is None:
        path = os.path.join(DATA_DIR, "ep_results.parquet")
        if os.path.exists(path):
            _portfolio_df = pd.read_parquet(path)
            print(f"Portfolio loaded: {len(_portfolio_df):,} loans")
        else:
            _portfolio_df = pd.DataFrame()
    return _portfolio_df


# ── Sub-grade ordinal mapping ─────────────────────────────────────────────────
_GRADE_MAP = {
    f"{g}{i}": (ci - 1) * 5 + i
    for ci, g in enumerate("abcdefg", 1)
    for i in range(1, 6)
}


def _sub_grade_num(sub_grade: str) -> float:
    return float(_GRADE_MAP.get(sub_grade.lower(), 0))


def _emp_length_num(emp_length: str) -> float:
    if not emp_length or emp_length.lower() in ("n/a", ""):
        return 5.0   # mode fallback
    s = emp_length.replace("< 1 year", "0")
    import re
    m = re.search(r"\d+", s)
    return float(m.group()) if m else 5.0


# ── Feature engineering for a single loan ────────────────────────────────────
def _build_feature_row(loan: "LoanInput") -> pd.DataFrame:
    """
    Replicates the pre-preprocessor feature engineering from the training notebooks.
    Returns a single-row DataFrame with all features the PKL preprocessors expect.
    """
    term_months    = float(loan.term)
    emp_len        = _emp_length_num(loan.emp_length)
    sub_grade_n    = _sub_grade_num(loan.sub_grade)
    log_inc        = float(np.log1p(loan.annual_inc))
    loan_to_income = loan.funded_amnt / (loan.annual_inc + 1)
    payment_burden = (loan.installment * 12) / (loan.annual_inc + 1)
    dti_x_term     = loan.dti * term_months
    inc_stability  = log_inc * emp_len

    # int_rate_residual: actual rate minus expected rate for this sub_grade
    expected_rate      = mean_rates_by_grade.get(int(sub_grade_n), loan.int_rate)
    int_rate_residual  = loan.int_rate - expected_rate

    row = {
        # Base numeric
        "funded_amnt":       loan.funded_amnt,
        "annual_inc":        loan.annual_inc,
        "dti":               loan.dti,
        "int_rate":          loan.int_rate,
        "emp_length":        emp_len,
        "term":              term_months,
        "sub_grade_num":     sub_grade_n,
        # Engineered numeric
        "loan_to_income":    loan_to_income,
        "payment_burden":    payment_burden,
        "log_annual_inc":    log_inc,
        "dti_x_term":        dti_x_term,
        "int_rate_residual": int_rate_residual,
        "inc_stability":     inc_stability,
        # Categorical (cleaned to lowercase_underscore)
        "home_ownership":    loan.home_ownership.lower().replace(" ", "_"),
        "purpose":           loan.purpose.lower().replace(" ", "_"),
    }
    return pd.DataFrame([row])


# ── Schemas ───────────────────────────────────────────────────────────────────
class LoanInput(BaseModel):
    funded_amnt:    float = Field(..., gt=0,    example=15000,   description="Loan amount ($)")
    term:           int   = Field(...,           example=36,      description="Loan term in months (36 or 60)")
    int_rate:       float = Field(..., gt=0,    example=13.99,   description="Interest rate (%, e.g. 13.99)")
    sub_grade:      str   = Field(...,           example="B3",    description="LC sub-grade (A1–G5)")
    emp_length:     str   = Field(...,           example="5 years", description="Employment length")
    home_ownership: str   = Field(...,           example="RENT",  description="RENT / OWN / MORTGAGE")
    annual_inc:     float = Field(..., gt=0,    example=65000,   description="Annual income ($)")
    purpose:        str   = Field(...,           example="debt_consolidation")
    dti:            float = Field(..., ge=0,    example=18.5,    description="Debt-to-income ratio")
    installment:    float = Field(..., gt=0,    example=492.0,   description="Monthly installment ($)")


class ScoreResponse(BaseModel):
    # Inputs echoed back
    funded_amnt: float
    int_rate:    float
    term:        int

    # Model outputs
    pd_score:       float = Field(description="Probability of Default (0–1)")
    lgd_hat:        float = Field(description="Loss Given Default (0–1)")
    ead_ratio_hat:  float = Field(description="EAD as fraction of funded amount (0–1)")
    ead_hat:        float = Field(description="Exposure at Default ($)")

    # Derived
    el:          float = Field(description="Expected Loss ($)")
    el_ratio:    float = Field(description="EL / funded_amnt")
    income:      float = Field(description="Expected interest income ($)")
    ep:          float = Field(description="Expected Profit ($)")
    ep_ratio:    float = Field(description="EP / funded_amnt")

    # Decision
    approved:       int   = Field(description="1 = Approve, 0 = Reject")
    hurdle_rate:    float = Field(description="Minimum EP ratio applied")
    pd_threshold:   float = Field(description="Max PD this loan can carry and still clear hurdle")
    risk_category:  str   = Field(description="LOW / MEDIUM / HIGH / VERY HIGH")


class SimulationRequest(BaseModel):
    hurdle_rate: float = Field(0.0,  ge=0,   le=1,   description="Minimum EP ratio (0–1)")
    pd_cutoff:   float = Field(1.0,  gt=0,   le=1,   description="Max PD to approve (0–1)")


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    models_ok = all([pd_model_obj, lgd_model_obj, ead_model_obj, approval_model])
    return {
        "status":       "active" if models_ok else "degraded",
        "models_loaded": models_ok,
        "pd":    pd_model_obj    is not None,
        "lgd":   lgd_model_obj   is not None,
        "ead":   ead_model_obj   is not None,
        "approval": approval_model is not None,
    }


@app.post("/predict", response_model=ScoreResponse)
def predict(loan: LoanInput):
    if not all([pd_preprocessor, pd_model_obj, lgd_preprocessor,
                lgd_model_obj, ead_preprocessor, ead_model_obj, approval_model]):
        raise HTTPException(503, "One or more models are not loaded.")

    try:
        X = _build_feature_row(loan)

        # ── PD ───────────────────────────────────────────────────────────
        X_pd      = pd_preprocessor.transform(X)
        pd_score  = float(pd_model_obj.predict_proba(X_pd)[0])

        # ── LGD ──────────────────────────────────────────────────────────
        X_lgd    = lgd_preprocessor.transform(X)
        lgd_hat  = float(np.clip(lgd_model_obj.predict(X_lgd)[0], 0, 1))

        # ── EAD ──────────────────────────────────────────────────────────
        X_ead        = ead_preprocessor.transform(X)
        ead_ratio_hat = float(np.clip(ead_model_obj.predict(X_ead)[0], 0, 1))
        ead_hat      = ead_ratio_hat * loan.funded_amnt

        # ── EL / EP ───────────────────────────────────────────────────────
        fa   = np.array([loan.funded_amnt])
        ir   = np.array([loan.int_rate])
        tm   = np.array([float(loan.term)])
        pd_a = np.array([pd_score])
        lg_a = np.array([lgd_hat])
        ea_a = np.array([ead_hat])

        el        = float(calculate_el(pd_a, lg_a, ea_a)[0])
        el_ratio  = float(calculate_el_ratio(np.array([el]), fa)[0])
        income    = float(calculate_income(fa, ir, tm)[0])
        ep        = float(calculate_ep(pd_a, lg_a, ea_a, fa, ir, tm)[0])
        ep_ratio  = float(calculate_ep_ratio(np.array([ep]), fa)[0])

        # ── Approval ──────────────────────────────────────────────────────
        approved     = int(approval_model.approve(pd_a, lg_a, ea_a, fa, ir, tm)[0])
        pd_threshold = float(approval_model.pd_threshold(fa, ir, lg_a, ea_a, tm)[0])

        # ── Risk category (based on PD) ───────────────────────────────────
        if pd_score < 0.15:
            risk = "LOW"
        elif pd_score < 0.30:
            risk = "MEDIUM"
        elif pd_score < 0.50:
            risk = "HIGH"
        else:
            risk = "VERY HIGH"

        return ScoreResponse(
            funded_amnt   = loan.funded_amnt,
            int_rate      = loan.int_rate,
            term          = loan.term,
            pd_score      = round(pd_score, 4),
            lgd_hat       = round(lgd_hat, 4),
            ead_ratio_hat = round(ead_ratio_hat, 4),
            ead_hat       = round(ead_hat, 2),
            el            = round(el, 2),
            el_ratio      = round(el_ratio, 4),
            income        = round(income, 2),
            ep            = round(ep, 2),
            ep_ratio      = round(ep_ratio, 4),
            approved      = approved,
            hurdle_rate   = round(approval_model.hurdle_rate, 4),
            pd_threshold  = round(pd_threshold, 4),
            risk_category = risk,
        )

    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/portfolio/simulate")
def portfolio_simulate(req: SimulationRequest):
    """
    Recompute approved-book stats for a given hurdle_rate + pd_cutoff.
    Reads from pre-computed ep_results.parquet.
    """
    df = _load_portfolio()
    if df.empty:
        raise HTTPException(503, "ep_results.parquet not found.")

    mask     = (df["ep_ratio"] > req.hurdle_rate) & (df["pd_score"] < req.pd_cutoff)
    approved = df[mask]
    total    = len(df)

    if len(approved) == 0:
        return {"n_approved": 0, "approval_rate": 0.0}

    by_grade = (
        approved.groupby("grade_label", observed=False)
        .agg(
            n_loans       = ("funded_amnt", "count"),
            mean_pd       = ("pd_score",    "mean"),
            mean_el_ratio = ("el_ratio",    "mean"),
            mean_ep_ratio = ("ep_ratio",    "mean"),
            total_ep      = ("ep",          "sum"),
        )
        .reset_index()
        .to_dict(orient="records")
    )

    return {
        "n_approved":          int(len(approved)),
        "approval_rate":       round(len(approved) / total, 4),
        "total_funded":        round(float(approved["funded_amnt"].sum()), 0),
        "total_ep":            round(float(approved["ep"].sum()), 0),
        "total_el":            round(float(approved["el"].sum()), 0),
        "mean_pd":             round(float(approved["pd_score"].mean()), 4),
        "mean_ep_ratio":       round(float(approved["ep_ratio"].mean()), 4),
        "mean_el_ratio":       round(float(approved["el_ratio"].mean()), 4),
        "obs_default_rate":    round(float(approved["is_default"].mean()), 4),
        "by_grade":            by_grade,
    }


@app.get("/portfolio/backtest")
def portfolio_backtest():
    path = os.path.join(DATA_DIR, "backtest_results.csv")
    if not os.path.exists(path):
        raise HTTPException(503, "backtest_results.csv not found.")
    return pd.read_csv(path).to_dict(orient="records")


@app.get("/portfolio/hurdle-calibration")
def hurdle_calibration():
    path = os.path.join(DATA_DIR, "hurdle_calibration.csv")
    if not os.path.exists(path):
        raise HTTPException(503, "hurdle_calibration.csv not found.")
    return pd.read_csv(path).to_dict(orient="records")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
