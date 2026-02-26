import streamlit as st
import requests
import pandas as pd
import numpy as np
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("API_URL", "http://localhost:8100")

# ── Helpers ───────────────────────────────────────────────────────────────────
def api_get(path: str):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error ({path}): {e}")
        return None


def api_post(path: str, payload: dict):
    try:
        r = requests.post(f"{API_URL}{path}", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error ({path}): {e}")
        return None


def fmt_pct(v):  return f"{v * 100:.2f}%"
def fmt_usd(v):  return f"${v:,.0f}"
def fmt_rate(v): return f"{v:.2f}%"       # int_rate already in pct units


# ── Risk colour ───────────────────────────────────────────────────────────────
RISK_COLOUR = {
    "LOW":       ("🟢", "normal"),
    "MEDIUM":    ("🟡", "off"),
    "HIGH":      ("🟠", "inverse"),
    "VERY HIGH": ("🔴", "inverse"),
}

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
st.title("🏦 Credit Risk System")
st.caption("Lending Club · PD · LGD · EAD · EL · EP · Approval Model")

tab1, tab2, tab3 = st.tabs(["📋 Loan Evaluator", "📊 Portfolio Analytics", "🔬 Backtest & Stress"])


# ───────────────────────────────────────────────────────────────────────────────
# TAB 1 — LOAN EVALUATOR
# ───────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("Single Loan Evaluation")
    st.markdown("Enter applicant data and score all four risk components in one call.")

    with st.form("loan_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Loan Terms")
            funded_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=40000,
                                           value=15000, step=500)
            term        = st.selectbox("Term (months)", [36, 60])
            int_rate    = st.number_input("Interest Rate (%)", min_value=5.0, max_value=31.0,
                                           value=13.99, step=0.01,
                                           help="Enter as percentage, e.g. 13.99 for 13.99%")
            sub_grade   = st.selectbox("Sub-Grade", [
                f"{g}{i}" for g in "ABCDEFG" for i in range(1, 6)
            ], index=7)   # default B3
            purpose     = st.selectbox("Purpose", [
                "debt_consolidation", "credit_card", "home_improvement",
                "other", "major_purchase", "medical", "small_business",
                "car", "vacation", "moving", "house", "wedding",
                "renewable_energy", "educational",
            ])

        with col2:
            st.subheader("Borrower Profile")
            annual_inc     = st.number_input("Annual Income ($)", min_value=5000,
                                              max_value=500000, value=65000, step=1000)
            dti            = st.number_input("Debt-to-Income Ratio", min_value=0.0,
                                              max_value=60.0, value=18.5, step=0.5)
            emp_length     = st.selectbox("Employment Length", [
                "< 1 year", "1 year", "2 years", "3 years", "4 years",
                "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years",
            ], index=5)
            home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])

        with col3:
            st.subheader("Payment")
            # Compute suggested installment (PMT formula)
            _r = (int_rate / 100) / 12
            _n = term
            if _r > 0:
                _suggested = funded_amnt * _r * (1 + _r) ** _n / ((1 + _r) ** _n - 1)
            else:
                _suggested = funded_amnt / _n
            installment = st.number_input(
                "Monthly Installment ($)",
                min_value=10.0,
                value=round(_suggested, 2),
                step=1.0,
                help="Auto-calculated from loan terms; adjust if needed.",
            )
            st.caption(f"Suggested by PMT formula: ${_suggested:.2f}")

        submitted = st.form_submit_button("⚡ Score Loan", use_container_width=True)

    # ── Results ───────────────────────────────────────────────────────────────
    if submitted:
        payload = {
            "funded_amnt":    funded_amnt,
            "term":           term,
            "int_rate":       int_rate,
            "sub_grade":      sub_grade,
            "emp_length":     emp_length,
            "home_ownership": home_ownership,
            "annual_inc":     annual_inc,
            "purpose":        purpose,
            "dti":            dti,
            "installment":    installment,
        }

        with st.spinner("Scoring loan..."):
            result = api_post("/predict", payload)

        if result:
            # ── Decision banner ───────────────────────────────────────────
            approved = result["approved"]
            risk     = result["risk_category"]
            emoji, _ = RISK_COLOUR.get(risk, ("⚪", "normal"))

            if approved:
                st.success(f"✅ **APPROVED** — Risk Category: {emoji} {risk}")
            else:
                st.error(f"❌ **REJECTED** — Risk Category: {emoji} {risk}  "
                         f"(PD {fmt_pct(result['pd_score'])} exceeds "
                         f"threshold {fmt_pct(result['pd_threshold'])})")

            # ── Four model outputs ────────────────────────────────────────
            st.markdown("---")
            st.subheader("Model Outputs")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("PD Score",         fmt_pct(result["pd_score"]),
                      delta=f"Threshold {fmt_pct(result['pd_threshold'])}")
            c2.metric("LGD (predicted)",  fmt_pct(result["lgd_hat"]))
            c3.metric("EAD Ratio",        fmt_pct(result["ead_ratio_hat"]))
            c3.caption(f"EAD: {fmt_usd(result['ead_hat'])}")
            c4.metric("EL",               fmt_usd(result["el"]),
                      delta=f"{fmt_pct(result['el_ratio'])} of funded amount",
                      delta_color="inverse")

            # ── Profitability ─────────────────────────────────────────────
            st.markdown("---")
            st.subheader("Profitability")
            p1, p2, p3 = st.columns(3)
            p1.metric("Interest Income",  fmt_usd(result["income"]))
            p2.metric("Expected Loss",    fmt_usd(result["el"]), delta_color="inverse",
                      delta=f"{fmt_pct(result['el_ratio'])}")
            p3.metric("Expected Profit",  fmt_usd(result["ep"]),
                      delta=f"EP ratio {fmt_pct(result['ep_ratio'])}",
                      delta_color="normal" if result["ep"] >= 0 else "inverse")

            hurdle = result["hurdle_rate"]
            ep_r   = result["ep_ratio"]
            margin = ep_r - hurdle
            st.progress(
                min(max(ep_r + 0.5, 0), 1),
                text=f"EP Ratio: {fmt_pct(ep_r)} | Hurdle: {fmt_pct(hurdle)} | "
                     f"Margin: {fmt_pct(margin)}"
            )

            # ── Debug expander ────────────────────────────────────────────
            with st.expander("Raw API response"):
                st.json(result)


# ───────────────────────────────────────────────────────────────────────────────
# TAB 2 — PORTFOLIO ANALYTICS
# ───────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("Portfolio Analytics")
    st.markdown(
        "Adjust the approval policy parameters and see how the approved book changes in real time. "
        "Data is read from the pre-computed `ep_results.parquet`."
    )

    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        hurdle_pct = st.slider(
            "Hurdle Rate (min EP ratio, %)",
            min_value=0.0, max_value=20.0, value=0.0, step=0.5,
            help="Approve only loans where EP/funded_amnt exceeds this threshold.",
        )
    with col_ctrl2:
        pd_cutoff_pct = st.slider(
            "PD Cutoff (max tolerated PD, %)",
            min_value=5.0, max_value=100.0, value=100.0, step=1.0,
            help="Secondary filter: reject any loan with PD above this value.",
        )

    sim_payload = {
        "hurdle_rate": hurdle_pct / 100,
        "pd_cutoff":   pd_cutoff_pct / 100,
    }

    with st.spinner("Recomputing approved book..."):
        sim = api_post("/portfolio/simulate", sim_payload)

    if sim and sim.get("n_approved", 0) > 0:
        # ── KPIs ──────────────────────────────────────────────────────────
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Approved Loans",    f"{sim['n_approved']:,}")
        k2.metric("Approval Rate",     fmt_pct(sim["approval_rate"]))
        k3.metric("Total EP",          fmt_usd(sim["total_ep"]))
        k4.metric("Mean EP Ratio",     fmt_pct(sim["mean_ep_ratio"]))
        k5.metric("Mean PD",           fmt_pct(sim["mean_pd"]))

        st.markdown("---")

        # ── Grade breakdown ───────────────────────────────────────────────
        if sim.get("by_grade"):
            grade_df = pd.DataFrame(sim["by_grade"])
            grade_df["mean_pd"]       = grade_df["mean_pd"].map(lambda x: f"{x*100:.1f}%")
            grade_df["mean_el_ratio"] = grade_df["mean_el_ratio"].map(lambda x: f"{x*100:.1f}%")
            grade_df["mean_ep_ratio"] = grade_df["mean_ep_ratio"].map(lambda x: f"{x*100:.1f}%")
            grade_df["total_ep"]      = grade_df["total_ep"].map(lambda x: f"${x:,.0f}")
            grade_df.columns = ["Grade", "# Loans", "Mean PD", "Mean EL Ratio",
                                  "Mean EP Ratio", "Total EP"]
            st.subheader("Approved Book by Grade")
            st.dataframe(grade_df, use_container_width=True, hide_index=True)

        # ── Hurdle calibration chart ──────────────────────────────────────
        st.markdown("---")
        st.subheader("Hurdle Rate Calibration (Historical)")
        hcal_data = api_get("/portfolio/hurdle-calibration")
        if hcal_data:
            hcal = pd.DataFrame(hcal_data)
            import altair as alt
            base = alt.Chart(hcal).encode(
                x=alt.X("hurdle:Q", axis=alt.Axis(format=".0%"), title="Hurdle Rate")
            )
            ep_line = base.mark_line(color="#4A90D9").encode(
                y=alt.Y("total_real_ep:Q", title="Realised EP ($)", axis=alt.Axis(format="$,.0f"))
            )
            pred_line = base.mark_line(color="#F5A623", strokeDash=[4, 2]).encode(
                y="total_pred_ep:Q"
            )
            st.altair_chart(
                alt.layer(ep_line, pred_line).resolve_scale(y="shared").properties(
                    title="Predicted vs. Realised Total EP by Hurdle Rate",
                    height=280,
                ),
                use_container_width=True,
            )
            st.caption("Blue = realised EP (actual outcomes) · Orange dashed = predicted EP")
    else:
        st.warning("No loans approved under these parameters, or portfolio data not available.")


# ───────────────────────────────────────────────────────────────────────────────
# TAB 3 — BACKTEST & STRESS
# ───────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("Backtest & Stress Analysis")
    st.markdown(
        "Results from the rolling-window backtest with Monte Carlo uncertainty bands. "
        "Data served from `backtest_results.csv`."
    )

    bt_data = api_get("/portfolio/backtest")

    if bt_data:
        bt = pd.DataFrame(bt_data)

        # ── Summary KPIs ──────────────────────────────────────────────────
        total_pred = bt["pred_ep_total"].sum()
        total_real = bt["real_ep_total"].sum()
        mean_app   = bt["approval_rate"].mean()
        mean_def   = bt["obs_default_rate"].mean()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Predicted EP",    fmt_usd(total_pred))
        k2.metric("Total Realised EP",      fmt_usd(total_real),
                  delta=f"{(total_real/total_pred - 1)*100:+.1f}% vs predicted",
                  delta_color="normal" if total_real >= total_pred else "inverse")
        k3.metric("Mean Approval Rate",     fmt_pct(mean_app))
        k4.metric("Mean Default Rate",      fmt_pct(mean_def))

        st.markdown("---")

        import altair as alt

        # ── Predicted vs Realised EP with MC band ─────────────────────────
        st.subheader("Predicted vs. Realised EP by Vintage (with MC Uncertainty)")
        band = alt.Chart(bt).mark_area(opacity=0.15, color="#4A90D9").encode(
            x=alt.X("year:O", title="Vintage Year"),
            y=alt.Y("mc_ep_p05:Q", title="Total EP ($)", axis=alt.Axis(format="$,.0f")),
            y2="mc_ep_p95:Q",
        )
        pred_line = alt.Chart(bt).mark_line(color="#4A90D9", strokeWidth=2).encode(
            x="year:O",
            y="pred_ep_total:Q",
        )
        real_line = alt.Chart(bt).mark_line(color="#E84040", strokeWidth=2,
                                             strokeDash=[4, 2]).encode(
            x="year:O",
            y="real_ep_total:Q",
        )
        st.altair_chart(
            alt.layer(band, pred_line, real_line).properties(height=300),
            use_container_width=True,
        )
        st.caption("Blue solid = predicted EP · Red dashed = realised EP · Band = MC 5th–95th pct")

        # ── Risk metrics ───────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Value at Risk & CVaR by Vintage")
        risk_df = bt[["year", "mc_ep_mean", "mc_var_5pct", "mc_cvar_5pct"]].copy()
        risk_df.columns = ["Year", "MC Mean EP ($)", "VaR 5% ($)", "CVaR 5% ($)"]
        for col in ["MC Mean EP ($)", "VaR 5% ($)", "CVaR 5% ($)"]:
            risk_df[col] = risk_df[col].map(lambda x: f"${x:,.0f}")
        st.dataframe(risk_df, use_container_width=True, hide_index=True)

        # ── Forecast error ────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Forecast Error (Predicted − Realised EP)")
        bt["forecast_error"] = bt["pred_ep_total"] - bt["real_ep_total"]
        err_chart = alt.Chart(bt).mark_bar().encode(
            x=alt.X("year:O", title="Vintage Year"),
            y=alt.Y("forecast_error:Q", title="Forecast Error ($)",
                    axis=alt.Axis(format="$,.0f")),
            color=alt.condition(
                alt.datum.forecast_error >= 0,
                alt.value("#F5A623"),   # overestimated
                alt.value("#4A90D9"),   # underestimated
            ),
        ).properties(height=250)
        st.altair_chart(err_chart, use_container_width=True)
        st.caption("Orange = model overestimated (pred > real) · Blue = underestimated")

        # ── Raw table ─────────────────────────────────────────────────────
        with st.expander("Full backtest table"):
            st.dataframe(bt, use_container_width=True, hide_index=True)

    else:
        st.info(
            "No backtest data available. Run `strategy_simulation.ipynb` first to generate "
            "`backtest_results.csv`, then mount it into the container's `/app/data` directory."
        )
