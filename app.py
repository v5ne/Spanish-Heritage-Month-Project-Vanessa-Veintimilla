"""
Hispanic Heritage Historical Data Explorer 2.0
A classroom Streamlit app for authentic World Bank data, polynomial modeling,
custom mathematical functions, calculus analysis, simulations, and exports.
"""
from __future__ import annotations

import io
import math
import re
import time
import ast
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
import sympy as sp
from sklearn.metrics import mean_absolute_error
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

st.set_page_config(page_title="Hispanic Heritage Data Lab", page_icon="🌎", layout="wide")

COUNTRIES = {
    "Mexico": "MEX", "Brazil": "BRA", "Argentina": "ARG", "Colombia": "COL",
    "Chile": "CHL", "Peru": "PER", "Venezuela": "VEN", "Ecuador": "ECU",
    "Uruguay": "URY"
}
INDICATORS = {
    "Life expectancy": ("SP.DYN.LE00.IN", "years", "Average number of years a newborn would live if current mortality patterns continued."),
    "Unemployment (national estimate)": ("SL.UEM.TOTL.NE.ZS", "% of labor force", "Share of the labor force without work but available for and seeking employment, using national estimates."),
    "GDP per capita": ("NY.GDP.PCAP.CD", "current US dollars per person", "Gross domestic product divided by midyear population. It is not average wealth or household income."),
    "Crude birth rate": ("SP.DYN.CBRT.IN", "births per 1,000 people per year", "Annual number of live births per 1,000 people. This is a rate, not a birth count."),
    "Net migration": ("SM.POP.NETM", "people", "Net number of migrants over the period; it is not an emigration count."),
    "Intentional homicide rate": ("VC.IHR.PSRC.P5", "intentional homicides per 100,000 people", "Intentional homicides per 100,000 population."),
    "Consumer price inflation": ("FP.CPI.TOTL.ZG", "% annual change", "Annual percentage change in the consumer price index."),
    "Central government debt": ("GC.DOD.TOTL.GD.ZS", "% of GDP", "Central government debt expressed as a share of GDP."),
    "Total reserves including gold": ("FI.RES.TOTL.CD", "current US dollars", "Total reserves including monetary gold, valued in current US dollars."),
    "External balance on goods/services": ("NE.RSB.GNFS.ZS", "% of GDP", "External balance on goods and services as a share of GDP.")
}
CONTEXTS = {
    "Business": ("profit = a*x**2 + b*x + c", "time", "profit", "years", "thousand dollars", "Profit changes over time; the derivative is the rate profit changes."),
    "Engineering": ("position = a*x**2 + b*x + c", "time", "position", "seconds", "meters", "The first derivative is velocity and the second derivative is acceleration."),
    "Medical": ("concentration = a*exp(-k*x)", "time", "concentration", "hours", "mg/L", "The function models a changing concentration; its derivative is the rate of concentration change."),
    "Sports science": ("distance = a*x**2 + b*x", "time", "distance", "seconds", "meters", "For distance versus time, the derivative represents speed when the model is increasing."),
    "Biology": ("population = a*exp(k*x)", "time", "population", "days", "organisms", "The derivative describes population growth rate."),
    "Bioengineering": ("response = a*sin(b*x) + c", "time", "response", "seconds", "response units", "The function can represent a periodic biological or engineered response."),
    "Economic forecasting": ("output = a*x**2 + b*x + c", "time", "output", "years", "index units", "The derivative describes the rate of change in the economic indicator."),
    "Weather": ("temperature = a*sin(b*x) + c", "time", "temperature", "hours", "degrees", "The derivative describes how quickly temperature changes."),
    "Custom": ("f(x) = x**2", "input", "output", "input units", "output units", "Describe what the function represents in your own context.")
}
X = sp.Symbol("x", real=True)
SAFE_FUNCS = {"sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp,
              "log": sp.log, "sqrt": sp.sqrt, "Abs": sp.Abs, "abs": sp.Abs,
              "pi": sp.pi, "E": sp.E, "ln": sp.log}

def wb_fetch(country: str, code: str, start: int, end: int) -> tuple[pd.DataFrame, str]:
    url = f"https://api.worldbank.org/v2/country/{country}/indicator/{code}"
    params = {"date": f"{start}:{end}", "format": "json", "per_page": 20000}
    try:
        response = requests.get(url, params=params, timeout=25)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
            return pd.DataFrame(columns=["year", "value"]), "The API returned no observations for this selection."
        rows = [{"year": int(r["date"]), "value": r["value"]} for r in payload[1]
                if r.get("date") and r.get("value") is not None]
        frame = pd.DataFrame(rows, columns=["year", "value"]).sort_values("year")
        if frame.empty:
            return frame, "The API response contained no non-null observations in this period."
        return frame.reset_index(drop=True), f"Retrieved {len(frame)} non-null observations."
    except requests.Timeout:
        return pd.DataFrame(columns=["year", "value"]), "Request timed out. Try again later."
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        return pd.DataFrame(columns=["year", "value"]), f"World Bank request failed: {exc}"

@st.cache_data(ttl=3600, show_spinner=False)
def cached_wb_fetch(country: str, code: str, start: int, end: int):
    return wb_fetch(country, code, start, end)

def validate_table(frame: pd.DataFrame, indicator: str) -> tuple[pd.DataFrame, list[str]]:
    errors = []
    if not {"year", "value"}.issubset(frame.columns):
        return pd.DataFrame(columns=["year", "value"]), ["Table must contain year and value columns."]
    out = frame[["year", "value"]].copy()
    numeric_year = pd.to_numeric(out["year"], errors="coerce")
    numeric_val = pd.to_numeric(out["value"], errors="coerce")
    if numeric_year.isna().any() or ((numeric_year % 1) != 0).any():
        errors.append("Years must be whole-number integers.")
    if numeric_val.isna().any():
        errors.append("Values must be numeric; remove blank or non-numeric rows.")
    out["year"] = numeric_year
    out["value"] = numeric_val
    out = out.dropna()
    if out["year"].duplicated().any():
        errors.append("Duplicate years are not allowed. Keep one row per year.")
    if ((out["year"] < 1800) | (out["year"] > 2200)).any():
        errors.append("Years must be between 1800 and 2200.")
    if indicator in ["Life expectancy", "Unemployment (national estimate)", "Crude birth rate",
                     "Intentional homicide rate", "Central government debt", "Total reserves including gold"]:
        if (out["value"] < 0).any():
            errors.append(f"{indicator} cannot be negative.")
    if indicator == "Life expectancy" and ((out["value"] < 0) | (out["value"] > 125)).any():
        errors.append("Life expectancy values must be between 0 and 125 years.")
    if indicator == "Unemployment (national estimate)" and ((out["value"] < 0) | (out["value"] > 100)).any():
        errors.append("Unemployment percentage must be between 0 and 100.")
    if indicator == "Consumer price inflation" and (out["value"] < -100).any():
        errors.append("Inflation below -100% is not mathematically valid for this indicator.")
    return out.sort_values("year").reset_index(drop=True), errors

def fit_polynomial(frame: pd.DataFrame, degree: int):
    clean = frame.dropna().sort_values("year")
    if clean["year"].nunique() < degree + 1:
        raise ValueError(f"Degree {degree} requires at least {degree + 1} distinct years; only {clean['year'].nunique()} are available.")
    base = int(clean["year"].min())
    t = clean["year"].to_numpy(dtype=float) - base
    y = clean["value"].to_numpy(dtype=float)
    coeff = np.polyfit(t, y, degree)
    pred = np.polyval(coeff, t)
    residuals = y - pred
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = float(1 - np.sum(residuals**2) / ss_tot) if ss_tot > 0 else float("nan")
    mae = float(mean_absolute_error(y, pred))
    return coeff, base, pred, residuals, r2, mae

def poly_expression(coeff, base):
    terms = []
    degree = len(coeff) - 1
    for i, c in enumerate(coeff):
        power = degree - i
        terms.append(f"({c:.10g})*(year - {base})**{power}")
    return " + ".join(terms)

def safe_parse(expr: str):
    cleaned = expr.strip().replace("^", "**")
    if not cleaned:
        raise ValueError("Enter a function, such as x**2 + 3*x + 1.")
    # Permit only a narrow mathematical character set before SymPy parsing.
    if not re.fullmatch(r"[0-9xX+\-*/()., _a-zA-Z**]+", cleaned):
        raise ValueError("Use numbers, x, arithmetic operators, and supported functions only.")
    cleaned = cleaned.replace("X", "x")
    try:
        result = sp.sympify(cleaned, locals={"x": X, **SAFE_FUNCS}, evaluate=True)
        if result.free_symbols - {X}:
            raise ValueError("Only x may be used as a variable.")
        if result.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
            raise ValueError("The expression contains an undefined or infinite constant.")
        return result
    except Exception as exc:
        raise ValueError(f"Could not read that function. Check parentheses and function names. Details: {exc}")

def numerical_real(expr, values):
    fn = sp.lambdify(X, expr, modules=["numpy"])
    with np.errstate(all="ignore"):
        vals = np.asarray(fn(values), dtype=float)
    if vals.ndim == 0:
        vals = np.full_like(np.asarray(values, dtype=float), float(vals))
    return vals

def make_plot(data, coeff, base, degree, horizon, sample_step, single=None, batch=None,
              custom_expr=None, plot_derivatives=None, point=None, secant=None, tangent=None,
              title="Historical data and fitted model", y_label="Value", x_label="Year"):
    fig, ax = plt.subplots(figsize=(10, 5.3))
    if data is not None and len(data):
        ax.scatter(data["year"], data["value"], label="Observed data", color="#1769aa", s=25, zorder=4)
    if coeff is not None:
        lo = int(data["year"].min())
        hi = int(data["year"].max()) + int(horizon)
        smooth_x = np.linspace(lo, max(lo + 1, hi), max(400, (hi-lo)*4))
        smooth_y = np.polyval(coeff, smooth_x - base)
        in_mask = smooth_x <= int(data["year"].max())
        if in_mask.any():
            ax.plot(smooth_x[in_mask], smooth_y[in_mask], label="Fitted curve (historical range)", linewidth=2)
        if (~in_mask).any():
            ax.plot(smooth_x[~in_mask], smooth_y[~in_mask], "--", label="Extrapolated segment", linewidth=2)
        sampled = np.arange(lo, int(data["year"].max()) + 1, sample_step)
        ax.scatter(sampled, np.polyval(coeff, sampled-base), marker="x", label=f"Model markers every {sample_step} year(s)", s=30)
        ax.axvline(int(data["year"].max()), color="gray", linestyle=":", label="Historical boundary")
    if custom_expr is not None and plot_derivatives:
        xx = np.linspace(plot_derivatives["xmin"], plot_derivatives["xmax"], 500)
        yy = numerical_real(custom_expr, xx)
        ax.plot(xx, yy, label="Selected function", linewidth=2)
        for key, expr in plot_derivatives["curves"].items():
            if plot_derivatives["show"].get(key, False):
                vals = numerical_real(expr, xx)
                ax.plot(xx, vals, label=key)
        if point is not None:
            px, py = point
            if np.isfinite(py):
                ax.scatter([px], [py], color="black", zorder=6, label="Current point")
        if tangent is not None:
            px, py, slope = tangent
            tx = np.linspace(plot_derivatives["xmin"], plot_derivatives["xmax"], 100)
            ax.plot(tx, py+slope*(tx-px), ":", label="Tangent line")
        if secant is not None:
            x1, x2, y1, y2 = secant
            ax.plot([x1, x2], [y1, y2], "--", label="Secant line")
    if single is not None:
        ax.scatter([single[0]], [single[1]], marker="*", s=150, label="Single-year estimate", zorder=7)
    if batch is not None and len(batch):
        ax.scatter(batch["year"], batch["estimate"], marker="D", s=35, label="Batch estimates")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig

def make_excel(summary, working, results):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        pd.DataFrame([summary]).to_excel(writer, sheet_name="Summary", index=False)
        working.to_excel(writer, sheet_name="Working Data", index=False)
        results.to_excel(writer, sheet_name="Calculated Years", index=False)
        for sheet, frame in [("Summary", pd.DataFrame([summary])), ("Working Data", working), ("Calculated Years", results)]:
            ws = writer.sheets[sheet]
            ws.freeze_panes(1, 0)
            for idx, col in enumerate(frame.columns):
                ws.set_column(idx, idx, min(35, max(14, len(str(col))+3)))
    output.seek(0)
    return output.getvalue()

def make_pdf(summary, working, results, fig):
    output = io.BytesIO()
    doc = SimpleDocTemplate(output, pagesize=landscape(letter), rightMargin=28, leftMargin=28, topMargin=25, bottomMargin=25)
    styles = getSampleStyleSheet()
    story = [Paragraph("Hispanic Heritage Historical Data Explorer", styles["Title"]), Spacer(1, 8)]
    for k, v in summary.items():
        story.append(Paragraph(f"<b>{k}:</b> {str(v)}", styles["BodyText"]))
    story.append(Spacer(1, 8))
    img = io.BytesIO()
    fig.savefig(img, format="png", dpi=140, bbox_inches="tight")
    img.seek(0)
    story.append(Image(img, width=690, height=350))
    story.append(Paragraph("Working data preview", styles["Heading2"]))
    preview = working.head(18).round(4)
    table_data = [list(preview.columns)] + preview.astype(str).values.tolist()
    tbl = Table(table_data, repeatRows=1)
    tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),.3,colors.grey),("FONTSIZE",(0,0),(-1,-1),7)]))
    story.append(tbl)
    story.append(Paragraph("Calculated years", styles["Heading2"]))
    if not results.empty:
        r = results.head(20)
        tbl2 = Table([list(r.columns)] + r.astype(str).values.tolist(), repeatRows=1)
        tbl2.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),.3,colors.grey),("FONTSIZE",(0,0),(-1,-1),7)]))
        story.append(tbl2)
    story.append(Spacer(1, 8))
    story.append(Paragraph("Interpretation and limitations: Polynomial models describe patterns in the selected observations. Extrapolation is uncertain and does not establish historical causes. Model fit metrics are in-sample unless explicitly identified otherwise.", styles["BodyText"]))
    doc.build(story)
    output.seek(0)
    return output.getvalue()

st.title("🌎 Hispanic Heritage Historical Data Explorer 2.0")
st.caption("HISPANIC HERITAGE DATA LAB | Authentic historical data • mathematical modeling • calculus • simulation")
with st.expander("Project setup and research question", expanded=True):
    research_question = st.text_input("Research question", "How have selected economic and social indicators changed over time in Latin American countries?")
    c1, c2, c3 = st.columns(3)
    with c1:
        country_name = st.selectbox("Featured country", list(COUNTRIES), index=0)
    with c2:
        indicator_name = st.selectbox("Historical indicator", list(INDICATORS), index=0)
    with c3:
        course = st.selectbox("Mathematics pathway", ["Honors Algebra 2", "Precalculus", "Calculus"])
    st.info("Brazil is included as a Latin American comparison country. The app does not treat Brazil as a Spanish-speaking country; it is included for regional comparison.")

country_code = COUNTRIES[country_name]
indicator_code, units, definition = INDICATORS[indicator_name]
current_year = datetime.now(timezone.utc).year
start_year, end_year = current_year - 70, current_year - 1
with st.spinner("Retrieving World Bank observations..."):
    source_data, fetch_message = cached_wb_fetch(country_code, indicator_code, start_year, end_year)
retrieved_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
st.caption(f"Source: World Bank API | Indicator: {indicator_code} | Requested years: {start_year}–{end_year} | Retrieved: {retrieved_at}")
st.write(definition)
st.markdown(f"[World Bank indicator metadata](https://api.worldbank.org/v2/indicator/{indicator_code}?format=json) · [Country data API](https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}?date={start_year}:{end_year}&format=json)")
st.write(fetch_message)
if source_data.empty:
    st.warning("No usable source observations are available. Check the connection or choose another indicator/country. The app will not invent replacement data.")
    source_data = pd.DataFrame(columns=["year", "value"])

signature = f"{country_code}|{indicator_code}|{course}"
if st.session_state.get("source_signature") != signature:
    st.session_state.source_signature = signature
    st.session_state.working_data = source_data.copy()
    st.session_state.user_modified = False
    st.session_state.calculated_results = pd.DataFrame()
    st.session_state.single_result = None

st.header("1. Historical data and validation")
if not source_data.empty:
    st.write(f"**Actual coverage:** {int(source_data.year.min())}–{int(source_data.year.max())} · **Non-null observations:** {len(source_data)}")
    st.write(f"**Source mean:** {source_data.value.mean():.5g} {units}")
else:
    st.write("**Actual coverage:** none · **Non-null observations:** 0")
st.caption("Edit the working copy below. Original API data remains separate. Edited data is a user-modified scenario, not an official World Bank series.")
edited = st.data_editor(st.session_state.working_data, num_rows="dynamic", use_container_width=True, key=f"editor_{signature}")
vdata, validation_errors = validate_table(edited, indicator_name)
if validation_errors:
    for err in validation_errors:
        st.error(err)
else:
    st.session_state.working_data = vdata
    st.session_state.user_modified = not vdata.equals(source_data.reset_index(drop=True))
if st.session_state.user_modified:
    st.warning("USER-MODIFIED SCENARIO: calculations and exports use your edited working data.")
if st.button("Reset to source data"):
    st.session_state.working_data = source_data.copy()
    st.session_state.user_modified = False
    st.session_state.calculated_results = pd.DataFrame()
    st.session_state.single_result = None
    st.rerun()

st.header("2. Historical polynomial model")
a, b, c = st.columns(3)
with a:
    degree = st.selectbox("Polynomial degree", list(range(3, 9)), index=0)
with b:
    sample_step = st.slider("Graph sample increment (years)", 1, 10, 2)
with c:
    horizon = st.slider("Forward display horizon (years)", 0, 50, 10)
model = None
fit_error = None
if not validation_errors and len(vdata):
    try:
        model = fit_polynomial(vdata, degree)
    except Exception as exc:
        fit_error = str(exc)
if fit_error:
    st.error(f"Model cannot be fitted: {fit_error}")
if model:
    coeff, base, fitted, residuals, r2, mae = model
    model_years = vdata.year.to_numpy(dtype=int)
    equation = poly_expression(coeff, base)
    metrics = {"Base year": base, "Degree": degree, "R-squared (in-sample)": f"{r2:.6f}",
               "MAE (in-sample)": f"{mae:.6g} {units}", "Observation count": len(vdata),
               "Historical range": f"{model_years.min()}–{model_years.max()}",
               "Mean": f"{vdata.value.mean():.6g} {units}"}
    st.code(f"t = year - {base}\nŷ = {equation}", language="text")
    mcols = st.columns(4)
    mcols[0].metric("R² · in-sample", f"{r2:.4f}" if np.isfinite(r2) else "undefined")
    mcols[1].metric("MAE · in-sample", f"{mae:.4g}")
    mcols[2].metric("Observations", len(vdata))
    mcols[3].metric("Historical mean", f"{vdata.value.mean():.4g}")
    with st.expander("Residuals and holdout comparison"):
        residual_table = pd.DataFrame({"year": model_years, "actual": vdata.value, "fitted_at_observed_years": fitted, "residual": residuals})
        st.dataframe(residual_table, use_container_width=True)
        if len(vdata) >= max(2*(degree+1), 12):
            split = max(degree+1, int(len(vdata)*0.8))
            train = vdata.iloc[:split]
            test = vdata.iloc[split:]
            if len(test):
                try:
                    hc, hb, *_ = fit_polynomial(train, degree)
                    holdout_pred = np.polyval(hc, test.year.to_numpy()-hb)
                    holdout_mae = float(np.mean(np.abs(test.value.to_numpy()-holdout_pred)))
                    st.write(f"Chronological holdout MAE (separate from in-sample metrics): **{holdout_mae:.5g} {units}** on {len(test)} later observations.")
                except Exception as exc:
                    st.caption(f"Holdout comparison unavailable: {exc}")
        else:
            st.caption("Not enough observations for a meaningful chronological holdout at this degree.")
    years_for_plot = vdata.year.to_numpy()
    fig_hist = make_plot(vdata, coeff, base, degree, horizon, sample_step, title=f"{country_name}: {indicator_name}", y_label=units)
    st.pyplot(fig_hist, use_container_width=True)
    st.caption("The sample increment changes the model-marker spacing. The smooth fitted curve is plotted separately. Dashed portions are extrapolations, not observed data.")
else:
    coeff = base = None
    equation = "Unavailable"
    metrics = {"Model": "Unavailable"}
    fig_hist, _ = plt.subplots()

# Clear stored calculations whenever the model inputs change.
model_signature = (
    country_code, indicator_code, degree,
    tuple((int(r.year), float(r.value)) for r in vdata.itertuples()) if len(vdata) else ()
)
if st.session_state.get("model_signature") != model_signature:
    st.session_state.model_signature = model_signature
    st.session_state.calculated_results = pd.DataFrame()
    st.session_state.single_result = None

st.header("3. Calculate individual and batch years")
single_year = st.number_input("Year to calculate", min_value=1950, max_value=2100, value=2025, step=1)
if st.button("Calculate Value", type="primary", disabled=model is None):
    val = float(np.polyval(coeff, single_year-base))
    st.session_state.single_result = {"year": int(single_year), "estimate": val, "within": int(vdata.year.min()) <= single_year <= int(vdata.year.max())}
if st.session_state.get("single_result") is not None and model:
    sr = st.session_state.single_result
    classification = "within fitted historical range" if sr["within"] else "outside fitted historical range (extrapolation)"
    st.success(f"Year {sr['year']}: **{sr['estimate']:.7g} {units}** — {classification}. This is a model estimate, not necessarily an actual observation.")
    if sr["year"] in set(vdata.year.astype(int)):
        st.info("This year has an actual observation in the working data. The fitted estimate and actual observed value are distinct.")
    else:
        st.caption("No actual working-data observation is recorded for this year.")

with st.expander("Batch year calculations", expanded=True):
    batch_mode = st.radio("Batch input type", ["Year range", "Custom years"], horizontal=True)
    batch_years = []
    if batch_mode == "Year range":
        bc1, bc2, bc3 = st.columns(3)
        with bc1: first_year = st.number_input("Start year", 1950, 2100, 2000, key="batch_start")
        with bc2: last_year = st.number_input("End year", 1950, 2100, 2030, key="batch_end")
        with bc3: batch_step = st.slider("Range step", 1, 10, 1)
        if first_year <= last_year:
            batch_years = list(range(int(first_year), int(last_year)+1, int(batch_step)))
        else:
            st.error("Start year must be less than or equal to end year.")
    else:
        raw_years = st.text_input("Enter years separated by commas", "2000, 2010, 2020, 2030")
        for token in raw_years.split(","):
            token = token.strip()
            if not token: continue
            try:
                val_year = float(token)
                if not val_year.is_integer() or not 1950 <= val_year <= 2100:
                    st.warning(f"Invalid year '{token}': use whole years from 1950 to 2100.")
                else:
                    batch_years.append(int(val_year))
            except ValueError:
                st.warning(f"Invalid entry '{token}': enter comma-separated whole years.")
        batch_years = list(dict.fromkeys(batch_years))
    if st.button("Calculate batch", disabled=model is None):
        if batch_years:
            results = pd.DataFrame({"year": batch_years, "estimate": [float(np.polyval(coeff, y-base)) for y in batch_years]})
            results["units"] = units
            results["classification"] = results.year.apply(lambda y: "within fitted range" if vdata.year.min() <= y <= vdata.year.max() else "outside fitted range (extrapolation)")
            st.session_state.calculated_results = results
        else:
            st.error("No valid years were entered.")
    batch_results = st.session_state.get("calculated_results", pd.DataFrame())
    if not batch_results.empty:
        st.dataframe(batch_results, use_container_width=True)

st.header("4. Country comparison")
comparison_names = st.multiselect("Compare with countries (same indicator)", [n for n in COUNTRIES if n != country_name], default=["Brazil", "Argentina"] if country_name == "Mexico" else ["Mexico"])
comparison_frames = []
for name in comparison_names:
    cc, msg = cached_wb_fetch(COUNTRIES[name], indicator_code, start_year, end_year)
    if not cc.empty:
        cc["country"] = name
        comparison_frames.append(cc)
if comparison_frames:
    comp_fig, comp_ax = plt.subplots(figsize=(10, 4.5))
    comp_ax.plot(vdata.year, vdata.value, label=f"{country_name} (working data)", linewidth=2)
    for cf in comparison_frames:
        comp_ax.plot(cf.year, cf.value, marker=".", label=cf.country)
    comp_ax.set_title(f"{indicator_name}: country comparison")
    comp_ax.set_xlabel("Year"); comp_ax.set_ylabel(units); comp_ax.grid(alpha=.25); comp_ax.legend()
    st.pyplot(comp_fig, use_container_width=True)
    st.caption("Countries may have different start/end years and missing observations. Gaps are not filled or treated as zero.")
else:
    st.info("Select comparison countries to display available same-indicator series.")

st.header("5. Context and function laboratory")
context = st.selectbox("Choose context", list(CONTEXTS))
suggestion, default_x, default_y, default_xu, default_yu, default_meaning = CONTEXTS[context]
st.caption("Brazil is a regional comparison rather than a Spanish-speaking country. Context functions are educational simulations, not claims about the historical indicator.")
context_description = st.text_input("Context description", default_meaning)
x_name = st.text_input("Independent variable name", default_x)
y_name = st.text_input("Dependent variable name", default_y)
x_units = st.text_input("Independent variable units", default_xu)
y_units = st.text_input("Dependent variable units", default_yu)
function_source = st.radio("Function source", ["Use historical curve of best fit", "Use suggested function", "Build parameterized function", "Enter my own function"], horizontal=False)
a_param = st.number_input("Parameter a", value=1.0, format="%.4f")
b_param = st.number_input("Parameter b", value=0.5, format="%.4f")
c_param = st.number_input("Parameter c", value=0.0, format="%.4f")
if function_source == "Use historical curve of best fit":
    if model:
        t_sym = X
        selected_expr = sum(sp.Float(float(co)) * t_sym**(len(coeff)-i-1) for i, co in enumerate(coeff))
        # This model uses t = year - base; x is explicitly the year offset.
        function_label = f"Historical fit in t = year - {base}"
        context_xmin, context_xmax = 0, max(1, int(vdata.year.max()-base))
    else:
        selected_expr = None
        function_label = "Historical fit unavailable"
        context_xmin, context_xmax = 0, 10
elif function_source == "Use suggested function":
    if context in ["Medical", "Biology"]:
        selected_expr = sp.Float(a_param) * sp.exp(-sp.Float(b_param)*X) if context == "Medical" else sp.Float(a_param)*sp.exp(sp.Float(b_param)*X)
    elif context in ["Weather", "Bioengineering"]:
        selected_expr = sp.Float(a_param)*sp.sin(sp.Float(b_param)*X)+sp.Float(c_param)
    elif context == "Engineering":
        selected_expr = sp.Float(a_param)*X**2+sp.Float(b_param)*X+sp.Float(c_param)
    else:
        selected_expr = sp.Float(a_param)*X**2+sp.Float(b_param)*X+sp.Float(c_param)
    function_label = "Suggested context function"
    context_xmin, context_xmax = 0, 20
elif function_source == "Build parameterized function":
    family = st.selectbox("Function family", ["Polynomial (quadratic)", "Rational", "Exponential", "Logarithmic", "Trigonometric"])
    if family == "Polynomial (quadratic)": selected_expr = a_param*X**2+b_param*X+c_param
    elif family == "Rational": selected_expr = (a_param*X+b_param)/(X-c_param) if c_param != 0 else (a_param*X+b_param)/(X-1)
    elif family == "Exponential": selected_expr = a_param*sp.exp(b_param*X)+c_param
    elif family == "Logarithmic": selected_expr = a_param*sp.log(X)+b_param
    else: selected_expr = a_param*sp.sin(b_param*X)+c_param
    function_label = f"Parameterized {family.lower()} function"
    context_xmin, context_xmax = (0.1, 20) if family == "Logarithmic" else (-10, 10)
else:
    user_expr = st.text_input("Enter f(x)", "x**3 - 3*x")
    try:
        selected_expr = safe_parse(user_expr)
        function_label = "Student-entered function"
    except ValueError as exc:
        selected_expr = None
        function_label = "Invalid function"
        st.error(str(exc))
    context_xmin, context_xmax = -10, 10

if selected_expr is not None:
    d1 = sp.diff(selected_expr, X)
    d2 = sp.diff(d1, X)
    st.latex(r"f(x) = " + sp.latex(selected_expr))
    if course in ["Precalculus", "Calculus"]:
        st.latex(r"f'(x) = " + sp.latex(d1))
    if course == "Calculus":
        st.latex(r"f''(x) = " + sp.latex(d2))
    st.caption("Symbolic derivatives are exact for the entered expression where defined. Numerical graph locations and root approximations are labeled as approximations.")
    xmin = st.number_input("Analysis interval minimum", value=float(context_xmin))
    xmax = st.number_input("Analysis interval maximum", value=float(context_xmax))
    if xmin >= xmax:
        st.error("Analysis interval minimum must be less than maximum.")
    else:
        x_point = st.slider("Current input value", min_value=float(xmin), max_value=float(xmax), value=float((xmin+xmax)/2))
        left, right = st.columns(2)
        with left:
            sec_x1 = st.slider("Average-rate interval start", float(xmin), float(xmax), float(xmin))
        with right:
            sec_x2 = st.slider("Average-rate interval end", float(xmin), float(xmax), float(xmax))
        show_function = st.checkbox("Show function f(x)", True)
        show_first = st.checkbox("Show first derivative f'(x)", course != "Honors Algebra 2")
        show_second = st.checkbox("Show second derivative f''(x)", course == "Calculus")
        try:
            y_point = float(sp.N(selected_expr.subs(X, x_point)))
            slope_point = float(sp.N(d1.subs(X, x_point)))
            y1 = float(sp.N(selected_expr.subs(X, sec_x1)))
            y2 = float(sp.N(selected_expr.subs(X, sec_x2)))
            avg_rate = (y2-y1)/(sec_x2-sec_x1) if sec_x2 != sec_x1 else float("nan")
            deriv_values = {"xmin": xmin, "xmax": xmax,
                "curves": {"f'(x)": d1, "f''(x)": d2},
                "show": {"f'(x)": show_first, "f''(x)": show_second}}
            sim_fig, sim_ax = plt.subplots(figsize=(10, 5))
            xx = np.linspace(xmin, xmax, 800)
            yy = numerical_real(selected_expr, xx)
            sim_ax.plot(xx, yy, label="f(x)", linewidth=2)
            if show_first: sim_ax.plot(xx, numerical_real(d1, xx), label="f'(x)")
            if show_second: sim_ax.plot(xx, numerical_real(d2, xx), label="f''(x)")
            if np.isfinite(y_point):
                sim_ax.scatter([x_point], [y_point], color="black", label="Current point", zorder=5)
            if np.isfinite(slope_point) and np.isfinite(y_point):
                tx = np.linspace(xmin, xmax, 100)
                sim_ax.plot(tx, y_point+slope_point*(tx-x_point), ":", label="Tangent line")
            if sec_x1 != sec_x2 and np.isfinite(y1) and np.isfinite(y2):
                sim_ax.plot([sec_x1, sec_x2], [y1, y2], "--", label="Secant line")
            sim_ax.axhline(0, color="gray", linewidth=.6); sim_ax.grid(alpha=.25)
            sim_ax.set_xlabel(f"{x_name} ({x_units})"); sim_ax.set_ylabel(f"{y_name} ({y_units})")
            sim_ax.legend(); sim_fig.tight_layout()
            st.pyplot(sim_fig, use_container_width=True)
            v1, v2, v3 = st.columns(3)
            v1.metric("Function value", f"{y_point:.6g} {y_units}")
            v2.metric("Instantaneous rate", f"{slope_point:.6g} {y_units}/{x_units}")
            v3.metric("Average rate", f"{avg_rate:.6g} {y_units}/{x_units}" if np.isfinite(avg_rate) else "Undefined")
            st.write(f"At x = {x_point:g}, the model output is {y_point:.6g} {y_units}. The derivative is {slope_point:.6g} {y_units} per {x_units}; it describes how quickly the output changes at that point.")
            st.write(f"From x = {sec_x1:g} to x = {sec_x2:g}, the average rate of change is {avg_rate:.6g} {y_units} per {x_units}." if np.isfinite(avg_rate) else "The average rate is undefined because both interval endpoints are equal.")
            st.caption(context_description)
            if context == "Engineering":
                st.info("Position model: f(x) is position, f'(x) is velocity, and f''(x) is acceleration, with corresponding units.")
            elif context in ["Business", "Economic forecasting"]:
                st.info("The first derivative describes growth or decline per input unit. The second derivative describes whether that rate is increasing or decreasing.")
            elif context == "Medical":
                st.info("The function can represent concentration; its first derivative is the concentration change per hour. This is a classroom model, not medical advice.")
            # Numerical calculus analysis: candidates are approximations, then checked.
            if course == "Calculus":
                st.subheader("Calculus analysis")
                roots1 = []
                roots2 = []
                for expr, target in [(d1, roots1), (d2, roots2)]:
                    try:
                        for root in sp.calculus.util.continuous_domain(expr, X, sp.Interval(float(xmin), float(xmax))).boundary:
                            pass
                        for root in sp.nroots(sp.together(expr).as_numer_denom()[0], maxsteps=100):
                            if abs(complex(root).imag) < 1e-8 and xmin <= float(sp.re(root)) <= xmax:
                                target.append(float(sp.re(root)))
                    except Exception:
                        pass
                roots1 = sorted(set(round(r, 6) for r in roots1))
                roots2 = sorted(set(round(r, 6) for r in roots2))
                critical = []
                for r in roots1:
                    try:
                        eps = max(1e-4, (xmax-xmin)*1e-5)
                        yl = float(sp.N(d1.subs(X, r-eps))); yr = float(sp.N(d1.subs(X, r+eps)))
                        if yl*yr < 0:
                            typ = "local maximum" if yl > 0 and yr < 0 else "local minimum"
                            critical.append((r, typ))
                    except Exception:
                        critical.append((r, "stationary candidate; classification uncertain"))
                st.write("**Critical-point approximations:**", critical if critical else "No classified critical points found in the interval.")
                st.write("**Zeros of f''(x) to investigate:**", roots2 if roots2 else "No real numerical candidates found.")
                verified_inflections = []
                for r in roots2:
                    eps = max(1e-4, (xmax-xmin)*1e-5)
                    try:
                        l = float(sp.N(d2.subs(X, r-eps))); rr = float(sp.N(d2.subs(X, r+eps)))
                        if l*rr < 0: verified_inflections.append(r)
                    except Exception: pass
                st.write("**Inflection-point approximations (concavity changes checked):**", verified_inflections if verified_inflections else "No verified concavity changes found.")
                st.write("**Concavity:** determined by the sign of f''(x) on intervals where the function is defined.")
                st.write("**Domain/discontinuities:** inspect denominators and logarithm arguments in the entered expression; graph gaps represent undefined regions.")
                st.caption("Numerical root-finding may miss repeated or difficult roots. A zero of f' is not automatically an extremum, and a zero of f'' is not automatically an inflection point.")
            elif course == "Precalculus":
                st.subheader("Precalculus interpretation")
                st.write("Compare average rates on different intervals to see how the rate changes. The degree and coefficients control end behavior, curvature, and possible turning points.")
            else:
                st.subheader("Honors Algebra 2 interpretation")
                st.write(f"Domain: the selected analysis interval {xmin:g} to {xmax:g} {x_units}, subject to any function restrictions. Range: inspect the plotted outputs over this interval. End behavior is determined by the leading term for polynomials. Average rate of change describes the slope between two points.")
            # Function-specific discontinuity and domain notices.
            denom = sp.denom(sp.together(selected_expr))
            if denom != 1:
                st.warning(f"Rational expression: denominator is {sp.latex(denom)}. Values making it zero are outside the domain; graph branches are not joined across those gaps.")
            if selected_expr.has(sp.log):
                st.warning("Logarithmic expression: its argument must be positive.")
        except Exception as exc:
            st.error(f"Could not evaluate this function at the selected point or interval. Check domain restrictions. Details: {exc}")

st.header("6. Interactive playback")
st.write("Playback advances the selected input point through the analysis interval. It is a visual teaching aid, not a physical prediction.")
play_cols = st.columns(4)
if "play_x" not in st.session_state:
    st.session_state.play_x = 0.0
if "is_playing" not in st.session_state:
    st.session_state.is_playing = False
with play_cols[0]:
    if st.button("▶ Play"):
        st.session_state.is_playing = True
with play_cols[1]:
    if st.button("⏸ Pause"):
        st.session_state.is_playing = False
with play_cols[2]:
    if st.button("↺ Reset"):
        st.session_state.is_playing = False
        st.session_state.play_x = 0.0
with play_cols[3]:
    play_speed = st.selectbox("Playback speed", ["Slow", "Normal", "Fast"], index=1)
if st.session_state.is_playing and selected_expr is not None:
    if st_autorefresh is not None:
        st_autorefresh(interval={"Slow":1000,"Normal":500,"Fast":200}[play_speed], key="function_playback")
        st.session_state.play_x += {"Slow":0.1,"Normal":0.25,"Fast":0.5}[play_speed]
        if st.session_state.play_x > float(context_xmax):
            st.session_state.play_x = float(context_xmin)
        st.info(f"Playback input: {st.session_state.play_x:.2f} {x_units}. Pause or reset using the controls above.")
    else:
        st.warning("Automatic refresh is unavailable. Install streamlit-autorefresh; use the point slider for manual exploration.")
if selected_expr is not None:
    st.session_state.play_x = float(st.session_state.play_x)
    st.slider("Playback input position", min_value=float(context_xmin), max_value=float(context_xmax),
              value=min(max(st.session_state.play_x, float(context_xmin)), float(context_xmax)),
              key="playback_manual")

st.header("7. Export your project")
status = "User-modified scenario" if st.session_state.user_modified else "Source data (unmodified)" 
summary = {"Research question": research_question, "Country": country_name, "Indicator": indicator_name,
           "Indicator code": indicator_code, "Units": units, "Mathematics pathway": course,
           "Data status": status, "Retrieved": retrieved_at, "Actual coverage": (f"{int(vdata.year.min())}–{int(vdata.year.max())}" if len(vdata) else "none"),
           "Observation count": len(vdata), "Function": str(selected_expr) if selected_expr is not None else "unavailable",
           "Historical polynomial equation": equation, **metrics}
results_export = st.session_state.get("calculated_results", pd.DataFrame())
csv_bytes = vdata.to_csv(index=False).encode("utf-8")
excel_bytes = make_excel(summary, vdata, results_export)
pdf_bytes = make_pdf(summary, vdata, results_export, fig_hist)
e1, e2, e3 = st.columns(3)
e1.download_button("Download working data CSV", csv_bytes, file_name="hispanic_heritage_working_data.csv", mime="text/csv")
e2.download_button("Download Excel workbook", excel_bytes, file_name="hispanic_heritage_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
e3.download_button("Download PDF report", pdf_bytes, file_name="hispanic_heritage_report.pdf", mime="application/pdf")

st.header("8. Printer-friendly report")
printer = st.checkbox("Printer-friendly view")
if printer:
    st.markdown("---")
    st.subheader("Project report")
    st.write(f"**Research question:** {research_question}")
    st.write(f"**Country / indicator:** {country_name} / {indicator_name} ({units})")
    st.write(f"**Mathematics pathway:** {course}")
    st.write(f"**Data status:** {status}")
    st.write(f"**Model:** {equation}")
    st.write(f"**Limitations:** In-sample fit does not prove causation. Polynomial extrapolations can become unrealistic. Missing years remain missing; no observations are fabricated.")
    st.dataframe(vdata.head(30), use_container_width=True)

st.divider()
st.caption("Data source: World Bank Open Data API. Retrieval date is displayed above. Historical data may be revised by the source. This classroom tool is not a substitute for expert statistical, medical, or engineering analysis.")
