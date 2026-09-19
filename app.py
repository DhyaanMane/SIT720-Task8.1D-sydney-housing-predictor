"""
SIT720 Task 8.1D — Sydney Housing Price Prediction
Streamlit decision-support prototype.

Run:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path("model/gb_model.joblib")
META_PATH  = Path("model/model_meta.json")

st.set_page_config(page_title="Sydney Housing Price Predictor",
                   page_icon="🏠", layout="centered")


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    return model, meta


KEYWORDS = {
    'kw_renovated'  : r'renovat|refurbish|updated|modern kitchen',
    'kw_original'   : r'original condition|needs work|renovator|potential to|as is',
    'kw_pool'       : r'\bpool\b',
    'kw_view'       : r'\bview[s]?\b|outlook|district view',
    'kw_transport'  : r'\bstation\b|\bmetro\b|light rail|bus stop',
    'kw_school'     : r'\bschool\b|catchment|education',
    'kw_luxury'     : r'luxur|prestig|premium|architect|bespoke',
    'kw_investment' : r'investor|rental return|tenant|yield',
    'kw_granny_flat': r'granny flat|dual occupancy|duplex potential',
    'kw_north'      : r'north[- ]?facing|northerly aspect',
}


def build_features(rec: dict) -> pd.DataFrame:
    """Reproduce the notebook's feature engineering for a single property."""
    desc = (rec.get("description") or "").lower()
    row = {
        "bedrooms": rec["bedrooms"],
        "bathrooms": rec["bathrooms"],
        "car_spaces": rec["car_spaces"],
        "size_m2": rec["size_m2"],
        "size_is_land": 1 if rec["property_type"] in ("House", "Townhouse") else 0,
        "total_rooms": rec["bedrooms"] + rec["bathrooms"],
        "bath_per_bed": rec["bathrooms"] / rec["bedrooms"] if rec["bedrooms"] else np.nan,
        "sale_month": rec["sale_month"],
        "desc_word_count": len(desc.split()),
        "desc_exclaims": desc.count("!"),
        "suburb": rec["suburb"],
        "property_type": rec["property_type"],
        "sale_method": rec["sale_method"],
    }
    for name, pat in KEYWORDS.items():
        row[name] = int(bool(pd.Series([desc]).str.contains(pat, regex=True).iloc[0]))
    return pd.DataFrame([row])


def predict(model, meta, rec):
    X = build_features(rec)[meta["features"]]
    log_pred = model.predict(X)[0]
    point = float(np.exp(log_pred))
    # Interval from the cross-validated residual SD in log space
    sd = meta["cv_resid_sd_log"]
    return point, float(np.exp(log_pred - 1.96*sd)), float(np.exp(log_pred + 1.96*sd))


st.title("🏠 Sydney Housing Price Predictor")
st.caption("Gradient boosting model trained on 102 sold listings in Chatswood, "
           "Parramatta and Blacktown. Estimates only — not a valuation.")

try:
    model, meta = load_model()
except FileNotFoundError:
    st.error("Model files not found. Run the notebook's export cell to create "
             "`model/gb_model.joblib` and `model/model_meta.json`.")
    st.stop()

tab_single, tab_batch = st.tabs(["Single property", "Batch upload (CSV)"])

with tab_single:
    c1, c2 = st.columns(2)
    with c1:
        suburb = st.selectbox("Suburb", ["Chatswood", "Parramatta", "Blacktown"])
        ptype = st.selectbox("Property type", ["House", "Apartment", "Townhouse"])
        beds = st.number_input("Bedrooms", 1, 8, 3)
        baths = st.number_input("Bathrooms", 1, 6, 2)
    with c2:
        cars = st.number_input("Car spaces", 0, 6, 1)
        size_label = "Land size (m²)" if ptype in ("House", "Townhouse") else "Internal area (m²)"
        size = st.number_input(size_label, 30.0, 3000.0, 550.0 if ptype == "House" else 95.0)
        method = st.selectbox("Sale method", ["Private treaty", "Auction"])
        month = st.slider("Sale month", 1, 12, 6)

    desc = st.text_area("Agent description (optional)",
                        placeholder="Paste the listing description — keyword features "
                                    "are extracted from it.", height=110)

    if st.button("Estimate price", type="primary", use_container_width=True):
        rec = dict(suburb=suburb, property_type=ptype, bedrooms=beds, bathrooms=baths,
                   car_spaces=cars, size_m2=size, sale_method=method,
                   sale_month=month, description=desc)
        point, lo, hi = predict(model, meta, rec)

        st.metric("Estimated sale price", f"${point:,.0f}")
        st.caption(f"95% interval: ${lo:,.0f} – ${hi:,.0f}")
        st.progress(min(1.0, point / 6_500_000))

        st.warning(
            f"**Interpret with caution.** Cross-validated error is "
            f"{meta['cv_mape']:.1f}% on average, rising to ~18% for Chatswood houses "
            "and ~31% for townhouses. The model has no access to zoning, frontage or "
            "redevelopment potential, so properties marketed as development sites are "
            "systematically under-valued."
        )

with tab_batch:
    st.write("Upload a CSV with columns: `suburb`, `property_type`, `bedrooms`, "
             "`bathrooms`, `car_spaces`, `size_m2`, `sale_method`, `sale_month`, "
             "and optionally `description`.")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        raw = pd.read_csv(up)
        required = ['suburb','property_type','bedrooms','bathrooms',
                    'car_spaces','size_m2','sale_method','sale_month']
        missing = [c for c in required if c not in raw.columns]

        if missing:
            st.error(f"Missing required column(s): {', '.join(missing)}. "
                     f"Found: {', '.join(raw.columns)}")
        else:
            if 'description' not in raw.columns:
                raw['description'] = ''
            st.write(f"{len(raw)} rows loaded.")
            out = raw.copy()
            preds = [predict(model, meta, row.to_dict()) for _, row in raw.iterrows()]
            out['predicted_price'] = [p[0] for p in preds]
            out['lower_95'] = [p[1] for p in preds]
            out['upper_95'] = [p[2] for p in preds]
            st.dataframe(out)
            st.download_button("Download predictions",
                               out.to_csv(index=False).encode(),
                               "predictions.csv", "text/csv")

st.divider()
st.caption("SIT720 Machine Learning Task 8.1D · Deakin University · Dhyaan Mane")
