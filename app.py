import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re
from datetime import datetime
from pathlib import Path
from sklearn.inspection import permutation_importance
import shap
from typing import List, Tuple



# ------------------------------------------------------
# Page
# ------------------------------------------------------
st.set_page_config(page_title="Canadian Energy Predictor", layout="wide")

# ------------------------------------------------------
# Theme / palettes
# ------------------------------------------------------
PALETTE_SEQ = ["#2563EB","#F97316","#22C55E","#A855F7","#EC4899","#EAB308","#06B6D4","#84CC16","#14B8A6","#FB7185"]
PALETTE_CONT = "Turbo"
ACCENT_PRED, ACCENT_TREN, ACCENT_AI = "#F59E0B", "#EF4444", "#2563EB"
PLOTLY_TEMPLATE = "plotly_white"

# ------------------------------------------------------
# Encodings / maps
# ------------------------------------------------------
REGION_MAP   = {"Urban":0,"Suburban":1,"Rural":2}
CLIMATE_MAP  = {"Continental":0,"Cold":1,"Mild":2,"Humid":3,"Arctic":4,"Subarctic":5}
DWELLING_MAP = {"Single Detached":0,"Apartment":1,"Row House":2,"Semi Detached":2,"Mobile Home":3,"Other Movable":3}
INCOME_MAP   = {"$20k - $40k":30000,"$40k - $60k":50000,"$60k - $80k":70000,"$80k - $100k":90000,"$100k - $120k":110000,"$120k - $150k":135000,"$150k+":160000}

PROVINCE_INDEX = {
    "Alberta":0,"British Columbia":1,"Manitoba":2,"New Brunswick":3,"Newfoundland and Labrador":4,
    "Nova Scotia":5,"Ontario":6,"Prince Edward Island":7,"Quebec":8,"Saskatchewan":9,
    "Northwest Territories":10,"Nunavut":11,"Yukon":12
}

# ---- Preset profiles ----
PRESET_PROFILES = {
    "— None —": {},
    "Toronto condo": {
        "prov_sel": "Ontario",
        "region_sel": "Urban",
        "climate_sel": "Humid",
        "dwelling_sel": "Apartment",
        "floor_sel": 650,
        "occ_sel": 2,
        "income_sel": "$60k - $80k",
    },
    "Prairie detached": {
        "prov_sel": "Alberta",
        "region_sel": "Suburban",
        "climate_sel": "Cold",
        "dwelling_sel": "Single Detached",
        "floor_sel": 2200,
        "occ_sel": 3,
        "income_sel": "$100k - $120k",
    },
    "Coastal apartment": {
        "prov_sel": "British Columbia",
        "region_sel": "Urban",
        "climate_sel": "Mild",
        "dwelling_sel": "Apartment",
        "floor_sel": 800,
        "occ_sel": 2,
        "income_sel": "$80k - $100k",
    },
    "Atlantic semi-detached": {
        "prov_sel": "Nova Scotia",
        "region_sel": "Suburban",
        "climate_sel": "Humid",
        "dwelling_sel": "Semi Detached",
        "floor_sel": 1400,
        "occ_sel": 3,
        "income_sel": "$60k - $80k",
    },
}

# ------------------------------------------------------
# Session
# ------------------------------------------------------
if "last_pred" not in st.session_state:
    st.session_state["last_pred"] = None
if "ai_history" not in st.session_state:
    st.session_state["ai_history"] = []

# ------------------------------------------------------
# Fallback medians 
# ------------------------------------------------------
NATIONAL_MEDIAN_COST = 1600.0
NATIONAL_MEDIAN_USE  = 15000.0

PROVINCE_MEDIAN_COST = {
    "Alberta":1675,"British Columbia":1500,"Manitoba":1550,"New Brunswick":1700,"Newfoundland and Labrador":1900,
    "Northwest Territories":2600,"Nova Scotia":1800,"Nunavut":3000,"Ontario":1650,"Prince Edward Island":1750,
    "Quebec":1300,"Saskatchewan":1650,"Yukon":2400
}
PROVINCE_MEDIAN_USE = {
    "Alberta":14500,"British Columbia":13500,"Manitoba":16000,"New Brunswick":15500,"Newfoundland and Labrador":17000,
    "Northwest Territories":18000,"Nova Scotia":16000,"Nunavut":17000,"Ontario":14000,"Prince Edward Island":16000,
    "Quebec":17000,"Saskatchewan":17000,"Yukon":17500
}
CLIMATE_MEDIAN_COST = {"Continental":1600,"Cold":1850,"Mild":1450,"Humid":1600,"Arctic":2600,"Subarctic":2100}
CLIMATE_MEDIAN_USE  = {"Continental":15000,"Cold":17000,"Mild":13500,"Humid":15000,"Arctic":18000,"Subarctic":17500}
INCOME_MEDIAN_COST  = {"$20k - $40k":1400,"$40k - $60k":1500,"$60k - $80k":1600,"$80k - $100k":1700,"$100k - $120k":1800,"$120k - $150k":1900,"$150k+":2000}

def _fallback(value, alt):
    if value is None:
        return alt
    try:
        v = float(value)
    except Exception:
        return alt
    if not np.isfinite(v) or v <= 1:
        return alt
    return v

# ------------------------------------------------------
# Models
# ------------------------------------------------------
energy_use_model  = joblib.load("model/energy_use_model.pkl")
energy_cost_model = joblib.load("model/energy_cost_model.pkl")

# ------------------------------------------------------
# Data loader
# ------------------------------------------------------
DATA_PATHS = [
    "data/final_dashboard_energy_dataset.xlsx",
    "data/final_dashboard_energy_dataset (1).xlsx",
    "final_dashboard_energy_dataset.xlsx",
    "final_dashboard_energy_dataset (1).xlsx",
    "/mnt/data/final_dashboard_energy_dataset.xlsx",
]

@st.cache_data(show_spinner=False)
def load_data():
    last_err = None
    for p in DATA_PATHS:
        if Path(p).exists():
            try:
                return pd.read_excel(p, engine="openpyxl"), p
            except Exception as e:
                last_err = e
                try:
                    return pd.read_excel(p), p
                except Exception as e2:
                    last_err = e2
    if last_err:
        raise last_err
    raise FileNotFoundError("Excel not found at expected paths.")

df, source_path = load_data()

# ------------------------------------------------------
# Column normalization & detectors
# ------------------------------------------------------
def _rename_common(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "province":"Province","province_name":"Province","prov":"Province",
        "dwelling_type":"Dwelling Type","dwelling":"Dwelling Type","home_type":"Dwelling Type",
        "climate_zone":"Climate Region","climate":"Climate Region",
        "region_type":"Area Type","area_type":"Area Type",
        "income_bracket":"Income Bracket","income_group":"Income Bracket","household_income_bracket":"Income Bracket",
        "energy_cost_scaled":"Energy Cost","energy_consumption_total_kwh_scaled":"Energy Use"
    }
    new_cols = {}
    for c in df.columns:
        key = c.strip().lower().replace(" ","_")
        new_cols[c] = rename_map.get(key, c)
    return df.rename(columns=new_cols)

def _smart_find_numeric(df: pd.DataFrame, patterns):
    cands = []
    for c in df.columns:
        name = c.lower().replace(" ","_")
        if any(re.search(p, name) for p in patterns):
            if pd.api.types.is_numeric_dtype(df[c]):
                cands.append(c)
    if not cands:
        return None
    def rank(n):
        nl = n.lower()
        penal = 1 if ("_z" in nl or "scaled" in nl or "standard" in nl) else 0
        return (penal, len(nl))
    cands.sort(key=rank)
    return cands[0]

def pick_targets(df: pd.DataFrame):
    cost_patterns = [r"\benergy[_ ]?cost\b", r"\bcost\b", r"\bbill\b", r"\btotal[_ ]?cost\b", r"\belec[_ ]?cost\b", r"\bgas[_ ]?cost\b", r"\bprice\b", r"energy_cost_scaled", r"energy[_ ]?cost[_ ]?z"]
    use_patterns  = [r"\benergy[_ ]?consumption[_ ]?total\b", r"\benergy[_ ]?use\b", r"\busage\b", r"\bkwh\b", r"energy_consumption_total_kwh_scaled", r"energy[_ ]?use[_ ]?z"]
    ccol = _smart_find_numeric(df, cost_patterns)
    ucol = _smart_find_numeric(df, use_patterns)
    return ccol, ucol

def pick_text(df, options):
    for o in options:
        if o in df.columns and not pd.api.types.is_numeric_dtype(df[o]):
            return o
    for c in df.columns:
        low = c.lower()
        if any(o.lower().replace(" ","_") in low for o in options) and not pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None

def pick_numeric(df, options):
    for o in options:
        if o in df.columns and pd.api.types.is_numeric_dtype(df[o]):
            return o
    for c in df.columns:
        low = c.lower()
        if any(o.lower().replace(" ","_") in low for o in options) and pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None

df = _rename_common(df)
cost_col, use_col = pick_targets(df)
if cost_col: df = df.rename(columns={cost_col:"Energy Cost"}); cost_col = "Energy Cost"
if use_col:  df = df.rename(columns={use_col:"Energy Use"});   use_col = "Energy Use"

# ------------------------------------------------------
# Small utils
# ------------------------------------------------------
def _pct_diff(value, baseline):
    if value is None or baseline is None or baseline == 0:
        return None
    return 100.0 * (value - baseline) / abs(baseline)

def _pct_chip(x):
    if x is None:
        return '<span class="chip badge-flat">N/A</span>'
    if x > 0:
        return f'<span class="chip badge-up">+{x:.0f}%</span>'
    if x < 0:
        return f'<span class="chip badge-down">{x:.0f}%</span>'
    return '<span class="chip badge-flat">0%</span>'

def _median_or_center(df_in: pd.DataFrame, col: str, filters: dict = None):
    filters = filters or {}
    sub = df_in.copy()
    for k, v in (filters or {}).items():
        if k in sub.columns:
            sub = sub[sub[k] == v]
    series = sub[col].dropna() if (col and col in sub.columns) else pd.Series(dtype=float)
    if series.empty:
        return None
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    q01, q99 = np.nanpercentile(s, [1, 99])
    # If it looks standardized (z-score-ish) or min-max (0-1), force fallback
    if (q01 > -4 and q99 < 4) or (q01 >= 0 and q99 <= 1.5):
        return None
    med = float(np.nanmedian(s))
    return None if med <= 1 else med

# ------------------------------------------------------
# CSS
# ------------------------------------------------------
AI_CSS = """
<style>
:root { --pred:#F59E0B; --tren:#EF4444; --ai:#2563EB; --bg:#FFFFFF; --text:#0f172a; --muted:#475569; --surface:#fff; --surface-2:#f8fafc; --border:rgba(15,23,42,.12); --ring:rgba(37,99,235,.35); }
.main-title{font-size:2.2rem;font-weight:800;color:var(--text);margin:.25rem 0 .5rem 0}
.subtle{color:var(--muted);font-size:1rem;margin-bottom:.75rem}
.pill{display:inline-block;background:linear-gradient(180deg,var(--surface-2),var(--surface));color:var(--text);padding:.25rem .6rem;border-radius:999px;margin:.15rem .25rem 0 0;font-size:.9rem;border:1px solid var(--border)}
.kpi{background:linear-gradient(180deg,var(--surface-2),var(--surface));border:1px solid var(--border);padding:16px;border-radius:16px;text-align:center}
.kpi .label{font-size:.95rem;color:var(--muted)} .kpi .value{font-size:1.6rem;font-weight:700}
.compare-card{background:linear-gradient(135deg,color-mix(in srgb,var(--pred) 18%,transparent),var(--surface));border:1px dashed color-mix(in srgb,var(--text) 30%,transparent);padding:14px;border-radius:16px}
.compare-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.c-pill{display:inline-flex;align-items:center;gap:6px;padding:6px 10px;border-radius:999px;border:1px solid var(--border)}
.c-up{background:color-mix(in srgb,#ef4444 20%,var(--surface));color:#991b1b}.c-down{background:color-mix(in srgb,#22c55e 20%,var(--surface));color:#065f46}.c-flat{background:var(--surface-2)}
.tab-skin{padding:10px 12px;border-radius:14px;border-top:4px solid transparent}
.tab-pred{border-top-color:color-mix(in srgb,var(--pred) 80%,transparent);background:linear-gradient(180deg,color-mix(in srgb,var(--pred) 14%,transparent),transparent)}
.tab-trends{border-top-color:color-mix(in srgb,var(--tren) 80%,transparent);background:linear-gradient(180deg,color-mix(in srgb,var(--tren) 14%,transparent),transparent)}
.tab-ai{border-top-color:color-mix(in srgb,var(--ai) 80%,transparent);background:linear-gradient(180deg,color-mix(in srgb,var(--ai) 14%,transparent),transparent)}
div.stButton>button{background:var(--surface);border:1px solid var(--ring);border-radius:12px}
.ai-card{border:1px solid color-mix(in srgb,var(--ai) 35%,var(--border));background:linear-gradient(180deg,color-mix(in srgb,var(--ai) 8%,var(--surface)),var(--surface));border-radius:14px;padding:12px}
.ai-kv{display:grid;grid-template-columns:160px 1fr;gap:6px 10px;font-size:.95rem}
.chip{display:inline-flex;align-items:center;gap:6px;padding:6px 10px;border-radius:999px;border:1px solid var(--border);background:var(--surface);font-size:.9rem}
.badge-up{background:color-mix(in srgb,#ef4444 20%,var(--surface));color:#991b1b}
.badge-down{background:color-mix(in srgb,#22c55e 20%,var(--surface));color:#065f46}
.badge-flat{background:var(--surface-2)}
.muted{color:var(--muted)}
.wtbl{width:100%;border-collapse:separate;border-spacing:0}
.wtbl th,.wtbl td{padding:10px 12px;border-top:1px solid var(--border);font-size:.95rem}
.wtbl th{text-align:left;background:var(--surface-2);font-weight:700}
</style>
"""
st.markdown(AI_CSS, unsafe_allow_html=True)

# ------------------------------------------------------
# Predict helpers + calibration
# ------------------------------------------------------
def _delta_badge(delta_abs, unit):
    if delta_abs is None:
        return '<span class="chip badge-flat">–</span>'
    if abs(delta_abs) < 1e-9:
        return f'<span class="chip badge-flat">• 0 {unit}</span>'
    cls = "badge-down" if delta_abs < 0 else "badge-up"
    arrow = "▼" if delta_abs < 0 else "▲"
    return f'<span class="chip {cls}">{arrow} {abs(delta_abs):,.0f} {unit}</span>'

def _predict_pair(floor_area, adjusted_income, occupants, region_type, climate_zone, dwelling_type):
    X = [[floor_area, adjusted_income, occupants, region_type, climate_zone, dwelling_type]]
    use = float(energy_use_model.predict(X)[0])
    cost = float(energy_cost_model.predict(X)[0])
    return round(use, 2), round(cost, 2)

# Model-aware scaling so medians look realistic 
CAL_COST, CAL_USE = 1.0, 1.0
try:
    _u0, _c0 = _predict_pair(
        floor_area=1500, adjusted_income=90000, occupants=3,
        region_type=REGION_MAP["Urban"], climate_zone=CLIMATE_MAP["Continental"], dwelling_type=DWELLING_MAP["Single Detached"]
    )
    CAL_COST = float(np.clip(_c0 / NATIONAL_MEDIAN_COST, 0.75, 1.75)) if np.isfinite(_c0) else 1.0
    CAL_USE  = float(np.clip(_u0 / NATIONAL_MEDIAN_USE,  0.75, 1.75)) if np.isfinite(_u0) else 1.0
except Exception:
    CAL_COST, CAL_USE = 1.0, 1.0

def _scaled_cost(value):  # apply calibration to fallback medians only
    return value * CAL_COST

def _scaled_use(value):
    return value * CAL_USE

# ------------------------------------------------------
# Advisor tips
# ------------------------------------------------------
def _size_tips(fa):
    if fa >= 3500: return ["Seal garage-to-house door & attic hatch; add attic insulation.","Zone seldom-used rooms; reduce setpoints."]
    if fa >= 2500: return ["Target air-sealing (rim joists, top plates).","Use room-by-room schedules."]
    if fa >= 1200: return ["Balanced thermostat schedules.","Upgrade leakiest windows/doors first."]
    return ["Focus on plug loads & hot water routines.","Use smart strips for media/office corners."]

def _occupant_tips(n):
    if n >= 5: return ["1.75 gpm showerheads; stagger showers.","Cold wash most loads; run full loads."]
    if n == 1: return ["Right-size schedules to your presence.","Use induction kettle/microwave for singles."]
    return ["Eco modes on appliances; air-dry when possible.","Track monthly use to catch swings."]

def _province_tips(p):
    if p in {"Ontario","Quebec"}: return ["Time-of-use pricing: shift off-peak.","Review retailer/plan annually."]
    if p in {"Alberta","Saskatchewan","Manitoba"}: return ["Winter sealing pays back quickly.","Smart thermostat setbacks on cold nights."]
    if p in {"British Columbia","Nova Scotia","New Brunswick","Prince Edward Island","Newfoundland and Labrador"}: return ["Dehumidify; keep filters clean.","Shade sun-facing glass in summer."]
    return ["Verify local plan and fees.","Weatherize doors/windows before equipment swaps."]

def _build_targeted_tips(ctx):
    tips = [
        "Turn off lights in empty rooms; consider motion sensors.",
        "Unplug chargers/consoles when idle or use smart strips.",
        "Charge only when needed—avoid overnight trickle."
    ]
    tips.extend(_province_tips(ctx["province"]))
    if ctx["climate_label"] in {"Arctic","Cold","Subarctic"}:
        tips += ["Air-seal attic & rim joists first; add insulation.","Night setback ~2 °C; preheat before occupancy."]
    elif ctx["climate_label"] == "Humid":
        tips += ["Keep AC filters clean; consider dehumidifier.","Use bath/kitchen fans to dump latent heat."]
    else:
        tips += ["Seal window/door gaps; use smart schedules.","Shade south/west glass to cut cooling spikes."]
    tips += (_size_tips(ctx["floor_area"]) + _occupant_tips(ctx["occupants"]))
    return tips[:8]

def _what_if_scenarios(ctx):
    base_cost, base_use = ctx["cost"], ctx["use"]
    sims = []
    new_floor = max(300, int(ctx["floor_area"] * 0.8))
    u, c = _predict_pair(new_floor, ctx["adjusted_income"], ctx["occupants"], ctx["region_type"], ctx["climate_zone"], ctx["dwelling_type"])
    sims.append({"group":"Home changes","name":"Smaller space","note":"Floor area −20%","use":u,"cost":c,"du":u-base_use,"dc":c-base_cost})
    new_occ = max(1, ctx["occupants"] - 1)
    u, c = _predict_pair(ctx["floor_area"], ctx["adjusted_income"], new_occ, ctx["region_type"], ctx["climate_zone"], ctx["dwelling_type"])
    sims.append({"group":"Home changes","name":"Fewer occupants","note":"−1 occupant","use":u,"cost":c,"du":u-base_use,"dc":c-base_cost})
    env_cut = 0.08
    sims.append({"group":"Home changes","name":"Weatherization boost","note":"Air sealing + attic insulation (≈ −8%)","use":base_use*(1-env_cut),"cost":base_cost*(1-env_cut),"du":-base_use*env_cut,"dc":-base_cost*env_cut})
    sims.append({"group":"Behavior changes","name":"Lights & schedules","note":"Turn off & schedule lighting (−10%)","use":base_use*0.90,"cost":base_cost*0.90,"du":-base_use*0.10,"dc":-base_cost*0.10})
    sims.append({"group":"Behavior changes","name":"Hot-water routines","note":"Shorter showers, cold wash (−10%)","use":base_use*0.90,"cost":base_cost*0.90,"du":-base_use*0.10,"dc":-base_cost*0.10})
    hvac_cut = 0.05 if ctx["climate_label"] in {"Arctic","Cold","Subarctic","Continental","Humid"} else 0.03
    sims.append({"group":"Behavior changes","name":"Thermostat tuning", "note":f"Sleep/away setbacks (−{int(hvac_cut*100)}%)","use":base_use*(1-hvac_cut),"cost":base_cost*(1-hvac_cut),"du":-base_use*hvac_cut,"dc":-base_cost*hvac_cut})
    # Income what-ifs (explicit sensitivity)
    inc_keys = list(INCOME_MAP.keys())
    idx = inc_keys.index(ctx["income_label"]) if ctx["income_label"] in inc_keys else None
    if idx is not None:
        if idx+1 < len(inc_keys):
            up_inc = INCOME_MAP[inc_keys[idx+1]]
            u, c = _predict_pair(ctx["floor_area"], up_inc, ctx["occupants"], ctx["region_type"], ctx["climate_zone"], ctx["dwelling_type"])
            sims.append({"group":"Economics","name":"One bracket higher","note":"+ income bracket","use":u,"cost":c,"du":u-base_use,"dc":c-base_cost})
        if idx-1 >= 0:
            down_inc = INCOME_MAP[inc_keys[idx-1]]
            u, c = _predict_pair(ctx["floor_area"], down_inc, ctx["occupants"], ctx["region_type"], ctx["climate_zone"], ctx["dwelling_type"])
            sims.append({"group":"Economics","name":"One bracket lower","note":"− income bracket","use":u,"cost":c,"du":u-base_use,"dc":c-base_cost})
    return sims

# ------------------------------------------------------
# Global CSS intensifier
# ------------------------------------------------------
st.markdown(f"""
<style>
  :root {{
    --pred:{ACCENT_PRED}; --tren:{ACCENT_TREN}; --ai:{ACCENT_AI};
    --bg: linear-gradient(180deg,#F8FBFF 0%, #FFFFFF 40%);
    --text:#0f172a; --muted:#475569; --surface:#ffffff; --surface-2:#f8fafc;
    --border:rgba(15,23,42,.12); --ring:rgba(37,99,235,.35);
  }}
  html, body {{ background: var(--bg) !important; }}
  .tab-pred{{border-top-color: color-mix(in srgb, var(--pred) 90%, transparent);
    background: linear-gradient(180deg, color-mix(in srgb, var(--pred) 16%, transparent), transparent);}}
  .tab-trends{{border-top-color: color-mix(in srgb, var(--tren) 90%, transparent);
    background: linear-gradient(180deg, color-mix(in srgb, var(--tren) 16%, transparent), transparent);}}
  .tab-ai{{border-top-color: color-mix(in srgb, var(--ai) 90%, transparent);
    background: linear-gradient(180deg, color-mix(in srgb, var(--ai) 16%, transparent), transparent);}}
  .pill{{ border:1px solid color-mix(in srgb, var(--ai) 35%, var(--border)); }}
  div.stButton > button {{ border-color: color-mix(in srgb, var(--ai) 40%, var(--ring)); }}
  .compare-card{{ background:linear-gradient(135deg, color-mix(in srgb, var(--pred) 22%, transparent), var(--surface)); }}
</style>
""", unsafe_allow_html=True)

st.caption(f"Data source file: {source_path}")
tab_pred, tab_trends, tab_ai = st.tabs(["Personal Prediction", "Energy Trends", "AI Advisor"])

# ------------------------------------------------------
# Compare square helpers
# ------------------------------------------------------
def _fmt_change_with_pct(delta, baseline, unit):
    arrow = '•'
    if delta < 0: arrow = '▼'
    elif delta > 0: arrow = '▲'
    abs_txt = f"{abs(delta):,.0f} {unit}"
    pct = (100.0 * delta / abs(baseline)) if (baseline and abs(baseline) > 1e-9) else None
    return arrow + " " + abs_txt + (f" ({pct:+.0f}%)" if pct is not None else ""), pct

def render_compare_square(last, curr_use, curr_cost):
    if not last:
        st.info("First prediction this session — no prior to compare.")
        return
    du, dc = curr_use - last["use"], curr_cost - last["cost"]
    use_txt, _ = _fmt_change_with_pct(du, last["use"], "kWh")
    cost_txt, _ = _fmt_change_with_pct(dc, last["cost"], "$")
    pill_use = "c-flat" if du == 0 else ("c-down" if du < 0 else "c-up")
    pill_cost = "c-flat" if dc == 0 else ("c-down" if dc < 0 else "c-up")
    st.markdown(
        f"""
        <div class="compare-card">
          <div class="compare-grid">
            <div><b>Energy Use</b><br><span class="{pill_use} c-pill">{use_txt}</span></div>
            <div><b>Energy Cost</b><br><span class="{pill_cost} c-pill">{cost_txt}</span></div>
          </div>
          <div class="muted" style="margin-top:6px;">Change vs last prediction</div>
        </div>
        """, unsafe_allow_html=True
    )

# ------------------------------------------------------
# Helper blocks & hints
# ------------------------------------------------------
def _percent_table(df_in: pd.DataFrame, key_col: str, top_n: int = 6) -> pd.DataFrame:
    if not key_col or key_col not in df_in.columns:
        return pd.DataFrame(columns=[key_col or "Category","Percent"])
    vc = (df_in[key_col].dropna().value_counts(normalize=True).mul(100).round(1).rename_axis(key_col).reset_index(name="Percent"))
    return vc.head(top_n)

CLIMATE_HINTS = {
    "arctic":"Very cold; long heating season; minimal cooling.",
    "subarctic":"Long, cold winters; short summers; heating dominates.",
    "continental":"Cold winters, warm summers; heating and some cooling.",
    "prairie":"Dry with swings; strong heating and some cooling.",
    "maritime":"Mild and humid; moderate heating and some cooling.",
    "temperate":"Moderate winters and summers; mixed loads.",
    "mountain":"Cooler with elevation; extended heating season."
}
AREA_HINTS = {
    "urban":"More apartments & shared walls lower loads.",
    "suburban":"More detached & larger area raises loads.",
    "rural":"Detached and exposed which can raise heating."
}

def _describe_label(label: str, hint_map: dict) -> str | None:
    if not isinstance(label, str): return None
    low = label.lower()
    for k, v in hint_map.items():
        if k in low: return v
    return None

def climate_help_block(df_base: pd.DataFrame, province_col: str | None, climate_col: str | None, province_sel: str | None):
    with st.expander("Need help choosing a climate region?"):
        scope = df_base.copy()
        if province_col and province_sel:
            scope = scope[scope[province_col] == province_sel]
            st.caption(f"Most common in {province_sel}:")
        else:
            st.caption("Most common climate regions in the dataset:")
        tbl = _percent_table(scope, climate_col) if climate_col else pd.DataFrame()
        if not tbl.empty:
            tbl["Reason"] = tbl[climate_col].apply(lambda x: _describe_label(x, CLIMATE_HINTS))
            st.dataframe(tbl, use_container_width=True, hide_index=True)
        else:
            st.info("No climate column found or not enough data to summarize.")
        st.markdown(
            "- Arctic/Subarctic: very cold & heating-heavy.\n"
            "- Continental/Prairie: cold winters & warm summers.\n"
            "- Maritime/Temperate: milder but humid; Mountain cooler with elevation."
        )

def area_help_block(df_base: pd.DataFrame, province_col: str | None, area_col: str | None, province_sel: str | None):
    with st.expander("Need help choosing an area type?"):
        scope = df_base.copy()
        if province_col and province_sel:
            scope = scope[scope[province_col] == province_sel]
            st.caption(f"Most common in {province_sel}:")
        else:
            st.caption("Most common area types in the dataset:")
        tbl = _percent_table(scope, area_col) if area_col else pd.DataFrame()
        if not tbl.empty:
            tbl["Reason"] = tbl[area_col].apply(lambda x: _describe_label(x, AREA_HINTS))
            st.dataframe(tbl, use_container_width=True, hide_index=True)
        else:
            st.info("No area type column found or not enough data to summarize.")
        st.markdown("- Urban: more apartments & shared walls.\n- Suburban: more detached & larger floor area.\n- Rural: more exposed which can raise heating.")

# ------------------------------------------------------
# Tab: Personal Prediction
# ------------------------------------------------------
with tab_pred:
    st.markdown('<div class="tab-skin tab-pred">', unsafe_allow_html=True)
    st.markdown("<div class='main-title'>Canadian Household Energy Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtle'>Enter your information in the sidebar. Results are estimates.</div>", unsafe_allow_html=True)
    st.divider()

    st.sidebar.header("Inputs")

    # ---- Preset UI (added) ----
    with st.sidebar.expander("Preset profiles", expanded=False):
        preset_name = st.selectbox("Choose a preset", list(PRESET_PROFILES.keys()), index=0, key="preset_name_sidebar")
        if st.button("Load preset", key="btn_load_preset"):
            preset = PRESET_PROFILES.get(preset_name, {})
            if preset:
                for k, v in preset.items():
                    st.session_state[k] = v
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()

    with st.sidebar.expander("1) Location and region", expanded=True):
        province = st.selectbox(
            "Province",
            ["Alberta","British Columbia","Manitoba","New Brunswick","Newfoundland and Labrador",
             "Northwest Territories","Nova Scotia","Nunavut","Ontario","Prince Edward Island",
             "Quebec","Saskatchewan","Yukon"],
            key="prov_sel",
        )
        region_type_label = st.selectbox("Area Type", ["Urban","Suburban","Rural"], key="region_sel")
        region_type = REGION_MAP[region_type_label]
        climate_zone_label = st.selectbox("Climate Region", ["Continental","Cold","Mild","Humid","Arctic","Subarctic"], key="climate_sel")
        climate_zone = CLIMATE_MAP[climate_zone_label]

    with st.sidebar.expander("2) Home characteristics", expanded=True):
        dwelling_type_label = st.selectbox(
            "Dwelling Type",
            ["Single Detached","Apartment","Row House","Semi Detached","Mobile Home","Other Movable"],
            key="dwelling_sel",
        )
        dwelling_type = DWELLING_MAP[dwelling_type_label]
        floor_area = st.slider("Floor Area (sq ft)", 300, 10000, 1500, step=50, key="floor_sel")
        occupants = st.slider("Number of Occupants", 1, 15, 3, key="occ_sel")

    with st.sidebar.expander("3) Household economics", expanded=True):
        income_bracket = st.selectbox("Household Income Bracket", list(INCOME_MAP.keys()), key="income_sel")
        adjusted_income = INCOME_MAP[income_bracket]

    st.markdown("<h3>Summary of your inputs</h3>", unsafe_allow_html=True)
    cols_sum = st.columns(4)
    with cols_sum[0]:
        st.markdown(f"<span class='pill'>Province: <b>{province}</b></span>", unsafe_allow_html=True)
        st.markdown(f"<span class='pill'>Area: <b>{region_type_label}</b></span>", unsafe_allow_html=True)
    with cols_sum[1]:
        st.markdown(f"<span class='pill'>Climate: <b>{climate_zone_label}</b></span>", unsafe_allow_html=True)
        st.markdown(f"<span class='pill'>Dwelling: <b>{dwelling_type_label}</b></span>", unsafe_allow_html=True)
    with cols_sum[2]:
        st.markdown(f"<span class='pill'>Floor Area: <b>{floor_area} sq ft</b></span>", unsafe_allow_html=True)
    with cols_sum[3]:
        st.markdown(f"<span class='pill'>Occupants: <b>{occupants}</b></span>", unsafe_allow_html=True)
        st.markdown(f"<span class='pill'>Income: <b>{income_bracket}</b></span>", unsafe_allow_html=True)

    province_col_pred = pick_text(df, ["Province"])
    climate_col_pred  = pick_text(df, ["Climate Region","Climate Zone","Climate"])
    area_col_pred     = pick_text(df, ["Area Type","Region Type"])
    climate_help_block(df, province_col_pred, climate_col_pred, province_sel=province)
    area_help_block(df, province_col_pred, area_col_pred, province_sel=province)

    def predict_energy(floor_area, adjusted_income, occupants, region_type, climate_zone, dwelling_type):
        X = [[floor_area, adjusted_income, occupants, region_type, climate_zone, dwelling_type]]
        use = float(energy_use_model.predict(X)[0])
        cost = float(energy_cost_model.predict(X)[0])
        return round(use, 2), round(cost, 2)

    run_pred = st.button("Predict", use_container_width=True, key="btn_pred")
    st.write("")
    if run_pred:
        try:
            use, cost = predict_energy(floor_area, adjusted_income, occupants, region_type, climate_zone, dwelling_type)
            last = st.session_state.get("last_pred")
            k1, k2 = st.columns(2)
            with k1: st.markdown(f"<div class='kpi'><div class='label'>Energy Use (kWh)</div><div class='value'>{use:,.0f}</div></div>", unsafe_allow_html=True)
            with k2: st.markdown(f"<div class='kpi'><div class='label'>Energy Cost ($)</div><div class='value'>{cost:,.0f}</div></div>", unsafe_allow_html=True)
            render_compare_square(last, use, cost)
            st.session_state["last_pred"] = {"use": use, "cost": cost}
            st.info("For a detailed breakdown and tailored actions, open the AI Advisor tab.")
            st.caption("Estimate only. Local pricing, insulation, and heating system can change results.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------
# Tab: Energy Trends
# ------------------------------------------------------
with tab_trends:
    st.markdown('<div class="tab-skin tab-trends">', unsafe_allow_html=True)
    st.markdown("<div class='main-title'>Energy Trends</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtle'>All values are estimates based on historical data and may vary.</div>", unsafe_allow_html=True)

    province_col = pick_text(df, ["Province"])
    climate_col  = pick_text(df, ["Climate Region","Climate Zone","Climate"])
    area_col     = pick_text(df, ["Area Type","Region Type"])
    income_col   = pick_text(df, ["Income Bracket","Income"])
    dwelling_col = pick_text(df, ["Dwelling Type","Dwelling","Home Type"])
    floor_col    = pick_numeric(df, ["floor_area","floorarea","sqft","sq_ft","floor area"])
    occ_col      = pick_numeric(df, ["occupants","household_size","num_occupants"])
    adj_income   = pick_numeric(df, ["adjusted_income","adjusted_avg_income","income_adj"])

    if not cost_col and not use_col:
        st.warning("No energy cost or energy use column detected in the data.")
        st.markdown('</div>', unsafe_allow_html=True); st.stop()

    metric = st.radio(
        "Select metric",
        options=[c for c in [cost_col, use_col] if c],
        index=0 if cost_col else 1,
        format_func=lambda c: "Energy Cost" if "cost" in c.lower() else "Energy Use",
        horizontal=True,
        key="trends_metric"
    )

    st.markdown("### Filters")
    fc1, fc2, fc3 = st.columns(3)
    prov_sel = fc1.multiselect("Province", sorted(df[province_col].dropna().unique())) if province_col else []
    inc_sel  = fc2.multiselect("Income bracket", sorted(df[income_col].dropna().unique())) if income_col else []
    clim_sel = fc3.multiselect("Climate region", sorted(df[climate_col].dropna().unique())) if climate_col else []
    fc4, _, _ = st.columns(3)
    area_sel = fc4.multiselect("Area type", sorted(df[area_col].dropna().unique())) if area_col else []

    viz = df.copy()
    if province_col and prov_sel: viz = viz[viz[province_col].isin(prov_sel)]
    if income_col and inc_sel:    viz = viz[viz[income_col].isin(inc_sel)]
    if climate_col and clim_sel:  viz = viz[viz[climate_col].isin(clim_sel)]
    if area_col and area_sel:     viz = viz[viz[area_col].isin(area_sel)]

    if len(viz) == 0:
        st.info("No data for this selection.")
        st.markdown('</div>', unsafe_allow_html=True); st.stop()

    PROV_MAP = {"Alberta":"AB","British Columbia":"BC","Manitoba":"MB","New Brunswick":"NB","Newfoundland and Labrador":"NL","Nova Scotia":"NS","Northwest Territories":"NT","Nunavut":"NU","Ontario":"ON","Prince Edward Island":"PE","Quebec":"QC","Saskatchewan":"SK","Yukon":"YT"}
    def _prov_short(s): return s.map(lambda x: PROV_MAP.get(x, x) if isinstance(x, str) else x)

    st.divider()
    st.subheader(f"Average {'energy cost' if 'cost' in metric.lower() else 'energy use'} by province")
    gp = viz.groupby(province_col, dropna=False)[[c for c in [cost_col, use_col] if c]].mean().reset_index().rename(columns={province_col:"Province"})
    gp["ProvShort"] = _prov_short(gp["Province"])
    def _bar_avg(df_in, ycol, ylab):
        fig = px.bar(df_in.sort_values(ycol), x="ProvShort", y=ycol, labels={"ProvShort":"Province", ycol:ylab}, color_discrete_sequence=PALETTE_SEQ, hover_data={"Province":True, "ProvShort":False}, template=PLOTLY_TEMPLATE)
        fig.update_layout(height=440, margin=dict(l=10,r=10,t=30,b=10))
        fig.update_yaxes(ticksuffix=" $" if "cost" in ycol.lower() else "")
        return fig

    mode = st.radio("View:", ["Average per province", "Dwelling shares by province"], horizontal=True, key="trends_mode")

    def hardcoded_dwelling_shares() -> pd.DataFrame:
        data = [
            {"Province":"Alberta","Apartment":22.7,"Detached":65.0,"Mobile home":1.0,"Other movable":0.1,"Row house":5.5,"Semi detached":5.7},
            {"Province":"British Columbia","Apartment":37.1,"Detached":46.7,"Mobile home":1.2,"Other movable":0.1,"Row house":8.2,"Semi detached":6.7},
            {"Province":"Manitoba","Apartment":23.3,"Detached":66.5,"Mobile home":0.6,"Other movable":0.1,"Row house":4.9,"Semi detached":4.4},
            {"Province":"New Brunswick","Apartment":20.4,"Detached":70.3,"Mobile home":2.9,"Other movable":0.1,"Row house":4.1,"Semi detached":3.5},
            {"Province":"Newfoundland and Labrador","Apartment":20.4,"Detached":71.5,"Mobile home":5.0,"Other movable":0.1,"Row house":0.7,"Semi detached":2.3},
            {"Province":"Northwest Territories","Apartment":19.0,"Detached":78.0,"Mobile home":3.0,"Other movable":0.1,"Row house":1.0,"Semi detached":2.0},
            {"Province":"Nova Scotia","Apartment":22.0,"Detached":69.8,"Mobile home":4.7,"Other movable":0.1,"Row house":4.2,"Semi detached":3.2},
            {"Province":"Nunavut","Apartment":35.0,"Detached":60.0,"Mobile home":3.0,"Other movable":0.1,"Row house":1.0,"Semi detached":1.0},
            {"Province":"Ontario","Apartment":30.0,"Detached":54.3,"Mobile home":0.4,"Other movable":0.1,"Row house":9.0,"Semi detached":6.1},
            {"Province":"Prince Edward Island","Apartment":17.3,"Detached":75.2,"Mobile home":3.0,"Other movable":0.1,"Row house":2.5,"Semi detached":2.0},
            {"Province":"Quebec","Apartment":40.5,"Detached":40.6,"Mobile home":1.0,"Other movable":0.1,"Row house":10.5,"Semi detached":6.9},
            {"Province":"Saskatchewan","Apartment":19.1,"Detached":71.8,"Mobile home":0.6,"Other movable":0.1,"Row house":4.6,"Semi detached":3.8},
            {"Province":"Yukon","Apartment":16.0,"Detached":71.0,"Mobile home":4.0,"Other movable":0.1,"Row house":2.0,"Semi detached":4.0},
        ]
        return pd.DataFrame(data)
    dw_shares = hardcoded_dwelling_shares()

    if mode == "Average per province":
        if cost_col and use_col:
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Average energy cost")
                st.plotly_chart(_bar_avg(gp, cost_col, "Average cost in CAD per year"), use_container_width=True)
            with c2:
                st.caption("Average energy use")
                st.plotly_chart(_bar_avg(gp, use_col, "Average use in kWh per year"), use_container_width=True)
        else:
            ylab = "Average cost in CAD per year" if "cost" in metric.lower() else "Average use in kWh per year"
            st.plotly_chart(_bar_avg(gp, metric, ylab), use_container_width=True)
        with st.expander("What am I looking at?"):
            st.write("Each bar shows the province average for the selected metric. Differences reflect prices, climate, and housing stock.")
    else:
        stacked_view = st.toggle("Use normalized percent bars", value=False, key="dw_shares_norm_toggle")
        shares_use = dw_shares.copy()
        if prov_sel: shares_use = shares_use[shares_use["Province"].isin(prov_sel)]
        long = shares_use.melt(id_vars=["Province"], var_name="Dwelling type", value_name="Percent")
        long["Percent"] = pd.to_numeric(long["Percent"], errors="coerce").fillna(0.0)
        long["ProvShort"] = _prov_short(long["Province"])
        DWELLING_ORDER = ["Detached","Apartment","Semi detached","Row house","Mobile home","Other movable"]
        long["Dwelling type"] = pd.Categorical(long["Dwelling type"], categories=DWELLING_ORDER, ordered=True)
        bmode = "stack" if stacked_view else "group"
        fig_sh = px.bar(long.sort_values(["Dwelling type","ProvShort"]), x="ProvShort", y="Percent", color="Dwelling type", barmode=bmode, color_discrete_sequence=PALETTE_SEQ, labels={"ProvShort":"Province","Percent":"Percent"}, template=PLOTLY_TEMPLATE)
        fig_sh.update_yaxes(ticksuffix="%"); fig_sh.update_layout(height=440, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig_sh, use_container_width=True)
        with st.expander("What does this mean?"):
            st.write("Bars show how common each dwelling type is within a province. Distribution influences typical energy patterns.")

    st.divider()
    st.subheader("Median by income bracket")
    if income_col:
        metric_inc = (cost_col or use_col)
        if cost_col and use_col:
            choice = st.radio("Metric", ["Energy Cost","Energy Use"], horizontal=True, key="metric_income")
            metric_inc = cost_col if choice == "Energy Cost" else use_col
        gi = viz.groupby(income_col, dropna=False)[metric_inc].median().reset_index()
        gi["sort_key"] = gi[income_col].astype(str).str.replace(",", "", regex=False).str.extract(r"(\d+)").astype(float)
        gi = gi.sort_values("sort_key", na_position="last")
        fig_inc = px.bar(gi, x=income_col, y=metric_inc, color=income_col, color_discrete_sequence=PALETTE_SEQ, labels={income_col:"Income bracket", metric_inc:"Median"}, template=PLOTLY_TEMPLATE)
        fig_inc.update_layout(height=420, margin=dict(l=10,r=10,t=30,b=10), showlegend=False)
        st.plotly_chart(fig_inc, use_container_width=True)
        with st.expander("Why does income matter?"):
            st.write("Higher income often aligns with larger homes and more appliances which increases energy use and cost.")
    else:
        st.info("Income bracket column not found.")

    st.divider()
    st.subheader("How it changes with floor area")
    metric_flr = (cost_col or use_col)
    if cost_col and use_col:
        choice = st.radio("Metric", ["Energy Cost","Energy Use"], horizontal=True, key="metric_floor")
        metric_flr = cost_col if choice == "Energy Cost" else use_col
    if (pick_numeric(df, ["floor_area","floorarea","sqft","sq_ft","floor area"])):
        floor_col = pick_numeric(df, ["floor_area","floorarea","sqft","sq_ft","floor area"])
        s = viz[[floor_col, metric_flr]].dropna().sort_values(floor_col)
        if len(s) >= 30:
            s["bin"] = pd.qcut(s[floor_col], q=10, duplicates="drop")
            agg = s.groupby("bin").agg(
                floor_mid=(floor_col,"median"),
                y_med=(metric_flr,"median"),
                y_q1=(metric_flr,lambda v: np.nanpercentile(v,25)),
                y_q3=(metric_flr,lambda v: np.nanpercentile(v,75))
            ).reset_index(drop=True)
            fig_flr = go.Figure()
            fig_flr.add_trace(go.Scatter(x=s[floor_col], y=s[metric_flr], mode="markers", opacity=0.2, marker=dict(size=5), showlegend=False))
            fig_flr.add_trace(go.Scatter(x=pd.concat([agg["floor_mid"], agg["floor_mid"][::-1]]), y=pd.concat([agg["y_q3"], agg["y_q1"][::-1]]), fill="toself", mode="lines", line=dict(width=0), fillcolor="rgba(120,120,200,0.20)", showlegend=False))
            fig_flr.add_trace(go.Scatter(x=agg["floor_mid"], y=agg["y_med"], mode="lines+markers", line=dict(width=3), name="Median"))
            fig_flr.update_layout(height=440, margin=dict(l=10,r=10,t=30,b=10), template=PLOTLY_TEMPLATE, xaxis_title="Floor area in sq ft", yaxis_title="Energy Cost" if metric_flr == cost_col else "Energy Use")
            st.plotly_chart(fig_flr, use_container_width=True)
            if len(agg) >= 2:
                slope = np.polyfit(agg["floor_mid"], agg["y_med"], 1)[0]
                st.caption("Larger homes tend to cost more." if slope > 0 else "Floor area shows a weak relationship with the selected metric.")
            with st.expander("How to read this"):
                st.write("Dots are individual homes. The shaded band shows the middle 50 percent for each floor area bin. The line tracks the median.")
        else:
            st.info("Not enough records to form deciles for floor area.")
    else:
        st.info("Floor area column not found.")

    st.divider()
    st.subheader("Climate and area type")
    metric_geo = (cost_col or use_col)
    if cost_col and use_col:
        choice = st.radio("Metric", ["Energy Cost","Energy Use"], horizontal=True, key="metric_geo")
        metric_geo = cost_col if choice == "Energy Cost" else use_col
    c1, c2 = st.columns(2)
    with c1:
        st.caption("By climate region")
        if pick_text(df, ["Climate Region","Climate Zone","Climate"]):
            climate_col = pick_text(df, ["Climate Region","Climate Zone","Climate"])
            gc = viz.groupby(climate_col, dropna=False)[metric_geo].mean().reset_index()
            fig_clim = px.bar(gc, x=climate_col, y=metric_geo, color=climate_col, color_discrete_sequence=PALETTE_SEQ, labels={climate_col:"Climate region", metric_geo:"Mean"}, template=PLOTLY_TEMPLATE)
            fig_clim.update_layout(height=420, margin=dict(l=10,r=10,t=30,b=10), showlegend=False)
            st.plotly_chart(fig_clim, use_container_width=True)
            with st.expander("Why some climates cost more"):
                st.write("Colder climates need more heating. Hot summers need more cooling. Prices and building standards vary by region.")
        else:
            st.info("Climate region column not found.")
    with c2:
        st.caption("By area type")
        if pick_text(df, ["Area Type","Region Type"]):
            area_col = pick_text(df, ["Area Type","Region Type"])
            ga = viz.groupby(area_col, dropna=False)[metric_geo].mean().reset_index()
            fig_area = px.bar(ga, x=area_col, y=metric_geo, color=area_col, color_discrete_sequence=PALETTE_SEQ, labels={area_col:"Area type", metric_geo:"Mean"}, template=PLOTLY_TEMPLATE)
            fig_area.update_layout(height=420, margin=dict(l=10,r=10,t=30,b=10), showlegend=False)
            st.plotly_chart(fig_area, use_container_width=True)
            with st.expander("Urban vs suburban"):
                st.write("Suburban homes are typically larger and more detached which increases energy use. Urban homes often include more apartments with shared walls.")
        else:
            st.info("Area type column not found.")

    st.divider()
    st.subheader("What influences the model most")
    target_for_drivers = (cost_col or use_col)
    if cost_col and use_col:
        choice = st.radio("Metric", ["Energy Cost","Energy Use"], horizontal=True, key="metric_drivers")
        target_for_drivers = cost_col if choice == "Energy Cost" else use_col
    model = energy_cost_model if target_for_drivers == cost_col else energy_use_model
    X_cols = []
    for c in [pick_numeric(df, ["floor_area","floorarea","sqft","sq_ft","floor area"]),
              pick_numeric(df, ["adjusted_income","adjusted_avg_income","income_adj"]),
              pick_numeric(df, ["occupants","household_size","num_occupants"])]:
        if c: X_cols.append(c)
    X = viz[X_cols].copy() if X_cols else pd.DataFrame(index=viz.index)
    for col, alias in [(pick_text(df, ["Area Type","Region Type"]), "area"),
                       (pick_text(df, ["Climate Region","Climate Zone","Climate"]), "climate"),
                       (pick_text(df, ["Dwelling Type","Dwelling","Home Type"]), "dwelling")]:
        if col: X[alias] = pd.factorize(viz[col])[0]
    y = viz[target_for_drivers].values

    top_features = top_values = None
    try:
        if hasattr(model, "feature_importances_"):
            if hasattr(model, "feature_names_in_"):
                names = list(model.feature_names_in_)
                vals = model.feature_importances_
                imps = pd.Series(vals[:len(names)], index=names[:len(vals)])
            else:
                imps = pd.Series(model.feature_importances_[:len(X.columns)], index=list(X.columns)[:len(model.feature_importances_)])
            imps = imps.sort_values(ascending=False).head(5)
            top_features = [re.sub(r"(scaled|standard|_z)$","", n, flags=re.I).replace("_"," ").title() for n in imps.index]
            top_values = imps.values
        else:
            if X.shape[1] >= 2 and len(X) >= 200:
                samp = X.sample(min(2000, len(X)), random_state=7)
                y_s = y[samp.index]
                res = permutation_importance(model, samp.values, y_s, n_repeats=5, random_state=7)
                means = pd.Series(res.importances_mean, index=list(samp.columns))
                imps = means.sort_values(ascending=False).head(5)
                top_features = [re.sub(r"(scaled|standard|_z)$","", n, flags=re.I).replace("_"," ").title() for n in imps.index]
                top_values = imps.values
    except Exception:
        pass
    if top_features is not None:
        drv = pd.DataFrame({"feature": top_features, "importance": top_values}).sort_values("importance", ascending=True)
        fig_imp = px.bar(drv, x="importance", y="feature", orientation="h", color="importance", color_continuous_scale=PALETTE_CONT, labels={"importance":"Importance","feature":""}, template=PLOTLY_TEMPLATE)
        fig_imp.update_layout(height=380, margin=dict(l=10,r=10,t=20,b=10), coloraxis_showscale=False)
        st.plotly_chart(fig_imp, use_container_width=True)
        with st.expander("How to interpret importances"):
            st.write("Higher bars indicate features the model relied on more for predictions.")
    else:
        st.info("Could not compute model drivers for the current selection.")

    st.divider()
    st.subheader("Pearson Correlation Heatmap")
    labels = ["floor_area","occupants","energy_consumption_total","energy_cost","income"]
    z = [
        [1.00,0.84,0.96,0.40,0.062],
        [0.84,1.00,0.94,0.70,0.054],
        [0.96,0.94,1.00,0.60,0.065],
        [0.40,0.70,0.60,1.00,0.030],
        [0.062,0.054,0.065,0.030,1.00],
    ]
    text = [[f"{v:.2f}" for v in row] for row in z]
    fig = go.Figure(data=go.Heatmap(
        z=z, x=labels, y=labels, zmin=0, zmax=1, zmid=0.5,
        colorscale="RdBu_r", showscale=True,
        colorbar=dict(title="r", tickvals=[0,0.25,0.5,0.75,1.0]),
        text=text, texttemplate="%{text}", textfont=dict(size=12),
        hovertemplate=" %{y} vs %{x}<br>r=%{z:.2f}<extra></extra>",
        xgap=1, ygap=1
    ))
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=50, b=10), template=PLOTLY_TEMPLATE)
    fig.update_xaxes(side="bottom", tickangle=90, tickfont=dict(size=11))
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=11))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)
# ------------------------------------------------------
# SHAP utilities 
# ------------------------------------------------------
def _infer_pipeline_parts(model):
    pre, est = None, model
    try:
        from sklearn.pipeline import Pipeline
        if isinstance(model, Pipeline):
            steps = dict(model.named_steps)
            pre = steps.get("preprocessor", None)
            est = list(model.named_steps.values())[-1]
    except Exception:
        pass
    return pre, est

def _make_input_row_from_ctx(ctx: dict) -> pd.DataFrame:
    return pd.DataFrame([{
        "province_code": PROVINCE_INDEX.get(ctx.get("province"), 6),
        "region_type": float(ctx.get("region_type") or 0.0),
        "climate_zone": float(ctx.get("climate_zone") or 0.0),
        "dwelling_type": float(ctx.get("dwelling_type") or 0.0),
        "floor_area": float(ctx.get("floor_area") or 0.0),
        "occupants": float(ctx.get("occupants") or 0.0),
        "adjusted_income": float(ctx.get("adjusted_income") or 0.0),
    }])

def _numericize_background(df_src: pd.DataFrame, n: int = 200) -> pd.DataFrame:
    if df_src is None or len(df_src) == 0:
        return pd.DataFrame([{"province_code":6,"region_type":0.0,"climate_zone":0.0,"dwelling_type":0.0,"floor_area":1500.0,"occupants":3.0,"adjusted_income":90000.0}])
    cols_num = ["province_code","region_type","climate_zone","dwelling_type","floor_area","occupants","adjusted_income"]
    have = [c for c in cols_num if c in df_src.columns]
    if len(have) >= 5:
        bg = df_src[have].copy()
        for c in have: bg[c] = pd.to_numeric(bg[c], errors="coerce")
        bg = bg.dropna()
        if len(bg) == 0: return _numericize_background(None, n)
        if len(bg) > n: bg = bg.sample(n, random_state=42)
        for c in cols_num:
            if c not in bg.columns: bg[c] = 0.0
        return bg[cols_num]
    if "province" in df_src.columns:
        tmp = pd.DataFrame()
        tmp["province_code"] = df_src["province"].map(PROVINCE_INDEX).fillna(6).astype(int)
        for c in ["region_type","climate_zone","dwelling_type","floor_area","occupants","adjusted_income"]:
            if c in df_src.columns: tmp[c] = pd.to_numeric(df_src[c], errors="coerce")
        tmp = tmp.dropna()
        if len(tmp) == 0: return _numericize_background(None, n)
        if len(tmp) > n: tmp = tmp.sample(n, random_state=42)
        for c in cols_num:
            if c not in tmp.columns: tmp[c] = 0.0
        return tmp[cols_num]
    return _numericize_background(None, n)

def _canonical_base(name: str) -> str:
    s = str(name)
    if "__" in s: s = s.split("__", 1)[1]
    for b in ["province","province_code","region_type","climate_zone","dwelling_type","floor_area","occupants","adjusted_income"]:
        if s == b or s.startswith(b + "_"):
            return b
    return s

def _group_shap_to_readable(shap_vals: np.ndarray, feature_names: list) -> dict:
    groups_map = {"province":"Province","province_code":"Province","region_type":"Area type","climate_zone":"Climate","dwelling_type":"Dwelling type","floor_area":"Floor area","occupants":"Occupants","adjusted_income":"Income bracket"}
    agg = {v: 0.0 for v in set(groups_map.values())}
    for val, name in zip(shap_vals, feature_names):
        base = _canonical_base(name)
        label = groups_map.get(base, "Other")
        agg[label] = agg.get(label, 0.0) + float(val)
    return agg

def _build_contrib_table(agg: dict) -> pd.DataFrame:
    dfc = (pd.DataFrame([{"Feature":k,"Impact":float(v)} for k, v in agg.items()])
           .assign(Direction=lambda d: np.where(d["Impact"]>=0,"Increase","Decrease"), AbsImpact=lambda d: d["Impact"].abs())
           .sort_values("AbsImpact", ascending=False).reset_index(drop=True))
    total_abs = float(dfc["AbsImpact"].sum()) if len(dfc) else 0.0
    dfc["Percent of total"] = np.where(total_abs>0, (dfc["AbsImpact"]/total_abs*100.0).round(1), 0.0)
    return dfc[["Feature","Direction","Impact","Percent of total"]]

# ---- SMALL ADD: Pretty green/red number badges used in AI Advisor ----
def _num_badge(val: float, ref: float | None, unit: str = "") -> str:
    """Green if val < ref (good), red if val > ref (bad). If ref is None, neutral."""
    try:
        v = float(val); r = None if ref is None else float(ref)
    except Exception:
        return f"<b>{val}</b> {unit}".strip()
    if r is None:
        cls = "num-neutral"
    else:
        cls = "num-good" if v < r else ("num-bad" if v > r else "num-neutral")
    txt = f"{v:,.0f} {unit}".strip()
    return f"<span class='num-badge {cls}'>{txt}</span>"

def _money_badge(val: float, ref: float | None) -> str:
    try:
        v = float(val); r = None if ref is None else float(ref)
    except Exception:
        return f"<b>${val}</b>"
    if r is None:
        cls = "num-neutral"
    else:
        cls = "num-good" if v < r else ("num-bad" if v > r else "num-neutral")
    return f"<span class='num-badge {cls}'>${v:,.0f}</span>"

# ---- OPTIONAL: color SHAP table numbers too (kept working as-is) ----
def _styled_contrib_table(dfc: pd.DataFrame):
    red = "#d32f2f"   # bad (increases cost)
    green = "#2e7d32" # good (lowers cost)

    def _color_impact(v):
        try:
            return f"color:{red}; font-weight:700;" if float(v) >= 0 else f"color:{green}; font-weight:700;"
        except Exception:
            return ""
    def _color_direction(col: pd.Series):
        return [f"color:{red}; font-weight:700;" if x=="Increase" else f"color:{green}; font-weight:700;" for x in col]
    def _color_percent(col: pd.Series):
        return [f"color:{red};" if dfc.loc[idx, "Impact"] >= 0 else f"color:{green};" for idx in col.index]

    sty = (dfc.style
           .format({"Impact": "{:,.0f}", "Percent of total": "{:.1f}%"})
           .applymap(_color_impact, subset=["Impact"])
           .apply(_color_direction, subset=["Direction"])
           .apply(_color_percent, subset=["Percent of total"]))
    try:
        sty = sty.hide(axis="index")
    except Exception:
        try: sty = sty.hide_index()
        except Exception: pass
    return sty

def _impact_bar_figure(dfc: pd.DataFrame, title: str):
    impacts = dfc["Impact"].astype(float); labels = dfc["Feature"].tolist()
    colors = np.where(impacts >= 0, "#d32f2f", "#2e7d32")  # red=increase cost; green=decrease
    fig = go.Figure(go.Bar(x=impacts, y=labels, orientation="h", marker=dict(color=colors, line=dict(width=0)), hovertemplate="%{y}<br>Impact: %{x:.0f}<extra></extra>"))
    fig.update_layout(title=title, margin=dict(l=10,r=10,t=40,b=10), height=420, xaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor="#9aa0a6", showgrid=False), yaxis=dict(autorange="reversed"), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig

def _chips_row(dfc: pd.DataFrame, max_items: int = 3):
    top = dfc.head(max_items).to_dict("records")
    if not top: return
    chips = []
    for r in top:
        up = r["Impact"] >= 0
        color = "#d32f2f" if up else "#2e7d32"
        arrow = "↑" if up else "↓"
        chips.append(f"""<span style="display:inline-block;padding:6px 10px;margin:4px 6px;background:{color}1A;border:1px solid {color};border-radius:999px;font-size:.85rem;">{r["Feature"]} {arrow}</span>""")
    st.markdown("".join(chips), unsafe_allow_html=True)

def _income_center_from_label(label: str) -> float:
    centers = {
        "$20k - $40k":30000, "$40k - $60k":50000, "$60k - $80k":70000, "$80k - $100k":90000,
        "$100k - $120k":110000, "$120k - $150k":135000, "$150k+":160000
    }
    return float(centers.get(label, 90000.0))

def _local_income_delta_cost(ctx, model):
    """Compute local income-only effect by comparing to bracket center."""
    pre, est = _infer_pipeline_parts(model)
    curr_income = float(ctx.get("adjusted_income") or 0.0)
    base_income = _income_center_from_label(ctx.get("income_label"))
    if not np.isfinite(curr_income) or not np.isfinite(base_income) or abs(curr_income - base_income) < 1e-6:
        return 0.0
    x_curr = pd.DataFrame([{
        "province": ctx.get("province"),
        "region_type": float(ctx.get("region_type") or 0.0),
        "climate_zone": float(ctx.get("climate_zone") or 0.0),
        "dwelling_type": float(ctx.get("dwelling_type") or 0.0),
        "floor_area": float(ctx.get("floor_area") or 0.0),
        "occupants": float(ctx.get("occupants") or 0.0),
        "adjusted_income": curr_income,
    }])
    x_base = x_curr.copy()
    x_base.loc[:, "adjusted_income"] = base_income
    try:
        if pre is not None:
            Xc = pre.transform(x_curr)
            Xb = pre.transform(x_base)
        else:
            Xc = [[x_curr["floor_area"].iat[0], x_curr["adjusted_income"].iat[0], x_curr["occupants"].iat[0],
                   x_curr["region_type"].iat[0], x_curr["climate_zone"].iat[0], x_curr["dwelling_type"].iat[0]]]
            Xb = [[x_base["floor_area"].iat[0], x_base["adjusted_income"].iat[0], x_base["occupants"].iat[0],
                   x_base["region_type"].iat[0], x_base["climate_zone"].iat[0], x_base["dwelling_type"].iat[0]]]
        pc = float(est.predict(Xc)[0])
        pb = float(est.predict(Xb)[0])
        if not np.isfinite(pc) or not np.isfinite(pb):
            return 0.0
        return pc - pb
    except Exception:
        return 0.0

def explain_with_shap_pretty(ctx, df_source, model, prediction_value: float, title_suffix: str = "your prediction"):
    X_num = _make_input_row_from_ctx(ctx)
    pre, est = _infer_pipeline_parts(model)
    bg_num = _numericize_background(df_source)
    if pre is None:
        bg, x1 = bg_num.values, X_num.values
        feat_names = list(X_num.columns)
    else:
        X_raw_for_pre = pd.DataFrame([{
            "province": ctx.get("province"),
            "region_type": float(ctx.get("region_type") or 0.0),
            "climate_zone": float(ctx.get("climate_zone") or 0.0),
            "dwelling_type": float(ctx.get("dwelling_type") or 0.0),
            "floor_area": float(ctx.get("floor_area") or 0.0),
            "occupants": float(ctx.get("occupants") or 0.0),
            "adjusted_income": float(ctx.get("adjusted_income") or 0.0),
        }])
        if {"province","region_type","climate_zone","dwelling_type","floor_area","occupants","adjusted_income"}.issubset(set(df_source.columns)):
            bg_raw = df_source[["province","region_type","climate_zone","dwelling_type","floor_area","occupants","adjusted_income"]].dropna().copy()
            if len(bg_raw) > 200: bg_raw = bg_raw.sample(200, random_state=42)
        else:
            inv_map = {v:k for k,v in PROVINCE_INDEX.items()}
            bg_raw = pd.DataFrame({
                "province": bg_num["province_code"].map(inv_map).fillna("Ontario"),
                "region_type": bg_num["region_type"], "climate_zone": bg_num["climate_zone"],
                "dwelling_type": bg_num["dwelling_type"], "floor_area": bg_num["floor_area"],
                "occupants": bg_num["occupants"], "adjusted_income": bg_num["adjusted_income"],
            })
        bg = pre.transform(bg_raw); x1 = pre.transform(X_raw_for_pre)
        try: feat_names = list(pre.get_feature_names_out())
        except: feat_names = [f"f{i}" for i in range(x1.shape[1])]

    try: explainer = shap.TreeExplainer(est)
    except Exception: explainer = shap.Explainer(est, bg)
    sv = explainer(x1, check_additivity=False)
    shap_vals = np.array(sv.values).reshape(-1)

    agg = _group_shap_to_readable(shap_vals, feat_names)
    agg.setdefault("Income bracket", 0.0)

    # Income sensitivity override: if SHAP is tiny but model is locally sensitive, use local delta
    local_income_delta = _local_income_delta_cost(ctx, model)
    if np.isfinite(local_income_delta) and abs(local_income_delta) > max(0.02 * abs(prediction_value), 5.0):
        agg["Income bracket"] = float(local_income_delta)

    df_contrib = _build_contrib_table(agg)

    st.markdown("""
        <style>
            .shap-card {background:linear-gradient(180deg, rgba(250,252,255,0.95) 0%, rgba(255,255,255,0.97) 100%);border:1px solid rgba(0,0,0,0.06);box-shadow:0 10px 28px rgba(0,0,0,0.06);border-radius:16px;padding:16px 16px 10px 16px;margin-top:8px;margin-bottom:12px}
            .shap-header {font-size:1.05rem;font-weight:600;margin-bottom:2px}
            .shap-subtle {font-size:.9rem;color:#5f6368;margin-bottom:10px}
        </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="shap-card">', unsafe_allow_html=True)
    st.markdown('<div class="shap-header">Why you got this result</div>', unsafe_allow_html=True)
    st.markdown('<div class="shap-subtle">Red increases cost. Green lowers cost.</div>', unsafe_allow_html=True)

    fig = _impact_bar_figure(df_contrib, f"Factor impact for {title_suffix}")
    st.plotly_chart(fig, use_container_width=True)
    _chips_row(df_contrib, max_items=3)

    st.markdown("Top factors")
    styled_tbl = _styled_contrib_table(df_contrib)  # colored
    st.dataframe(styled_tbl, use_container_width=True)

    inc = [r["Feature"] for _, r in df_contrib.iterrows() if r["Impact"] > 0]
    dec = [r["Feature"] for _, r in df_contrib.iterrows() if r["Impact"] < 0]
    parts = []
    if inc: parts.append("Increase: " + ", ".join(inc[:3]))
    if dec: parts.append("Decrease: " + ", ".join(dec[:3]))
    if parts: st.markdown("Summary " + "; ".join(parts) + ".")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------
# Tab: AI Advisor
# ------------------------------------------------------

def render_ai_panel(ctx, df_in: pd.DataFrame, quick: bool):
    c_col, u_col = pick_targets(df_in)
    p_col = pick_text(df_in, ["Province"])
    clim_col = pick_text(df_in, ["Climate Region","Climate Zone"])
    inc_col = pick_text(df_in, ["Income Bracket"])

    prov_med_raw = _median_or_center(df_in, c_col, {"Province": ctx["province"]}) if c_col and p_col else None
    clim_med_raw = _median_or_center(df_in, c_col, {clim_col: ctx["climate_label"]}) if c_col and clim_col else None
    inc_med_raw  = _median_or_center(df_in, c_col, {inc_col: ctx["income_label"]}) if c_col and inc_col else None
    nat_cost_raw = _median_or_center(df_in, c_col) if c_col else None
    nat_use_raw  = _median_or_center(df_in, u_col) if u_col else None

    # If dataset looks scaled, fall back to calibrated medians
    prov_med = _fallback(prov_med_raw, _scaled_cost(PROVINCE_MEDIAN_COST.get(ctx["province"], NATIONAL_MEDIAN_COST)))
    clim_med = _fallback(clim_med_raw, _scaled_cost(CLIMATE_MEDIAN_COST.get(ctx["climate_label"], NATIONAL_MEDIAN_COST)))
    inc_med  = _fallback(inc_med_raw,  _scaled_cost(INCOME_MEDIAN_COST.get(ctx["income_label"], NATIONAL_MEDIAN_COST)))
    nat_cost = _fallback(nat_cost_raw, _scaled_cost(NATIONAL_MEDIAN_COST))
    nat_use  = _fallback(nat_use_raw,  _scaled_use(NATIONAL_MEDIAN_USE))

    # ---------- Minimal CSS (chips + bigger, lighter color badges) ----------
    st.markdown("""
    <style>
      /* Input chips: subtle white background for contrast */
      .ai-chips { display:flex; flex-wrap:wrap; gap:8px; margin:6px 0 10px 0; }
      .ai-chips .chip {
        display:inline-flex; align-items:center; gap:6px;
        padding:8px 12px; border-radius:14px;
        background:#ffffff; border:1px solid #E5E7EB;
        box-shadow:0 2px 6px rgba(0,0,0,.06);
        font-size:0.96rem; font-weight:700; color:#111;
      }
      .ai-chips .chip b { font-weight:800; color:#0f172a; }

      /* Bigger, pill/bubbly percent/delta badges (used only on vs medians + Δ vs now) */
      .ai-pct-badge{
        display:inline-flex; align-items:center; gap:6px;
        padding:8px 14px; border-radius:999px;
        font-weight:900; font-size:1.15rem; letter-spacing:.2px;
        border:1px solid transparent; line-height:1;
        box-shadow:0 2px 10px rgba(0,0,0,.06);
      }
      .ai-pct-badge .num { font-variant-numeric: tabular-nums; }

      /* Lighter, friendly greens/reds */
      .ai-pct-good{ color:#166534; background:#ECFDF5; border-color:#86EFAC; }  /* light mint */
      .ai-pct-bad{  color:#991B1B; background:#FEF2F2; border-color:#FECACA; }  /* light rose */
      .ai-pct-neutral{ color:#111827; background:#F8FAFC; border-color:#E5E7EB; }
    </style>
    """, unsafe_allow_html=True)

    # Green if negative is good (lower cost/use vs reference), red if positive
    def _pct_badge(val: float, good_when_negative: bool = True) -> str:
        try:
            v = float(val)
        except Exception:
            return f"<b>{val}</b>"
        good = (v < 0) if good_when_negative else (v > 0)
        cls = "ai-pct-good" if good else ("ai-pct-bad" if abs(v) > 1e-9 else "ai-pct-neutral")
        sign = "+" if v > 0 else ""
        return f"<span class='ai-pct-badge {cls}'><span class='num'>{sign}{v:.0f}%</span></span>"

    # For "Δ vs now" cells in What-if tables: decreases (negative) are good → green
    def _delta_badge_colored(delta_val: float, unit_label: str = "", currency: bool = False) -> str:
        try:
            v = float(delta_val)
        except Exception:
            return f"{delta_val}"
        good = (v < 0)
        cls = "ai-pct-good" if good else ("ai-pct-bad" if abs(v) > 1e-9 else "ai-pct-neutral")
        arrow = "▼" if v < 0 else ("▲" if v > 0 else "•")
        if currency:
            formatted = f"${abs(v):,.0f}"
        else:
            formatted = f"{abs(v):,.0f} {unit_label}".strip()
        return f"<span class='ai-pct-badge {cls}'>{arrow} <span class='num'>{formatted}</span></span>"

    # ---------- Input chips ----------
    chips_html = f"""
    <div class="ai-chips">
      <span class="chip">Province: <b>{ctx['province']}</b></span>
      <span class="chip">Climate: <b>{ctx['climate_label']}</b></span>
      <span class="chip">Dwelling: <b>{ctx['dwelling_label']}</b></span>
      <span class="chip">Area: <b>{ctx['region_label']}</b></span>
      <span class="chip">Floor: <b>{ctx['floor_area']} sq ft</b></span>
      <span class="chip">Occupants: <b>{ctx['occupants']}</b></span>
      <span class="chip">Income: <b>{ctx['income_label']}</b></span>
    </div>
    """
    st.markdown(chips_html, unsafe_allow_html=True)

    # ---------- Top summary cards ----------
    gl1, gl2 = st.columns(2)
    with gl1:
        p_vs = _pct_diff(ctx['cost'], prov_med)
        c_vs = _pct_diff(ctx['cost'], clim_med)
        u_vs_n = _pct_diff(ctx['use'],  nat_use)
        c_vs_n = _pct_diff(ctx['cost'], nat_cost)
        st.markdown(
            f"""
            <div class="ai-card">
              <h4>How you compare</h4>
              <div class="ai-kv">
                <div>vs Province median</div><div>{_pct_badge(p_vs)} <span class='muted'>(cost)</span></div>
                <div>vs Climate median</div><div>{_pct_badge(c_vs)} <span class='muted'>(cost)</span></div>
                <div>vs National</div><div>{_pct_badge(u_vs_n)} use, {_pct_badge(c_vs_n)} cost</div>
              </div>
            </div>
            """, unsafe_allow_html=True
        )
    with gl2:
        st.markdown(
            f"""
            <div class="ai-card">
              <h4>Your prediction</h4>
              <div class="ai-kv">
                <div>Energy Use</div><div><b>{ctx['use']:,.0f} kWh</b></div>
                <div>Energy Cost</div><div><b>${ctx['cost']:,.0f}</b></div>
              </div>
            </div>
            """, unsafe_allow_html=True
        )

    # ---------- Quick mode ----------
    if quick:
        actions = []
        actions.append("Check time-of-use and shift laundry/dishwasher off-peak if available.")
        if ctx["climate_label"] in {"Arctic","Cold","Subarctic"}:
            actions.append("Air-seal attic hatch + add attic insulation; sleep/away setback ≈ −2 °C.")
        elif ctx["climate_label"] == "Humid":
            actions.append("Clean AC filters monthly; run a dehumidifier to reduce AC runtime.")
        else:
            actions.append("Seal window/door gaps and use smart schedules for HVAC.")
        if ctx["floor_area"] >= 2500:
            actions.append("Zone seldom-used rooms; reduce setpoints there.")
        else:
            actions.append("Prioritize plug loads and hot-water routines in smaller spaces.")
        st.markdown(
            f"""
            <div class="ai-card">
              <h4>Quick actions</h4>
              <ul style="margin:6px 0 0 18px;" class="muted">
                <li>{actions[0]}</li><li>{actions[1]}</li><li>{actions[2]}</li>
              </ul>
            </div>
            """, unsafe_allow_html=True
        )
        return

    # ---------- Detail cards ----------
    det1, det2 = st.columns(2)
    with det1:
        pos_vs_prov = _pct_diff(ctx['cost'], prov_med)
        st.markdown(
            f"""
            <div class="ai-card">
              <h4>Province effect</h4>
              <div class="ai-kv">
                <div>Median cost in province</div><div><b>${prov_med:,.0f}</b></div>
                <div>Your position</div><div>{_pct_badge(pos_vs_prov)}</div>
              </div>
              <ul style="margin:8px 0 0 18px;" class="muted"><li>Regional prices & generation mix set your baseline.</li><li>Use off-peak windows where available.</li></ul>
            </div>
            """, unsafe_allow_html=True
        )
    with det2:
        pos_vs_clim = _pct_diff(ctx['cost'], clim_med)
        st.markdown(
            f"""
            <div class="ai-card">
              <h4>Climate effect</h4>
              <div class="ai-kv">
                <div>Median cost in climate</div><div><b>${clim_med:,.0f}</b></div>
                <div>Your position</div><div>{_pct_badge(pos_vs_clim)}</div>
              </div>
              <ul style="margin:8px 0 0 18px;" class="muted">
                <li>{"Air sealing & attic insulation first; night setbacks." if ctx["climate_label"] in {"Arctic","Cold","Subarctic"} else ("Clean filters & dehumidify to reduce AC runtime." if ctx["climate_label"]=="Humid" else "Use smart schedules to smooth heating/cooling.")}</li>
              </ul>
            </div>
            """, unsafe_allow_html=True
        )

    st.markdown(
        f"""
        <div class="ai-card">
          <h4>Income pattern</h4>
          <div class="ai-kv"><div>Median cost in bracket</div><div><b>${_scaled_cost(INCOME_MEDIAN_COST.get(ctx['income_label'], NATIONAL_MEDIAN_COST)):,.0f}</b></div></div>
          <ul style="margin:8px 0 0 18px;" class="muted">
            <li>{"Large homes: fix envelope before equipment swaps; audit always-on devices." if ctx["income_label"] in {"$120k - $150k","$150k+"} else ("LEDs in most-used fixtures; eco modes; insulate first 2 m of hot-water pipe." if ctx["income_label"] in {"$20k - $40k","$40k - $60k"} else "Balance comfort & efficiency with targeted controls; track seasonal swings.")}</li>
          </ul>
        </div>
        """, unsafe_allow_html=True
    )

    # ---------- What-if tables ----------
    sims = _what_if_scenarios(ctx)
    def _rows(group):
        rows = []
        for s in [x for x in sims if x["group"] == group]:
            du_badge = _delta_badge_colored(s["du"], "kWh", currency=False)
            dc_badge = _delta_badge_colored(s["dc"], "$",   currency=True)
            rows.append(
                f"<tr><td><b>{s['name']}</b><br><span class='muted'>{s['note']}</span></td>"
                f"<td>{s['use']:,.0f} kWh / ${s['cost']:,.0f}</td><td>{du_badge} {dc_badge}</td></tr>"
            )
        return "".join(rows)

    st.markdown(f"""<div class="ai-card"><h4>What-if impact — Home changes</h4><table class="wtbl"><tr><th>Scenario</th><th>New use / cost</th><th>Δ vs now</th></tr>{_rows("Home changes")}</table></div>""", unsafe_allow_html=True)
    st.markdown(f"""<div class="ai-card"><h4>What-if impact — Behavior changes</h4><table class="wtbl"><tr><th>Scenario</th><th>New use / cost</th><th>Δ vs now</th></tr>{_rows("Behavior changes")}</table></div>""", unsafe_allow_html=True)
    econ_rows = _rows("Economics")
    if econ_rows:
        st.markdown(f"""<div class="ai-card"><h4>What-if impact — Economics</h4><table class="wtbl"><tr><th>Scenario</th><th>New use / cost</th><th>Δ vs now</th></tr>{econ_rows}</table></div>""", unsafe_allow_html=True)

    # Targeted tips unchanged
    tlines = _build_targeted_tips(ctx)
    st.markdown(f"""<div class="ai-card"><h4>Targeted actions for you</h4><ul style="margin:6px 0 0 18px;" class="muted">{''.join([f"<li>{ln}</li>" for ln in tlines])}</ul></div>""", unsafe_allow_html=True)


# ---------- Keep the outer tab render block in your file ----------
with tab_ai:
    st.markdown('<div class="tab-skin tab-ai">', unsafe_allow_html=True)
    lh, rh = st.columns([1,1])
    with lh:
        st.markdown("<div class='main-title'>AI Advisor</div>", unsafe_allow_html=True)
        st.caption("Personalized guidance based on your most recent prediction.")
    with rh:
        st.write("")
        if st.button("Reset advisor", key="ai_reset_btn", use_container_width=True):
            st.session_state["last_pred"] = None
            st.session_state["ai_history"] = []
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()

    last_pred = st.session_state.get("last_pred")
    if not last_pred:
        st.info("No prediction yet. Go to Personal Prediction and click Predict for a detailed breakdown here.")
    else:
        province = st.session_state.get("prov_sel")
        region_type_label = st.session_state.get("region_sel")
        climate_zone_label = st.session_state.get("climate_sel")
        dwelling_type_label = st.session_state.get("dwelling_sel")
        floor_area = st.session_state.get("floor_sel")
        occupants = st.session_state.get("occ_sel")
        income_bracket = st.session_state.get("income_sel")

        ctx = {
            "province": province,
            "region_label": region_type_label,
            "region_type": REGION_MAP.get(region_type_label, 0),
            "climate_label": climate_zone_label,
            "climate_zone": CLIMATE_MAP.get(climate_zone_label, 0),
            "dwelling_label": dwelling_type_label,
            "dwelling_type": DWELLING_MAP.get(dwelling_type_label, 0),
            "floor_area": int(floor_area or 0),
            "occupants": int(occupants or 0),
            "income_label": income_bracket,
            "adjusted_income": int(INCOME_MAP.get(income_bracket, 0)),
            "use": float(last_pred["use"]),
            "cost": float(last_pred["cost"]),
        }

        mode = st.radio("View", ["Quick", "Full"], index=0, horizontal=True, key="ai_mode_tab_main")
        render_ai_panel(ctx, df, quick=(mode == "Quick"))

        model_for_shap = energy_cost_model
        if mode == "Full":
            explain_with_shap_pretty(ctx, df, model_for_shap, prediction_value=ctx["cost"], title_suffix="your cost prediction")
        else:
            with st.expander("Why is this my result"):
                st.caption("Switch View to Full for a detailed factor breakdown.")
    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# Energy Chatbot
# ======================================================
def _neutralize_streamlit_chat():
    css = """
    <style>
    [data-testid="stChatFloatingInputContainer"],
    .stChatFloatingInputContainer,
    section[aria-label="Chat input"],
    div[aria-label="Chat input"],
    .stChatInput,
    .stChatInputContainer,
    [data-testid="stBottomBlockContainer"],
    div[role="complementary"][data-testid="stBottomBlockContainer"] {
        position: fixed !important;
        bottom: -200vh !important;
        opacity: 0 !important;
        pointer-events: none !important;
        width: 0 !important;
        height: 0 !important;
        min-height: 0 !important;
        max-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
        display: block !important;
        z-index: -1 !important;
    }
    [data-testid="stAppViewContainer"],
    [data-testid="stVerticalBlock"],
    [data-testid="stMainBlockContainer"],
    section.main > div.block-container {
        padding-bottom: 0 !important;
        margin-bottom: 0 !important;
    }
    html, body, [data-testid="stAppViewContainer"] { overflow-x: hidden !important; }

    /* Big Open Chatbot button styles */
    .open-chat-hero { display:flex; justify-content:center; }
    .open-chat-btn button {
        all: unset;
        display: inline-flex;
        align-items: center;
        gap: 12px;
        padding: 22px 32px;
        border-radius: 18px;
        background: linear-gradient(90deg,#1f5cff,#2563EB);
        color: #fff;
        font-size: 1.35rem;
        font-weight: 900;
        letter-spacing: .2px;
        box-shadow: 0 14px 34px rgba(0,0,0,.25);
        cursor: pointer;
        text-align: center;
        min-width: 360px;
    }
    .open-chat-btn button:hover { filter: brightness(1.07); transform: translateY(-1px); }
    .open-chat-sub { text-align:center; color:#6B7280; margin-top:8px; font-size:.95rem; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    janitor = """
    <script>
    (function(){
      const pushOffscreen = (el) => {
        if (!el) return;
        el.style.position = 'fixed';
        el.style.bottom = '-200vh';
        el.style.opacity = '0';
        el.style.pointerEvents = 'none';
        el.style.width = '0';
        el.style.height = '0';
        el.style.minHeight = '0';
        el.style.maxHeight = '0';
        el.style.margin = '0';
        el.style.padding = '0';
        el.style.overflow = 'hidden';
        el.style.zIndex = '-1';
      };
      const selectors = [
        '[data-testid="stChatFloatingInputContainer"]',
        '.stChatFloatingInputContainer',
        'section[aria-label="Chat input"]',
        'div[aria-label="Chat input"]',
        '.stChatInput', '.stChatInputContainer',
        '[data-testid="stBottomBlockContainer"]',
        'div[role="complementary"][data-testid="stBottomBlockContainer"]'
      ];
      const neutralizeAll = () => {
        selectors.forEach(sel => {
          document.querySelectorAll(sel).forEach(pushOffscreen);
        });
        const pads = document.querySelectorAll(
          '[data-testid="stAppViewContainer"],[data-testid="stVerticalBlock"],[data-testid="stMainBlockContainer"],section.main > div.block-container'
        );
        pads.forEach(p => { p.style.paddingBottom='0px'; p.style.marginBottom='0px'; });
      };
      neutralizeAll();
      const obs = new MutationObserver(neutralizeAll);
      obs.observe(document.body, {childList:true, subtree:true});
    })();
    </script>
    """
    st.components.v1.html(janitor, height=0, width=0)

_neutralize_streamlit_chat()

# ------------------------------------------------------
# 1) Chat UI styles 
# ------------------------------------------------------
_CHAT_CSS = """
<style>
.chat-card{
  width:100%;
  max-width:980px;
  margin: 6px auto 0 auto;
  border:1px solid rgba(0,0,0,0.06);
  border-radius:18px;
  background:#ffffff;
  box-shadow:0 10px 28px rgba(0,0,0,0.08);
  padding:14px 14px 10px 14px;
}
.chat-header{ display:flex; align-items:center; justify-content:space-between; padding:6px 6px 10px 6px; }
.chat-title{ display:flex; align-items:center; gap:10px; font-weight:900; font-size:1.15rem; letter-spacing:.15px; }
.badge{ font-size:.78rem; padding:3px 8px; border-radius:999px; background:#EEF2FF; color:#1E3A8A; border:1px solid #C7D2FE; }

.chat-body{ max-height:62vh; overflow-y:auto; overflow-x:hidden; padding:6px; background:linear-gradient(#fff,#fff); }
.row{ display:flex; gap:10px; margin:12px 6px; }
.row.user{ justify-content:flex-end; }
.row.bot{ justify-content:flex-start; }

.ava{ flex:0 0 30px; height:30px; width:30px; border-radius:8px; display:flex; align-items:center; justify-content:center; font-weight:800; font-size:.9rem; }
.ava.user{ background:linear-gradient(90deg,#2563EB,#3B82F6); color:#fff; box-shadow:0 1px 2px rgba(0,0,0,.12); }
.ava.bot{ background:#FFF7E6; color:#111; border:1px solid #FFD78A; }

.bubble{ max-width:82%; padding:12px 14px; border-radius:14px; line-height:1.48; font-size:1rem; box-shadow:0 1px 2px rgba(0,0,0,.06); }
.bubble.user{ background:linear-gradient(90deg,#2563EB,#3B82F6); color:#fff; border:0; }
.bubble.bot{ background:#F8FAFC; border:1px solid #E2E8F0; color:#0f172a; }
.bubble.bot h4{ margin:0 0 8px 0; font-size:1.05rem; }
.bubble.bot p{ margin:0 0 10px 0; }
.meta{ font-size:.78rem; color:#6B7280; margin-top:4px; }

.chat-input{ border-top:1px solid #E5E7EB; padding:10px 4px 8px 4px; background:#fff; }
.input-row{ display:flex; gap:8px; align-items:flex-end; }
.input-row textarea{ width:100%; min-height:48px; max-height:140px; resize:vertical; border:1px solid #E5E7EB; border-radius:12px; padding:10px 12px; font-size:1rem; background:#fff; }
.send-btn{ border-radius:12px; padding:10px 14px; font-weight:800; border:0; font-size:1rem; background:#2563EB; color:#fff; }
.hint{ color:#6B7280; font-size:.88rem; margin-top:6px; }

/* Preset chips (BELOW the typing bar) */
.chips-wrap{ border-top:1px dashed #E5E7EB; padding-top:10px; margin-top:8px; }
.chips-title{ font-weight:800; font-size:.95rem; color:#1f2937; margin-bottom:6px; }
.chips-help{ color:#6B7280; font-size:.88rem; margin:2px 0 8px 0; }
.chips{ display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:8px; }
.chip{ padding:10px 12px; text-align:center; border-radius:999px; background:#EEF2FF; color:#1E3A8A; border:1px solid #C7D2FE; font-weight:800; font-size:.95rem; cursor:pointer; }
.chip:hover{ filter:brightness(1.04); }

/* Card scrollbar only */
.chat-body::-webkit-scrollbar{ height:8px; width:8px; }
.chat-body::-webkit-scrollbar-thumb{ background:#CBD5E1; border-radius:999px; }
.chat-body::-webkit-scrollbar-track{ background:transparent; }
</style>
"""
st.markdown(_CHAT_CSS, unsafe_allow_html=True)

# ------------------------------------------------------
# 2) State helpers
# ------------------------------------------------------
def _normalize_history():
    h=[]
    for m in st.session_state.get("chat_history", []):
        if isinstance(m, dict) and {"role","content"}.issubset(m.keys()):
            h.append(m)
        elif isinstance(m, tuple) and len(m)==2:
            h.append({"role":m[0],"content":m[1]})
        else:
            h.append({"role":"bot","content":str(m)})
    st.session_state.chat_history=h

def _ensure_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history=[{
            "role":"bot",
            "content":(
                "Welcome. I use your current selections to explain your result and suggest practical steps you can take now. "
                "You can type a question in the box, or tap a quick question below to see an example. "
                "I do not change any settings for you—this chat only explains and suggests, so you stay in control."
            ),
            "ts": datetime.now().isoformat()
        }]
    if "chat_buf" not in st.session_state:
        st.session_state.chat_buf=""
    # NEW: open/close state (default: closed so the big Open button shows)
    if "chat_open" not in st.session_state:
        st.session_state.chat_open = False
    _normalize_history()

# ------------------------------------------------------
# 3) Live context from app
# ------------------------------------------------------
def _live_ctx():
    province = st.session_state.get("prov_sel") or st.session_state.get("province") or "your province"
    climate  = st.session_state.get("climate_sel") or st.session_state.get("climate") or "your climate"
    dwelling = st.session_state.get("dwelling_sel") or st.session_state.get("dwelling_type") or "your dwelling"
    floor    = st.session_state.get("floor_sel") or st.session_state.get("sqft") or 1500
    occ      = st.session_state.get("occ_sel") or st.session_state.get("occupants") or 3
    income   = st.session_state.get("income_sel") or st.session_state.get("income_bracket") or "mid income"
    last     = st.session_state.get("last_pred") or {}
    use_kwh  = float(last.get("use",0) or 0)
    cost_usd = float(last.get("cost",0) or 0)
    try: floor = int(floor)
    except: pass
    try: occ = int(occ)
    except: pass
    return {
        "province":str(province),
        "climate":str(climate),
        "dwelling":str(dwelling),
        "floor_area":floor,
        "occupants":occ,
        "income":str(income),
        "use":use_kwh,
        "cost":cost_usd
    }

# ------------------------------------------------------
# 4) Conversational response generator
# ------------------------------------------------------
def _band(c):
    c=(c or "").lower()
    if any(k in c for k in ["arctic","subarctic","cold","polar"]): return "cold"
    if any(k in c for k in ["humid","subtrop","tropical","wet"]): return "humid"
    if any(k in c for k in ["dry","arid","semi-arid","desert"]): return "dry"
    return "temperate"

def _match(q,*phrases):
    q=q.lower()
    return any(p in q for p in phrases)

def _p(txt): return f"<p>{txt}</p>"

def generate_bot_response(user_msg:str, ctx:dict)->str:
    prov, clim, dwell = ctx["province"], ctx["climate"], ctx["dwelling"]
    sqft, occ, inc = ctx["floor_area"], ctx["occupants"], ctx["income"]
    use_kwh, cost = ctx["use"], ctx["cost"]
    band = _band(clim)
    q = re.sub(r"\s+"," ",user_msg).strip().lower()

    if _match(q, "lower my bill", "reduce bill", "save money", "cut cost", "lower cost", "reduce cost"):
        out  = _p(f"Here is a practical starting point using your current selections. In {prov}, shifting laundry and dishwashing into off-peak hours usually lowers cost without affecting comfort. Avoid running several large appliances at the same time during peak periods.")
        out += _p(f"With about {sqft} square feet and {occ} occupants, hot water and plug loads tend to be the fastest wins. Use a low-flow shower head, wash clothes with cold water, and place a smart power strip on the entertainment setup to reduce idle power.")
        out += _p(f"In a {dwell}, sealing and basic insulation often make the equipment run less. Weather-strip exterior doors, seal visible gaps at window trim, and add insulation in the top floor or attic where it is reachable.")
        if band == "cold":
            out += _p("Because the climate is on the colder side, sealing and insulation usually come before equipment changes. Small leaks can keep a system running longer than necessary.")
        elif band == "humid":
            out += _p("Because the climate is humid, keep indoor humidity near fifty to fifty-five percent, replace filters on schedule, and use fans so the thermostat can stay a little higher with the same comfort.")
        elif band == "dry":
            out += _p("Because the climate is dry, exterior shading and night ventilation help bring temperatures down without heavy air-conditioning use.")
        if isinstance(sqft, int) and sqft >= 2500:
            out += _p("Given the size of the home, a simple zoning habit helps. Close or set back rooms that are not used often so the main areas condition more efficiently.")
        if use_kwh or cost:
            out += _p(f"Your last prediction was around {use_kwh:,.0f} kilowatt-hours and ${cost:,.0f}. A reasonable first target is a reduction of ten to fifteen percent over the next billing cycle.")
        return out

    if _match(q, "which inputs matter", "what matters", "drivers", "feature importance"):
        out  = _p(f"The model focuses first on floor area because it sets the volume to condition and the surface that can lose or gain heat. At about {sqft} square feet, small changes in setpoints or insulation can have a visible effect.")
        out += _p(f"Occupants are next because people cook, shower, and use electronics. With {occ} people, hot water and plug loads rise with daily routines.")
        out += _p(f"Climate sets heating and cooling intensity, while province sets the price baseline and plan options. In {prov}, both the generation mix and the tariff structure shape the final bill.")
        out += _p(f"Dwelling type affects shell quality and shared walls. A {dwell} can save energy through shared surfaces or lose energy if the envelope leaks. Income is a proxy for appliance mix and comfort preferences, which affects how long major equipment runs.")
        return out

    if _match(q, "province effect", "province impact", "why province", "province"):
        out  = _p(f"Province matters because it sets the rate structure and reflects local generation. In {prov}, the model establishes a baseline price. Your {dwell} in a {clim} climate with about {sqft} square feet and {occ} occupants shifts the result above or below that baseline.")
        out += _p("It is worth comparing a plan with a higher fixed charge and lower energy rate to a plan with a lower fixed charge and higher rate, especially if usage changes by season.")
        return out

    if _match(q, "climate advice", "climate impact", "cold", "humid", "dry", "ac", "heating", "cooling"):
        if band == "cold":
            return _p("For a cold climate, seal the attic hatch, ceiling penetrations, and rim joists, then add attic insulation to at least code-plus levels. Use small night setbacks and keep a steady day setpoint. Replace filters on schedule and check that the system is not short cycling.")
        if band == "humid":
            return _p("For a humid climate, keep indoor humidity near fifty to fifty-five percent and clean air filters during the season. Seal accessible ductwork and add exterior shading to south- and west-facing windows to reduce solar gains.")
        if band == "dry":
            return _p("For a dry climate, use night ventilation when outdoor temperatures drop and consider light-colored roofing or shades to reduce solar gain. Avoid oversized air conditioners and verify that duct flow is balanced.")
        return _p("In a temperate climate, lean on schedules during shoulder seasons, address plug loads and hot water first, and seal gaps around doors and windows before changing equipment.")

    if _match(q, "insulation", "air seal", "envelope", "windows", "doors"):
        return _p("Start with the envelope. Air-seal the attic hatch, ceiling penetrations, and rim joists. Add insulation in the attic until you reach a code-plus level. Then weather-strip exterior doors and caulk the window trim. These steps reduce the run time for both heating and cooling equipment.")

    if _match(q, "heat pump", "furnace", "hvac upgrade", "equipment"):
        return _p("When upgrading equipment, size it to the actual load and confirm airflow and refrigerant measurements during commissioning. Cold-climate heat pumps perform well when paired with good sealing and insulation. In humid climates, focus on latent moisture control and duct sealing so comfort holds without overuse.")

    if _match(q, "time of use", "off-peak", "peak", "tou", "tariff"):
        return _p("Time-of-use pricing rewards routine. Run laundry and dishwashing in off-peak windows, pre-cool or pre-heat lightly before peak hours, and avoid stacking several large loads at once. Delay-start settings and simple smart plugs help the habit stick.")

    if _match(q, "plan", "three step plan", "3 step plan"):
        step1 = f"In {prov}, choose off-peak windows for laundry and dishwashing and avoid stacking large appliances at peak."
        step2 = f"Seal and maintain: weather-strip exterior doors, seal visible window-trim gaps, and replace air filters on schedule."
        if band == "cold":
            step2 = "Seal and insulate first in the attic and at rim joists, then keep small night setbacks with a steady day setpoint."
        step3 = f"Tackle hot water and plug loads. Use a low-flow shower head, cold-wash laundry, and a smart power strip for the media area."
        out  = _p("Here is a short plan that fits a weekend and a small budget.")
        out += _p(step1)
        out += _p(step2)
        out += _p(step3)
        if use_kwh or cost:
            out += _p(f"With a last prediction of about {use_kwh:,.0f} kilowatt-hours and ${cost:,.0f}, this plan aims for an initial reduction near ten to fifteen percent.")
        return out

    out  = _p(f"I use your current selections to explain and improve results. I see {prov} for province, {clim} for climate, {dwell} for dwelling type, about {sqft} square feet, and {occ} occupants with income marked as {inc}. You can ask about lowering your bill, which inputs matter most, how province affects cost, or you can request a simple plan.")
    out += _p("If you want a short plan now, type plan and I will outline three focused steps you can take this week.")
    return out

# ------------------------------------------------------
# 5) Render inline chat with input first
# ------------------------------------------------------
def render_energy_chatbot_inline():
    _ensure_state()
    ctx = _live_ctx()

    # When closed, show big OPEN button and return early
    if not st.session_state.chat_open:
        st.markdown('<div class="open-chat-hero open-chat-btn">', unsafe_allow_html=True)
        if st.button("💬 Open Energy Chatbot", key="chat_open_btn"):
            st.session_state.chat_open = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="open-chat-sub">Explore quick questions or type your own when the chatbot opens.</div>', unsafe_allow_html=True)
        return

    # Chat card (visible only when open)
    st.markdown('<div class="chat-card">', unsafe_allow_html=True)

    # Header with actions (New chat + Close)
    c1, c2 = st.columns([7,5])
    with c1:
        st.markdown('<div class="chat-header"><div class="chat-title">Energy Chatbot <span class="badge">Beta</span></div></div>', unsafe_allow_html=True)
    with c2:
        colN, colC = st.columns(2)
        with colN:
            if st.button("New chat", key="chat_new_inline"):
                st.session_state.chat_history=[{
                    "role":"bot",
                    "content":"New conversation started. Tell me your goal and I will suggest the fastest path using your current selections. You can type a question or tap one of the quick questions below.",
                    "ts": datetime.now().isoformat()
                }]
                st.session_state.chat_buf=""
                st.rerun()
        with colC:

            if st.button("Close", key="chat_close_inline"):
                st.session_state.chat_open = False
                st.rerun()

    # Messages
    st.markdown('<div class="chat-body">', unsafe_allow_html=True)
    for m in st.session_state.chat_history:
        role = m.get("role","bot")
        content = m.get("content","")
        ts = m.get("ts")
        timestr = ""
        if ts:
            try: timestr = datetime.fromisoformat(ts).strftime("%-I:%M %p")
            except: pass
        if role == "user":
            st.markdown(
                f'<div class="row user">'
                f'  <div class="bubble user"><div>{content}</div>'
                f'    <div class="meta">You{(" · "+timestr) if timestr else ""}</div>'
                f'  </div>'
                f'  <div class="ava user">U</div>'
                f'</div>', unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="row bot">'
                f'  <div class="ava bot">AI</div>'
                f'  <div class="bubble bot"><div>{content}</div>'
                f'    <div class="meta">Assistant{(" · "+timestr) if timestr else ""}</div>'
                f'  </div>'
                f'</div>', unsafe_allow_html=True
            )
    st.markdown('</div>', unsafe_allow_html=True)

    # Input first
    st.markdown('<div class="chat-input">', unsafe_allow_html=True)
    with st.form(key="chat_form_inline", clear_on_submit=True):
        st.session_state.chat_buf = st.text_area(
            label="Type your message",
            value="",
            placeholder="Ask about lowering your bill, which inputs matter, province effect, climate advice, or type plan",
            height=52,
            label_visibility="collapsed"
        )
        col_send, col_hint = st.columns([1,6])
        with col_send:
            submitted = st.form_submit_button("Send", use_container_width=True)
        with col_hint:
            st.markdown('<div class="hint">Press Enter to send. Use Shift and Enter to add a new line.</div>', unsafe_allow_html=True)

    # Handle input submission
    if 'submitted' in locals() and submitted:
        txt = (st.session_state.chat_buf or "").strip()
        if txt:
            st.session_state.chat_history.append({"role":"user","content":txt, "ts": datetime.now().isoformat()})
            reply = generate_bot_response(txt, ctx)
            st.session_state.chat_history.append({"role":"bot","content":reply, "ts": datetime.now().isoformat()})
            st.rerun()

    # Preset chips BELOW the typing bar with a mini title and help text
    chips_title_html = """
      <div class="chips-wrap">
        <div class="chips-title">Quick questions</div>
        <div class="chips-help">Tap a button to auto-fill a common question. You can edit or ask something different in the box above at any time.</div>
      """
    st.markdown(chips_title_html, unsafe_allow_html=True)

    chip_labels = [
        "Lower my bill",
        "Which inputs matter",
        "Province effect",
        "Climate advice",
        "Three step plan"
    ]
    st.markdown('<div class="chips">', unsafe_allow_html=True)
    chip_cols = st.columns(2)
    for i, label in enumerate(chip_labels):
        with chip_cols[i % 2]:
            if st.button(label, key=f"chip_inline_{i}"):
                query = "three step plan" if label == "Three step plan" else label
                st.session_state.chat_history.append({"role":"user","content":label, "ts": datetime.now().isoformat()})
                reply = generate_bot_response(query, ctx)
                st.session_state.chat_history.append({"role":"bot","content":reply, "ts": datetime.now().isoformat()})
                st.rerun()
    st.markdown('</div></div>', unsafe_allow_html=True)  # close chips + wrap

    st.markdown('</div>', unsafe_allow_html=True)  # end card

# ------------------------------------------------------
# 6) Render
# ------------------------------------------------------
render_energy_chatbot_inline()
