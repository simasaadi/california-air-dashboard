import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ===============================
# Page setup (single-screen layout)
# ===============================
st.set_page_config(
    page_title="California Air Quality — Scientific Animated",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Tighten padding & header spacing
st.markdown("""
<style>
.block-container {padding-top:0.6rem; padding-bottom:0.6rem; max-width:1400px;}
h1, h2, h3 { margin:0.4rem 0 0.6rem 0; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='margin-top:0'>California Air Quality — Scientific Animated Dashboard</h2>",
            unsafe_allow_html=True)

# Figure heights (tune if needed to fit your screen)
MAP_H = 420     # animated map
FIG_H = 340     # correlation / regional / 3D

# =======================
# Data loader (cached)
# =======================
@st.cache_data(show_spinner=False)
def load_data(file):
    df = pd.read_csv(file)
    required = ["Date","Pollutant","Concentration","County","Site Latitude","Site Longitude","Site ID"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date","Concentration","Site Latitude","Site Longitude","Pollutant","County"])
    df["MonthLabel"] = df["Date"].dt.to_period("M").astype(str)
    return df

# =======================
# Sidebar (optional data input)
# =======================
st.sidebar.header("Data")
use_uploader = st.sidebar.checkbox("Upload a CSV instead", value=False)
if use_uploader:
    up = st.sidebar.file_uploader("Upload a CSV", type=["csv"])
    if up is not None:
        df = load_data(up)
    else:
        st.stop()
else:
    # Default: CSV in repo root with this exact name
    df = load_data("California_NO2_CO_Combined.csv")

st.sidebar.header("Animation/Data Filters")
min_obs = st.sidebar.slider("Min observations per month (for animations)", 200, 2000, 800, 50)
months_counts = df["MonthLabel"].value_counts()
rich_months = sorted(months_counts[months_counts >= min_obs].index.tolist())
df = df[df["MonthLabel"].isin(rich_months)].copy()

# Wide format for computed views
wide = (
    df.pivot_table(
        index=["Date","Site ID","Site Latitude","Site Longitude","County"],
        columns="Pollutant",
        values="Concentration",
        aggfunc="mean"
    )
    .reset_index()
)
wide["MonthLabel"] = wide["Date"].dt.to_period("M").astype(str)
months_sorted = sorted(wide["MonthLabel"].unique())

if not months_sorted:
    st.warning("No months meet the minimum observations threshold. Lower the slider in the sidebar.")
    st.stop()

# =======================
# Global synchronized controls
# =======================
if "play" not in st.session_state:
    st.session_state.play = False
if "month_idx" not in st.session_state:
    st.session_state.month_idx = 0
if "speed_ms" not in st.session_state:
    st.session_state.speed_ms = 900  # ms between frames

def toggle_play():
    st.session_state.play = not st.session_state.play

def next_frame():
    st.session_state.month_idx = (st.session_state.month_idx + 1) % len(months_sorted)

# Top control bar
c1, c2, c3, c4 = st.columns([0.1, 0.7, 0.15, 0.15])
with c1:
    st.button("▶️ / ⏸", on_click=toggle_play, help="Play/Pause all charts")
with c2:
    st.session_state.month_idx = st.slider(
        "Month",
        min_value=0, max_value=len(months_sorted)-1,
        value=st.session_state.month_idx,
        format="%d",
        label_visibility="collapsed",
    )
with c3:
    speed = st.select_slider("Speed", options=[300, 600, 900, 1200, 1500], value=st.session_state.speed_ms,
                             format_func=lambda x: f"{int(x/1000)}s", label_visibility="collapsed")
    st.session_state.speed_ms = speed
with c4:
    st.markdown(f"<div style='text-align:right;padding-top:8px'><b>{months_sorted[st.session_state.month_idx]}</b></div>", unsafe_allow_html=True)

# Auto-advance when playing
if st.session_state.play:
    time.sleep(st.session_state.speed_ms / 1000.0)
    next_frame()
    st.experimental_rerun()

# Current month label
M = months_sorted[st.session_state.month_idx]

# ===============================
# Precompute pieces needed per-month
# ===============================

# 1) Composite Map data (for month M ONLY)
county_month_all = (
    wide.groupby(["County","MonthLabel"], as_index=False)
        .agg(lat=("Site Latitude","mean"),
             lon=("Site Longitude","mean"),
             NO2=("NO2","mean"),
             CO=("CO","mean"),
             NO2_sd=("NO2","std"),
             CO_sd=("CO","std"),
             n=("Site ID","nunique"))
)
cm = county_month_all[county_month_all["MonthLabel"]==M].copy()
# Composite & size
if not cm.empty:
    # Rank within month only
    cm["rNO2"] = cm["NO2"].rank(pct=True)
    cm["rCO"]  = cm["CO"].rank(pct=True)
    cm["Composite"] = 100.0*(0.5*cm["rNO2"] + 0.5*cm["rCO"])
    cm["VarMag"] = np.sqrt(cm["NO2_sd"].fillna(0)**2 + cm["CO_sd"].fillna(0)**2)
    cm["size"] = 8 + 20*(cm["VarMag"]/cm["VarMag"].max()) if cm["VarMag"].max() > 0 else 10.0

# 2) Correlation Heatmap for M
corr_vars = ["NO2","CO","Site Latitude","Site Longitude"]
sub_corr = wide[wide["MonthLabel"]==M][corr_vars].dropna()
if len(sub_corr) < 3:
    C_M = np.zeros((len(corr_vars), len(corr_vars)))
else:
    C_M = sub_corr.corr().values

# 3) Regional trends up to current month (cumulative story)
county_lat = wide.groupby("County", as_index=False)["Site Latitude"].mean().rename(columns={"Site Latitude":"centroid_lat"})
q1, q2 = county_lat["centroid_lat"].quantile([0.33, 0.66]).tolist()
def region_from_lat(lat):
    if lat <= q1: return "South"
    if lat <= q2: return "Central"
    return "North"
county_lat["Region"] = county_lat["centroid_lat"].apply(region_from_lat)

cm_region_all = (
    wide.merge(county_lat[["County","Region"]], on="County", how="left")
        .groupby(["Region","MonthLabel"], as_index=False)
        .agg(NO2=("NO2","mean"), CO=("CO","mean"))
)
# keep months <= M
months_to_plot = [m for m in months_sorted if m <= M]
cm_region = cm_region_all[cm_region_all["MonthLabel"].isin(months_to_plot)].copy()
cm_region["_date"] = pd.to_datetime(cm_region["MonthLabel"]+"-01")

# 4) 3D surface for M
def surface_for_month(m, bins=45):
    sub = wide[wide["MonthLabel"]==m][["NO2","CO"]].dropna()
    if len(sub) < 50:
        H = np.zeros((bins, bins))
        xe = np.linspace(0,1,bins+1); ye = np.linspace(0,1,bins+1)
        return H.T, xe, ye
    x = sub["NO2"].values; y = sub["CO"].values
    xbins = np.linspace(np.nanmin(x), np.nanmax(x), bins+1)
    ybins = np.linspace(np.nanmin(y), np.nanmax(y), bins+1)
    H, xe, ye = np.histogram2d(x, y, bins=[xbins, ybins], density=True)
    return H.T, xe, ye

Z, xe, ye = surface_for_month(M, bins=45)
xmid = 0.5*(xe[:-1] + xe[1:]); ymid = 0.5*(ye[:-1] + ye[1:])

# ===============================
# Build figures (no per-fig animations)
# ===============================

# Map
if cm.empty:
    fig_map = go.Figure()
    fig_map.update_layout(height=MAP_H, margin=dict(l=5,r=5,t=30,b=0))
else:
    fig_map = px.scatter_mapbox(
        cm, lat="lat", lon="lon",
        color="Composite", size="size",
        hover_name="County",
        hover_data={"lat":":.3f","lon":":.3f","NO2":":.2f","CO":":.2f","Composite":":.1f","n":True,"size":False},
        zoom=4.5
    )
    fig_map.update_layout(mapbox_style="open-street-map",
                          height=MAP_H, margin=dict(l=5,r=5,t=30,b=0))

# Correlation
fig_corr = go.Figure(data=[go.Heatmap(z=C_M, x=corr_vars, y=corr_vars, zmin=-1, zmax=1, colorbar=dict(title="r"))])
fig_corr.update_layout(height=FIG_H, margin=dict(l=5,r=5,t=30,b=0))

# Regional lines (NO2 by default with toggle)
regions = ["North","Central","South"]
fig_regions = go.Figure()
for reg in regions:
    sub = cm_region[cm_region["Region"]==reg].sort_values("_date")
    fig_regions.add_trace(go.Scatter(x=sub["_date"], y=sub["NO2"], mode="lines+markers", name=reg))
fig_regions.update_layout(height=FIG_H, margin=dict(l=5,r=5,t=30,b=0),
                          xaxis_title="Month", yaxis_title="Concentration")

# 3D surface
fig_surface = go.Figure(data=[go.Surface(z=Z, x=xmid, y=ymid, showscale=True)])
fig_surface.update_layout(height=FIG_H, margin=dict(l=5,r=5,t=30,b=0),
                          scene=dict(xaxis_title="NO₂", yaxis_title="CO", zaxis_title="Density"))

# =========================
# 2×2 GRID (no scrolling)
# =========================
col1, col2 = st.columns(2, gap="small")
with col1:
    st.subheader(f"Composite Index — Map  |  {M}")
    st.plotly_chart(fig_map, use_container_width=True)
with col2:
    st.subheader("Correlation Heatmap")
    st.plotly_chart(fig_corr, use_container_width=True)

col3, col4 = st.columns(2, gap="small")
with col3:
    st.subheader("Regional Trends (NO₂)")
    st.plotly_chart(fig_regions, use_container_width=True)
with col4:
    st.subheader("3D Exposure Surface")
    st.plotly_chart(fig_surface, use_container_width=True)

st.caption("One global timeline controls all charts. Play to auto-advance; move the slider to scrub months.")
