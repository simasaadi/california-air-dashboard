import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from functools import lru_cache

# ---------- Page setup ----------
st.set_page_config(
    page_title="California Air Quality — Scientific Animated",
    layout="wide",
)

st.markdown(
    "<h2 style='margin-top:0'>California Air Quality — Scientific Animated Dashboard</h2>",
    unsafe_allow_html=True,
)

# ---------- Data loader ----------
@st.cache_data(show_spinner=False)
def load_data(file):
    df = pd.read_csv(file)
    # Normalize/validate columns expected from your CSV
    # Expected: Date, Pollutant (NO2/CO), Concentration, County, Site Latitude, Site Longitude, Site ID
    required = ["Date","Pollutant","Concentration","County","Site Latitude","Site Longitude","Site ID"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Concentration", "Site Latitude", "Site Longitude", "Pollutant", "County"])
    df["MonthLabel"] = df["Date"].dt.to_period("M").astype(str)
    return df

# ---------- Sidebar controls ----------
st.sidebar.header("Data")
use_uploader = st.sidebar.checkbox("Upload a CSV instead", value=False)
if use_uploader:
    up = st.sidebar.file_uploader("Upload a CSV", type=["csv"])
    if up is not None:
        df = load_data(up)
    else:
        st.stop()
else:
    # Local bundled file path:
    df = load_data("California_NO2_CO_Combined.csv")

st.sidebar.header("Animation/Data Filters")
min_obs = st.sidebar.slider("Min observations per month (for animations)", 200, 2000, 800, 50)
months = df["MonthLabel"].value_counts()
rich_months = months[months >= min_obs].index.tolist()
df = df[df["MonthLabel"].isin(rich_months)].copy()

# Prepare wide format for some charts
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

# =========================================
# Chart 1) Composite Pollution Map (Animated)
# =========================================
with st.container():
    st.subheader("Composite Pollution Index — Monthly Animated Map (NO₂ & CO)")
    county_month = (
        wide.groupby(["County","MonthLabel"], as_index=False)
            .agg(lat=("Site Latitude","mean"),
                 lon=("Site Longitude","mean"),
                 NO2=("NO2","mean"),
                 CO=("CO","mean"),
                 NO2_sd=("NO2","std"),
                 CO_sd=("CO","std"),
                 n=("Site ID","nunique"))
    )
    # Percentile-based composite (unitless, robust)
    p_NO2 = county_month["NO2"].rank(pct=True)
    p_CO  = county_month["CO"].rank(pct=True)
    county_month["Composite"] = 100.0*(0.5*p_NO2 + 0.5*p_CO)

    # Marker size by variability magnitude
    county_month["VarMag"] = np.sqrt(county_month["NO2_sd"].fillna(0)**2 + county_month["CO_sd"].fillna(0)**2)
    if county_month["VarMag"].max() > 0:
        county_month["size"] = 8 + 20*(county_month["VarMag"]/county_month["VarMag"].max())
    else:
        county_month["size"] = 10.0

    fig_map = px.scatter_mapbox(
        county_month,
        lat="lat", lon="lon",
        color="Composite", size="size",
        hover_name="County",
        hover_data={"lat":":.3f","lon":":.3f","NO2":":.2f","CO":":.2f","Composite":":.1f","n":True,"size":False},
        animation_frame="MonthLabel",
        zoom=4.5, height=560
    )
    fig_map.update_layout(mapbox_style="open-street-map", margin=dict(l=5,r=5,t=40,b=5))
    st.plotly_chart(fig_map, use_container_width=True)

# =========================================
# Chart 2) Monthly Correlation Heatmap (Animated)
# =========================================
with st.container():
    st.subheader("Monthly Correlation Structure — NO₂, CO, Latitude, Longitude")

    corr_vars = ["NO2","CO","Site Latitude","Site Longitude"]
    def corr_matrix_for_month(m):
        sub = wide[wide["MonthLabel"]==m][corr_vars].dropna()
        if len(sub) < 3:
            C = np.zeros((len(corr_vars), len(corr_vars)))
        else:
            C = sub.corr().values
        return C

    frames = []
    C0 = corr_matrix_for_month(months_sorted[0]) if months_sorted else np.zeros((4,4))
    fig_corr = go.Figure(data=[go.Heatmap(z=C0, x=corr_vars, y=corr_vars, zmin=-1, zmax=1, colorbar=dict(title="r"))])
    for m in months_sorted:
        C = corr_matrix_for_month(m)
        frames.append(go.Frame(name=m, data=[go.Heatmap(z=C, x=corr_vars, y=corr_vars, zmin=-1, zmax=1)]))
    fig_corr.frames = frames
    fig_corr.update_layout(
        height=480, margin=dict(l=5,r=5,t=10,b=5),
        updatemenus=[{
            "type":"buttons",
            "buttons":[
                {"label":"Play","method":"animate","args":[None, {"fromcurrent":True,"frame":{"duration":900,"redraw":True},"transition":{"duration":300}}]},
                {"label":"Pause","method":"animate","args":[[None], {"mode":"immediate","frame":{"duration":0,"redraw":False},"transition":{"duration":0}}]}
            ],
            "x":0.0,"y":1.1,"xanchor":"left","yanchor":"top"
        }],
        sliders=[{
            "active": 0, "pad":{"t":20}, "x":0.1, "len":0.8,
            "currentvalue":{"prefix":"Month: "},
            "steps":[{"args":[[fr.name], {"frame":{"duration":0,"redraw":True},"transition":{"duration":0}}], "label":fr.name, "method":"animate"} for fr in frames]
        }]
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# =========================================
# Chart 3) Regional Trends (North/Central/South) — Animated by Pollutant
# =========================================
with st.container():
    st.subheader("Regional Trends — North vs Central vs South (Animated by Pollutant)")

    county_lat = wide.groupby("County", as_index=False)["Site Latitude"].mean().rename(columns={"Site Latitude":"centroid_lat"})
    q1, q2 = county_lat["centroid_lat"].quantile([0.33, 0.66]).tolist()
    def region_from_lat(lat):
        if lat <= q1: return "South"
        if lat <= q2: return "Central"
        return "North"
    county_lat["Region"] = county_lat["centroid_lat"].apply(region_from_lat)

    cm_region = (
        wide.merge(county_lat[["County","Region"]], on="County", how="left")
            .groupby(["Region","MonthLabel"], as_index=False)
            .agg(NO2=("NO2","mean"), CO=("CO","mean"))
    )

    regions = ["North","Central","South"]

    def lines_for_pollutant(pol):
        traces = []
        for reg in regions:
            sub = cm_region[cm_region["Region"]==reg].sort_values("MonthLabel")
            x = pd.to_datetime(sub["MonthLabel"]+"-01")
            y = sub[pol]
            traces.append(go.Scatter(x=x, y=y, mode="lines+markers", name=reg))
        return traces

    fig_regions = go.Figure()
    for tr in lines_for_pollutant("NO2"):
        fig_regions.add_trace(tr)
    frames_regions = [
        go.Frame(name="NO2", data=lines_for_pollutant("NO2")),
        go.Frame(name="CO",  data=lines_for_pollutant("CO")),
    ]
    fig_regions.frames = frames_regions
    fig_regions.update_layout(
        height=480, margin=dict(l=5,r=5,t=10,b=5),
        xaxis_title="Month", yaxis_title="Concentration",
        updatemenus=[{
            "type":"buttons",
            "buttons":[
                {"label":"NO₂","method":"animate","args":[["NO2"], {"mode":"immediate","frame":{"duration":0,"redraw":True},"transition":{"duration":200}}]},
                {"label":"CO", "method":"animate","args":[["CO"],  {"mode":"immediate","frame":{"duration":0,"redraw":True},"transition":{"duration":200}}]}
            ],
            "x":0.0,"y":1.1,"xanchor":"left","yanchor":"top"
        }]
    )
    st.plotly_chart(fig_regions, use_container_width=True)

# =========================================
# Chart 4) 3D Exposure Surface (NO2×CO density) — Animated Monthly
# =========================================
with st.container():
    st.subheader("3D Exposure Surface — NO₂ × CO Density (Animated Monthly)")

    def surface_for_month(m, bins=45):
        sub = wide[wide["MonthLabel"]==m][["NO2","CO"]].dropna()
        if len(sub) < 50:
            H = np.zeros((bins, bins))
            xe = np.linspace(0,1,bins+1)
            ye = np.linspace(0,1,bins+1)
            return H.T, xe, ye
        x = sub["NO2"].values
        y = sub["CO"].values
        xbins = np.linspace(np.nanmin(x), np.nanmax(x), bins+1)
        ybins = np.linspace(np.nanmin(y), np.nanmax(y), bins+1)
        H, xe, ye = np.histogram2d(x, y, bins=[xbins, ybins], density=True)
        return H.T, xe, ye

    m0 = months_sorted[0] if months_sorted else None
    if m0:
        Z0, xe0, ye0 = surface_for_month(m0, bins=45)
        xmid0 = 0.5*(xe0[:-1] + xe0[1:])
        ymid0 = 0.5*(ye0[:-1] + ye0[1:])
    else:
        Z0 = np.zeros((10,10)); xmid0 = np.arange(10); ymid0 = np.arange(10)

    fig_surface = go.Figure(data=[go.Surface(z=Z0, x=xmid0, y=ymid0, showscale=True)])
    frames_surf = []
    for m in months_sorted:
        Z, xe, ye = surface_for_month(m, bins=45)
        xmid = 0.5*(xe[:-1] + xe[1:])
        ymid = 0.5*(ye[:-1] + ye[1:])
        frames_surf.append(go.Frame(name=m, data=[go.Surface(z=Z, x=xmid, y=ymid)]))
    fig_surface.frames = frames_surf
    fig_surface.update_layout(
        height=560, margin=dict(l=5,r=5,t=10,b=5),
        scene=dict(
            xaxis_title="NO₂",
            yaxis_title="CO",
            zaxis_title="Density",
            # You can tweak these to 'zoom in' quickly:
            # camera=dict(eye=dict(x=1.7, y=1.7, z=0.9))
        ),
        updatemenus=[{
            "type":"buttons",
            "buttons":[
                {"label":"Play","method":"animate","args":[None, {"fromcurrent":True,"frame":{"duration":900,"redraw":True},"transition":{"duration":300}}]},
                {"label":"Pause","method":"animate","args":[[None], {"mode":"immediate","frame":{"duration":0,"redraw":False},"transition":{"duration":0}}]}
            ],
            "x":0.0,"y":1.05,"xanchor":"left","yanchor":"top"
        }],
        sliders=[{
            "active": 0, "pad":{"t":20}, "x":0.1, "len":0.8,
            "currentvalue":{"prefix":"Month: "},
            "steps":[{"args":[[fr.name], {"frame":{"duration":0,"redraw":True},"transition":{"duration":0}}], "label":fr.name, "method":"animate"} for fr in frames_surf]
        }]
    )
    st.plotly_chart(fig_surface, use_container_width=True)

st.caption("Tip: Use the sliders or Play buttons on each chart to animate through months.")
