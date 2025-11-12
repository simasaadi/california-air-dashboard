import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ---------- Page ----------
st.set_page_config(page_title="California Air Quality — Scientific Animated", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
.block-container{padding-top:0.9rem; padding-bottom:0.6rem; max-width:1400px;}
h1{margin:0.2rem 0 0.9rem 0; text-align:center;}
hr{margin:0.2rem 0 1.0rem 0; border:1px solid #999;}
</style>
""", unsafe_allow_html=True)
st.markdown("<h1>California Air Quality — Scientific Animated Dashboard</h1><hr/>", unsafe_allow_html=True)

# ---------- Data ----------
def find_csv():
    p = Path("California_NO2_CO_Combined.csv")
    if p.exists(): return str(p)
    for q in list(Path(".").glob("*.csv")) + list(Path("data").glob("*.csv")):
        return str(q)
    return None

src = st.sidebar.selectbox("Data source", ["Auto-detect CSV","Upload a CSV"])
if src == "Auto-detect CSV":
    path = find_csv()
    if not path:
        st.sidebar.warning("No CSV found. Please upload instead.")
        src = "Upload a CSV"

if src == "Upload a CSV":
    up = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if up is None: st.stop()
    df = pd.read_csv(up)
else:
    df = pd.read_csv(path)

req = ["Date","Pollutant","Concentration","County","Site Latitude","Site Longitude","Site ID"]
missing = [c for c in req if c not in df.columns]
if missing:
    st.error(f"CSV missing required columns: {missing}")
    st.stop()

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date","Pollutant","Concentration","County","Site Latitude","Site Longitude"])
df["MonthLabel"] = df["Date"].dt.to_period("M").astype(str)

min_obs = st.sidebar.slider("Min observations per month (for animations)", 200, 2000, 800, 50)
rich = df["MonthLabel"].value_counts()
rich_months = sorted(rich[rich >= min_obs].index.tolist())
df = df[df["MonthLabel"].isin(rich_months)].copy()
if not len(rich_months):
    st.warning("No month meets the threshold. Lower the slider.")
    st.stop()

# Wide table
wide = (df.pivot_table(
            index=["Date","Site ID","Site Latitude","Site Longitude","County"],
            columns="Pollutant",
            values="Concentration",
            aggfunc="mean")
        .reset_index())
wide["MonthLabel"] = wide["Date"].dt.to_period("M").astype(str)
months = sorted(wide["MonthLabel"].unique())

# County monthly (for map)
county_month = (wide.groupby(["County","MonthLabel"], as_index=False)
                    .agg(lat=("Site Latitude","mean"),
                         lon=("Site Longitude","mean"),
                         NO2=("NO2","mean"),
                         CO=("CO","mean"),
                         NO2_sd=("NO2","std"),
                         CO_sd=("CO","std"),
                         n=("Site ID","nunique")))

# Regions
county_lat = wide.groupby("County", as_index=False)["Site Latitude"].mean().rename(columns={"Site Latitude":"centroid_lat"})
q1, q2 = county_lat["centroid_lat"].quantile([0.33, 0.66]).tolist()
def region_from_lat(v):
    if v <= q1: return "South"
    if v <= q2: return "Central"
    return "North"
county_lat["Region"] = county_lat["centroid_lat"].apply(region_from_lat)
reg_series = (wide.merge(county_lat[["County","Region"]], on="County", how="left")
                 .groupby(["Region","MonthLabel"], as_index=False)
                 .agg(NO2=("NO2","mean"), CO=("CO","mean")))
regions = ["North","Central","South"]
corr_vars = ["NO2","CO","Site Latitude","Site Longitude"]

# Choose the trend pollutant with more variance
no2_var = reg_series.groupby("MonthLabel")["NO2"].mean().var()
co_var  = reg_series.groupby("MonthLabel")["CO"].mean().var()
trend_pol = "CO" if (co_var > no2_var) else "NO2"
trend_title = f"Regional Trends ({'CO' if trend_pol=='CO' else 'NO₂'})"

def per_month_tables(m):
    cm = county_month[county_month["MonthLabel"]==m].copy()
    if len(cm):
        cm["rNO2"] = cm["NO2"].rank(pct=True)
        cm["rCO"]  = cm["CO"].rank(pct=True)
        cm["Composite"] = 100.0*(0.5*cm["rNO2"] + 0.5*cm["rCO"])
        cm["VarMag"] = np.sqrt(cm["NO2_sd"].fillna(0)**2 + cm["CO_sd"].fillna(0)**2)
        cm["size"] = 8 + 20*(cm["VarMag"]/cm["VarMag"].max()) if cm["VarMag"].max()>0 else 10.0

    sub = wide[wide["MonthLabel"]==m][corr_vars].dropna()
    C = sub.corr().values if len(sub)>=3 else np.zeros((4,4))

    months_to_plot = [x for x in months if x <= m]
    rs = reg_series[reg_series["MonthLabel"].isin(months_to_plot)].copy()
    rs["_date"] = pd.to_datetime(rs["MonthLabel"]+"-01")
    rs.sort_values(["Region","_date"], inplace=True)
    for reg in regions:
        mask = rs["Region"]==reg
        rs.loc[mask, trend_pol+"_SMA3"] = rs.loc[mask, trend_pol].rolling(3, min_periods=1).mean()

    S = wide[wide["MonthLabel"]==m][["NO2","CO"]].dropna()
    bins = 35
    if len(S) < 50:
        H = np.zeros((bins, bins))
        xe = np.linspace(0,1,bins+1); ye = np.linspace(0,1,bins+1)
    else:
        x = S["NO2"].values; y = S["CO"].values
        xbins = np.linspace(np.nanmin(x), np.nanmax(x), bins+1)
        ybins = np.linspace(np.nanmin(y), np.nanmax(y), bins+1)
        H, xe, ye = np.histogram2d(x, y, bins=[xbins, ybins], density=True)
    xmid = 0.5*(xe[:-1] + xe[1:]); ymid = 0.5*(ye[:-1] + ye[1:])
    return cm, C, rs, xmid, ymid, H.T

# ---------- Figure ----------
fig = make_subplots(
    rows=2, cols=2,
    specs=[[{"type":"mapbox"}, {"type":"heatmap"}],
           [{"type":"xy"},      {"type":"surface"}]],
    column_widths=[0.5, 0.5], row_heights=[0.5, 0.5],
    horizontal_spacing=0.08, vertical_spacing=0.12,
    subplot_titles=("Composite Index — Map", "Correlation Heatmap",
                    trend_title, "3D Exposure Surface")
)

m0 = months[0]
cm0, C0, rs0, x0, y0, Z0 = per_month_tables(m0)

# Map
if len(cm0):
    map_trace = go.Scattermapbox(
        lat=cm0["lat"], lon=cm0["lon"],
        mode="markers", showlegend=False,
        marker=dict(size=cm0["size"], color=cm0["Composite"], colorscale="Blues", showscale=False),
        text=cm0["County"],
        customdata=np.c_[cm0["NO2"], cm0["CO"]],
        hovertemplate="County=%{text}<br>NO₂=%{customdata[0]:.2f}<br>CO=%{customdata[1]:.2f}<extra></extra>",
    )
    fig.add_trace(map_trace, row=1, col=1)
fig.update_mapboxes(row=1, col=1, style="open-street-map", center=dict(lat=36.5, lon=-119.5), zoom=4.5)

# Heatmap (keep a single colorbar). NOTE: this trace index is 1.
heat_trace = go.Heatmap(z=C0, x=corr_vars, y=corr_vars, zmin=-1, zmax=1,
                        coloraxis="coloraxis", showscale=True)
fig.add_trace(heat_trace, row=1, col=2)
fig.update_layout(coloraxis=dict(colorscale="Blues", colorbar=dict(title="r", y=0.9)))

# Regional lines (SMA3 only)
for reg in regions:
    sub = rs0[rs0["Region"]==reg].sort_values("_date")
    fig.add_trace(go.Scatter(x=sub["_date"], y=sub[f"{trend_pol}_SMA3"], mode="lines", name=reg),
                  row=2, col=1)

# 3D surface
surf_trace = go.Surface(z=Z0, x=x0, y=y0, showscale=False)
fig.add_trace(surf_trace, row=2, col=2)

fig.update_layout(
    height=880, margin=dict(l=10, r=10, t=80, b=90),
    scene=dict(xaxis_title="NO₂", yaxis_title="CO", zaxis_title="Density"),
    legend=dict(orientation="h", yanchor="bottom", y=0.08, xanchor="left", x=0.05),
    title=dict(text=f"Month: {m0}", x=0.5, y=0.99)
)

# ---------- Frames: update only what changes ----------
# Trace order: [0]=map, [1]=heatmap, [2-4]=regional lines, [5]=surface
frames = []
for m in months:
    cm, C, rs, xmid, ymid, Z = per_month_tables(m)
    frame_data = []

    # Map update
    if len(cm):
        frame_data.append(go.Scattermapbox(
            lat=cm["lat"], lon=cm["lon"],
            marker=dict(size=cm["size"], color=cm["Composite"], colorscale="Blues", showscale=False),
            text=cm["County"], customdata=np.c_[cm["NO2"], cm["CO"]],
        ))
    else:
        frame_data.append(go.Scattermapbox(lat=[], lon=[]))

    # Heatmap: **only update Z** to avoid axis/colorbar re-layout jitter
    frame_data.append({"z": C})

    # Regional lines
    for reg in regions:
        sub = rs[rs["Region"]==reg].sort_values("_date")
        frame_data.append(go.Scatter(x=sub["_date"], y=sub[f"{trend_pol}_SMA3"]))

    # Surface
    frame_data.append(go.Surface(z=Z, x=xmid, y=ymid, showscale=False))

    # Map the "z-only" update to the heatmap trace index (1)
    frames.append(go.Frame(
        name=m,
        data=frame_data,
        traces=list(range(len(frame_data))),   # aligns updates to trace indices
        layout=go.Layout(title=dict(text=f"Month: {m}", x=0.5, y=0.99))
    ))

fig.frames = frames

# Controls bottom
fig.update_layout(
    updatemenus=[{
        "type":"buttons",
        "buttons":[
            {"label":"Play","method":"animate",
             "args":[None, {"fromcurrent":True,
                            "frame":{"duration":600, "redraw":True},
                            "transition":{"duration":200}}]},
            {"label":"Pause","method":"animate",
             "args":[[None], {"mode":"immediate",
                              "frame":{"duration":0,"redraw":False},
                              "transition":{"duration":0}}]}
        ],
        "direction":"left", "x":0.01, "y":0.03, "xanchor":"left", "yanchor":"bottom",
        "pad":{"r":10,"t":30}
    }],
    sliders=[{
        "active":0, "x":0.14, "y":0.03, "len":0.75,
        "currentvalue":{"prefix":"Month: "},
        "pad":{"t":40, "b":0},
        "steps":[{"args":[[m], {"frame":{"duration":0,"redraw":True},
                                 "transition":{"duration":0}}],
                  "label":m, "method":"animate"} for m in months]
    }]
)

st.plotly_chart(fig, use_container_width=True)
