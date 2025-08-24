import io
import os
import zipfile
import tempfile
from datetime import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static
import hashlib


# ============================= Small helpers
def _df_sig(df: pd.DataFrame, cols=None):
    """Stable signature for caching: hash selected columns of df."""
    if df is None or df.empty:
        return "EMPTY"
    if cols is None:
        cols = list(df.columns)
    h = pd.util.hash_pandas_object(df[cols], index=True).values.tobytes()
    return hashlib.md5(h).hexdigest()


def _get_qparam(name, default=None, cast=str):
    """URL query param getter for deep links (Streamlit v1.30+ compatible)."""
    try:
        # Prefer new API if available
        if hasattr(st, "query_params"):
            val = st.query_params.get(name, default)
            if isinstance(val, list):
                val = val[0] if val else default
            return cast(val)
        # Fallback
        return cast(st.experimental_get_query_params().get(name, [default])[0])
    except Exception:
        return default


def _set_qparams(**kwargs):
    """URL query param setter for deep links (Streamlit v1.30+ compatible)."""
    try:
        if hasattr(st, "query_params"):
            st.query_params.clear()
            for k, v in kwargs.items():
                if v is not None:
                    st.query_params[k] = v
        else:
            st.experimental_set_query_params(**kwargs)
    except Exception:
        pass


def _clean(s: str) -> str:
    return str(s).strip().lower().replace(" ", "_")


def _make_week_date(df: pd.DataFrame, year_col: str, week_col: str) -> pd.DataFrame:
    out = df.copy()

    def safe_week_start(y, w):
        try:
            y = int(y)
            w = int(w)
            if w < 1:
                w = 1
            if w > 53:
                w = 53
            return datetime.fromisocalendar(y, w, 1)
        except Exception:
            return pd.NaT

    out["week_start"] = [safe_week_start(y, w) for y, w in zip(out[year_col], out[week_col])]
    return out


def _auto_alias_map(cols):
    """Very simple automatic aliasing for common column names."""
    base = {c.lower(): c for c in cols}

    def get(*opts):
        for o in opts:
            if o in base:
                return base[o]
        return None

    auto = {
        "district": get("district", "upazila", "subdistrict"),
        "division": get("division", "region", "state", "province"),
        "year": get("year", "yr"),
        "week": get("week", "epiweek", "epi_week", "epi_week_number"),
        "cases": get("weekly_hospitalised_cases", "weekly_hospitalized_cases", "cases", "weekly_cases"),
        "population": get("population", "pop"),
        "temp": get("meantemperature", "temperature", "temp"),
        "rain": get("rainsum", "rainfall", "rain"),
        "hum": get("rhdailymean", "humidity", "rh", "relative_humidity"),
    }
    # Return (auto_map, aliases_dict) to match earlier calling style
    return auto, {k: [v] for k, v in auto.items() if v}


def _district_norm(s):
    return _clean(s)


def _norm_key(s):
    return _clean(s)


# ---- Baseline / threshold helpers (EWARS-style)
@st.cache_data(show_spinner=False)
def _compute_threshold_by_week(df: pd.DataFrame, cases_col: str, method: str) -> pd.DataFrame:
    """Return [week, threshold] using chosen method on counts supplied in df (filtered to baseline years)."""

    def _baseline_func(s: pd.Series) -> float:
        if method.startswith("Endemic"):  # median + 2*IQR
            return float(s.median() + 2.0 * (s.quantile(0.75) - s.quantile(0.25)))
        if method.startswith("Mean"):
            return float(s.mean() + 2.0 * s.std(ddof=0))
        return float(s.quantile(0.95))

    tmp = df.copy()
    if "week" not in tmp.columns:
        # assume second column is week if not provided; safer to require caller to set
        raise ValueError("Column 'week' must exist in the input DataFrame for threshold computation.")

    thr = (
        tmp.groupby("week", as_index=False)[cases_col]
        .agg(threshold=_baseline_func)
        .reset_index(drop=True)
    )
    return thr[["week", "threshold"]]


# ============================= Page & Title
st.set_page_config(page_title="EWARS Dashboard (Python)", layout="wide")
st.markdown(
    """
    <style>
        .block-container {padding-top: 1.0rem;}
        h1.app-title {font-size: 28px; font-weight: 700; margin: 0 0 0.15rem 0;}
        .subtitle {color:#5c6370; font-size:0.95rem; margin-bottom:0.6rem}
        h1, .app-title {letter-spacing: 0 !important; font-variant-ligatures: none !important; -webkit-font-smoothing: antialiased;}
        .smallnote {color:#556; font-size:0.9rem}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<h1 class="app-title">🦟 EWARS Dashboard — Python/Streamlit (v0.8)</h1>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Early Warning for Dengue — national overview, district drilldown, spatial risk, seasonality, DLNM-style forecasts, and EWARS alerts.</div>', unsafe_allow_html=True)


# ============================= Readers & utils
@st.cache_data(show_spinner=False)
def _read_geojson(bytes_buf: bytes) -> gpd.GeoDataFrame:
    return gpd.read_file(io.BytesIO(bytes_buf))


@st.cache_data(show_spinner=False)
def _read_zipped_shapefile(bytes_buf: bytes) -> gpd.GeoDataFrame:
    with tempfile.TemporaryDirectory() as tmp:
        zpath = os.path.join(tmp, "shape.zip")
        with open(zpath, "wb") as f:
            f.write(bytes_buf)
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(tmp)
        shp_candidates = [os.path.join(tmp, f) for f in os.listdir(tmp) if f.lower().endswith(".shp")]
        if not shp_candidates:
            raise ValueError("No .shp file found inside the zip.")
        return gpd.read_file(shp_candidates[0])


@st.cache_data(show_spinner=False)
def _read_surv_csv(bytes_buf: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(bytes_buf))


# ============================= Sidebar — Uploads only (no mapping UI here)
with st.sidebar:
    st.header("Inputs")
    bfile = st.file_uploader(
        "Boundary: GeoJSON or zipped Shapefile (.zip)",
        type=["geojson", "zip"],
        accept_multiple_files=False,
    )
    sfile = st.file_uploader(
        "Surveillance CSV",
        type=["csv"],
        accept_multiple_files=False,
        help=(
            "Columns: year, division, district, population, week, "
            "weekly_hospitalised_cases, rhdailymean, rainsum, meantemperature"
        ),
    )
    st.markdown("<div class='smallnote'>Tip: Upload files here. Column mapping now lives in the <b>Settings</b> tab.</div>", unsafe_allow_html=True)


# Track freshness paths if files are uploaded from disk
if sfile and hasattr(sfile, "name"):
    st.session_state["data_source_path_surveillance"] = getattr(sfile, "name", None)
if bfile and hasattr(bfile, "name"):
    # we reuse this slot; boundary freshness usually not needed
    st.session_state["data_source_path_weather"] = getattr(bfile, "name", None)


GDF, DF = None, None
geo_ok = dat_ok = False

if bfile:
    try:
        GDF = _read_geojson(bfile.getvalue()) if bfile.name.lower().endswith(".geojson") else _read_zipped_shapefile(bfile.getvalue())
        geo_ok = True
    except Exception as e:
        st.sidebar.error(f"Boundary load failed: {e}")

if sfile:
    try:
        DF = _read_surv_csv(sfile.getvalue())
        DF.columns = [_clean(c) for c in DF.columns]
        dat_ok = True
    except Exception as e:
        st.sidebar.error(f"Data load failed: {e}")


# --- Quick load status under uploaders (helps verify something actually loaded)
with st.sidebar:
    if dat_ok:
        st.success(f"✅ CSV loaded: {len(DF):,} rows x {len(DF.columns)} cols")
        st.caption(f"Columns: {', '.join(DF.columns[:12])}{'…' if len(DF.columns)>12 else ''}")
    else:
        st.info("Upload a surveillance CSV to proceed.")

    if geo_ok:
        try:
            st.success(f"✅ Boundary loaded: {len(GDF):,} features")
        except Exception:
            st.success("✅ Boundary loaded")
    else:
        st.info("Upload a boundary file (GeoJSON/ZIP).")


# ------------------------------ Column mapping & canonicalization
year_col = "year"
week_col = "week"
cases_col = "weekly_hospitalised_cases"
district_key = "district"
pop_col = "population"
temp_col = "meantemperature"
rain_col = "rainsum"
hum_col = "rhdailymean"


if dat_ok:
    # Ensure session container exists
    if "CAN_MAPPING" not in st.session_state:
        st.session_state["CAN_MAPPING"] = {}

    CAN = st.session_state["CAN_MAPPING"]

    # If no user-defined mapping yet, try auto-mapping
    if not CAN:
        auto, _ALIASES = _auto_alias_map(DF.columns.tolist())
        CAN.update(
            {
                "district": auto.get("district"),
                "division": auto.get("division"),
                "year": auto.get("year"),
                "week": auto.get("week"),
                "weekly_hospitalised_cases": auto.get("cases"),
                "population": auto.get("population"),
                "meantemperature": auto.get("temp"),
                "rainsum": auto.get("rain"),
                "rhdailymean": auto.get("hum"),
            }
        )

    # Determine if mapping is sufficient for homepage usage
    required_keys = [
        "district",
        "year",
        "week",
        "weekly_hospitalised_cases",
        "population",
        "meantemperature",
        "rainsum",
        "rhdailymean",
    ]
    needs_mapping = any(CAN.get(k) is None for k in required_keys)
    st.session_state["NEEDS_MAPPING"] = needs_mapping

    # Safe rename: only rename columns that exist and have a target key
    rename_map = {v: k for k, v in CAN.items() if v and v in DF.columns}
    if rename_map:
        DF = DF.rename(columns=rename_map).copy()

    # Enforce numeric types & add week_start if core fields are present
    core_cols = [year_col, week_col, cases_col, pop_col]
    if all(c in DF.columns for c in core_cols):
        for c in core_cols:
            DF[c] = pd.to_numeric(DF[c], errors="coerce")
        DF = _make_week_date(DF, year_col, week_col)

    # Optional banner so you immediately know why homepage might be empty
    if st.session_state["NEEDS_MAPPING"]:
        st.warning("Columns need mapping. Open the **Settings** tab and complete the mapping to enable the Overview.")


# ====== Tabs ======
tabs = st.tabs([
    "Overview",
    "Drilldown",
    "Spatial",
    "Seasonality",
    "Forecasts",
    "Alerts",
    "Data quality",
    "Threshold Tuner",
])


# ====== Overview (non-technical landing) ======
if dat_ok:
    with tabs[0]:
        method = st.selectbox(
            "Baseline method",
            ["Endemic channel (median + 2*IQR)", "Mean + 2*SD", "95th percentile"],
            key="ovr_method",
            help=(
                "**How thresholds are computed**"
                "• **Endemic channel (Median + 2×IQR):** robust to outliers; good for noisy history."
                "• **Mean + 2×SD:** average plus margin; fine for stable data but can be pulled by past spikes."
                "• **95th percentile:** flags only top 5% extremes; most sensitive to recent surges if you have enough years."
                "**Tips:** Use Endemic channel for stability; Mean+2SD for balance; 95th pct for early warnings."
            ),
        )
        with st.popover("ℹ️ Baseline method explainer"):
            st.markdown(
                """
                **In plain language**
                - **Endemic channel**: "typical" level + robust spread (ignores weird years). Good to avoid false comfort.
                - **Mean + 2×SD**: average + safety margin. Works when past years are steady.
                - **95th percentile**: alert if this week is higher than ~95% of past years.

                **Example (same data: 20, 25, 30, 40, 200)**
                - Mean+2×SD ≈ **201**
                - Median+2×IQR = **225**
                - 95th percentile ≈ **168**
                """
            )
        exclude_current = st.toggle("Exclude current year from baseline", value=True, key="ovr_excl")

        # National weekly sums
        latest_year = int(DF[year_col].max())
        latest_week = int(DF.loc[DF[year_col] == latest_year, week_col].max())

        nat = DF.groupby([year_col, week_col], as_index=False)[cases_col].sum()
        nat = nat.rename(columns={cases_col: "cases_nat"})

        # Baseline on historical national sums by epi-week
        base_nat = nat.copy()
        if exclude_current:
            base_nat = base_nat[base_nat[year_col] < latest_year]
        base_nat = base_nat.rename(columns={"cases_nat": cases_col}).copy()
        base_nat["week"] = base_nat[week_col]
        base_thr = _compute_threshold_by_week(base_nat, cases_col, method)

        nat2 = nat.copy()
        nat2["week"] = nat2[week_col]
        nat2 = nat2.merge(base_thr, on="week", how="left")

        # KPIs
        this_wk = nat2[(nat2[year_col] == latest_year) & (nat2[week_col] == latest_week)]
        prev_wk = nat2[(nat2[year_col] == latest_year) & (nat2[week_col] == latest_week - 1)]
        total_this = float(this_wk["cases_nat"].sum()) if len(this_wk) else np.nan
        total_prev = float(prev_wk["cases_nat"].sum()) if len(prev_wk) else np.nan
        pct_change = (
            (total_this - total_prev) / total_prev * 100.0 if np.isfinite(total_prev) and total_prev > 0 else np.nan
        )

        # District alerts for latest week
        base = DF.copy()
        if exclude_current:
            base = base[base[year_col] < latest_year]
        base["week"] = base[week_col]

        def _baseline_func(s: pd.Series) -> float:
            if method.startswith("Endemic"):
                return float(s.median() + 2.0 * (s.quantile(0.75) - s.quantile(0.25)))
            if method.startswith("Mean"):
                return float(s.mean() + 2.0 * s.std(ddof=0))
            return float(s.quantile(0.95))

        thr = (
            base.groupby([district_key, "week"], as_index=False)
            .agg({cases_col: _baseline_func})
            .rename(columns={cases_col: "threshold"})
        )
        cur = DF[[district_key, year_col, week_col, cases_col]].copy()
        cur["week"] = cur[week_col]
        alert = cur.merge(thr, on=[district_key, "week"], how="left")
        alert["alert"] = alert[cases_col] > alert["threshold"]
        A = alert[(alert[year_col] == latest_year) & (alert[week_col] == latest_week)]
        n_alert = int(A["alert"].sum()) if len(A) else 0
        n_districts = int(A[district_key].nunique()) if len(A) else int(DF[district_key].nunique())
        pct_alert = 100.0 * n_alert / max(n_districts, 1) if n_districts else 0.0

        # KPI cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Current week (national cases)",
            f"{total_this:,.0f}" if np.isfinite(total_this) else "–",
            delta=(f"{pct_change:+.1f}% vs last week" if np.isfinite(pct_change) else None),
        )
        c2.metric("Districts on alert (this week)", f"{n_alert} / {n_districts}")
        c3.metric("Latest week", f"Year {latest_year}, W{latest_week}")
        risk_text = "LOW" if pct_alert < 20 else ("MODERATE" if pct_alert < 50 else "HIGH")
        c4.metric("National risk level", risk_text)

        # Risk gauge
        g1, g2 = st.columns([1.2, 2.2])
        with g1:
            fig_g = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=pct_alert,
                    number={"suffix": "%"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#d95f0e"},
                        "steps": [
                            {"range": [0, 20], "color": "#eaf4f4"},
                            {"range": [20, 50], "color": "#fee391"},
                            {"range": [50, 100], "color": "#fec44f"},
                        ],
                    },
                    title={"text": "Share of districts on alert"},
                )
            )
            fig_g.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_g, use_container_width=True)

        with g2:
            # National time series vs threshold (overlay)
            nat_plot = nat2.sort_values([year_col, week_col]).copy()
            nat_plot = _make_week_date(nat_plot, year_col, week_col)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=nat_plot["week_start"], y=nat_plot["cases_nat"], mode="lines", name="National cases")
            )
            fig.add_trace(
                go.Scatter(
                    x=nat_plot["week_start"],
                    y=nat_plot["threshold"],
                    mode="lines",
                    name="Threshold (baseline)",
                    line=dict(dash="dot"),
                )
            )
            fig.update_layout(title="National weekly cases vs baseline threshold", xaxis_title="Week", yaxis_title="Cases")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Top districts (recent 12 weeks)**")
        DF_sorted = DF.sort_values([year_col, week_col])
        recent = DF_sorted[(DF_sorted[year_col] == latest_year) & (DF_sorted[week_col] >= max(1, latest_week - 11))]
        top = (
            recent.groupby(district_key, as_index=False)[cases_col]
            .sum()
            .sort_values(cases_col, ascending=False)
            .head(6)[district_key]
            .tolist()
        )
        cols = st.columns(3)
        for i, d in enumerate(top):
            sub = DF[DF[district_key].astype(str) == str(d)].sort_values([year_col, week_col]).tail(30)
            sub = _make_week_date(sub, year_col, week_col)
            fig_s = go.Figure()
            fig_s.add_trace(go.Scatter(x=sub["week_start"], y=sub[cases_col], mode="lines", name=str(d)))
            fig_s.update_layout(title=f"{d}", height=220, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
            cols[i % 3].plotly_chart(fig_s, use_container_width=True)


# ====== Drilldown (district focus) ======
if dat_ok:
    with tabs[1]:
        st.markdown("#### District drilldown")
        dlist = DF[district_key].dropna().astype(str).sort_values().unique().tolist()
        sel_district = st.selectbox("District", dlist)
        Dd = DF[DF[district_key].astype(str) == str(sel_district)].copy()
        if "week_start" not in Dd.columns or Dd["week_start"].isna().all():
            Dd = _make_week_date(Dd, year_col, week_col)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total cases", int(Dd[cases_col].sum()))
        k2.metric("Median weekly", float(np.nanmedian(Dd[cases_col])))
        inc = 1e5 * Dd[cases_col].sum() / max(Dd[pop_col].dropna().mean() if pop_col in Dd.columns else 1, 1)
        k3.metric("Cumulative incidence /100k", f"{inc:,.2f}")
        k4.metric("Rows", len(Dd))

        st.markdown("**Cases vs weather**")
        for col, label in [(temp_col, "Temperature"), (rain_col, "Rainfall"), (hum_col, "Humidity")]:
            dfp = Dd.sort_values("week_start")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dfp["week_start"], y=dfp[cases_col], mode="lines", name="Cases"))
            fig.add_trace(go.Scatter(x=dfp["week_start"], y=dfp[col], mode="lines", yaxis="y2", name=label))
            fig.update_layout(
                title=f"Cases vs {label}",
                xaxis_title="Week",
                yaxis_title="Cases",
                yaxis2=dict(overlaying="y", side="right", title=label),
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- Combined chart: Cases vs Temperature, Rainfall, Humidity (one figure)
        st.markdown("**Combined: cases vs temperature, rainfall and humidity**")
        normalize_weather = st.checkbox(
            "Normalize weather (z-score) for comparison",
            value=True,
            help="If on, temp/rain/humidity are standardized so all share a common scale.",
        )

        dfp = Dd.sort_values("week_start").copy()

        def _z(x):
            m = np.nanmean(x)
            s = np.nanstd(x)
            s = s if (np.isfinite(s) and s > 0) else 1.0
            return (x - m) / s

        # Choose raw vs normalized weather series for plotting
        if normalize_weather:
            t_series = _z(dfp[temp_col].astype(float))
            r_series = _z(dfp[rain_col].astype(float))
            h_series = _z(dfp[hum_col].astype(float))
            t_title = "Temperature (z)"
            r_title = "Rainfall (z)"
            h_title = "Humidity (z)"
        else:
            t_series = dfp[temp_col].astype(float).values
            r_series = dfp[rain_col].astype(float).values
            h_series = dfp[hum_col].astype(float).values
            t_title = "Temperature"
            r_title = "Rainfall"
            h_title = "Humidity"

        fig_all = go.Figure()
        # Left axis: cases
        fig_all.add_trace(go.Scatter(x=dfp["week_start"], y=dfp[cases_col], name="Cases", mode="lines", line=dict(width=2)))
        # Right-side stacked axes for the three weather series
        fig_all.add_trace(go.Scatter(x=dfp["week_start"], y=t_series, name="Temp", mode="lines", yaxis="y2"))
        fig_all.add_trace(go.Scatter(x=dfp["week_start"], y=r_series, name="Rainfall", mode="lines", yaxis="y3"))
        fig_all.add_trace(go.Scatter(x=dfp["week_start"], y=h_series, name="Humidity", mode="lines", yaxis="y4"))

        # Layout with multiple right axes (y2, y3, y4)
        fig_all.update_layout(
            title=f"Cases vs weather — {sel_district}",
            xaxis=dict(title="Week"),
            yaxis=dict(title="Cases"),
            yaxis2=dict(title=t_title, overlaying="y", side="right", position=1.00, showgrid=False),
            yaxis3=dict(title=r_title, overlaying="y", side="right", position=0.95, showgrid=False),
            yaxis4=dict(title=h_title, overlaying="y", side="right", position=0.90, showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=40, r=120, t=60, b=40),
            height=420,
        )
        st.plotly_chart(fig_all, use_container_width=True)


# ====== Spatial (hotspot-like choropleth) ======
if dat_ok:
    with tabs[2]:
        if not geo_ok:
            st.info("Upload a boundary file to enable spatial plotting.")
        else:
            st.markdown("#### District-level dengue burden (hotspot style)")

            # --------- Helpers (cached) ----------
            if "_prep_boundary" not in globals():

                @st.cache_data(show_spinner=False)
                def _prep_boundary(_gdf_raw, tol_m=800, keep_cols=None):
                    """
                    Prepare boundary for web mapping:
                    - project to EPSG:3857 (meters), fix & simplify by tol_m meters
                    - back to EPSG:4326
                    - keep only requested attribute columns + geometry
                    - precompute representative center
                    NOTE: leading underscore arg name makes it cacheable with st.cache_data.
                    """
                    import warnings

                    if _gdf_raw is None or len(_gdf_raw) == 0:
                        return _gdf_raw

                    g = _gdf_raw.copy()
                    g.columns = [_clean(c) for c in g.columns]

                    # Ensure CRS then project to meters
                    try:
                        if g.crs is None:
                            g.set_crs(4326, inplace=True, allow_override=True)
                        g = g.to_crs(3857)
                    except Exception:
                        pass

                    # Fix invalid geometries in projected CRS (more robust)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            g["geometry"] = g["geometry"].buffer(0)
                        except Exception:
                            pass

                    # Simplify in meters (big win for size)
                    try:
                        g["geometry"] = g["geometry"].simplify(float(tol_m), preserve_topology=True)
                    except Exception:
                        pass

                    # Back to WGS84 for web maps
                    try:
                        g = g.to_crs(4326)
                    except Exception:
                        pass

                    # Keep only essential columns
                    if keep_cols:
                        keep = [c for c in keep_cols if c in g.columns]
                        g = g[keep + ["geometry"]].copy()
                    else:
                        non_geom = [c for c in g.columns if c != "geometry"]
                        keep = non_geom[:1]
                        g = g[keep + ["geometry"]].copy()

                    # Representative points as stable centers
                    try:
                        rp = g.geometry.representative_point()
                        g["_centroid_y"] = rp.y
                        g["_centroid_x"] = rp.x
                    except Exception:
                        b = g.geometry.bounds
                        g["_centroid_y"] = b[["miny", "maxy"]].mean(axis=1)
                        g["_centroid_x"] = b[["minx", "maxx"]].mean(axis=1)

                    return g

            if "_map_center_from_gdf" not in globals():

                def _map_center_from_gdf(g):
                    if g is None or len(g) == 0:
                        return [23.6850, 90.3563]
                    return [float(g["_centroid_y"].mean()), float(g["_centroid_x"].mean())]

            # --------- Optional performance mode ----------
            if "perf_mode" in st.session_state:
                perf_mode = st.session_state["perf_mode"]
            else:
                perf_mode = st.sidebar.checkbox("⚡ Performance mode (faster maps)", value=True, key="perf_mode")

            # --------- Controls (unique keys) ----------
            GDF.columns = [_clean(c) for c in GDF.columns]
            gcols = [c for c in GDF.columns if c != "geometry"]
            gdf_key = st.selectbox("Boundary column matching your district key", gcols, key="sp_gdf_key")

            years = sorted(DF[year_col].dropna().unique().tolist())
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                year_filter = st.selectbox("Year", ["All"] + years, index=len(years), key="sp_year")
            with c2:
                metric = st.selectbox("Metric", ["Sum of cases", "Incidence per 100k"], key="sp_metric")
            with c3:
                scheme = st.selectbox("Color bins", ["EWARS custom", "Quantiles (5)"], key="sp_scheme")

            # --------- Aggregate data (minimal columns) ----------
            dmap = DF[[district_key, year_col, cases_col, pop_col]].copy()
            if year_filter != "All":
                dmap = dmap[dmap[year_col] == year_filter]

            agg = (
                dmap.groupby(district_key, as_index=False)[cases_col]
                .sum()
                .rename(columns={cases_col: "value"})
            )
            if metric == "Incidence per 100k":
                pop_agg = dmap.groupby(district_key, as_index=False)[pop_col].mean().rename(columns={pop_col: "_p"})
                agg = agg.merge(pop_agg, on=district_key, how="left")
                agg["value"] = 1e5 * agg["value"] / agg["_p"].replace(0, np.nan)
                caption = "Dengue incidence per 100k"
            else:
                caption = "Hospitalized Dengue Cases (Summed)"

            # --------- Prepare boundary once; simplify in meters (adaptive) ----------
            base_tol = 1600 if perf_mode else 800
            GDF_SIMPL = _prep_boundary(GDF, tol_m=base_tol, keep_cols=[gdf_key])

            g = GDF_SIMPL.copy()
            g[gdf_key] = g[gdf_key].apply(_district_norm)
            agg[district_key] = agg[district_key].apply(_district_norm)
            g = g.merge(agg[[district_key, "value"]], left_on=gdf_key, right_on=district_key, how="left")
            g["value"] = g["value"].fillna(0)

            # If still too big, re-simplify more aggressively
            try:
                geojson_str = g[[gdf_key, "value", "geometry"]].to_json()
                if len(geojson_str) > 8_000_000:
                    st.info("Large boundary detected — simplifying further for stability…")
                    GDF_SIMPL2 = _prep_boundary(GDF, tol_m=2400, keep_cols=[gdf_key])
                    g = GDF_SIMPL2.copy()
                    g[gdf_key] = g[gdf_key].apply(_district_norm)
                    g = g.merge(agg[[district_key, "value"]], left_on=gdf_key, right_on=district_key, how="left")
                    g["value"] = g["value"].fillna(0)
                    geojson_str = g[[gdf_key, "value", "geometry"]].to_json()
            except Exception:
                geojson_str = g[[gdf_key, "value", "geometry"]].to_json()

            # --------- Binning ----------
            import branca

            if scheme == "EWARS custom":
                bins = [0, 500, 2500, 10000, 50000, float("inf")]
            else:
                qs = g["value"].quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]).tolist()
                bins = [float(qs[0])] + [float(x) + 1e-9 * i for i, x in enumerate(qs[1:])]
            colors = ["#fff7bc", "#fee391", "#fec44f", "#fe9929", "#d95f0e", "#993404"]

            # --------- Render ----------
            def _render_map(geojson_payload, _g):
                center = _map_center_from_gdf(_g)
                fmap = folium.Map(location=center, zoom_start=6, tiles="CartoDB positron")

                def style_fn(feat):
                    v = feat["properties"].get("value", 0)
                    col = colors[-1]
                    for i in range(len(bins) - 1):
                        if bins[i] <= v < bins[i + 1]:
                            col = colors[i]
                            break
                    return {"fillColor": col, "color": "#777", "weight": 0.6, "fillOpacity": 0.85}

                folium.GeoJson(
                    data=geojson_payload,
                    style_function=style_fn,
                    tooltip=folium.features.GeoJsonTooltip(fields=[gdf_key, "value"], aliases=["District", "Value"]),
                    name="Dengue burden",
                ).add_to(fmap)

                branca.colormap.StepColormap(colors=colors, index=bins, caption=caption).add_to(fmap)
                return fmap

            if perf_mode:
                if st.button("Render map (fast)", key="sp_render_btn"):
                    fmap = _render_map(geojson_str, g)
                    folium_static(fmap, height=520)
                else:
                    st.caption("⚡ Performance mode is ON. Click **Render map (fast)** to draw the map.")
            else:
                fmap = _render_map(geojson_str, g)
                folium.LayerControl(collapsed=True).add_to(fmap)
                folium_static(fmap, height=560)


# ====== Seasonality ======
if dat_ok:
    with tabs[3]:
        seas = DF.copy()
        seas["week"] = seas[week_col]
        grp = seas.groupby("week")[cases_col]
        out = pd.DataFrame(
            {
                "week": grp.mean().index,
                "mean_cases": grp.mean().values,
                "lo": (grp.mean() - 1.96 * grp.std(ddof=0) / np.sqrt(grp.count())).values,
                "hi": (grp.mean() + 1.96 * grp.std(ddof=0) / np.sqrt(grp.count())).values,
            }
        )
        fig = go.Figure()
        fig.add_trace(go.Bar(x=out["week"], y=out["mean_cases"], name="Mean"))
        fig.add_trace(go.Scatter(x=out["week"], y=out["hi"], mode="lines", name="Upper 95%"))
        fig.add_trace(
            go.Scatter(x=out["week"], y=out["lo"], mode="lines", name="Lower 95%", fill="tonexty")
        )
        fig.update_layout(title="Average cases by Epi week (±95% CI)", xaxis_title="Epi week", yaxis_title="Cases")
        st.plotly_chart(fig, use_container_width=True)


# ====== Forecasts (DLNM-style Python, robust) ======
if dat_ok:
    with tabs[4]:
        st.markdown("##### Forecasts — choose a model")
        model_choice = st.selectbox(
            "Select forecasting model",
            ["DLNM‑style Poisson GLM", "Zero‑Inflated Poisson (ZIP)"],
            key="fc_model_select",
            help="DLNM‑style uses splines × lags + Poisson. ZIP handles excess zeros by combining a Poisson count model with a separate zero‑inflation process.",
        )

        # Common controls
        subset_only = st.checkbox("Run on selected district only (faster)", value=True, key="fc_subset")
        max_lag = st.slider("Maximum lag (weeks)", 1, 12, 8, key="fc_maxlag")
        spline_df = st.slider("Spline degrees of freedom (per lag)", 2, 6, 3, key="fc_splinedf")
        add_time_smooth = st.checkbox("Add smooth time term (controls slow trends)", value=True, key="fc_timesmooth")
        use_pop_offset = st.checkbox("Use population as log-offset (incidence model)", value=False, key="fc_offset")
        run_btn = st.button("▶️ Run selected model", key="fc_run_btn")

        import statsmodels.api as sm
        from patsy import bs
        from numpy.linalg import LinAlgError
        from statsmodels.discrete.count_model import ZeroInflatedPoisson

        # --- Helper for Python version differences of isocalendar()
        if "_iso_year_week" not in globals():

            def _iso_year_week(d):
                iso = d.isocalendar()
                try:
                    return int(iso.year), int(iso.week)
                except AttributeError:
                    return int(iso[0]), int(iso[1])

        weather_vars = [temp_col, rain_col, hum_col]

        def _safe_bs(x: np.ndarray, df: int):
            finite = x[np.isfinite(x)]
            if finite.size == 0 or np.nanmin(finite) == np.nanmax(finite):
                return None
            try:
                return bs(x, df=df, degree=3, include_intercept=False)
            except Exception:
                return None

        def _build_design_matrix(df: pd.DataFrame, max_lag: int, spline_df: int, add_time_smooth: bool):
            """Cross-basis (splines × lags) + optional smooth time. Returns X and index kept (due to lagging)."""
            mats = []
            keep_idx = df.index[max_lag:] if len(df) > max_lag else df.index[:0]

            for var in weather_vars:
                if var not in df.columns:
                    continue
                series = df[var].astype(float).values
                for lag in range(max_lag + 1):
                    lagged = np.roll(series, lag)
                    if lag > 0:
                        lagged[:lag] = np.nan
                    B = _safe_bs(lagged, df=spline_df)
                    if B is None:
                        col = pd.Series(lagged, index=df.index, name=f"{var}_lag{lag}")
                        mats.append(col.to_frame())
                    else:
                        cols = [f"{var}_lag{lag}_s{i}" for i in range(B.shape[1])]
                        mats.append(pd.DataFrame(B, index=df.index, columns=cols))

            if add_time_smooth:
                tvals = df["time"].astype(float).values
                T = _safe_bs(tvals, df=6)
                if T is None:
                    mats.append(pd.Series(tvals, index=df.index, name="time_linear").to_frame())
                else:
                    tcols = [f"time_s{i}" for i in range(T.shape[1])]
                    mats.append(pd.DataFrame(T, index=df.index, columns=tcols))
            else:
                mats.append(df[["time"]])

            if not mats:
                return pd.DataFrame(index=keep_idx), keep_idx

            X = pd.concat(mats, axis=1)
            X = X.loc[keep_idx].copy()
            X = X.loc[:, X.notna().any(axis=0)]
            if X.isna().any().any():
                X = X.fillna(X.median(numeric_only=True))
            nunique = X.nunique(dropna=False)
            X = X.loc[:, nunique > 1]
            return X, keep_idx

        # ---------------- Fitting functions ----------------
        def _fit_poisson_glm(df: pd.DataFrame):
            df = df.sort_values([year_col, week_col]).copy()
            df["time"] = np.arange(len(df), dtype=int)
            if "week_start" not in df.columns or df["week_start"].isna().all():
                df = _make_week_date(df, year_col, week_col)
            X, keep_idx = _build_design_matrix(df, max_lag, spline_df, add_time_smooth)
            if X.shape[0] == 0 or X.shape[1] == 0:
                raise ValueError(
                    "Design matrix is empty. Reduce max_lag or check data coverage for weather variables."
                )
            y = df.loc[keep_idx, cases_col].astype(float).values
            if len(y) < 10:
                raise ValueError("Not enough rows after lagging to fit reliably.")
            offset = None
            if use_pop_offset and pop_col in df.columns:
                pop_vals = df.loc[keep_idx, pop_col].astype(float).values
                pop_vals = np.where(np.isfinite(pop_vals), pop_vals, np.nan)
                if np.all(~np.isfinite(pop_vals)):
                    pop_vals = np.nanmean(df[pop_col].astype(float).values) * np.ones_like(y)
                offset = np.log(np.maximum(pop_vals, 1.0))
            Xc = sm.add_constant(X, has_constant="add")
            model = sm.GLM(y, Xc, family=sm.families.Poisson(), offset=offset)
            res = model.fit()
            out = df.loc[keep_idx, [district_key, year_col, week_col, cases_col, "week_start"]].copy()
            out["fitted_dlnm_py"] = res.fittedvalues
            return out, res, Xc, offset

        def _fit_zip(df: pd.DataFrame):
            """Zero-Inflated Poisson using the SAME DLNM design matrix as exog.
            Inflation model: intercept + (optional) time smooth if available."""
            df = df.sort_values([year_col, week_col]).copy()
            df["time"] = np.arange(len(df), dtype=int)
            if "week_start" not in df.columns or df["week_start"].isna().all():
                df = _make_week_date(df, year_col, week_col)

            X, keep_idx = _build_design_matrix(df, max_lag, spline_df, add_time_smooth)
            if X.shape[0] == 0 or X.shape[1] == 0:
                raise ValueError(
                    "Design matrix is empty. Reduce max_lag or check data coverage for weather variables."
                )
            y = df.loc[keep_idx, cases_col].astype(float).values
            if len(y) < 10:
                raise ValueError("Not enough rows after lagging to fit reliably.")

            # ZIP does not use offsets directly; if offset requested, add as a column.
            X_zip = X.copy()
            if use_pop_offset and pop_col in df.columns:
                pop_vals = df.loc[keep_idx, pop_col].astype(float).values
                pop_vals = np.where(np.isfinite(pop_vals), pop_vals, np.nan)
                if np.all(~np.isfinite(pop_vals)):
                    pop_vals = np.nanmean(df[pop_col].astype(float).values) * np.ones_like(y)
                X_zip["_log_pop_offset"] = np.log(np.maximum(pop_vals, 1.0))

            # Inflation covariates: simple intercept + (optional) time smooth if present in X
            infl_cols = [c for c in X_zip.columns if c.startswith("time_")] or None
            if infl_cols:
                exog_infl = sm.add_constant(X_zip[infl_cols], has_constant="add")
            else:
                exog_infl = np.ones((X_zip.shape[0], 1))  # intercept-only inflation

            exog_count = sm.add_constant(X_zip, has_constant="add")
            zip_model = ZeroInflatedPoisson(endog=y, exog=exog_count, exog_infl=exog_infl, inflation="logit")
            zip_res = zip_model.fit(method="bfgs", maxiter=500, disp=False)

            fitted = zip_res.predict(exog=exog_count, exog_infl=exog_infl, which="mean")
            out = df.loc[keep_idx, [district_key, year_col, week_col, cases_col, "week_start"]].copy()
            out["fitted_zip"] = fitted
            return out, zip_res, exog_count, exog_infl

        # Cache wrapper (so reruns don’t recompute)
        @st.cache_data(show_spinner=True, max_entries=64)
        def _cached_fit(
            df_one: pd.DataFrame,
            model_choice: str,
            max_lag: int,
            spline_df: int,
            add_time_smooth: bool,
            use_pop_offset: bool,
        ):
            cols_needed = [year_col, week_col, cases_col, temp_col, rain_col, hum_col, pop_col]
            dfh = df_one.reindex(columns=cols_needed)
            sig = (
                pd.util.hash_pandas_object(dfh, index=True).values.tobytes(),
                model_choice,
                max_lag,
                spline_df,
                bool(add_time_smooth),
                bool(use_pop_offset),
            )
            if model_choice.startswith("DLNM"):
                out, res, Xc, offset = _fit_poisson_glm(df_one)
                return sig, out, res, ("poisson", Xc, offset)
            else:
                out, res, exog_count, exog_infl = _fit_zip(df_one)
                return sig, out, res, ("zip", exog_count, exog_infl)

        # ---------------- Run (single district) ----------------
        if subset_only:
            dlist = DF[district_key].dropna().astype(str).sort_values().unique().tolist()
            d_current = st.selectbox("Select district", dlist, index=0, key="fc_district")
            if run_btn:
                df_one = DF[DF[district_key].astype(str) == str(d_current)][
                    [
                        district_key,
                        year_col,
                        week_col,
                        cases_col,
                        pop_col,
                        temp_col,
                        rain_col,
                        hum_col,
                        "week_start",
                    ]
                ].copy()
                try:
                    _sig, fitted_df, res, pack = _cached_fit(
                        df_one, model_choice, max_lag, spline_df, add_time_smooth, use_pop_offset
                    )

                    # Plot observed vs fitted
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(x=fitted_df["week_start"], y=fitted_df[cases_col], mode="lines", name="Observed")
                    )
                    if model_choice.startswith("DLNM"):
                        fig.add_trace(
                            go.Scatter(
                                x=fitted_df["week_start"],
                                y=fitted_df["fitted_dlnm_py"],
                                mode="lines",
                                name="DLNM‑Poisson fitted",
                            )
                        )
                    else:
                        fig.add_trace(
                            go.Scatter(
                                x=fitted_df["week_start"], y=fitted_df["fitted_zip"], mode="lines", name="ZIP fitted"
                            )
                        )
                    fig.update_layout(title=f"{model_choice} — {d_current}", xaxis_title="Week", yaxis_title="Cases")
                    st.plotly_chart(fig, use_container_width=True)

                    # Accuracy metrics
                    y_true = fitted_df[cases_col].astype(float).values
                    if model_choice.startswith("DLNM"):
                        y_pred = fitted_df["fitted_dlnm_py"].astype(float).values
                        dev_info = (
                            f"Deviance: {res.deviance:,.2f} | Pearson χ²: {res.pearson_chi2:,.2f} | n={int(res.nobs)}"
                        )
                    else:
                        y_pred = fitted_df["fitted_zip"].astype(float).values
                        dev_info = f"AIC: {res.aic:,.2f} | LogLik: {res.llf:,.2f} | n={len(y_true)}"

                    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
                    mae = float(np.mean(np.abs(y_true - y_pred)))
                    mask = y_true > 0
                    mape = (
                        float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else np.nan
                    )

                    st.caption(dev_info)
                    m1, m2, m3 = st.columns(3)
                    m1.metric("RMSE", f"{rmse:,.2f}")
                    m2.metric("MAE", f"{mae:,.2f}")
                    m3.metric("MAPE (%)", "NA" if not np.isfinite(mape) else f"{mape:,.1f}")

                    st.dataframe(fitted_df.head(20))

                    # Forecast (both models)
                    with st.expander("🔮 Forecast next weeks (experimental)"):
                        horizon = st.slider("Weeks ahead", 1, 8, 4, key="fc_horizon")
                        scen = st.selectbox(
                            "Weather scenario",
                            [
                                "Persistence (use last observed)",
                                "Set constants manually",
                                "Percentile of past (by district)",
                            ],
                            key="fc_scen",
                        )
                        # Scenario values
                        if scen == "Persistence (use last observed)":
                            t_const = (
                                float(df_one[temp_col].dropna().iloc[-1]) if df_one[temp_col].notna().any() else 0.0
                            )
                            r_const = (
                                float(df_one[rain_col].dropna().iloc[-1]) if df_one[rain_col].notna().any() else 0.0
                            )
                            h_const = (
                                float(df_one[hum_col].dropna().iloc[-1]) if df_one[hum_col].notna().any() else 0.0
                            )
                        elif scen == "Percentile of past (by district)":
                            p = st.slider("Percentile for weather (past distribution)", 5, 95, 75, key="fc_pct")
                            t_const = float(np.nanpercentile(df_one[temp_col].values, p))
                            r_const = float(np.nanpercentile(df_one[rain_col].values, p))
                            h_const = float(np.nanpercentile(df_one[hum_col].values, p))
                        else:
                            t_const = st.number_input(
                                "Temperature for forecast",
                                value=float(np.nanmean(df_one[temp_col].tail(8))),
                                format="%.3f",
                                key="fc_t",
                            )
                            r_const = st.number_input(
                                "Rainfall for forecast",
                                value=float(np.nanmean(df_one[rain_col].tail(8))),
                                format="%.3f",
                                key="fc_r",
                            )
                            h_const = st.number_input(
                                "Humidity for forecast",
                                value=float(np.nanmean(df_one[hum_col].tail(8))),
                                format="%.3f",
                                key="fc_h",
                            )

                        base_df = df_one.sort_values([year_col, week_col]).copy()
                        if base_df["week_start"].notna().any():
                            last_date = pd.to_datetime(base_df["week_start"].iloc[-1])
                        else:
                            last_year = int(base_df[year_col].iloc[-1])
                            last_week = int(base_df[week_col].iloc[-1])
                            last_date = datetime.fromisocalendar(last_year, min(max(last_week, 1), 53), 1)

                        fut_rows = []
                        for k in range(1, horizon + 1):
                            d = last_date + pd.Timedelta(days=7 * k)
                            y_i, w_i = _iso_year_week(d)
                            fut_rows.append(
                                {
                                    district_key: d_current,
                                    year_col: y_i,
                                    week_col: w_i,
                                    cases_col: np.nan,
                                    pop_col: float(base_df[pop_col].dropna().iloc[-1])
                                    if pop_col in base_df.columns and base_df[pop_col].notna().any()
                                    else np.nan,
                                    temp_col: t_const,
                                    rain_col: r_const,
                                    hum_col: h_const,
                                    "week_start": d,
                                }
                            )
                        df_future = pd.concat([base_df, pd.DataFrame(fut_rows)], ignore_index=True).sort_values(
                            [year_col, week_col]
                        )
                        df_future["time"] = np.arange(len(df_future), dtype=int)

                        # Build future design matrix aligned to training columns
                        X_all, _ = _build_design_matrix(df_future, max_lag, spline_df, add_time_smooth)
                        future_idx = df_future.index[-horizon:]
                        X_future = X_all.loc[X_all.index.intersection(future_idx)].copy()

                        if pack[0] == "poisson":
                            # Align to training columns
                            X_train = pack[1][pack[1].columns.difference(["const"])].copy()
                            X_future = X_future.reindex(columns=X_train.columns, fill_value=0.0)
                            Xc_future = sm.add_constant(X_future, has_constant="add")
                            offset_future = pack[2]
                            if use_pop_offset and pop_col in df_future.columns:
                                pop_vals = df_future.loc[X_future.index, pop_col].values
                                pop_vals = np.where(np.isfinite(pop_vals), pop_vals, np.nan)
                                if np.all(~np.isfinite(pop_vals)):
                                    pop_vals = np.nanmean(base_df[pop_col].astype(float).values) * np.ones(len(X_future))
                                offset_future = np.log(np.maximum(pop_vals, 1.0))
                            yhat_future = res.predict(Xc_future, offset=offset_future)
                        else:
                            # ZIP: build exog_count & exog_infl with same columns as training
                            exog_count_train = pack[1]
                            exog_infl_train = pack[2]

                            X_zip_future = X_future.copy()
                            if use_pop_offset and pop_col in df_future.columns:
                                pop_vals = df_future.loc[X_future.index, pop_col].values
                                pop_vals = np.where(np.isfinite(pop_vals), pop_vals, np.nan)
                                if np.all(~np.isfinite(pop_vals)):
                                    pop_vals = np.nanmean(base_df[pop_col].astype(float).values) * np.ones(len(X_future))
                                X_zip_future["_log_pop_offset"] = np.log(np.maximum(pop_vals, 1.0))

                            count_cols = [c for c in exog_count_train.columns if c != "const"]
                            infl_cols = [c for c in exog_infl_train.columns if c != "const"]
                            X_zip_future_count = X_zip_future.reindex(columns=count_cols, fill_value=0.0)
                            if len(infl_cols) > 0:
                                X_zip_future_infl = X_zip_future.reindex(columns=infl_cols, fill_value=0.0)
                                exog_infl_future = sm.add_constant(X_zip_future_infl, has_constant="add")
                            else:
                                exog_infl_future = np.ones((X_zip_future.shape[0], 1))
                            exog_count_future = sm.add_constant(X_zip_future_count, has_constant="add")

                            yhat_future = res.predict(
                                exog=exog_count_future, exog_infl=exog_infl_future, which="mean"
                            )

                        fc = pd.DataFrame(
                            {
                                "week_start": df_future.loc[X_future.index, "week_start"].values,
                                "forecast": yhat_future,
                            }
                        )

                        fig_fc = go.Figure()
                        recent = fitted_df.tail(26)
                        fig_fc.add_trace(
                            go.Scatter(x=recent["week_start"], y=recent[cases_col], mode="lines", name="Observed")
                        )
                        fitted_name = "fitted_dlnm_py" if "fitted_dlnm_py" in fitted_df.columns else "fitted_zip"
                        fig_fc.add_trace(
                            go.Scatter(x=recent["week_start"], y=recent[fitted_name], mode="lines", name="Fitted")
                        )
                        fig_fc.add_trace(
                            go.Scatter(
                                x=fc["week_start"], y=fc["forecast"], mode="lines", name="Forecast", line=dict(dash="dash")
                            )
                        )
                        fig_fc.update_layout(
                            title=f"Forecast next {horizon} week(s) — {model_choice}", xaxis_title="Week", yaxis_title="Cases"
                        )
                        st.plotly_chart(fig_fc, use_container_width=True)
                        st.download_button(
                            "⬇️ Download forecast CSV",
                            data=fc.to_csv(index=False).encode("utf-8"),
                            file_name=f"forecast_{str(d_current).replace(' ','_')}_{horizon}w_{'ZIP' if pack[0]=='zip' else 'DLNM'}.csv",
                            mime="text/csv",
                        )

                    st.download_button(
                        "⬇️ Download fitted (this district)",
                        data=fitted_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"fitted_{str(d_current).replace(' ','_')}_{'ZIP' if model_choice.startswith('Zero') else 'DLNM'}.csv",
                        mime="text/csv",
                    )

                except (ValueError, LinAlgError) as e:
                    st.warning(f"Model could not run: {e}")

        # ---------------- Run (all districts) ----------------
        else:
            if run_btn:
                districts = DF[district_key].dropna().astype(str).sort_values().unique().tolist()
                prog = st.progress(0)
                outs = []
                for i, d in enumerate(districts, 1):
                    df_one = DF[DF[district_key].astype(str) == str(d)][
                        [
                            district_key,
                            year_col,
                            week_col,
                            cases_col,
                            pop_col,
                            temp_col,
                            rain_col,
                            hum_col,
                            "week_start",
                        ]
                    ].copy()
                    try:
                        _sig, fitted_df, res, _ = _cached_fit(
                            df_one, model_choice, max_lag, spline_df, add_time_smooth, use_pop_offset
                        )
                        fitted_df = fitted_df.copy()
                        fitted_df[district_key] = str(d)
                        outs.append(fitted_df)
                    except Exception:
                        pass
                    prog.progress(int(100 * i / max(1, len(districts))))
                prog.empty()

                if outs:
                    all_out = pd.concat(outs, axis=0).reset_index(drop=True)
                    st.success(f"Fitted {model_choice} for {all_out[district_key].nunique()} districts.")
                    y_true_all = all_out[cases_col].astype(float).values
                    pred_col = "fitted_dlnm_py" if "fitted_dlnm_py" in all_out.columns else "fitted_zip"
                    y_pred_all = all_out[pred_col].astype(float).values
                    rmse_all = float(np.sqrt(np.mean((y_true_all - y_pred_all) ** 2)))
                    mae_all = float(np.mean(np.abs(y_true_all - y_pred_all)))
                    m_mask = y_true_all > 0
                    mape_all = (
                        float(np.mean(np.abs((y_true_all[m_mask] - y_pred_all[m_mask]) / y_true_all[m_mask])) * 100)
                        if m_mask.any()
                        else np.nan
                    )

                    c1, c2, c3 = st.columns(3)
                    c1.metric("RMSE (all districts)", f"{rmse_all:,.2f}")
                    c2.metric("MAE (all districts)", f"{mae_all:,.2f}")
                    c3.metric("MAPE (%)", "NA" if not np.isfinite(mape_all) else f"{mape_all:,.1f}")

                    st.dataframe(all_out.head(25))
                    st.download_button(
                        "⬇️ Download fitted (all districts)",
                        data=all_out.to_csv(index=False).encode("utf-8"),
                        file_name=f"fitted_all_{'ZIP' if model_choice.startswith('Zero') else 'DLNM'}.csv",
                        mime="text/csv",
                    )
                else:
                    st.warning(
                        "No districts produced a fit. Try reducing max_lag, lowering spline df, or disabling time smooth."
                    )
            else:
                st.info("Set parameters and click **Run selected model** to process all districts.")


# ====== Alerts & Report ======
if dat_ok:
    with tabs[5]:
        st.markdown("#### EWARS-style alerts")
        st.caption("Select baseline method and generate weekly alerts by district (seasonally adjusted by epi week).")

        method = st.selectbox(
            "Baseline method",
            ["Endemic channel (median + 2*IQR)", "Mean + 2*SD", "95th percentile"],
            help=(
                "**How thresholds are computed**"
                "• **Endemic channel (Median + 2×IQR):** robust to outliers; recommended in many EWARS contexts."
                "• **Mean + 2×SD:** average plus margin; okay when history is stable."
                "• **95th percentile:** triggers only when above ~95% of history (stricter for small N)."
            ),
        )
        with st.popover("ℹ️ Baseline method explainer", use_container_width=True):
            st.markdown(
                """
                **Quick scenarios**
                - *Single huge outbreak in one year*: Endemic channel resists that spike; 95th pct may still be lower than the spike; Mean+2SD can be pulled up.
                - *Stable history*: 95th pct sits near the max; Mean+2SD and Endemic channel are close.
                - *Several elevated years*: all methods move up; 95th pct remains the most sensitive.
                """
            )
        min_years = st.slider("Min years of history to compute baseline", 1, 10, 3)
        exclude_current = st.toggle("Exclude current year from baseline", value=True)

        latest = int(DF[year_col].max())
        base = DF.copy()
        if exclude_current:
            base = base[base[year_col] < latest]

        base["week"] = base[week_col]
        years_per_district = base.groupby([district_key])[year_col].nunique()
        ok_districts = years_per_district[years_per_district >= min_years].index.astype(str).tolist()
        base = base[base[district_key].astype(str).isin(ok_districts)]

        def _baseline_func(s: pd.Series) -> float:
            if method.startswith("Endemic"):
                return float(s.median() + 2.0 * (s.quantile(0.75) - s.quantile(0.25)))
            if method.startswith("Mean"):
                return float(s.mean() + 2.0 * s.std(ddof=0))
            return float(s.quantile(0.95))

        thr = (
            base.groupby([district_key, "week"], as_index=False)
            .agg({cases_col: _baseline_func})
            .rename(columns={cases_col: "threshold"})
        )

        cur = DF[[district_key, year_col, week_col, cases_col]].copy()
        cur["week"] = cur[week_col]
        alert = cur.merge(thr, on=[district_key, "week"], how="left")
        alert["alert"] = alert[cases_col] > alert["threshold"]
        st.dataframe(alert.sort_values([district_key, year_col, week_col]).reset_index(drop=True))

        # Quick alert map for latest year
        if geo_ok:
            st.markdown(f"**Hotspots — {latest} (weeks alerted)**")
            A = (
                alert[alert[year_col] == latest]
                .groupby(district_key, as_index=False)["alert"]
                .sum()
                .rename(columns={"alert": "weeks_alerted"})
            )
            graw = GDF.copy()
            graw.columns = [_clean(c) for c in graw.columns]
            gcols = [c for c in graw.columns if c != "geometry"]
            gkey2 = gcols[0] if len(gcols) == 1 else st.selectbox("Boundary column for alerts map", gcols, key="alerts_gkey")
            g = _prep_boundary(graw, tol_m=800, keep_cols=[gkey2]).copy()
            g[gkey2] = g[gkey2].apply(_norm_key)
            A[district_key] = A[district_key].apply(_norm_key)
            g = g.merge(A, left_on=gkey2, right_on=district_key, how="left")
            g["weeks_alerted"] = g["weeks_alerted"].fillna(0)
            g = g.to_crs(4326)

            center = _map_center_from_gdf(g)
            fmap = folium.Map(location=center, zoom_start=6, tiles="CartoDB positron")
            folium.Choropleth(
                geo_data=g.to_json(),
                data=g,
                columns=[g.index, "weeks_alerted"],
                key_on="feature.id",
                fill_opacity=0.85,
                line_opacity=0.7,
            ).add_to(fmap)
            folium_static(fmap, height=420)

        # Export buttons
        st.markdown("### Export")
        csv_bytes = alert.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download alerts CSV", data=csv_bytes, file_name="ewars_alerts.csv", mime="text/csv")

        try:
            # If a spatial summary exists from Spatial tab, allow export (best-effort)
            choro_candidate = "agg" in locals()
            if choro_candidate:
                choro = agg.rename(columns={"value": "map_value"})
                st.download_button(
                    "⬇️ Download map summary CSV",
                    data=choro.to_csv(index=False).encode("utf-8"),
                    file_name="ewars_map_summary.csv",
                    mime="text/csv",
                )
        except Exception:
            pass

        html = f"""
        <html><head><meta charset='utf-8'><style>body{{font-family:Arial}} table{{border-collapse:collapse}} td,th{{border:1px solid #ccc;padding:4px}}</style></head>
        <body>
        <h2>EWARS Alert Report</h2>
        <p>Method: {method} | Exclude current year: {exclude_current} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        <p>Canonical columns: year, division, district, population, week, weekly_hospitalised_cases, rhdailymean, rainsum, meantemperature</p>
        </body></html>
        """
        st.download_button(
            "⬇️ Download brief HTML report",
            data=html.encode("utf-8"),
            file_name="ewars_alert_report.html",
            mime="text/html",
        )


# ====== Data Quality ======
if dat_ok:
    with tabs[6]:
        st.markdown("#### Data quality checks")
        st.caption("Quick checks: missing values, plausible ranges, and outlier flags.")

        # Missingness summary
        miss = DF.isna().mean().rename("missing_rate").reset_index().rename(columns={"index": "column"})
        miss["missing_rate"] = (100 * miss["missing_rate"]).round(1)
        st.subheader("Missing values (%) by column")
        st.dataframe(miss)

        # Range checks for key fields
        rng = {
            cases_col: (0, np.nanmax(DF[cases_col].values) if cases_col in DF else np.nan),
            temp_col: (np.nanmin(DF[temp_col].values), np.nanmax(DF[temp_col].values)),
            rain_col: (np.nanmin(DF[rain_col].values), np.nanmax(DF[rain_col].values)),
            hum_col: (np.nanmin(DF[hum_col].values), np.nanmax(DF[hum_col].values)),
        }
        st.subheader("Observed ranges (min, max)")
        st.write({k: (float(v[0]) if np.isfinite(v[0]) else None, float(v[1]) if np.isfinite(v[1]) else None) for k, v in rng.items()})

        # Simple outlier count by district using z-score on cases
        st.subheader("Potential outliers by district (|z| > 4)")

        def _z_outliers(x: pd.Series):
            z = (x - x.mean()) / (x.std(ddof=0) + 1e-9)
            return int((z.abs() > 4).sum())

        out_tbl = DF.groupby(district_key)[cases_col].apply(_z_outliers).reset_index().rename(columns={cases_col: "n_outliers"})
        st.dataframe(out_tbl.sort_values("n_outliers", ascending=False))

        st.info("These automated checks are indicative. Please review health MIS logs for confirmed anomalies.")


# ====== Threshold Tuner (CIDARS-style Moving Percentile Method) ======
if dat_ok:
    with tabs[7]:
        st.markdown("#### Threshold Tuner (CIDARS-style Moving Percentile Method)")
        st.caption(
            "Pick a calibration year, sweep percentiles (P40–P95), and select the proper threshold that balances sensitivity and false alerts. Uses same-week ±k window over prior Y years."
        )

        # -------------------- Controls
        years_back = st.slider("Baseline history (years back)", 3, 7, 5, help="How many prior years to use for the reference window.")
        k_weeks = st.slider("Week window (±k)", 0, 3, 2, help="Use same epi-week ±k weeks to build the reference set.")
        use_incidence = st.checkbox(
            "Use incidence per 100k (recommended for fair comparison)", value=True
        )
        min_ref = st.slider(
            "Minimum reference points required", 5, 25, 10, help="Skip a week if fewer historical blocks are available."
        )
        percentiles = list(range(40, 100, 5))  # P40..P95
        gold_method = st.selectbox(
            "Gold standard (for comparison)", ["Mean + 2*SD"], index=0,
            help="China study used 'mean + 2SD' as the temporary gold standard in calibration.",
        )

        years = sorted(int(y) for y in DF[year_col].dropna().unique())
        latest_year_all = years[-1]

        def is_complete_year(df, y, min_weeks=50):
            return df.loc[df[year_col] == y, week_col].nunique() >= min_weeks

        candidate_years = [y for y in years if y < latest_year_all]
        if is_complete_year(DF, latest_year_all):
            candidate_years.append(latest_year_all)
        options = candidate_years or [latest_year_all]
        calib_year = st.selectbox(
            "Calibration year (evaluate thresholds against this year)",
            options=options,
            index=len(options) - 1,
        )

        # -------------------- Helpers
        def _to_metric(series_cases, series_pop):
            """Return counts or incidence per 100k depending on toggle."""
            if not use_incidence:
                return series_cases.astype(float).values
            p = series_pop.astype(float).values
            p = np.where(np.isfinite(p) & (p > 0), p, np.nan)
            return 1e5 * series_cases.astype(float).values / np.where(np.isfinite(p), p, np.nan)

        def _get_ref_values(gdf, y, w, years_back, k_weeks, metric_col):
            """Collect historical values: prior years [y-years_back..y-1], weeks [w-k..w+k]."""
            vals = []
            for yy in range(int(y) - years_back, int(y)):
                for ww in range(max(1, w - k_weeks), min(53, w + k_weeks) + 1):
                    hit = gdf[(gdf[year_col] == yy) & (gdf[week_col] == ww)][metric_col].values
                    if hit.size:
                        vals.append(hit[0])
            vals = np.array(vals, dtype=float)
            vals = vals[np.isfinite(vals)]
            return vals

        def _gold_standard_threshold(ref_vals):
            if ref_vals.size == 0:
                return np.nan
            mu = np.mean(ref_vals)
            sd = np.std(ref_vals, ddof=0)
            return mu + 2.0 * sd

        def _event_onsets(arr):
            """Return indices where a 0->1 transition occurs in a binary sequence."""
            arr = np.asarray(arr, dtype=int)
            onsets = []
            prev = 0
            for i, v in enumerate(arr):
                if prev == 0 and v == 1:
                    onsets.append(i)
                prev = v
            return onsets

        def _tune_for_district(df_d):
            """Return evaluation table for a single district across P40..P95."""
            df_d = df_d[[district_key, year_col, week_col, cases_col, pop_col]].copy()
            df_d = df_d.sort_values([year_col, week_col]).reset_index(drop=True)

            # Metric vector (counts or incidence)
            df_d["metric"] = _to_metric(df_d[cases_col], df_d[pop_col])

            # Slice calibration year rows
            d_cal = df_d[df_d[year_col] == calib_year].copy()
            if d_cal.empty:
                return None

            records = []
            for _, row in d_cal.iterrows():
                y, w = int(row[year_col]), int(row[week_col])
                ref_vals = _get_ref_values(df_d, y, w, years_back, k_weeks, "metric")
                if ref_vals.size < min_ref:
                    continue
                obs = float(row["metric"])
                thr_gold = _gold_standard_threshold(ref_vals)
                gold_alert = int(obs > thr_gold)

                thr_p = {p: np.nanpercentile(ref_vals, p) for p in percentiles}
                alerts_p = {p: int(obs > thr_p[p]) for p in percentiles}
                records.append(
                    {
                        year_col: y,
                        week_col: w,
                        "obs": obs,
                        "gold": gold_alert,
                        **{f"p{p}": alerts_p[p] for p in percentiles},
                    }
                )

            if not records:
                return None

            E = pd.DataFrame(records).sort_values([year_col, week_col]).reset_index(drop=True)

            # Compute performance per percentile vs gold standard
            rows = []
            for p in percentiles:
                pred = E[f"p{p}"].values.astype(int)
                gold = E["gold"].values.astype(int)
                N = len(gold)
                TP = int(((pred == 1) & (gold == 1)).sum())
                TN = int(((pred == 0) & (gold == 0)).sum())
                FP = int(((pred == 1) & (gold == 0)).sum())
                FN = int(((pred == 0) & (gold == 1)).sum())
                Se = TP / max(TP + FN, 1)
                Sp = TN / max(TN + FP, 1)
                YI = Se + Sp - 1.0
                CR = (TP + TN) / max(N, 1)

                # Timeliness: compare onset weeks
                on_gold = _event_onsets(gold)
                on_pred = _event_onsets(pred)
                lead_list = []
                j = 0
                for g_idx in on_gold:
                    while j < len(on_pred) and on_pred[j] < g_idx - 4:
                        j += 1
                    if j < len(on_pred):
                        lead_list.append(on_pred[j] - g_idx)
                med_lead = np.median(lead_list) if lead_list else np.nan

                rows.append(
                    {
                        "district": str(df_d[district_key].iloc[0]),
                        "percentile": p,
                        "N_weeks": N,
                        "TP": TP,
                        "FP": FP,
                        "FN": FN,
                        "TN": TN,
                        "Sensitivity": round(Se, 3),
                        "Specificity": round(Sp, 3),
                        "Youden": round(YI, 3),
                        "ConsistencyRate": round(CR, 3),
                        "MedianLeadWeeks": float(med_lead) if np.isfinite(med_lead) else None,
                    }
                )
            return pd.DataFrame(rows)

        # -------------------- Run tuning (per district), then summarize
        run = st.button("Run tuning (per district)")

        if run:
            dlist = DF[district_key].dropna().astype(str).sort_values().unique().tolist()
            all_eval = []
            prog = st.progress(0)
            for i, d in enumerate(dlist, 1):
                df_d = DF[DF[district_key].astype(str) == str(d)]
                tbl = _tune_for_district(df_d)
                if tbl is not None and len(tbl):
                    all_eval.append(tbl)
                prog.progress(int(100 * i / max(1, len(dlist))))
            prog.empty()

            if not all_eval:
                st.warning(
                    "No districts produced evaluation tables. Try lowering 'Minimum reference points' or selecting another calibration year."
                )
            else:
                EALL = pd.concat(all_eval, axis=0).reset_index(drop=True)

                # District-wise recommended P: choose percentile with max Youden; tie-break by Consistency Rate
                EALL["rk"] = EALL.groupby("district")["Youden"].rank(method="first", ascending=False)
                rec = (
                    EALL.sort_values(["district", "rk", "ConsistencyRate"], ascending=[True, True, False])
                    .groupby("district")
                    .head(1)
                    .drop(columns=["rk"])
                )
                rec = rec.rename(columns={"percentile": "RecommendedP"})

                st.subheader("Recommended percentile per district (maximize Youden)")
                st.dataframe(
                    rec[
                        [
                            "district",
                            "RecommendedP",
                            "Sensitivity",
                            "Specificity",
                            "Youden",
                            "ConsistencyRate",
                            "MedianLeadWeeks",
                        ]
                    ].reset_index(drop=True)
                )
                st.download_button(
                    "⬇️ Download recommended P per district",
                    data=rec.to_csv(index=False).encode("utf-8"),
                    file_name=f"threshold_recommendations_calib{calib_year}_Y{years_back}_k{k_weeks}.csv",
                    mime="text/csv",
                )

                st.subheader("Aggregate performance by percentile (all districts)")
                agg = (
                    EALL.groupby("percentile")
                    .apply(
                        lambda g: pd.Series(
                            {
                                "N_weeks_total": int(g["N_weeks"].sum()),
                                "Sensitivity_avg": g["Sensitivity"].mean(),
                                "Specificity_avg": g["Specificity"].mean(),
                                "Youden_avg": g["Youden"].mean(),
                                "ConsistencyRate_avg": g["ConsistencyRate"].mean(),
                                "MedianLeadWeeks_med": g["MedianLeadWeeks"].median(skipna=True),
                            }
                        )
                    )
                    .reset_index()
                )
                st.dataframe(agg)

                # Simple plot: Youden vs percentile
                figp = go.Figure()
                figp.add_trace(
                    go.Scatter(x=agg["percentile"], y=agg["Youden_avg"], mode="lines+markers", name="Youden (avg)")
                )
                figp.add_trace(
                    go.Scatter(
                        x=agg["percentile"], y=agg["ConsistencyRate_avg"], mode="lines+markers", name="Consistency (avg)"
                    )
                )
                figp.update_layout(
                    title=f"Threshold sweep (calibration {calib_year}; Y={years_back}, ±k={k_weeks})",
                    xaxis_title="Percentile (P)",
                    yaxis_title="Score",
                )
                st.plotly_chart(figp, use_container_width=True)

                st.info(
                    "Tip: Start with the district-wise recommendations above. If you must set a single national default, pick the percentile with the highest average Youden (or a balance between Youden and Consistency)."
                )

