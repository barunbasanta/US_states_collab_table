
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon as MplPolygon
from bokeh.sampledata.us_states import data as BOKEH_STATES

st.set_page_config(page_title="Collaboration Table Generator", page_icon="📊", layout="wide")

STATE_ABBR = {
    "Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA",
    "Colorado":"CO","Connecticut":"CT","Delaware":"DE","Florida":"FL","Georgia":"GA",
    "Hawaii":"HI","Idaho":"ID","Illinois":"IL","Indiana":"IN","Iowa":"IA","Kansas":"KS",
    "Kentucky":"KY","Louisiana":"LA","Maine":"ME","Maryland":"MD","Massachusetts":"MA",
    "Michigan":"MI","Minnesota":"MN","Mississippi":"MS","Missouri":"MO","Montana":"MT",
    "Nebraska":"NE","Nevada":"NV","New Hampshire":"NH","New Jersey":"NJ","New Mexico":"NM",
    "New York":"NY","North Carolina":"NC","North Dakota":"ND","Ohio":"OH","Oklahoma":"OK",
    "Oregon":"OR","Pennsylvania":"PA","Rhode Island":"RI","South Carolina":"SC",
    "South Dakota":"SD","Tennessee":"TN","Texas":"TX","Utah":"UT","Vermont":"VT",
    "Virginia":"VA","Washington":"WA","West Virginia":"WV","Wisconsin":"WI","Wyoming":"WY"
}

SCHEMES = {
    "red": [("1–2", "#FFF7D6"), ("3–10", "#FFD07A"), ("11–20", "#F89C3D"), ("21+", "#D62828")],
    "green": [("1–2", "#EAF9EF"), ("3–10", "#98E2B8"), ("11–20", "#1F8A5B"), ("21+", "#063B25")],
    "blue": [("1–2", "#E6F7FF"), ("3–10", "#7FD3FF"), ("11–20", "#1565D8"), ("21+", "#062B6F")],
}

P_COLOR = "#0057E7"
G_COLOR = "#14833B"
T_COLOR = "#8B4513"

def state_fill(total, bins):
    t = int(total)
    if t <= 2:
        return bins[0][1]
    if t <= 10:
        return bins[1][1]
    if t <= 20:
        return bins[2][1]
    return bins[3][1]

@st.cache_resource
def load_state_geometries():
    """
    Load actual US state outline coordinates from Bokeh sample data.

    This avoids the Streamlit Cloud basemap dependency problem and prevents
    falling back to same-looking colored rectangles. Each table icon is drawn
    from real state polygon coordinates.
    """
    geoms = {}
    for abbr, item in BOKEH_STATES.items():
        geoms[abbr] = {
            "lons": np.array(item["lons"], dtype=float),
            "lats": np.array(item["lats"], dtype=float),
        }
    return geoms


def normalize_table(df):
    lower = {str(c).strip().lower(): c for c in df.columns}
    def find(names):
        for n in names:
            if n.lower() in lower:
                return lower[n.lower()]
        return None

    state_col = find(["State", "State Name"])
    if state_col is None:
        raise ValueError("Need a State column.")

    out = pd.DataFrame()
    out["State"] = df[state_col].astype(str).str.strip()

    for target, names in {
        "P": ["P", "Publications", "Publication"],
        "G": ["G", "Grants", "Grant"],
        "T": ["T", "Trials", "Clinical Trials", "Clinical Trial"],
    }.items():
        c = find(names)
        out[target] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int) if c else 0

    total_col = find(["Total", "Total Collaborations"])
    if total_col:
        out["Total"] = pd.to_numeric(df[total_col], errors="coerce").fillna(out["P"] + out["G"] + out["T"]).astype(int)
    else:
        out["Total"] = out["P"] + out["G"] + out["T"]

    out = out[out["State"].isin(STATE_ABBR)].copy()
    out["Abbr"] = out["State"].map(STATE_ABBR)
    return out.sort_values("Total", ascending=False).reset_index(drop=True)

def _split_nan_parts(xs, ys):
    """Split coordinate arrays into separate polygon rings at NaN gaps."""
    parts = []
    cur_x, cur_y = [], []
    for x, y in zip(xs, ys):
        if np.isnan(x) or np.isnan(y):
            if len(cur_x) >= 3:
                parts.append((np.array(cur_x), np.array(cur_y)))
            cur_x, cur_y = [], []
        else:
            cur_x.append(x)
            cur_y.append(y)
    if len(cur_x) >= 3:
        parts.append((np.array(cur_x), np.array(cur_y)))
    return parts


def draw_icon(ax, geom, x0, y0, w, h, fill, linewidth=1.0):
    """
    Draw a real state outline inside the icon box.

    Uses true state polygon coordinates from Bokeh's built-in US states data.
    If a state is missing, only then a small square fallback is drawn.
    """
    if geom is None:
        ax.add_patch(Rectangle((x0 + w*0.20, y0 + h*0.20), w*0.60, h*0.60,
                               facecolor=fill, edgecolor="#404040", linewidth=linewidth))
        return

    lons = geom["lons"]
    lats = geom["lats"]
    parts = _split_nan_parts(lons, lats)

    if not parts:
        return

    all_x = np.concatenate([p[0] for p in parts])
    all_y = np.concatenate([p[1] for p in parts])

    minx, maxx = np.nanmin(all_x), np.nanmax(all_x)
    miny, maxy = np.nanmin(all_y), np.nanmax(all_y)

    sx = (w * 0.88) / max(maxx - minx, 1e-9)
    sy = (h * 0.88) / max(maxy - miny, 1e-9)

    for px, py in parts:
        xs = (px - minx) * sx + x0 + w * 0.06
        ys = (py - miny) * sy + y0 + h * 0.06

        ax.add_patch(
            MplPolygon(
                np.c_[xs, ys],
                closed=True,
                facecolor=fill,
                edgecolor="#404040",
                linewidth=linewidth,
                joinstyle="round"
            )
        )


def render_double(df, bins):
    geoms = load_state_geometries()
    table_df = df[(df["Total"] > 0) & (df["State"] != "Massachusetts")].copy()
    half = int(np.ceil(len(table_df) / 2))
    left_df, right_df = table_df.iloc[:half], table_df.iloc[half:]

    row_h = 0.82
    fig_h = max(len(left_df), len(right_df)) * row_h + 2.0
    fig, ax = plt.subplots(figsize=(18.5, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    ax.add_patch(FancyBboxPatch((0.01,0.01),0.98,0.98,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.25, edgecolor="#DADADA", facecolor="white", transform=ax.transAxes))

    top_y = fig_h - 0.48
    for part, x0, w in [(left_df, 0.03, 0.44), (right_df, 0.515, 0.44)]:
        shape_x = x0 + 0.045
        state_x = x0 + 0.092
        pub_x = x0 + 0.255
        grant_x = x0 + 0.350
        trial_x = x0 + 0.420
        total_x = x0 + 0.470

        for txt, x, col, ha in [
            ("State", shape_x, "black", "center"),
            ("State Name", state_x, "black", "left"),
            ("Publications", pub_x, P_COLOR, "center"),
            ("Grants", grant_x, G_COLOR, "center"),
            ("Trials", trial_x, T_COLOR, "center"),
            ("Total", total_x, "black", "center"),
        ]:
            ax.text(x, top_y, txt, ha=ha, va="center", fontsize=13.2, fontweight="bold", color=col)

        tile_w = 0.028
        tile_h = row_h * 0.52

        for i, (_, row) in enumerate(part.iterrows()):
            yc = top_y - 0.60 - i * row_h
            ax.plot([x0, x0+w], [yc-row_h*0.45, yc-row_h*0.45], color="#ECECEC", lw=0.8)
            tx = shape_x - tile_w/2
            ty = yc - tile_h/2
            ax.add_patch(FancyBboxPatch((tx,ty), tile_w, tile_h,
                boxstyle="round,pad=0.002,rounding_size=0.004",
                linewidth=0.8, edgecolor="#D3D3D3", facecolor="#FAFAFA"))
            draw_icon(ax, geoms.get(row["Abbr"]), tx, ty, tile_w, tile_h, state_fill(row["Total"], bins))
            ax.text(state_x, yc, row["State"], ha="left", va="center", fontsize=12.8, fontweight="bold")
            for col, x, color in [("P", pub_x, P_COLOR), ("G", grant_x, G_COLOR), ("T", trial_x, T_COLOR), ("Total", total_x, "black")]:
                ax.text(x, yc, str(int(row[col])), ha="center", va="center", fontsize=12.8, fontweight="bold", color=color)

    legend_y = 0.46
    legend_x = 0.31
    legend_w = 0.40
    ax.text(legend_x, legend_y + 0.42, "Total Collaborations", fontsize=13.0, fontweight="bold")
    sw = legend_w / len(bins)
    for i, (lab, color) in enumerate(bins):
        x = legend_x + i*sw
        ax.add_patch(Rectangle((x, legend_y), sw, 0.22, facecolor=color, edgecolor="#9E9E9E"))
        ax.text(x+sw/2, legend_y-0.08, lab, ha="center", va="top", fontsize=10.5, fontweight="bold")
    return fig

def render_mass(df, bins):
    geoms = load_state_geometries()
    ma = df[df["State"] == "Massachusetts"]
    row = ma.iloc[0] if not ma.empty else pd.Series({"P":0, "G":0, "T":0, "Total":0})

    fig, ax = plt.subplots(figsize=(10.8, 1.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(FancyBboxPatch((0.01,0.12),0.98,0.76,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.4, edgecolor="#DADADA", facecolor="white"))

    tx, ty = 0.045, 0.34
    tile_w, tile_h = 0.065, 0.24
    ax.add_patch(FancyBboxPatch((tx,ty), tile_w, tile_h,
        boxstyle="round,pad=0.003,rounding_size=0.006",
        linewidth=0.9, edgecolor="#D3D3D3", facecolor="#FAFAFA"))

    draw_icon(ax, geoms.get("MA"), tx, ty, tile_w, tile_h, state_fill(row["Total"], bins), linewidth=1.6)

    ax.text(0.16, 0.46, "Massachusetts (Non KI)", fontsize=16, fontweight="bold", ha="left", va="center")
    for val, x, color in [(row["P"],0.60,P_COLOR), (row["G"],0.73,G_COLOR), (row["T"],0.84,T_COLOR), (row["Total"],0.94,"black")]:
        ax.text(x, 0.46, str(int(val)), fontsize=16, fontweight="bold", color=color, ha="center", va="center")
    return fig

def fig_bytes(fig, fmt):
    b = io.BytesIO()
    fig.savefig(b, format=fmt, dpi=700 if fmt == "png" else None,
                bbox_inches="tight", facecolor="white", pad_inches=0.08)
    b.seek(0)
    plt.close(fig)
    return b.getvalue()

def make_zip(files):
    import zipfile
    z = io.BytesIO()
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zipf:
        for name, data in files.items():
            zipf.writestr(name, data)
    z.seek(0)
    return z.getvalue()

st.title("Collaboration Table Generator")
st.caption("Generate 3 two-column tables and 3 Massachusetts legend rows in Red, Green, and Blue; each exported as high-quality PNG and PDF.")

uploaded = st.file_uploader("Upload CSV or Excel counts table", type=["csv", "xlsx", "xls"])
st.write("Required columns: `State`, `P`, `G`, `T`; `Total` is optional.")

if uploaded:
    try:
        raw = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
        df = normalize_table(raw)
        st.dataframe(df, use_container_width=True, hide_index=True)

        if st.button("Generate all files", type="primary"):
            outputs = {}
            for scheme, bins in SCHEMES.items():
                outputs[f"double_table_{scheme}.png"] = fig_bytes(render_double(df, bins), "png")
                outputs[f"double_table_{scheme}.pdf"] = fig_bytes(render_double(df, bins), "pdf")
                outputs[f"massachusetts_legend_{scheme}.png"] = fig_bytes(render_mass(df, bins), "png")
                outputs[f"massachusetts_legend_{scheme}.pdf"] = fig_bytes(render_mass(df, bins), "pdf")

            st.download_button(
                "Download ZIP",
                data=make_zip(outputs),
                file_name="collaboration_tables_3_schemes.zip",
                mime="application/zip",
            )

            st.success("Done. The ZIP contains 12 files.")
    except Exception as e:
        st.error("Could not process the file.")
        st.exception(e)
else:
    st.info("Upload your table to begin.")
