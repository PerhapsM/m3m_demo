from pathlib import Path
import re
import math
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
import rasterio
from tifffile import TiffFile


DATA_DIR = Path(r"Multispectral Photo Folder")

BAND_LABELS = {
    "G": "Green",
    "R": "Red",
    "RE": "Red Edge",
    "NIR": "NIR",
}


@st.cache_data(show_spinner=False)
def discover_missions(data_dir: Path):
    tif_files = list(data_dir.glob("*.TIF")) + list(data_dir.glob("*.tif"))
    jpg_files = list(data_dir.glob("*.JPG")) + list(data_dir.glob("*.jpg"))
    image_map = {}

    for file in tif_files:
        match = re.search(r"DJI_(\d{8}\d{6})_(\d{4})_MS_([A-Z]+)\.TIF", file.name, re.IGNORECASE)
        if not match:
            continue

        flight_tag, image_number, band = match.groups()
        core_name = file.name.split("_MS_")[0]
        entry = image_map.setdefault(
            core_name,
            {
                "image_number": image_number,
                "flight_tag": flight_tag,
                "files": {},
                "source_folder": str(data_dir),
                "rgb_files": [],
            },
        )
        entry["files"][band] = file

    for file in jpg_files:
        match = re.search(r"DJI_(\d{8}\d{6})_(\d{4})_D\.JPG", file.name, re.IGNORECASE)
        if not match:
            continue
        core_name = file.name.split("_D.")[0]
        for mission_key, meta in image_map.items():
            if mission_key == core_name:
                meta["rgb_files"].append(file)

    rows = []
    for mission_key, meta in image_map.items():
        files = meta["files"]
        rows.append(
            {
                "mission_id": mission_key,
                "flight_time": meta["flight_tag"],
                "capture_index": meta["image_number"],
                "bands": sorted(files.keys()),
                "raw_files": [str(f) for f in files.values()],
                "files": files,
                "rgb_files": meta["rgb_files"],
            }
        )

    return sorted(rows, key=lambda r: r["mission_id"])


@st.cache_data(show_spinner=False)
def load_band(file_path: str):
    with rasterio.open(file_path) as src:
        arr = src.read(1).astype(np.float32)
    return arr


@st.cache_data(show_spinner=False)
def read_tiff_metadata(file_path: str):
    meta = {}
    with rasterio.open(file_path) as src:
        meta.update(
            {
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "dtype": str(src.dtypes[0]),
                "crs": str(src.crs) if src.crs else "Unknown",
                "driver": src.driver,
                "bounds": list(src.bounds),
                "resolution": src.res,
            }
        )
    with TiffFile(file_path) as tif:
        tags = tif.pages[0].tags
        meta["tags"] = {tag.name: tag.value for _, tag in tags.items()}
    return meta


@st.cache_data(show_spinner=False)
def renderable_image(file_path: str):
    try:
        img = Image.open(file_path)
        mode = img.mode

        if mode in {"I;16", "I;16B", "I;16L", "I;16N", "I"}:
            arr = np.asarray(img).astype(np.float32)
            arr_min = np.nanmin(arr)
            arr_max = np.nanmax(arr)

            if not np.isfinite(arr_min) or not np.isfinite(arr_max) or arr_max == arr_min:
                arr_u8 = np.zeros(arr.shape, dtype=np.uint8)
            else:
                arr_norm = (arr - arr_min) / (arr_max - arr_min)
                arr_norm = np.clip(arr_norm, 0, 1)
                arr_u8 = (arr_norm * 255).astype(np.uint8)

            return Image.fromarray(arr_u8)

        if mode not in {"RGB", "RGBA", "L", "P"}:
            img = img.convert("RGB")
        return img
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def compute_mission_ndvi(files: dict):
    if "NIR" not in files or "R" not in files:
        return None, None

    nir = load_band(str(files["NIR"]))
    red = load_band(str(files["R"]))
    denom = nir + red
    ndvi = np.full(nir.shape, np.nan, dtype=np.float32)
    mask = denom != 0
    ndvi[mask] = (nir[mask] - red[mask]) / denom[mask]

    valid = ndvi[np.isfinite(ndvi)]
    health_stats = {
        "mean": float(np.nanmean(valid)) if valid.size else None,
        "median": float(np.nanmedian(valid)) if valid.size else None,
        "std": float(np.nanstd(valid)) if valid.size else None,
        "min": float(np.nanmin(valid)) if valid.size else None,
        "max": float(np.nanmax(valid)) if valid.size else None,
        "health_area_pct": float(np.mean((valid >= 0.1) & (valid <= 0.7)) * 100) if valid.size else 0.0,
    }

    return ndvi, health_stats


@st.cache_data(show_spinner=False)
def describe_band_stats(file_path: str):
    arr = load_band(file_path)
    valid = arr[np.isfinite(arr)]
    return {
        "min": float(np.nanmin(valid)) if valid.size else 0.0,
        "max": float(np.nanmax(valid)) if valid.size else 0.0,
        "mean": float(np.nanmean(valid)) if valid.size else 0.0,
        "median": float(np.nanmedian(valid)) if valid.size else 0.0,
        "std": float(np.nanstd(valid)) if valid.size else 0.0,
    }



def build_dashboard():
    st.logo("images/logo.png")
    st.set_page_config(page_title="M3M Precision Agriculture Demo", layout="wide", page_icon="images/icon.png")

    st.markdown(
        """
        <style>
            .platform-header {
                background: linear-gradient(120deg, #09213d, #155e75);
                border-radius: 14px;
                padding: 24px 26px;
                color: white;
                margin-bottom: 16px;
            }
            .platform-header h1 {
                margin: 0 0 10px;
                font-weight: 700;
                font-size: 2.2rem;
            }
            .platform-header p {
                margin: 0;
                color: #d7f8e7;
            }
            .kpi-card {
                background: #f8fafc;
                border-radius: 12px;
                border: 1px solid #d1e6dd;
                padding: 14px 14px;
                margin-bottom: 10px;
                min-height: 96px;
            }
            .kpi-label {
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.12em;
                color: #476172;
                text-transform: uppercase;
            }
            .kpi-value {
                font-size: 25px;
                font-weight: 800;
                color: #153a3a;
                margin-top: 4px;
            }
            .platform-panel {
                border-radius: 14px;
                padding: 12px;
                border: 1px solid #bedbcc;
                background: linear-gradient(180deg, #eefcf7, #fafffb);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="platform-header">
            <h1>DJI M3M Precision Agriculture Platform</h1>
            <p>Mission Intelligence Dashboard · Multispectral Field Intelligence</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.title("M3M Mission Console")
    st.sidebar.info("Raw data folder: Multispectral Photo Folder")

    raw_base = st.sidebar.text_input(
        "Mission Folder",
        value=str(DATA_DIR),
        help="This demo reads the DJI M3M raw data folder directly.",
    )

    try:
        data_dir = Path(raw_base)
        missions = discover_missions(data_dir)
    except Exception as exc:
        st.sidebar.error(f"Could not discover mission files: {exc}")
        missions = []

    if not missions:
        st.warning("No supported DJI M3M multispectral image files were found in the selected folder.")
        return

    selected_mission = st.sidebar.selectbox(
        "Select Mission",
        options=[m["mission_id"] for m in missions],
        index=0,
    )

    selected = next(item for item in missions if item["mission_id"] == selected_mission)

    flight_year = selected["flight_time"][:4] if selected["flight_time"] else "Unknown"
    # flight_tag = selected["flight_time"]

    # st.sidebar.markdown("---")
    # st.sidebar.metric("Mission Count", len(missions))
    # st.sidebar.metric("Selected Flight", flight_tag)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Platform Navigation")
    page = st.sidebar.radio(
        "Go to",
        options=["Upload Mission", "Explore Images", "Analytics", "Future Platform"],
        index=0,
        label_visibility="collapsed",
    )

    mission_files = {}
    for file in data_dir.glob("*.TIF"):
        match = re.search(r"DJI_\d{14}_\d{4}_MS_([A-Z]+)\.TIF", file.name, re.IGNORECASE)
        if match:
            image_band = match.group(1).upper()
            core = file.name.split("_MS_")[0]
            if core == selected_mission:
                mission_files[image_band] = file

    if not mission_files:
        st.warning("No matching raw multispectral files found for the selected mission.")
        return

    selected_files = next((m["files"] for m in missions if m["mission_id"] == selected_mission), {})
    ndvi_map, ndvi_stats = compute_mission_ndvi(selected_files)

    st.subheader("Control Center")
    kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
    with kpi_c1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Mission</div>
                <div class="kpi-value">{selected_mission}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_c2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Images</div>
                <div class="kpi-value">{len(missions)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_c3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Band Channels</div>
                <div class="kpi-value">{len(mission_files)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_c4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Year</div>
                <div class="kpi-value">{flight_year}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="platform-panel">
            <b>Mission detected automatically</b>: DJI M3M raw files were scanned and mapped to a campaign mission profile.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    if page == "Upload Mission":
        st.subheader("1. Upload Mission")
        # summary_cols = st.columns(5)
        summary_cols = st.columns([1, 1, 1, 1, 2])
        with summary_cols[0]:
            st.metric("Dataset", "M3M")
        with summary_cols[1]:
            st.metric("Flight Source", "DJI PPK")
        with summary_cols[2]:
            st.metric("Frame Capture", selected["capture_index"])
        with summary_cols[3]:
            st.metric("Raw Channels", f"{len(mission_files)}")
        with summary_cols[4]:
            st.metric("Platform", "Agriculture Intelligence")

        st.markdown("---")
        st.subheader("Mission Metadata")
        mission_meta_rows = []
        for band, file_path in selected_files.items():
            meta = read_tiff_metadata(str(file_path))
            mission_meta_rows.append(
                {
                    "Band": BAND_LABELS.get(band, band),
                    "File": file_path.name,
                    "Width": meta.get("width"),
                    "Height": meta.get("height"),
                    "Channels": meta.get("count"),
                    "dtype": meta.get("dtype"),
                    "CRS": meta.get("crs"),
                }
            )
        st.dataframe(pd.DataFrame(mission_meta_rows), use_container_width=True)

    elif page == "Explore Images":
        st.subheader("2. Explore Images")
        band_tabs = st.tabs(["RGB", "Green", "Red", "Red Edge", "NIR"])

        for tab, band_name in zip(band_tabs, ["RGB", "G", "R", "RE", "NIR"]):
            with tab:
                if band_name == "RGB":
                    rgb_files = [data_dir / f.name for f in data_dir.glob("*.JPG") if selected_mission in f.name]
                    if rgb_files:
                        rgb_path = rgb_files[0]
                        rgb_img = Image.open(rgb_path)
                        st.image(rgb_img, caption=rgb_path.name, use_container_width=True)
                    else:
                        st.info("No RGB JPG preview found for this mission.")
                else:
                    band_file = mission_files.get(band_name)
                    if band_file:
                        img = Image.open(band_file)
                        preview = renderable_image(str(band_file))
                        if preview is not None:
                            st.image(preview, caption=band_file.name, use_container_width=True)

                        meta = {
                            "File": band_file.name,
                            "Band": BAND_LABELS.get(band_name, band_name),
                            "Mode": img.mode,
                            "Size": img.size,
                            "Format": img.format,
                        }
                        st.dataframe(pd.DataFrame([meta]), use_container_width=True)
                    else:
                        st.info(f"No {band_name} channel available for the selected mission.")

        st.subheader("Pixel Inspector")
        inspector_col, inspector_plot = st.columns([1, 3])

        selected_band = inspector_col.selectbox(
            "Channel",
            options=["G", "R", "RE", "NIR"],
            index=1,
        )

        pixel_x = inspector_col.slider("Pixel X", 0, 500, 250)
        pixel_y = inspector_col.slider("Pixel Y", 0, 500, 250)

        band_file = mission_files.get(selected_band)
        if band_file:
            img = Image.open(band_file)
            arr = np.asarray(img)
            height, width = arr.shape[:2]
            px_val = arr[min(height - 1, pixel_y), min(width - 1, pixel_x)]
            inspector_plot.metric("Selected Channel", BAND_LABELS.get(selected_band, selected_band))
            inspector_plot.metric("Value", f"{float(px_val):.2f}")

            preview = renderable_image(str(band_file))
            if preview is not None:
                inspector_plot.image(preview, caption=band_file.name, use_container_width=True)
            else:
                inspector_plot.info("Image preview could not be rendered.")
        else:
            inspector_plot.info("No pixel data is available for the selected band.")

    elif page == "Analytics":
        st.subheader("3. Analytics")

        report_col, map_col = st.columns([1.1, 1.9])
        with report_col:
            st.markdown(
                """
                <div class="platform-panel">
                    <h3>Field Report</h3>
                    <p><b>Crop Zone:</b> North Block A</p>
                    <p><b>Field Status:</b> Monitoring / Green-up</p>
                    <p><b>Recommended Action:</b> Verify NDVI pockets in South range</p>
                    <p><b>Sample Time:</b> 2023-02-15 10:39</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with map_col:
            st.markdown(
                """
                <div class="platform-panel">
                    <h3>Field Map / Controller</h3>
                    <p>Field Grid: Block A-02 · Flight 0003</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            map_df = pd.DataFrame(
                {
                    "x": [10, 20, 30, 40, 50],
                    "y": [20, 15, 35, 25, 45],
                    "value": [0.12, 0.34, 0.52, 0.39, 0.71],
                }
            )
            fig = px.scatter(map_df, x="x", y="y", size="value", hover_data=["value"], title="Field Grid Heat Map")
            st.plotly_chart(fig, use_container_width=True)

        ndvi_tab, health_tab, charts_tab = st.tabs(["NDVI", "Health Statistics", "Charts"])

        ndvi_data = None
        with ndvi_tab:
            if "NIR" in mission_files and "R" in mission_files:
                nir = load_band(str(mission_files["NIR"]))
                red = load_band(str(mission_files["R"]))
                denom = nir + red
                ndvi = np.full(nir.shape, np.nan, dtype=np.float32)
                mask = denom != 0
                ndvi[mask] = (nir[mask] - red[mask]) / denom[mask]

                valid = ndvi[np.isfinite(ndvi)]
                mean_ndvi = float(np.nanmean(valid)) if valid.size else 0.0
                median_ndvi = float(np.nanmedian(valid)) if valid.size else 0.0

                st.metric("Mean NDVI", f"{mean_ndvi:.4f}")
                st.metric("Median NDVI", f"{median_ndvi:.4f}")
                st.color_picker("NDVI Color Scale", value="#7dd3fc")

                ndvi_fig = px.imshow(
                    ndvi,
                    color_continuous_scale="Viridis",
                    labels=dict(color="NDVI"),
                    title="NDVI Heatmap",
                    aspect="auto",
                )
                st.plotly_chart(ndvi_fig, use_container_width=True)

                hist_fig = px.histogram(valid, nbins=40, title="NDVI Distribution")
                st.plotly_chart(hist_fig, use_container_width=True)

                ndvi_data = pd.DataFrame({"NDVI": valid})
            else:
                st.warning("NDVI requires both Red and NIR channels to be present.")

        with health_tab:
            if ndvi_data is not None:
                health = ndvi_data["NDVI"]
                health_healthy = health[(health >= 0.35) & (health <= 0.75)]
                health_stressed = health[health < 0.2]
                health_fig = px.pie(
                    values=[health_healthy.count(), health_stressed.count(), (health.count() - health_healthy.count() - health_stressed.count())],
                    names=["Healthy", "Stressed", "Other"],
                    title="Crop Health Composition",
                )
                st.plotly_chart(health_fig, use_container_width=True)

                st.dataframe(
                    pd.DataFrame(
                        {
                            "Metric": ["Healthy Zone %", "Stressed Zone %", "Mean NDVI", "Median NDVI"],
                            "Value": [
                                f"{(len(health_healthy) / len(health)) * 100:.2f}%",
                                f"{(len(health_stressed) / len(health)) * 100:.2f}%",
                                f"{float(np.nanmean(health)):.4f}",
                                f"{float(np.nanmedian(health)):.4f}",
                            ],
                        }
                    ),
                    use_container_width=True,
                )
            else:
                st.info("Health statistics will be calculated after the required bands are available.")

        with charts_tab:
            band_stats = []
            for band, path in selected_files.items():
                stats = describe_band_stats(str(path))
                band_stats.append(
                    {
                        "Band": BAND_LABELS.get(band, band),
                        "Min": stats["min"],
                        "Max": stats["max"],
                        "Mean": stats["mean"],
                        "Median": stats["median"],
                    }
                )
            if band_stats:
                stats_df = pd.DataFrame(band_stats)
                st.dataframe(stats_df, use_container_width=True)
                chart = px.bar(
                    stats_df,
                    x="Band",
                    y="Mean",
                    title="Mean Pixel Value by Band",
                    color="Band",
                )
                st.plotly_chart(chart, use_container_width=True)
            else:
                st.info("No raw band statistics are available.")

    elif page == "Future Platform":
        st.subheader("4. Future Platform")
        st.markdown(
            """
            The current demo is a starting point for a Precision Agriculture Platform. The roadmap is to evolve from raw M3M data ingestion into an operational intelligence layer for field decision support.
            """
        )

        future_cols = st.columns(4)
        capabilities = [
            ("AI Crop Diagnosis", "Disease and pest visualization, anomaly ranking, monitoring intelligence"),
            ("Variable Rate Fertilizer", "Prescription maps linked to real field variability"),
            ("Temporal Monitoring", "Seasonal growth curves and crop performance trends"),
            ("Auto Reports", "Executive PDF and field operation reports"),
        ]
        for idx, (title, desc) in enumerate(capabilities):
            with future_cols[idx % 4]:
                st.markdown(
                    f"""
                    <div style="padding:12px;border-radius:10px;background:#eef2ff;border:1px solid #a5b4fc;min-height:150px">
                    <b>{title}</b><br>{desc}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.info("This demo is not the end state — it is a first-step Precision Agriculture Platform layer for M3M mission intelligence.")


if __name__ == "__main__":
    build_dashboard()
