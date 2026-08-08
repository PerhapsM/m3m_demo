# M3M Precision Agriculture Dashboard

This project is a Streamlit-based demo dashboard for visualizing DJI M3M multispectral mission data and presenting a first-step Precision Agriculture Platform experience.

## Purpose

The dashboard is designed to support the following workflow:

1. Upload or discover an M3M mission directory.
2. Inspect metadata and raw mission files.
3. Explore RGB, Green, Red, Red Edge, and NIR image layers.
4. Calculate basic vegetation indicators such as NDVI.
5. Present field-level health, chart, and operational intelligence panels.
6. Showcase a future platform roadmap for AI, prescription mapping, temporal monitoring, and report automation.

## Project Structure

- `m3m_dashboard.py` — main Streamlit application entry point.
- `filelist.txt` — inventory of files discovered in the mission dataset.
- `Multispectral Photo Folder/` — sample raw mission folder containing DJI-style TIFF/JPG data.

## Features

- Automatic mission detection and mission folder selection.
- Mission metadata summary table.
- Band-level image tabs and file metadata views.
- Pixel inspection for selected multispectral bands.
- NDVI heatmap and histogram visualization.
- Crop health statistic views.
- Field report and controller-style analytics mockups.
- Future platform capability section.

## Run Locally

From this folder, run:

```powershell
python -m streamlit run m3m_dashboard.py
```

Or with a specific port:

```powershell
python -m streamlit run m3m_dashboard.py --server.port 8504
```

## Dependencies

The dashboard expects common geospatial and visualization packages such as:

- streamlit
- pandas
- numpy
- plotly
- pillow
- rasterio
- tifffile

## Notes

This project is a demo and demonstration platform layer. It is not a complete operational field intelligence stack, but it demonstrates a path from raw M3M capture data to a decision-support dashboard.
