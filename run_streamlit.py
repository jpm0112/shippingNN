import streamlit as st
import pandas as pd
from PIL import Image
import glob
import os
import pathlib
from datetime import datetime

st.set_page_config(layout="wide", page_title="Project Results Browser")

ROOT = pathlib.Path(".")
PLOTS_DIR = ROOT / "plots"
RESULTS_DIR = ROOT / "results"

# discover files
image_patterns = [
    str(PLOTS_DIR / "**" / "*.png"),
    str(PLOTS_DIR / "**" / "*.jpg"),
    str(PLOTS_DIR / "**" / "*.jpeg"),
]
images = []
for p in image_patterns:
    images.extend(glob.glob(p, recursive=True))

# CSVs: search in root + results + plots (dedup)
csv_patterns = [
    str(ROOT / "*.csv"),
    str(RESULTS_DIR / "**" / "*.csv"),
    str(PLOTS_DIR / "**" / "*.csv"),
]
csv_files = []
for p in csv_patterns:
    csv_files.extend(glob.glob(p, recursive=True))
csv_files = sorted(set(csv_files))

# infer methods from filenames/paths using common keywords
keywords = ["lasso", "lstm", "tft", "transformer", "transformers", "sarima", "naive", "xai", "tuning", "bayesopt"]
methods = set()
for f in images + csv_files:
    hay = f.lower()
    for k in keywords:
        if k in hay:
            methods.add(k)
methods = sorted(methods)
methods.insert(0, "all")

st.sidebar.title("Controls")
method = st.sidebar.selectbox("Filter by method", methods)
show_images = st.sidebar.checkbox("Show images", value=True)
show_csvs = st.sidebar.checkbox("Show CSVs", value=True)
max_rows = st.sidebar.slider("CSV preview rows", 5, 500, 100)

def file_meta(path):
    info = pathlib.Path(path)
    stat = info.stat()
    return f"{stat.st_size/1024:.1f} KB • {datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')}"

st.title("Project Results Browser")
st.markdown("Displays plots and CSV outputs produced by scripts in the repository.")

# Images section
if show_images:
    st.header("Plots")
    filtered_imgs = [p for p in images if method == "all" or method in p.lower()]
    if not filtered_imgs:
        st.info("No plot images found for the selected method.")
    else:
        cols = st.columns(2)
        for i, img_path in enumerate(sorted(filtered_imgs, key=os.path.getmtime, reverse=True)):
            with cols[i % 2]:
                st.subheader(os.path.basename(img_path))
                try:
                    img = Image.open(img_path)
                    st.image(img, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not open image: {e}")
                st.caption(file_meta(img_path))
                with open(img_path, "rb") as f:
                    st.download_button(
                        "Download image",
                        f.read(),
                        file_name=os.path.basename(img_path),
                        key=f"dl_img_{img_path}",
                    )

# CSV section
if show_csvs:
    st.header("CSV outputs")
    filtered_csvs = [p for p in csv_files if method == "all" or method in p.lower()]
    if not filtered_csvs:
        st.info("No CSVs found for the selected method.")
    else:
        for csv_path in sorted(filtered_csvs, key=os.path.getmtime, reverse=True):
            st.subheader(os.path.basename(csv_path))
            st.write(f"Preview ({file_meta(csv_path)})")

            # preview only (fast)
            try:
                df_preview = pd.read_csv(csv_path, nrows=max_rows, encoding_errors="replace")
            except TypeError:
                # for older pandas/python: fallback without encoding_errors
                df_preview = pd.read_csv(csv_path, nrows=max_rows)

            except Exception as e:
                st.error(f"Could not read CSV preview: {e}")
                continue

            st.dataframe(df_preview)

            # raw file download (no need to load full dataframe)
            try:
                with open(csv_path, "rb") as f:
                    st.download_button(
                        "Download CSV",
                        f.read(),
                        file_name=os.path.basename(csv_path),
                        mime="text/csv",
                        key=f"dl_csv_{csv_path}",
                    )
            except Exception as e:
                st.error(f"Could not prepare CSV download: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("Run: `streamlit run streamlit_dashboard.py`")
