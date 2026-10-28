import json
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="What makes a photo popular?", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,300..800&display=swap');
html, body, p, label, input, button, h1, h2, h3, .stMarkdown { font-family: 'Archivo', sans-serif; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding: 3rem 1.5rem 2rem 1.5rem; max-width: 100%; }

/* top bar */
.st-key-topbar { background: #EAE3D8; border: 1px solid #DDD5C8; border-radius: 8px; padding: 14px 20px 16px 20px; }
.st-key-topbar [data-testid="stWidgetLabel"] p { font-size: 13px; color: #66615C; font-weight: 500; }
.tb-name { font-size: 26px; font-weight: 750; font-stretch: 122%; color: #292929; line-height: 1; margin: 0 0 6px 0; }
.tb-desc { font-size: 13px; line-height: 1.45; color: #454140; margin: 0; }
.tb-count .num { font-size: 26px; font-weight: 700; font-stretch: 115%; color: #292929; line-height: 1; }
.tb-count .of { font-size: 12px; color: #66615C; margin: 4px 0 8px 0; }
.tb-bar { height: 6px; background: #DDD5C8; border-radius: 3px; overflow: hidden; }
.tb-bar span { display: block; height: 100%; background: #C96A2F; border-radius: 3px; }
.tb-warn { font-size: 12px; color: #A6412B; margin-top: 6px; }
.foot { font-size: 12px; line-height: 1.5; color: #99938B; margin-top: 10px; }
.foot a { color: #66615C; }
div[role="dialog"]:has(.wide-dialog) { width: min(1200px, 94vw); max-width: 94vw; }
</style>
""",
    unsafe_allow_html=True,
)


# what the photo shows, guessed from words in its AI description
# the order matters: a photo of a man on a mountain counts as "People"
SUBJECTS = {
    "People": ["man", "woman", "person", "people", "girl", "boy", "men", "women", "child", "couple"],
    "Night sky": ["star", "milky way", "galaxy", "night sky", "aurora"],
    "Animals": ["dog", "cat", "bird", "animal", "horse", "deer", "fox", "puppy", "kitten"],
    "Flowers & plants": ["flower", "plant", "leaf", "leaves", "succulent", "rose"],
    "City & buildings": ["building", "city", "street", "road", "bridge", "architecture", "tower"],
    "Sea & beach": ["beach", "sea", "ocean", "wave", "shore", "seashore"],
    "Mountains": ["mountain"],
    "Forest & trees": ["tree", "forest", "woods"],
    "Sky & sunset": ["sky", "sunset", "sunrise", "cloud", "moon"],
}


@st.cache_data
def load_data():
    df = pd.read_csv("data/clean_data.csv")
    df["download_pct"] = df["download_rate"] * 100

    desc = df["ai_description"].fillna("").str.lower()
    conditions = [desc.str.contains(r"\b(?:" + "|".join(words) + r")s?\b") for words in SUBJECTS.values()]
    df["subject"] = np.select(conditions, list(SUBJECTS.keys()), default="Other")
    return df


@st.cache_resource
def train_model():
    data = load_data()
    data = data.assign(high_rate=(data["download_rate"] > data["download_rate"].median()).astype(int))

    X = pd.get_dummies(data[["orientation", "camera_type", "year", "megapixels"]], dtype=int)
    y = data["high_rate"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, model.predict(X_test)) * 100
    train_acc = accuracy_score(y_train, model.predict(X_train)) * 100
    return model, list(X.columns), test_acc, train_acc


def shutter_text(seconds):
    if seconds < 1:
        return f"1/{round(1 / seconds)}"
    return f"{seconds:g}s"


def top_values(series, label_fn, n=8):
    # the n most common values, sorted by the value itself (like a camera dial)
    counts = series.dropna().value_counts().head(n)
    counts = counts.sort_index()
    return [{"label": label_fn(v), "count": int(c)} for v, c in counts.items()]


def story_data(filtered, total):
    model, cols, test_acc, train_acc = train_model()
    if len(filtered) == 0:
        return {"n": 0}

    # most downloaded photos, used for the photo wall and the mosaic
    top = filtered.sort_values("downloads", ascending=False)
    photos = []
    for _, row in top.head(150).iterrows():
        camera = row["camera_model"] if pd.notna(row["camera_model"]) else "Camera unknown"
        photos.append({
            "img": row["photo_image_url"] + "?w=320&q=60",
            "url": row["photo_url"],
            "d": int(row["downloads"]),
            "desc": str(row["ai_description"]).capitalize(),
            "camera": camera,
        })

    # share of downloads taken by the top photos
    downloads = top["downloads"]
    total_dl = downloads.sum()
    n_top1 = max(1, round(len(downloads) * 0.01))
    n_top10 = max(1, round(len(downloads) * 0.10))
    top1_sum = downloads.head(n_top1).sum()
    top10_sum = downloads.head(n_top10).sum()
    mosaic = photos[: min(150, n_top1)]
    mosaic_sum = sum(p["d"] for p in mosaic)

    device = filtered.groupby("camera_type").agg(rate=("download_pct", "median"), count=("photo_id", "count"))
    device = device.sort_values("count", ascending=False).reset_index()
    orientation = filtered.groupby("orientation").agg(rate=("download_pct", "median"), count=("photo_id", "count"))
    orientation = orientation.reindex([o for o in ["landscape", "square", "portrait"] if o in orientation.index])
    years = filtered.groupby("year").agg(median=("downloads", "median"), count=("photo_id", "count")).reset_index()
    focal = filtered.groupby("camera_type")["focal_length"].median()

    # camera brands (photos with camera info only), with the device type each brand is mostly used as
    known = filtered[filtered["brand"] != "unknown"]
    brands = known.groupby("brand").agg(count=("photo_id", "count"),
                                        type=("camera_type", lambda s: s.mode().iloc[0]))
    brands = brands.sort_values("count", ascending=False).head(10)

    # download rate by subject, with the most downloaded photo of each as an example
    subject_rows = []
    for name, group in filtered[filtered["subject"] != "Other"].groupby("subject"):
        if len(group) < 20:
            continue
        best = group.sort_values("downloads", ascending=False).iloc[0]
        subject_rows.append({"name": name, "rate": float(group["download_pct"].median()), "count": len(group),
                             "img": best["photo_image_url"] + "?w=160&h=160&fit=crop&q=60", "url": best["photo_url"]})
    subject_rows = sorted(subject_rows, key=lambda r: r["rate"], reverse=True)

    tree = model.tree_
    names = {"year": "Upload year", "orientation_portrait": "Portrait or not", "megapixels": "Megapixels",
             "orientation_landscape": "Landscape or not", "orientation_square": "Square or not"}
    importance = pd.Series(model.feature_importances_, index=cols)
    used = importance[importance > 0].sort_values(ascending=False)
    importance_rows = [{"name": names.get(k, k), "value": float(v)} for k, v in used.items()]
    importance_rows.append({"name": "Device type", "value": 0.0})

    return {
        "n": len(filtered),
        "total": total,
        "median_views": int(filtered["views"].median()),
        "median_downloads": int(filtered["downloads"].median()),
        "median_rate": float(filtered["download_pct"].median()),
        "photos": photos[:40],
        "mosaic": mosaic,
        "rest_top1": int(top1_sum - mosaic_sum),
        "top1sum": int(top1_sum),
        "n_top1": int(n_top1),
        "next9": int(top10_sum - top1_sum),
        "other90": int(total_dl - top10_sum),
        "top1": float(top1_sum / total_dl * 100),
        "top10": float(top10_sum / total_dl * 100),
        "device": [{"name": r["camera_type"], "rate": float(r["rate"]), "count": int(r["count"])} for _, r in device.iterrows()],
        "orientation": [{"name": o, "rate": float(r["rate"]), "count": int(r["count"])} for o, r in orientation.iterrows()],
        "years": [{"year": int(r["year"]), "median": float(r["median"]), "count": int(r["count"])} for _, r in years.iterrows()],
        "settings": {
            "iso": top_values(filtered["iso"], lambda v: f"{v:g}"),
            "aperture": top_values(filtered["aperture"], lambda v: f"f/{v:g}"),
            "shutter": top_values(filtered["shutter"], shutter_text),
        },
        "focal": {k: round(float(v), 1) for k, v in focal.dropna().items()},
        "brands": [{"name": b, "count": int(r["count"]), "type": r["type"]} for b, r in brands.iterrows()],
        "brands_known": int(len(known)),
        "subjects": subject_rows,
        "overall_rate": float(load_data()["download_pct"].median()),
        "model": {
            "test_acc": test_acc,
            "train_acc": train_acc,
            "importance": importance_rows,
            "tree": {
                "left": tree.children_left.tolist(),
                "right": tree.children_right.tolist(),
                "feature": [cols[f] if f >= 0 else None for f in tree.feature],
                "threshold": tree.threshold.tolist(),
                "prob": (tree.value[:, 0, 1] / tree.value[:, 0].sum(axis=1)).tolist(),
                "samples": tree.n_node_samples.tolist(),
            },
        },
    }


df = load_data()
show_cols = ["photo_id", "year", "brand", "camera_model", "camera_type", "orientation", "subject",
             "iso", "aperture", "focal_length", "views", "downloads", "download_pct"]
stats_cols = ["views", "downloads", "download_pct", "iso", "aperture", "focal_length", "megapixels"]


@st.dialog("The data behind the story", width="large")
def show_data(data):
    st.markdown('<div class="wide-dialog"></div>', unsafe_allow_html=True)
    st.caption(f"The first 100 of the {len(data):,} photos in your selection.")
    st.dataframe(data[show_cols].head(100), width="stretch", height=480)


@st.dialog("Summary statistics", width="large")
def show_stats(data):
    st.markdown('<div class="wide-dialog"></div>', unsafe_allow_html=True)
    st.caption(f"Count, mean, spread and percentiles for the {len(data):,} photos in your selection.")
    st.dataframe(data[stats_cols].describe().round(2), width="stretch")


# ---------------- Top bar: filters + data buttons ----------------
with st.container(key="topbar"):
    c_name, c_device, c_year, c_orient, c_count, c_buttons = st.columns(
        [2.0, 2.6, 2.1, 1.3, 1.4, 1.5], gap="medium", vertical_alignment="center")

    c_name.markdown(
        """<div class="tb-name">Unsplash Lite</div>
<div class="tb-desc">25,000 free photos, the camera settings behind each one, and how many people viewed and downloaded them.</div>""",
        unsafe_allow_html=True,
    )
    devices = c_device.pills(
        "Taken with",
        options=["camera", "phone", "drone", "unknown"],
        selection_mode="multi",
        default=["camera", "phone", "drone", "unknown"],
    )
    years = c_year.slider(
        "Upload year",
        min_value=int(df["year"].min()),
        max_value=int(df["year"].max()),
        value=(int(df["year"].min()), int(df["year"].max())),
    )
    orientation = c_orient.selectbox("Orientation", ["All", "landscape", "portrait", "square"])

    filtered = df[df["camera_type"].isin(devices)]
    filtered = filtered[(filtered["year"] >= years[0]) & (filtered["year"] <= years[1])]
    if orientation != "All":
        filtered = filtered[filtered["orientation"] == orientation]

    share = len(filtered) / len(df) * 100
    warning = '<div class="tb-warn">Pick at least one device.</div>' if not devices else ""
    c_count.markdown(
        f"""<div class="tb-count">
  <div class="num">{len(filtered):,}</div>
  <div class="of">of {len(df):,} photos in the story</div>
  <div class="tb-bar"><span style="width:{share:.1f}%"></span></div>
  {warning}
</div>""",
        unsafe_allow_html=True,
    )
    if c_buttons.button("Data table", icon=":material/table_chart:", width="stretch"):
        show_data(filtered)
    if c_buttons.button("Summary stats", icon=":material/functions:", width="stretch"):
        show_stats(filtered)

# ---------------- The story (one frame per insight) ----------------
with open("story.html", encoding="utf-8") as f:
    story = f.read()
story = story.replace("/*DATA*/null", json.dumps(story_data(filtered, len(df))))
st.iframe(story, height=780)

st.markdown(
    """<div class="foot">The prediction model always learns from all 25,000 photos.
Source: <a href="https://github.com/unsplash/datasets" target="_blank">github.com/unsplash/datasets</a></div>""",
    unsafe_allow_html=True,
)
