# What makes a photo popular?

An exploratory data analysis of 25,000 free photos from the [Unsplash Lite dataset](https://github.com/unsplash/datasets), told from a photographer's point of view. It looks at the camera, settings, framing and upload date behind each photo, and how many people viewed and downloaded it.

**Main finding:** phone and camera photos get downloaded at almost the same rate. Framing (landscape vs portrait) and upload date matter more than the camera.

## Project files

| File | What it is |
|---|---|
| `notebook.ipynb` | Data loading, cleaning, EDA charts, insights and the decision tree model |
| `data/clean_data.csv` | The cleaned dataset used by the dashboard |
| `app.py` | The Streamlit dashboard |
| `story.html` | The dashboard's interactive story, loaded by `app.py` |
| `requirements.txt` | Python packages for the dashboard |
| `.streamlit/config.toml` | The dashboard's color theme |

## Run the dashboard locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Run the notebook

The notebook needs the raw data. Download the **Lite** version from [github.com/unsplash/datasets](https://github.com/unsplash/datasets), unzip it into `data/raw/`, then run the cells in order. It also uses `seaborn`, `matplotlib` and `scikit-learn`.

## Tools

Python, pandas, matplotlib, seaborn, scikit-learn, Streamlit, D3.js
