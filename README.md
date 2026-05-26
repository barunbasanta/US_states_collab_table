# Collaboration Table Generator

Streamlit Cloud fixed version.

This app bundles `us_state_shapes.json`, so the state icons render as real
map shapes without requiring `bokeh.sampledata`, `basemap`, `shapely`, or `pyshp`
on Streamlit Cloud.

Input columns:
`State`, `P`, `G`, `T`, optional `Total`.

Run:
```bash
pip install -r requirements.txt
streamlit run app.py
```
