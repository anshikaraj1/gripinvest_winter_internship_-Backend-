import streamlit as st

st.set_page_config(page_title="Theme Toggle", layout="centered")

# Use query params to retain theme state
query_params = st.query_params
theme = query_params.get("theme", "Dark")

# Sidebar radio
selected = st.sidebar.radio("Choose Theme", ["Dark", "Light"], index=0 if theme == "Dark" else 1)

# Update param & rerun if changed
if selected != theme:
    st.query_params["theme"] = selected
    st.rerun()

# Custom theme application
if selected == "Dark":
    st.markdown("""
        <style>
        body {
            background-color: #0d1117 !important;
            color: #c9d1d9 !important;
        }
        [data-testid="stAppViewContainer"] {
            background-color: #0d1117 !important;
            color: #c9d1d9 !important;
        }
        .stTextInput > div > input,
        .stNumberInput input {
            background-color: #161b22 !important;
            color: #c9d1d9 !important;
        }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        body {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        [data-testid="stAppViewContainer"] {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        .stTextInput > div > input,
        .stNumberInput input {
            background-color: #f0f0f0 !important;
            color: #000000 !important;
        }
        </style>
    """, unsafe_allow_html=True)

# Display theme state
st.title("🌓 Dynamic Theme Toggle")
st.write(f"Current Theme: **{selected}**")

