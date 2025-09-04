import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import datetime
import time
import streamlit as st
from PIL import Image
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
from datetime import datetime
from gtts import gTTS
from deep_translator import GoogleTranslator
import base64
import tempfile
import os
import streamlit.components.v1 as components
import streamlit as st
import streamlit.components.v1 as components

# ✅ Voice + Welcome shown only once per session
if "intro_shown" not in st.session_state:
    st.session_state.intro_shown = False

if not st.session_state.intro_shown:
    # CSS and HTML styling
    st.markdown("""
        <style>
        .intro-section {
            background-image: url('https://cdn.wallpapersafari.com/88/25/flhNYm.jpg');
            background-size: cover;
            background-position: left;
            background-repeat: no-repeat;
            min-height: 65vh;
            width: 110%;
            height: 65%;
            display: flex;
            justify-content: center;
            align-items: center;
            position: relative;
            z-index: 1;
            text align: center;
            flex-direction: column;

        }

        .intro-overlay {
            background-color: rgba(0, 0, 0, 0.6);
            border-radius: 16px;
            text-align: center;
            color: white;
            max-width: 80%;
            padding: 2rem;
        }

        .intro-title {
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }

        .intro-sub {
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }

        .start-button {
            background-color: #ff4b4b;
            color: white;
            font-size: 1.1rem;
            padding: 0.7rem 1.6rem;
            border-radius: 10px;
            font-weight: bold;
            border: none;
            cursor: pointer;
        }
        </style>

        <div class="intro-section">
            <div class="intro-overlay">
                <div class="intro-title">Welcome to Predictive Maintenance AI Dashboard</div>
                <div class="intro-sub">This platform harnesses the power of AI and Machine Learning to predict machine failures, reduce downtime, and revolutionize industrial reliability!</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ✅ Voice using reliable speechSynthesis (as in tabs)
    components.html(
        """
        <script>
            const msg = new SpeechSynthesisUtterance("Welcome to Predictive Maintenance AI Dashboard. This platform harnesses the power of AI and Machine Learning to predict machine failures, reduce downtime, and revolutionize industrial reliability!");
            msg.volume = 1;
            msg.rate = 0.95;
            msg.pitch = 1.1;
            window.speechSynthesis.speak(msg);
        </script>
        """,
        height=0,
    )

    # Show "Start" button separately using Streamlit
    st.markdown("<br><br><center>", unsafe_allow_html=True)
    if st.button("🌠 Start Experience"):
        st.session_state.intro_shown = True
        st.rerun()
    st.markdown("</center>", unsafe_allow_html=True)

    st.stop()  # Stop execution of rest of the app until "Start" is pressed



def speak(text, lang="English"):
    lang_codes = {
        "English": "en",
        "Hindi": "hi",
        "French": "fr",
        "Spanish": "es",
        "German": "de"
    }

    target_lang_code = lang_codes.get(lang, "en")

    # Translate if not English
    if lang != "English":
        text = GoogleTranslator(source='auto', target=target_lang_code).translate(text)

    # Generate speech using gTTS
    tts = gTTS(text=text, lang=target_lang_code)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
        tts.save(tmpfile.name)
        audio_path = tmpfile.name

    # Read audio bytes and encode
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
        b64_audio = base64.b64encode(audio_bytes).decode()

    # Remove temporary file
    os.remove(audio_path)

    # Inject HTML and JS to play audio automatically
    audio_html = f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3">
        </audio>
        <script>
            var audio = document.querySelector("audio");
            audio.play();
        </script>
    """
    components.html(audio_html, height=0)


# ----------------------------
# FUNCTION: Analyze Feedback
# ----------------------------
def analyze_feedback(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.2:
        return "Positive"
    elif polarity < -0.2:
        return "Negative"
    else:
        return "Neutral"

# ----------------------------
# FUNCTION: Save Feedback
# ----------------------------
def save_feedback(text, sentiment):
    feedback_file = "user_feedback.csv"
    data = {
        "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "feedback": [text],
        "sentiment": [sentiment]
    }
    df = pd.DataFrame(data)
    if os.path.exists(feedback_file):
        df.to_csv(feedback_file, mode='a', header=False, index=False)
    else:
        df.to_csv(feedback_file, index=False)


# ✅ Page configuration
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .tab-content {
        animation: slideIn 0.5s ease-in-out;
    }

    @keyframes slideIn {
        0% { 
            opacity: 0; 
            transform: translateX(-20px); 
        }
        100% { 
            opacity: 1; 
            transform: translateX(0); 
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ✅ Persist theme using query params
query_params = st.query_params
theme = query_params.get("theme", "Dark")

# ✅ Sidebar Theme Toggle
selected = st.sidebar.radio("Choose Theme:", ["Dark", "Light"], index=0 if theme == "Dark" else 1)

# ✅ Trigger theme update and rerun
if selected != theme:
    st.query_params["theme"] = selected
    st.rerun()

# ✅ Apply the selected theme with safe styling for all elements
if selected == "Dark":
    st.markdown("""
        <style>
        body, [data-testid="stAppViewContainer"] {
            background-image: url("https://cdn.pixabay.com/photo/2016/02/14/06/54/stars-1199060_1280.png");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
            color: #c9d1d9 !important;
        }

        body::before {
            content: "";
            position: fixed;
            top: 0; left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(13, 17, 23, 0.6); /* Optional dark overlay */
            z-index: -1;
        }
        [data-testid="stAppViewContainer"] {
            background-color: #0d1117 !important;
            color: #c9d1d9 !important;
        }
        .stTextInput > div > input,
        .stNumberInput input {
            background-color: #0d1117 !important;
            color: #c9d1d9 !important;
        }
        .stTextInput > div > input,
        .stNumberInput input {
            background-color: #161b22 !important;
            color: #c9d1d9 !important;
        }
        .stTabs [role="tab"] {
            color: #ffffff !important;
        }
        .stAlert {
            color: #c9d1d9 !important;
        }
        </style>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
        <style>
        body, [data-testid="stAppViewContainer"] {
            background-image: url("https://images.pexels.com/photos/4790056/pexels-photo-4790056.jpeg");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
            color: #000000 !important;
        }

        /* Optional light overlay for readability */
        body::before {
            content: "";
            position: fixed;
            top: 0; left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(255, 255, 255, 0.6);
            z-index: -1;
        }
        section[data-testid="stSidebar"] label {
            color: white !important;
            font-weight: bold !important;
        }
        [data-testid="stAppViewContainer"] {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        .stTextInput > div > input,
        .stNumberInput input {
            background-color: #a3a990 !important;
            color: #000000 !important;
        }
        .stTabs [role="tab"] {
            color: #000000 !important;
        }
        .stAlert {
            border: 1px solid #cccccc !important;
            background-color: #000000 !important;
            color: #000000 !important;
        }
        .stButton>button, .stDownloadButton>button, .stForm>form>div>button {
            background-color: #000000 !important;
            color: white !important;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            padding: 0.5rem 1rem !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
        }
        .stDownloadButton button:hover {
        background-color: #3e5f3e  !important;
        color: white !important;
        }
        label, .stNumberInput label {
            color: #000000 !important;
            font-weight: 600 !important;
        }
        .stForm button {
            background-color: #000000 !important;
            color: white !important;
            font-weight: bold !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.5rem 1rem !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
        }
        .stForm button:hover {
        background-color: #3e5f3e !important;
        color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)

# Display theme state
st.write(f"Current Theme: **{selected}**")
# Load data
@st.cache_data
def load_data(option, uploaded_file=None):
    if option == "Main Dataset":
        df = pd.read_csv("data.csv")
        model = joblib.load("rul_model.pkl")
    elif option == "BDL Dataset":
        df = pd.read_csv("bdl_torpedo_data.csv")
        model = joblib.load("bdl_rul_model.pkl")
    elif option == "NASA Dataset":
        df = pd.read_csv("nasa_cleaned.csv")
        model = joblib.load("nasa_model.pkl")
    elif option == "Upload Your Own" and uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        model = None
    else:
        df, model = None, None
    return df, model

def preprocess(df):
    features = df.drop(columns=["RUL"])
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    return features_scaled, df["RUL"], scaler, features.columns

# UI and logic
st.title("Predictive Maintenance Dashboard")

option = st.sidebar.selectbox("Choose Dataset:", ["Main Dataset", "BDL Dataset", "NASA Dataset", "Upload Your Own"])

# Move the file uploader OUTSIDE the cached function
uploaded_file = None
if option == "Upload Your Own":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

with st.spinner("⏳ Loading dataset and model..."):
    time.sleep(1.2)
    df, model = load_data(option, uploaded_file)

# Language selection
language = st.sidebar.selectbox(
    "Select Language for Voice Explanation",
    options=["English", "Hindi", "French", "Spanish", "German"]
)

if df is not None:
    tabs = st.tabs(["Data Preview", "Prediction", "Heatmap", "Visual Analysis", "Model Comparison", "Live Inference", "Model Info", "Maintenance Log Analyzer"])

    with tabs[0]:  # Data Preview
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)

        st.subheader("Sample Data")
        st.write(df.head())

        st.markdown('</div>', unsafe_allow_html=True)


        # Optional: Add voice explanation button
        st.button("🔊 About Tab", on_click=lambda: speak(
            "This is the Data Preview tab. It shows you a quick glimpse of the uploaded or selected dataset. This helps you verify the data format and understand the structure before analysis begins.", lang=language
        ))

    if "RUL" in df.columns:
        X_scaled, y_true, scaler, feature_names = preprocess(df)

        if model:
            with st.spinner("⏳ Running prediction..."):
                time.sleep(1.5)
                y_pred = model.predict(X_scaled)

            with tabs[1]:
                st.subheader("Predicted vs Actual RUL")
   
                n_points = 50
                actual = y_true[:n_points].values
                predicted = y_pred[:n_points]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(range(n_points)), y=[None]*n_points, mode='lines', name='Actual', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=list(range(n_points)), y=[None]*n_points, mode='lines', name='Predicted', line=dict(color='orange')))

                frames = []
                for i in range(1, n_points + 1):
                    frames.append(go.Frame(
                        data=[
                            go.Scatter(x=list(range(i)), y=actual[:i], mode='lines', line=dict(color='blue')),
                            go.Scatter(x=list(range(i)), y=predicted[:i], mode='lines', line=dict(color='orange')),
                        ],
                        name=str(i)
                    ))

                fig.frames = frames
                fig.update_layout(
                    title="RUL Prediction Over Time",
                    xaxis_title="Sample Index",
                    yaxis_title="RUL",
                    updatemenus=[dict(
                        type="buttons",
                        showactive=False,
                        buttons=[
                            dict(label="Play", method="animate", args=[None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}]),
                            dict(label="Pause", method="animate", args=[None, {"frame": {"duration": 0}, "mode": "immediate"}])
                        ],
                        x=1.05, y=1.2
                    )]
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### 🔍 Scaled Input Features")
                st.dataframe(pd.DataFrame(X_scaled[:5], columns=feature_names))
                st.success("✅ RUL prediction completed!")
                st.info(f"⚙️ Model Confidence (simulated): {round(np.random.uniform(88, 96), 2)}%")
                st.button("🔊 About Tab", key="prediction_audio" , on_click=lambda: speak(
                    "This is the RUL Prediction tab. RUL means Remaining Useful Life. This tab shows how long a machine component is likely to run before failure using predictive models.", lang=language
                ))

            with tabs[2]:
                st.subheader("Feature Correlation Heatmap")
                

                numeric_cols = df.select_dtypes(include='number')
                if not numeric_cols.empty:
                    with st.spinner("📊 Creating heatmap..."):
                        time.sleep(1)
                        corr_matrix = numeric_cols.corr().round(2).reset_index().melt(id_vars='index')
                        corr_matrix.columns = ['Feature 1', 'Feature 2', 'Correlation']

                        fig = px.density_heatmap(
                            corr_matrix,
                            x="Feature 1",
                            y="Feature 2",
                            z="Correlation",
                            color_continuous_scale="Viridis",
                            text_auto=True,
                            title="Interactive Correlation Heatmap"
                        )
                        fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No numeric columns available for heatmap.")
                st.button("🔊 About tab", key="heatmap_audio", on_click=lambda: speak(
                    "This is the heatmap tab. It shows relationships between features using a color-coded matrix. Strong correlations help in identifying useful predictors.", lang=language
                ))

            with tabs[3]:
                st.subheader("Scatter Matrix (Visual Analysis)")

                numeric_cols = df.select_dtypes(include='number')
                if not numeric_cols.empty:
                    with st.spinner("🔍 Generating scatter matrix..."):
                        time.sleep(1)
                        fig = px.scatter_matrix(numeric_cols,
                                                dimensions=numeric_cols.columns[:5],
                                                title="Scatter Matrix (Top 5 Features)",
                                                height=700,
                                                color_discrete_sequence=["#58a6ff"])
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Not enough numeric data for scatter matrix.")
                st.button("🔊 About Tab", key="visual_audio" , on_click=lambda: speak(
                    "This tab includes visual plots like scatter matrix and feature trends to explore data distributions and potential anomalies.", lang=language
                ))
                

            with tabs[4]:
                st.subheader("Model Comparison")
                
                st.write("Comparison between Random Forest and XGBoost models")
                st.write("🧪 Dashboard feature count (X_scaled):", X_scaled.shape[1])

                with st.spinner("🔄 Loading and comparing models..."):
                    time.sleep(1.5)
                    try:
                        if option == "Main Dataset":
                            rf_model = joblib.load("rul_model.pkl")
                            xgb_model = joblib.load("xgb_model.pkl")
                        elif option == "BDL Dataset":
                            rf_model = joblib.load("bdl_rf_model.pkl")
                            xgb_model = joblib.load("bdl_xgb_model.pkl")
                        elif option == "NASA Dataset":
                            rf_model = joblib.load("nasa_rf_model.pkl")
                            xgb_model = joblib.load("nasa_xgb_model.pkl")
                        else:
                            st.warning("Model comparison is not available for uploaded datasets.")
                            rf_model, xgb_model = None, None

                        if rf_model and xgb_model:
                            rf_pred = rf_model.predict(X_scaled)
                            xgb_pred = xgb_model.predict(X_scaled)

                            fig_cmp = go.Figure()
                            frames = []
                            n = 50
                            for i in range(1, n + 1):
                                frames.append(go.Frame(
                                    data=[
                                        go.Scatter(x=list(range(i)), y=y_true[:i], mode="lines", name="Actual", line=dict(color="black")),
                                        go.Scatter(x=list(range(i)), y=rf_pred[:i], mode="lines", name="Random Forest", line=dict(color="blue")),
                                        go.Scatter(x=list(range(i)), y=xgb_pred[:i], mode="lines", name="XGBoost", line=dict(color="green"))
                                    ],
                                    name=str(i)
                                ))

                            fig_cmp.add_trace(go.Scatter(x=list(range(n)), y=[None]*n, mode="lines", name="Actual", line=dict(color="black")))
                            fig_cmp.add_trace(go.Scatter(x=list(range(n)), y=[None]*n, mode="lines", name="Random Forest", line=dict(color="blue")))
                            fig_cmp.add_trace(go.Scatter(x=list(range(n)), y=[None]*n, mode="lines", name="XGBoost", line=dict(color="green")))

                            fig_cmp.frames = frames
                            fig_cmp.update_layout(
                                title="Model Comparison 📈",
                                xaxis_title="Sample Index",
                                yaxis_title="RUL",
                                updatemenus=[dict(
                                    type="buttons",
                                    showactive=False,
                                    buttons=[
                                        dict(label="Play", method="animate", args=[None, {"frame": {"duration": 100}, "fromcurrent": True}]),
                                        dict(label="Pause", method="animate", args=[None, {"frame": {"duration": 0}, "mode": "immediate"}])
                                    ],
                                    x=1.05, y=1.2
                                )]
                            )
                            st.plotly_chart(fig_cmp)

                    except Exception as e:
                        st.error("Error loading comparison models: " + str(e))

                result_df = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})
                st.download_button("Download Predictions", result_df.to_csv(index=False), "predictions.csv", "text/csv")
                st.button("🔊 About Tab", key="model_comparison_audio" , on_click=lambda: speak(
                    "This is the model comparison tab. It lets you compare performance of different machine learning models based on accuracy, error, and execution time.", lang=language
                ))

            with tabs[5]:
                st.subheader("Live Inference")

                with st.form("manual_form"):
                    st.write("Enter feature values:")
                    user_input = [st.number_input(col, value=float(df[col].mean())) for col in feature_names]
                    submitted = st.form_submit_button("Predict")

                if submitted:
                    with st.spinner("🧠 Predicting..."):
                        time.sleep(1.5)
                        user_scaled = scaler.transform([user_input])
                        pred_val = model.predict(user_scaled)[0]
                        st.success(f"Predicted RUL: {pred_val:.2f}")

                tab5 = st.container()
                with tab5:
                    st.title("🔍 Live Inference")

                    # Simulate user input
                    temp = st.slider("Temperature", 0.0, 100.0, 25.0)
                    vib = st.slider("Vibration", 0.0, 10.0, 2.5)
                    hrs = st.slider("Hours", 0.0, 1000.0, 500.0)

                    input_data = np.array([[temp, vib, hrs]])

                    if st.button("Predict"):
                        try:
                            model = joblib.load("rul_model.pkl")  # example path
                            prediction = model.predict(input_data)
                            st.success(f"Predicted RUL: {prediction[0]:.2f} hours")
                        except Exception as e:
                            st.error("⚠️ Prediction failed. Check model path and inputs.")

                    # ----------------------
                    # Feature 2: Feedback
                    # ----------------------
                    st.subheader("🔁 Feedback Assistant")

                    feedback = st.text_area("💬 Please share your feedback on this prediction (optional):")
                    submit_feedback = st.button("Submit Feedback")

                    if submit_feedback:
                        if feedback.strip() == "":
                            st.warning("⚠️ Feedback is empty.")
                        else:
                            sentiment = analyze_feedback(feedback)
                            st.success(f"✅ Thank you! Detected sentiment: **{sentiment}**")
                            save_feedback(feedback, sentiment)

                    # ----------------------
                    # Feature 3: Auto Comment Generator
                    # ----------------------
                    st.subheader("🧠 Auto-Comment Generator")

                    if 'prediction' in locals():
                        comment = ""
                        if prediction[0] > 700:
                            comment = "Machine is running optimally. No maintenance needed."
                        elif prediction[0] > 300:
                            comment = "Performance dropping. Schedule inspection soon."
                        else:
                            comment = "Urgent: Immediate maintenance required."

                        st.info(f"💡 Suggested Comment: {comment}")
                if st.button("🔊 About Tab", key="live_inference_audio"):
                    speak(
                        "This tab is for live inference. You enter machine readings in real time and the model predicts the remaining useful life instantly.", lang=language
                    )

            with tabs[6]:
                st.subheader("Model Info & Metadata")
                
                st.markdown(f"""
                - **Model Type**: Random Forest/XGBoost  
                - **Dataset Used**: {option}  
                - **Input Features**: {len(feature_names)}  
                - **Model Loaded**: {datetime.now().strftime("%Y-%m-%d %H:%M")}  
                - **Note**: All inputs are scaled before prediction.
                """)
                st.info("📌 This section gives basic metadata of your ML model.")
                st.warning("⚠️ Always retrain your model when dataset changes significantly.")
                st.success("✅ AI model is currently working with the selected dataset.")
                if st.button("🔊 About Tab", key="model_info_audio"):
                    speak(
                        "This section shows details about the machine learning model: its type, parameters, training results, and input features. It helps users understand how the model   works behind the scenes.", lang=language
                         )

            with tabs[7]:  # New tab for Maintenance Logs
                st.subheader("🛠 Maintenance Log Analyzer")

                uploaded_log = st.file_uploader("Upload a maintenance log (.txt or .csv)", type=["txt", "csv"])

                if uploaded_log:
                    text = uploaded_log.read().decode("utf-8")
                    st.text_area("📄 Log Preview", value=text, height=200)

                    if st.button("Analyse Log"):
                        import spacy
                        from collections import Counter

                        nlp = spacy.load("en_core_web_sm")
                        doc = nlp(text)

                        # Extract common noun chunks (e.g., "engine problem", "sensor failure")
                        noun_chunks = [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text) > 3]
                        top_chunks = Counter(noun_chunks).most_common(10)

                        st.markdown("### 🔧 Most Common Issues Mentioned")
                        for phrase, count in top_chunks:
                            st.write(f"- {phrase} ({count} times)")

                        # Optional: extract named entities like systems/components
                        entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "GPE"]]
                        top_entities = Counter(entities).most_common(5)

                        if top_entities:
                            st.markdown("### 🧠 Components/Entities Mentioned")
                            for ent, count in top_entities:
                                st.write(f"- {ent} ({count} times)")
                if st.button("🔊 About Tab", key="maintenance_log_audio"):
                    speak(
                        "This tool analyzes maintenance logs using NLP to highlight common issues and patterns. It turns text-based logs into actionable insights.", lang=language
                    )

        else:
            with tabs[1]:
                st.warning("No model available. Only data preview shown.")
    else:
        st.error("Dataset does not contain 'RUL'. Model training skipped.")
else:
    st.info("Please select or upload a valid dataset.")




