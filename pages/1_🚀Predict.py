import streamlit as st
import pandas as pd
import joblib

# ─── 1) MUST be the very first Streamlit call ───
st.set_page_config(
    page_title="Mining Site Prediction",
    page_icon="🛰️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load the trained model
try:
    model = joblib.load("FINAL_mining_model_v2.pkl")
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'FINAL_mining_model_v2.pkl' is in this folder.")
    st.stop()

# ─── 2) Monkey-patch scikit-learn 1.6.1 internal check ───
if not hasattr(model, "_preprocessor"):
    model._preprocessor = None

# ─── Figure out which columns the model actually expects ───
# If you trained with a Pipeline, you might have `model.named_steps['pre'].feature_names_in_`
# or your estimator might have `feature_names_in_`; otherwise fall back to n_features_in_.
if hasattr(model, "feature_names_in_"):
    FEATURE_ORDER = list(model.feature_names_in_)
elif hasattr(model, "n_features_in_"):
    # assume the first n_features_in_ of the sidebar dict order
    sidebar_keys = [
        'Distance from Earth (M km)',
        'Iron (%)',
        'Nickel (%)',
        'Water Ice (%)',
        'Other Minerals (%)',
        'Estimated Value (B USD)',
        'Sustainability Index',
        'Efficiency Index'
    ]
    FEATURE_ORDER = sidebar_keys[: model.n_features_in_]
else:
    st.error("Cannot determine which features the model expects.")
    st.stop()

# Custom CSS for styling
st.markdown("""
    <style>
        body { background-color: #F5F5F5; color: #333; }
        .sidebar .sidebar-content { background-color: #EEEEEE; padding: 10px; border-radius: 10px; }
        .stSlider { color: #83e4f7; }
        .stButton>button { background-color: #042380; color: white; border-radius: 5px; font-weight: bold; }
        .stButton>button:hover { background-color: #000000; }
    </style>
""", unsafe_allow_html=True)

def show_decide_page():
    st.title("Mining Site Prediction 🛰️")
    st.write("""
        **Discover the potential of your mining site!**  
        Enter the details in the sidebar to find out if your site is worth mining.
    """)
    
    st.sidebar.header("🔍 Input Features")
    # 8 sliders total
    data = {
        'Distance from Earth (M km)': st.sidebar.slider("Distance from Earth (M km)", 1.0, 1000.0, 100.0),
        'Iron (%)':                  st.sidebar.slider("Iron (%)",                0.0, 100.0,  50.0),
        'Nickel (%)':                st.sidebar.slider("Nickel (%)",              0.0, 100.0,  50.0),
        'Water Ice (%)':             st.sidebar.slider("Water Ice (%)",           0.0, 100.0,  50.0),
        'Other Minerals (%)':        st.sidebar.slider("Other Minerals (%)",      0.0, 100.0,  50.0),
        'Estimated Value (B USD)':   st.sidebar.slider("Estimated Value (B USD)", 0.0, 500.0, 100.0),
        'Sustainability Index':      st.sidebar.slider("Sustainability Index",    0.0, 100.0,  50.0),
        'Efficiency Index':          st.sidebar.slider("Efficiency Index",        0.0, 100.0,  50.0),
    }

    if st.button("🔮 Predict"):
        df = pd.DataFrame(data, index=[0])
        st.subheader("🔍 User Input Features")
        st.table(df)

        # ─── 3) Subset to exactly what model expects ───
        try:
            X = df[FEATURE_ORDER]
        except KeyError:
            st.error(f"Feature mismatch! Model expects: {FEATURE_ORDER}")
            return

        pred = model.predict(X)
        st.subheader("📊 Prediction Result")
        if pred[0] == 1:
            st.success("✅ **This is a Potential Mining Site!**")
        else:
            st.error("❌ **This is Not a Potential Mining Site.**")

    st.markdown("""
        <div style="margin-top:20px; padding:15px; border:2px solid #ccc; border-radius:10px;">
            <strong>Note:</strong> The prediction is based on the model’s analysis of key features. 
            Use this as a guide; further domain analysis may be required.
        </div>
    """, unsafe_allow_html=True)

show_decide_page()
