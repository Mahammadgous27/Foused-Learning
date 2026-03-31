# app_tracker_simple.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION & INIT ---
st.set_page_config(layout="wide", page_title="Focused Learning")

# Ensure 'df' is defined globally to avoid NameError on initial run
df = pd.DataFrame() 

# --- FUNCTIONS ---

@st.cache_data
def load_data(file):
    """Load data and simulate simple classification if 'Label' is missing."""
    if file:
        df = pd.read_csv(file)
        if 'Label' not in df.columns:
            # Simple rule: 'Social', 'Entertainment', 'Games', 'Communication' are Distracting
            distracting_cats = ['Social', 'Entertainment', 'Games', 'Communication']
            df['Label'] = np.where(df['Category'].isin(distracting_cats), 'Distracting', 'Working')
        
        # Clean data types
        for col in ['Usage Time (min)', 'Frequency']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df.dropna(subset=['Usage Time (min)', 'Frequency']).copy()
    return pd.DataFrame()

def display_bar_chart(df):
    """Display Top Apps by Usage Time chart."""
    df_sorted = df.sort_values(by='Usage Time (min)', ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = df_sorted['Label'].apply(lambda x: 'tab:red' if x == 'Distracting' else 'tab:blue')
    
    ax.bar(df_sorted['App Name'], df_sorted['Usage Time (min)'], color=colors)
    ax.set_title('Top Apps by Usage Time'); ax.tick_params(axis='x', rotation=45, labelsize=9)
    plt.tight_layout()
    st.pyplot(fig)

# --- APP LAYOUT ---
st.title("Focused Learning")
st.markdown("ML-powered app classification with real-time usage alerts")

# --- Report Button (HTML-based for simplicity) ---
st.markdown("""
    <div style="text-align: right; margin-top: -60px;">
        <button style="background-color: #0078d4; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer;">
            Report Results
        </button>
    </div>
    """, unsafe_allow_html=True)
st.markdown("---")


# 1. Upload & Alert Threshold
col_upload, col_threshold = st.columns([2, 1])

with col_upload:
    st.subheader("Upload App Usage Data")
    uploaded_file = st.file_uploader("Choose File", type=['csv'], help="App Name, Category, Usage Time (min), Frequency, Label")

# Load data (using a local file for default view if no file is uploaded)
if uploaded_file:
    df = load_data(uploaded_file)
else:
    # Use default 'Book1.csv' for demonstration if available
    try:
        df = load_data('Book1.csv')
    except:
        st.info("Please upload a CSV file or ensure 'Book1.csv' is in the directory.")

# Alert Threshold Logic
with col_threshold:
    st.subheader("Alert Threshold")
    # Use session state for persistent max_minutes
    if 'max_minutes' not in st.session_state:
        st.session_state.max_minutes = 120 
    
    st.session_state.max_minutes = st.number_input(
        "Maximum Minutes:",
        min_value=1,
        value=st.session_state.max_minutes,
        key="minutes_input_simple",
        label_visibility="collapsed"
    )
    if st.button("Apply"):
        st.success(f"Threshold set to {st.session_state.max_minutes} minutes.")

st.markdown("---")

# 2. Main Dashboard Content
if not df.empty and 'Label' in df.columns:
    max_minutes = st.session_state.max_minutes
    df_distracting = df[df['Label'] == 'Distracting']
    alerts = df_distracting[df_distracting['Usage Time (min)'] > max_minutes].sort_values(by='Usage Time (min)', ascending=False)
    alert_count = len(alerts)

    # --- Usage Alerts Section ---
    with st.expander(f"Usage Alerts ({alert_count}) - Threshold: {max_minutes} min", expanded=True):
        if alert_count > 0:
            for _, row in alerts.iterrows():
                st.markdown(
                    f"**<span style='color:#ff7f0e;'>{row['App Name']}</span>** ({row['Category']}) used for **{int(row['Usage Time (min)'])} min** ({int(row['Frequency'])} opens).", 
                    unsafe_allow_html=True
                )
        else:
            st.info("No distracting apps have exceeded the usage threshold!")

    st.markdown("---")

    # --- Metrics ---
    total_usage = int(df['Usage Time (min)'].sum())
    distracting_count = len(df_distracting)
    working_count = len(df[df['Label'] == 'Working'])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Usage", f"{total_usage} min", "all apps")
    col2.metric("Distracting Apps", distracting_count, "classified")
    col3.metric("Working Apps", working_count, "classified")
    col4.metric("Alerts Triggered", alert_count, "Threshold")

    st.markdown("---")

    # --- Chart and ML Performance ---
    col_chart, col_ml = st.columns([3, 1])

    with col_chart:
        display_bar_chart(df)

    with col_ml:
        st.markdown("##### ML Model Performance (Simulated)")
        st.markdown("Accuracy: **88.0%**")
        st.progress(0.88)
        st.markdown("Precision: **80.0%**")
        st.progress(0.80)
        # Removed Recall and F1 for brevity