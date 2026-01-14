import streamlit as st
import pandas as pd
import os

# ====================================================
# PAGE CONFIG
# ====================================================
st.set_page_config(
    page_title="Nubank Fraud Detection – Mixture of Experts",
    layout="wide"
)

st.title("💳 Nubank Fraud Detection – Mixture of Experts (MoE)")
st.write(
    "This UI simulates Nubank’s real-time fraud defense platform, "
    "where multiple expert models are combined to produce an "
    "transparent fraud risk decision."
)

# ====================================================
# LOAD DATA (SAFE, RELATIVE TO app.py)
# ====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "final_nubank_fraud_output.csv")

if not os.path.exists(CSV_FILE):
    st.error("❌ final_nubank_fraud_output.csv not found in the app directory.")
    st.stop()

df = pd.read_csv(CSV_FILE)

# ====================================================
# REQUIRED COLUMNS CHECK
# ====================================================
required_cols = [
    "is_fraud",
    "final_moe_score",
    "fraud_risk_level",
    "lstm_score",
    "transformer_score",
    "autoencoder_score",
    "xgb_score",
    "ada_score"
]

missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    st.error(f"❌ Missing required columns: {missing_cols}")
    st.stop()

# ====================================================
# SIDEBAR – TRANSACTION SELECTION (SAFE)
# ====================================================
st.sidebar.header("🔍 Select Transaction")

if "transaction_id" in df.columns:
    df["transaction_id"] = (
        df["transaction_id"]
        .astype(str)
        .str.strip()
    )

    txn_id = st.sidebar.selectbox(
        "Transaction ID",
        df["transaction_id"].unique().tolist()
    )

    filtered = df[df["transaction_id"] == txn_id]

    if filtered.empty:
        st.error("❌ Selected transaction not found.")
        st.stop()

    txn = filtered.iloc[0]
else:
    txn_index = st.sidebar.selectbox(
        "Transaction Row Index",
        df.index.tolist()
    )
    txn = df.loc[txn_index]

# ====================================================
# MAIN OUTPUT – FINAL DECISION
# ====================================================
st.subheader("🚨 Final Fraud Decision")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Final Fraud Score",
    round(float(txn["final_moe_score"]), 3)
)

col2.metric(
    "Fraud Risk Level",
    str(txn["fraud_risk_level"])
)

col3.metric(
    "Actual Fraud (Ground Truth)",
    "Yes" if int(txn["is_fraud"]) == 1 else "No"
)

# ====================================================
# ✅ BUSINESS DECISION MESSAGE (NEW)
# ====================================================
risk_level = str(txn["fraud_risk_level"]).upper()

st.subheader("📢 Transaction Status")

if risk_level == "LOW":
    st.success("✅ Your transaction was successful.")
elif risk_level == "MEDIUM":
    st.warning("⚠️ Your transaction is on hold. Please contact the bank.")
elif risk_level == "HIGH":
    st.error("🚫 Your transaction is blocked. Please contact the bank immediately.")
else:
    st.info("ℹ️ Transaction status unavailable.")

# ====================================================
# EXPERT MODEL SCORES
# ====================================================
st.subheader("🧠 Expert Model Contributions")

expert_scores = {
    "LSTM (Behavioral)": txn["lstm_score"],
    "Transformer (Feature Interaction)": txn["transformer_score"],
    "Autoencoder (Anomaly)": txn["autoencoder_score"],
    "XGBoost (Tabular)": txn["xgb_score"],
    "AdaBoost (Ensemble)": txn["ada_score"]
}

expert_df = pd.DataFrame.from_dict(
    expert_scores,
    orient="index",
    columns=["Score"]
)

st.bar_chart(expert_df)

# ====================================================
# EXPLANATION SECTION
# ====================================================
st.subheader("📘 How This Decision Was Made")

st.markdown("""
- **LSTM** analyzes sequential transaction behavior  
- **Transformer** models complex feature interactions  
- **Autoencoder** detects anomalous patterns  
- **XGBoost & AdaBoost** provide strong tabular signals  
- **Gating Network (MoE)** combines all expert outputs dynamically  
""")

# ====================================================
# OPTIONAL DATASET INSIGHTS
# ====================================================
with st.expander("📊 Dataset Insights"):
    st.write("Fraud Risk Level Distribution")
    st.bar_chart(df["fraud_risk_level"].value_counts())

    st.write("Final MoE Score Distribution (Sample)")
    st.line_chart(
        df["final_moe_score"]
        .sample(min(1000, len(df)))
        .sort_values()
        .reset_index(drop=True)
    )

# ====================================================
# FOOTER
# ====================================================
st.markdown("---")
st.caption(
    "This application demonstrates a Mixture-of-Experts fraud detection "
    "architecture inspired by Nubank’s real-time defense platform."
)
