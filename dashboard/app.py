import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://admin:password@localhost:5435/fraud_db")
engine = create_engine(DATABASE_URL)

st.set_page_config(page_title="Real-Time Fraud Telemetry", layout="wide")
st.title("Financial Transaction Fraud Telemetry")

try:
    query = "SELECT * FROM transactions ORDER BY timestamp DESC LIMIT 100"
    df = pd.read_sql(query, engine)

    if not df.empty:
        total_txns = len(df)
        flagged_txns = df[df['is_fraud'] == 1]
        fraud_rate = (len(flagged_txns) / total_txns) * 100 if total_txns > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Evaluated Transactions", total_txns)
        col2.metric("System Fraud Rate", f"{fraud_rate:.2f}%")
        col3.metric("Total Fraud Losses Prevented", f"${flagged_txns['amount'].sum():,.2f}")

        st.divider()

        chart_col, table_col = st.columns(2)

        with chart_col:
            st.subheader("Inference Probability Distribution")
            
            bins = pd.cut(df['fraud_probability'], bins=[0, 0.25, 0.5, 0.75, 1.0])
            binned_counts = bins.value_counts().sort_index()
            
            binned_counts.index = binned_counts.index.astype(str)
            st.bar_chart(binned_counts)

        with table_col:
            st.subheader("Critical Explanations (High Risk Flagged)")
            
            flagged_display = flagged_txns[['user_id', 'amount', 'merchant', 'fraud_probability', 'explanation']].head(10)
            st.dataframe(flagged_display, hide_index=True)

    else:
        st.info("System fully operational. Waiting for transaction events...")

except Exception as e:
    st.error(f"Error establishing telemetry stream to DB: {e}")