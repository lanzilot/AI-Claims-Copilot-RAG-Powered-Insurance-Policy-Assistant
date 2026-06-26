import streamlit as st
from rag import answer_claim_question

from pathlib import Path
from pdf_ingest import ingest_pdf

st.markdown("""
<style>
.stApp {
    background-color: #EAF4FF;
}
</style>
""", unsafe_allow_html=True)

st.subheader("Upload HMO Policy PDF")

uploaded_file = st.file_uploader("Upload policy PDF", type=["pdf"])

if uploaded_file is not None:
    save_path = Path("data") / uploaded_file.name

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("Ingest PDF into Knowledge Base"):
        with st.spinner("Reading PDF and updating vector database..."):
            chunk_count = ingest_pdf(save_path)

        st.success(f"PDF ingested successfully. {chunk_count} chunks added.")
        

st.set_page_config(page_title="AI Claims Copilot", layout="wide")

st.title("AI Claims Copilot")
st.write("Ask questions about HMO benefits, claims, exclusions, and coverage.")

question = st.text_input("Enter your claims or policy question:")

if st.button("Ask AI"):
    if question:
        st.write("Button clicked.")
        
        with st.spinner("Checking policy documents..."):
            answer, sources, context = answer_claim_question(question)

        st.write("AI function completed.")

        st.subheader("Answer")
        st.success(answer)

        st.subheader("Sources")
        st.write(sources)

        st.subheader("Retrieved Policy Context")
        st.text_area("Context used by AI", context, height=250)
    else:
        st.warning("Please enter a question.")

import pandas as pd
from pathlib import Path

st.divider()
st.subheader("Audit Logs")

audit_file = Path("audit_logs/claims_audit.csv")

if audit_file.exists():
    df = pd.read_csv(audit_file)
    st.dataframe(df)
else:
    st.info("No audit logs yet.")

st.divider()
st.subheader("Evaluation Results")

eval_file = Path("evaluation/evaluation_results.csv")

if eval_file.exists():
    eval_df = pd.read_csv(eval_file)

    accuracy = eval_df["passed"].mean() * 100

    st.metric("AI Accuracy", f"{accuracy:.2f}%")
    st.dataframe(eval_df)
else:
    st.info("No evaluation results yet. Run: python evals.py")