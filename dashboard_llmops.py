import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="LLMOps Control Center", layout="wide")
st.title("LLMOps Observability & Evaluation Control Center")

tab1, tab2 = st.tabs(["📊 Benchmarks & Drift (v1 vs v2 vs v3)", "⚡ Live Playground & Traces"])

with tab1:
    try:
        with open("eval_history.jsonl", "r", encoding="utf-8") as f:
            runs = [json.loads(line) for line in f]
        df_runs = pd.DataFrame(runs)
        
        latest = df_runs.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Version Active", latest["tag"], f"Commit: {latest['git_commit']}")
        c2.metric("Faithfulness (Fidélité)", f"{latest['avg_faithfulness']*100:.1f}%")
        c3.metric("Answer Relevance", f"{latest['avg_relevance']*100:.1f}%")
        c4.metric("Latence Moyenne", f"{latest['avg_latency_ms']:.0f} ms")

        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Évolution de la Qualité par Version")
            fig_qual = px.bar(
                df_runs, x="tag", y=["avg_faithfulness", "avg_relevance"],
                barmode="group", range_y=[0, 1.1],
                labels={"value": "Score (0-1)", "tag": "Version du Pipeline"}
            )
            st.plotly_chart(fig_qual, use_container_width=True)

        with col_right:
            st.subheader("Décomposition de la Latence (Dernier Run)")
            df_det = pd.DataFrame(latest["details"])
            fig_lat = px.bar(
                df_det, x="category", y=["retrieval_ms", "generation_ms"],
                labels={"value": "Latence (ms)", "category": "Catégorie de Test"}
            )
            st.plotly_chart(fig_lat, use_container_width=True)

        st.subheader("Détail des tests d'évaluation")
        st.dataframe(pd.DataFrame(latest["details"]), use_container_width=True)
    except FileNotFoundError:
        st.warning("Aucune donnée trouvée. Lance d'abord `python3 eval_pipeline.py`.")

with tab2:
    st.markdown("🔗 **Arize Phoenix Tracing UI :** [http://localhost:6006](http://localhost:6006)")
    q = st.text_input("Tester une requête sur le pipeline instrumenté :", "Quelles sont les compétences de Soulaiman en IA ?")
    if st.button("Exécuter & Tracer"):
        res = requests.post("http://localhost:8006/ask", json={"question": q}).json()
        st.json(res)