import requests
import pandas as pd
import plotly.express as px
import streamlit as st

API_URL = "http://128.226.29.112:8000"

st.title("Semantic Drift Heatmap")
st.markdown("""
### What is Semantic Drift?

Semantic drift measures **how much the meaning of memes changes** as they move from 4chan to Reddit.  
It is computed using **SBERT embeddings**, comparing the centroid of texts from each platform.

- **Higher drift (lighter color)** → memes on 4chan and Reddit differ significantly in meaning  
- **Lower drift (darker color)** → content is more semantically aligned across platforms  

This plot helps identify which 4chan boards create memes that transform the most before appearing on Reddit.
""")


# UI options
board_options = ["b", "pol", "gif"]
subreddit_options = ["memes", "dankmemes", "funny"]

boards = st.multiselect("Select 4chan boards", board_options, default=["b", "pol"])
subs = st.multiselect("Select Reddit subreddits", subreddit_options, default=["memes", "dankmemes"])
days = st.slider("Days Window", 3, 30, 14)

# Initialize session state to persist results
if "drift_result" not in st.session_state:
    st.session_state["drift_result"] = None


# ================================================================
# 1️⃣ Compute Drift Button
# ================================================================
if st.button("Compute Drift"):
    payload = {
        "boards": boards,
        "subreddits": subs,
        "days": days
    }

    try:
        resp = requests.post(f"{API_URL}/semantic_drift", json=payload)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        st.error(f"Backend error: {e}")
        st.stop()

    if data.get("drift"):
        # Create dataframe
        df = pd.DataFrame(
            data["drift"],
            index=data["boards"],
            columns=data["subreddits"]
        )

        # Save results to session_state for explanation
        st.session_state["drift_result"] = {
            "boards": data["boards"],
            "subreddits": data["subreddits"],
            "drift": data["drift"]
        }

        # Show results
        st.subheader("Semantic Drift Matrix")
        st.write(df)

        # Drift heatmap
        fig = px.imshow(
            df,
            text_auto=True,
            labels=dict(
                x="Subreddit",
                y="4chan Board",
                color="Drift (1 - cosine similarity)"
            ),
            title="Semantic Drift (4chan → Reddit)"
        )
        st.plotly_chart(fig, use_container_width=True, key="drift_chart_main")

    else:
        st.warning(data.get("message", "No drift data returned"))



# ================================================================
# 2️⃣ If drift was previously computed → show LLM explanation button
# ================================================================
stored = st.session_state.get("drift_result", None)

if stored:
    st.markdown("---")
    st.subheader("Explain the Drift Pattern")

    if st.button("Explain Drift (LLM Generated)"):
        explain_payload = {
            "analysis_type": "drift",
            "data": stored   # send original boards, subs, matrix
        }

        try:
            res = requests.post(f"{API_URL}/explain", json=explain_payload)
            res.raise_for_status()
            explanation = res.json().get("explanation", "No explanation available.")
        except Exception as e:
            st.error(f"LLM backend error: {e}")
            st.stop()

        st.subheader("LLM Explanation")
        st.write(explanation)
