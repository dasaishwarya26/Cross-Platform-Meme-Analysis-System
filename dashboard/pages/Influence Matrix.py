import requests
import pandas as pd
import plotly.express as px
import streamlit as st

API_URL = "http://128.226.29.112:8000"

st.title("Influence Matrix (4chan → Reddit)")

st.markdown("""
### What is Influence?

The influence score estimates **how closely aligned** the content from a 4chan board is to a Reddit subreddit.  
It uses **cosine similarity** between SBERT embedding centroids.

- **Higher influence (brighter color)** → strong directional similarity from 4chan → Reddit  
- **Lower influence** → limited cross-platform resemblance  

This matrix answers the core research question:  
**Which 4chan boards exert the strongest memetic influence on Reddit communities?**
""")


board_options = ["b", "pol", "gif"]
subreddit_options = ["memes", "dankmemes", "funny"]

boards = st.multiselect("Select 4chan Boards", board_options, default=["b", "pol"])
subs = st.multiselect("Select Reddit Subreddits", subreddit_options, default=["memes", "dankmemes"])
days = st.slider("Days Window", 3, 30, 14)

# Initialize session state
if "influence_result" not in st.session_state:
    st.session_state["influence_result"] = None


# ================================================================
# 1️⃣ Compute Influence Button
# ================================================================
if st.button("Compute Influence"):
    payload = {
        "boards": boards,
        "subreddits": subs,
        "days": days
    }

    try:
        resp = requests.post(f"{API_URL}/influence_matrix", json=payload)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        st.error(f"Backend error: {e}")
        st.stop()

    rows = data.get("rows", [])

    if rows:
        df = pd.DataFrame(rows)

        pivot = df.pivot(
            index="board_name",
            columns="subreddit",
            values="influence_score"
        )

        # Save result for LLM explanation
        st.session_state["influence_result"] = {
            "boards": pivot.index.tolist(),
            "subreddits": pivot.columns.tolist(),
            "influence_matrix": pivot.values.tolist()
        }

        st.subheader("Influence Matrix")
        st.write(pivot)

        fig = px.imshow(
            pivot,
            text_auto=True,
            labels=dict(
                x="Subreddit",
                y="4chan Board",
                color="Influence (Cosine Similarity)"
            ),
            title="Influence Matrix (4chan → Reddit)"
        )
        st.plotly_chart(fig, use_container_width=True, key="influence_chart_main")

    else:
        st.warning("No influence data returned.")


# ================================================================
# 2️⃣ LLM Explanation Section
# ================================================================
stored = st.session_state.get("influence_result", None)

if stored:
    st.markdown("---")
    st.subheader("Explain the Influence Pattern")

    if st.button("Explain Influence (LLM Generated)"):
        explain_payload = {
            "analysis_type": "influence",
            "data": stored
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
