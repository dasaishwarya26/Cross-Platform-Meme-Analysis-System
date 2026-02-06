import requests
import pandas as pd
import plotly.express as px
import streamlit as st

API_URL = "http://128.226.29.112:8000"

st.title("Temporal Co-occurrence")

st.markdown("""
### What is Temporal Co-occurrence?

Temporal co-occurrence tracks **which platform posts first** by comparing daily activity.  
It measures the difference:

\[
\{lag}(t) = \{Reddit posts count}(t) - \{4chan posts count}(t)
\]

- **Negative lag** → 4chan leads (4chan posts appear before Reddit’s)  
- **Positive lag** → Reddit is more active or reacting later  

This visualization helps identify whether **4chan drives early meme creation** relative to Reddit.
""")


board_options = ["b", "pol", "gif"]
subreddit_options = ["memes", "dankmemes", "funny"]

board = st.selectbox("Select 4chan Board", board_options, index=0)
sub = st.selectbox("Select Subreddit", subreddit_options, index=0)
days = st.slider("Days Window", 3, 60, 30)

# Initialize session state
if "temporal_result" not in st.session_state:
    st.session_state["temporal_result"] = None


# ================================================================
# 1️⃣ Compute Temporal Activity Button
# ================================================================
if st.button("Compute Temporal Co-occurrence"):
    payload = {
        "board": board,
        "subreddit": sub,
        "days": days
    }

    try:
        resp = requests.post(f"{API_URL}/temporal_cooccurrence", json=payload)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        st.error(f"Backend error: {e}")
        st.stop()

    points = data.get("points", [])

    if points:
        df = pd.DataFrame(points)

        # Save result for LLM explanation
        st.session_state["temporal_result"] = {
            "board": board,
            "subreddit": sub,
            "time": df["time"].tolist(),
            "lag_values": df["lag_value"].tolist()
        }

        st.subheader("Temporal Lag Data")
        st.write(df)

        fig = px.line(
            df,
            x="time",
            y="lag_value",
            markers=True,
            labels=dict(
                time="Day Index",
                lag_value="Reddit - 4chan Activity"
            ),
            title=f"Temporal Activity Difference ({board} → {sub})"
        )
        st.plotly_chart(fig, use_container_width=True, key="temporal_chart_main")

    else:
        st.warning("No temporal data returned.")


# ================================================================
# 2️⃣ LLM Explanation Section
# ================================================================
stored = st.session_state.get("temporal_result", None)

if stored:
    st.markdown("---")
    st.subheader("Explain the Temporal Pattern")

    if st.button("Explain Temporal (LLM Generated)"):
        explain_payload = {
            "analysis_type": "temporal",
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
