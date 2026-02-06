import streamlit as st

# Set page config
st.set_page_config(page_title="Meme Influence Dashboard", layout="wide")

# -------- LEFT-ALIGN EVERYTHING WITH CSS --------
st.markdown(
    """
    <style>
    .left-align {
        text-align: left !important;
        padding-left: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------- CONTENT --------
st.markdown('<div class="left-align">', unsafe_allow_html=True)

st.title("Cross-Platform Meme Influence : AplusA Group")

st.markdown("### Navigate using the left sidebar to explore analyses.")

st.markdown("### Team Members")
st.markdown(
"""
- **Aishwarya Das**  
- **Annie Wu**
"""
)

st.markdown("</div>", unsafe_allow_html=True)
