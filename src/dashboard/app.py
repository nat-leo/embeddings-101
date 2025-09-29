from dashboard import api

import streamlit as st
import plotly.graph_objects as go

PALETTE = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
    "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
    "#393b79","#637939","#8c6d31","#843c39","#7b4173",
    "#3182bd","#e6550d","#31a354","#756bb1","#636363",
]

st.set_page_config(page_title="Dynamic Compare Form", layout="centered")

# --- Session state setup ---
if "inputs" not in st.session_state:
    st.session_state.inputs = [None]

if "submitted" not in st.session_state:
    st.session_state.submitted = False

if "vectors" not in st.session_state:
    st.session_state.vectors = []

if "last_response" not in st.session_state:
    st.session_state.last_response = {}

st.title("Text Embeddings Comparision")

# Grab the vector embeddings
# data = embed([text1, text2])
# st.text(data.content)

# --- Controls to add/remove fields (outside the form) ---
if st.session_state.submitted:
    items = [s for s in st.session_state.inputs if s is not None and s.strip()]
    if not items:
        st.warning("Please enter at least one non-empty item.")

    with st.spinner("Embedding…"):
        try:
            resp = api.embed(items)
            if resp.status_code == 200:
                st.session_state.last_response = resp.json()
                # Show a small summary and the raw payload
                st.session_state.vectors = [text["embedding"] for text in st.session_state.last_response["data"]]
            else:
                st.error(f"API error {resp.status_code}")
                with st.expander("Response body"):
                    st.text(resp.text)
        except Exception as e:
            st.error("Request failed.")
            st.exception(e)

        # Project to 2D and plot immediately
        coords = api.to_2d(st.session_state.vectors)
        fig = go.Figure()
        for i, (label, (x, y)) in enumerate(zip(items, coords)):
            fig.add_trace(
                go.Scatter(
                    x=[0,x],
                    y=[0,y],
                    mode="lines",
                    line=dict(width=2, color=PALETTE[i % len(PALETTE)]),
                    name=label,
                    text=label,  # hover
                )
            )

            fig.add_annotation(
                x=x, y=y, ax=0, ay=0,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=3, arrowsize=1, arrowwidth=2,
                arrowcolor=PALETTE[i % len(PALETTE)],
            )

        fig.update_layout(
            xaxis_title="PC 1",
            yaxis_title="PC 2",
            legend_title="Text Embeddings",
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    fig = go.Figure()
    st.plotly_chart(fig, use_container_width=True)

# -- Cute dividing line --
# st.divider()
st.success("Received embeddings.")

# --- The form ---

# Dynamically Remove unused form inputs
# TODO this is broken. I need to remove all Nones or empty strings except the last one,
#and I need to do so without needing to embed everything all over again.
for i, text_element in enumerate(st.session_state.inputs):
    if text_element in (None, ""):
        st.session_state.inputs.pop(i)
st.session_state.inputs.append(None)

# The actual form is built from the st.session_state.inputs session state.
with st.form("compare_form", clear_on_submit=False):
    st.subheader("Enter items to compare")
    for i in range(len(st.session_state.inputs)):
        st.session_state.inputs[i] = st.text_input(
            f"Item {i+1}",
            value=st.session_state.inputs[i],
            key=f"item_{i}",
            placeholder="Type something…",
        )
            
    # Submit button (onSubmit intentionally empty)
    st.session_state.submitted = st.form_submit_button("Compare")

if st.session_state.vectors and isinstance(st.session_state.vectors, list) and isinstance(st.session_state.vectors[0], list):
    dim = len(st.session_state.vectors[0])
    st.caption(f"Returned {len(st.session_state.vectors)} vectors · dimension {dim}")
st.json(st.session_state.last_response)