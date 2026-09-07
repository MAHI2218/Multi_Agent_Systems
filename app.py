"""
Streamlit UI for the multi-agent research pipeline (search -> read -> write -> critique).

Run with:
    streamlit run app.py

This file expects `agents.py` (build_search_agent, build_reader_agent, writer_chain,
critic_chain) to be importable from the same folder as pipeline.py.
"""

import streamlit as st  # type: ignore[import-not-found]
from agents import build_search_agent, build_reader_agent, writer_chain, critic_chain


# ---------- Helpers ----------

def to_text(x) -> str:
    """Normalize LLM/chain output (str, AIMessage, dict, etc.) into plain text."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if hasattr(x, "content"):
        return x.content
    if isinstance(x, dict):
        # common langchain shapes
        for key in ("output_text", "output", "content", "text"):
            if key in x:
                return to_text(x[key])
        return str(x)
    return str(x)


def run_pipeline_streamlit(topic: str) -> dict:
    """Same 4 steps as run_reasearch_pipeline, but reporting progress to the Streamlit UI."""
    state = {}
    progress = st.progress(0, text="Starting pipeline...")

    # Step 1 - Search agent
    with st.status("Step 1/4 — Search agent is working...", expanded=True) as status:
        search_agent = build_search_agent()
        search_result = search_agent.invoke(
            {"messages": [("user", f"Search for recent and reliable information on the topic: {topic}")]}
        )
        state["search_results"] = to_text(search_result["messages"][-1].content)
        st.write(state["search_results"])
        status.update(label="Step 1/4 — Search complete", state="complete")
    progress.progress(25, text="Search complete")

    # Step 2 - Reader agent
    with st.status("Step 2/4 — Reader agent is scraping the top resource...", expanded=True) as status:
        reader_agent = build_reader_agent()
        reader_result = reader_agent.invoke(
            {
                "messages": [
                    (
                        "user",
                        f"Based on the following search results about '{topic}', "
                        f"pick the most relevant URL and scrape it for deeper content.\n\n"
                        f"Search Results:\n{state['search_results'][:800]}",
                    )
                ]
            }
        )
        state["scraped_content"] = to_text(reader_result["messages"][-1].content)
        st.write(state["scraped_content"])
        status.update(label="Step 2/4 — Scraping complete", state="complete")
    progress.progress(50, text="Scraping complete")

    # Step 3 - Writer chain
    with st.status("Step 3/4 — Writer agent is generating the report...", expanded=True) as status:
        research_combined = (
            f"SEARCH_RESULTS :\n{state['search_results']}\n\n"
            f"DETAILED SCRAPED_CONTENT :\n{state['scraped_content']}"
        )
        report_raw = writer_chain.invoke({"topic": topic, "research": research_combined})
        state["report"] = to_text(report_raw)
        st.write(state["report"])
        status.update(label="Step 3/4 — Report generated", state="complete")
    progress.progress(75, text="Report generated")

    # Step 4 - Critic chain
    with st.status("Step 4/4 — Critic agent is reviewing the report...", expanded=True) as status:
        feedback_raw = critic_chain.invoke({"report": state["report"]})
        state["feedback"] = to_text(feedback_raw)
        st.write(state["feedback"])
        status.update(label="Step 4/4 — Review complete", state="complete")
    progress.progress(100, text="Pipeline complete")

    return state


# ---------- Page ----------

st.set_page_config(page_title="Multi-Agent Research Pipeline", page_icon="🔎", layout="wide")

st.title("🔎 Multi-Agent Research Pipeline")
st.caption("Search agent → Reader agent → Writer chain → Critic chain")

if "history" not in st.session_state:
    st.session_state.history = {}

with st.form("topic_form"):
    topic = st.text_input("Enter the research topic", placeholder="e.g. Latest advances in solid-state batteries")
    submitted = st.form_submit_button("Run pipeline", type="primary")

if submitted:
    if not topic.strip():
        st.warning("Please enter a topic first.")
    else:
        with st.spinner("Running the full pipeline — this can take a bit..."):
            try:
                result = run_pipeline_streamlit(topic.strip())
                st.session_state.history[topic.strip()] = result
                st.success("Done! Final report and feedback are below.")
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                st.stop()

        st.divider()
        st.subheader("📄 Final Report")
        st.markdown(result["report"])
        st.download_button(
            "Download report (.md)",
            data=result["report"],
            file_name=f"{topic.strip().replace(' ', '_')}_report.md",
            mime="text/markdown",
        )

        st.subheader("🧐 Critic Feedback")
        st.markdown(result["feedback"])

        with st.expander("🔍 Raw search results"):
            st.write(result["search_results"])
        with st.expander("📚 Raw scraped content"):
            st.write(result["scraped_content"])

# Sidebar: past runs in this session
if st.session_state.history:
    st.sidebar.header("Past runs (this session)")
    for past_topic in reversed(list(st.session_state.history.keys())):
        if st.sidebar.button(past_topic, key=f"hist_{past_topic}"):
            past = st.session_state.history[past_topic]
            st.subheader(f"📄 Report — {past_topic}")
            st.markdown(past["report"])
            st.subheader("🧐 Critic Feedback")
            st.markdown(past["feedback"])