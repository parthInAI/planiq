"""
PlanIQ — Streamlit MVP Interface (Step 4)
==========================================
The first user-facing interface. Calls the FastAPI backend
and renders the structured response in a clean, trustworthy UI.

Run: streamlit run app.py
Requires the FastAPI server running on localhost:8000
"""

import sys
import json
import requests
from pathlib import Path

import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "PlanIQ — Irish Planning AI",
    page_icon  = "🏠",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

API_BASE = "http://localhost:8000"

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 600;
        color: #1D9E75;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #6B7280;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .confidence-high   { color: #065F46; background: #D1FAE5; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 500; }
    .confidence-medium { color: #92400E; background: #FEF3C7; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 500; }
    .confidence-low    { color: #991B1B; background: #FEE2E2; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 500; }
    .disclaimer-box {
        background: #F9FAFB;
        border-left: 3px solid #D1D5DB;
        padding: 0.75rem 1rem;
        font-size: 0.8rem;
        color: #6B7280;
        margin-top: 1.5rem;
        border-radius: 0 4px 4px 0;
    }
    .citation-card {
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }
    .escalation-box {
        background: #FFF7ED;
        border: 1px solid #FED7AA;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://via.placeholder.com/150x50/1D9E75/FFFFFF?text=PlanIQ", width=150)
    st.markdown("---")

    st.markdown("**Select your council**")
    council_options = {
        "National (all Ireland)": "national",
        "Dublin City Council":    "dublin_city",
        "Fingal County Council":  "fingal",
        "South Dublin":           "south_dublin",
        "Dún Laoghaire-Rathdown": "dun_laoghaire_rathdown",
        "Cork City":              "cork_city",
        "Cork County":            "cork_county",
        "Galway City":            "galway_city",
        "Galway County":          "galway_county",
        "Limerick":               "limerick",
        "Waterford":              "waterford",
    }
    selected_council_name = st.selectbox(
        "Council",
        list(council_options.keys()),
        label_visibility="collapsed",
    )
    council_slug = council_options[selected_council_name]

    st.markdown("---")
    st.markdown("**About PlanIQ**")
    st.markdown(
        "PlanIQ uses AI to help you understand Irish planning law. "
        "It draws from the Planning and Development Acts, exempted "
        "development regulations, and your council's development plan."
    )

    # Check API health
    st.markdown("---")
    st.markdown("**System status**")
    try:
        health = requests.get(f"{API_BASE}/health", timeout=3).json()
        st.success(f"API online — {health['kb_chunks']:,} chunks loaded")
    except Exception:
        st.error("API offline — start the FastAPI server first")
        st.code("uvicorn api.main:app --reload --port 8000")


# ── Main content ───────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🏠 PlanIQ</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">AI-powered Irish planning permission guidance — built on Irish planning legislation, 8 ministerial guidelines, and 25 council development plans.</div>',
    unsafe_allow_html=True
)

# Example questions
st.markdown("**Try one of these questions:**")
examples = [
    "Do I need planning permission to build a rear extension in Dublin?",
    "Is a garden shed exempt from planning permission?",
    "Are solar panels on my roof exempt from planning permission?",
    "How long do I have to appeal a planning decision?",
    "What is the maximum size for an exempt rear extension?",
]

cols = st.columns(len(examples))
selected_example = None
for col, example in zip(cols, examples):
    with col:
        if st.button(example[:35] + "...", use_container_width=True):
            selected_example = example

# Query input
query = st.text_area(
    "Your planning question",
    value=selected_example or "",
    placeholder="e.g. Do I need planning permission to build a rear extension on my house in Dublin?",
    height=100,
)

col1, col2 = st.columns([1, 4])
with col1:
    submit = st.button("Ask PlanIQ →", type="primary", use_container_width=True)

if submit and query:
    if len(query.strip()) < 10:
        st.warning("Please enter a more detailed question.")
    else:
        with st.spinner("Searching Irish planning law..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/query",
                    json={
                        "query":  query.strip(),
                        "council": council_slug,
                        "top_k":  5,
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to PlanIQ API. "
                    "Start the server with: `uvicorn api.main:app --reload --port 8000`"
                )
                st.stop()
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        answer = data["answer"]

        # ── Blocked response ──────────────────────
        if answer["is_blocked"]:
            st.error(f"⚠️ {answer['summary']}")
            st.markdown(
                f'<div class="disclaimer-box">{answer["disclaimer"]}</div>',
                unsafe_allow_html=True
            )
            st.stop()

        # ── Escalation warning ────────────────────
        if answer["escalation"]:
            st.markdown(
                f'<div class="escalation-box">⚠️ <strong>Professional advice recommended</strong><br>'
                f'{answer["warning"]}</div>',
                unsafe_allow_html=True
            )

        # ── Confidence badge ──────────────────────
        conf      = answer["confidence"]
        conf_css  = f"confidence-{conf}"
        conf_label = {"high": "High confidence", "medium": "Medium confidence", "low": "Low confidence"}.get(conf, conf)

        st.markdown("---")
        col_a, col_b = st.columns([5, 1])
        with col_a:
            st.markdown(f"### Answer")
        with col_b:
            st.markdown(f'<span class="{conf_css}">{conf_label}</span>', unsafe_allow_html=True)

        # ── Answer summary ────────────────────────
        st.info(f"**{answer['summary']}**")

        # ── Full structured answer ────────────────
        full = answer.get("full", {})
        query_type = data.get("query_type", "ELIGIBILITY")

        if query_type == "EXEMPTION" and "is_exempt" in full:
            exempt_val = full.get("is_exempt")
            if exempt_val is True:
                st.success("✅ Your works appear to qualify as **exempted development** — planning permission may not be required.")
            elif exempt_val is False:
                st.error("❌ Your works do **not** appear to qualify as exempted development.")
            else:
                st.warning("⚠️ Exemption status is **uncertain** for your specific circumstances.")

            if full.get("thresholds"):
                st.markdown("**Key thresholds that apply:**")
                for t in full["thresholds"]:
                    st.markdown(f"- {t.get('threshold', '')} _{t.get('citation', '')}_")

            if full.get("conditions"):
                st.markdown("**Conditions that must be met:**")
                for c in full["conditions"]:
                    condition = c.get("condition", c) if isinstance(c, dict) else c
                    st.markdown(f"- {condition}")

            if full.get("disqualifiers"):
                st.markdown("**⚠️ Things that would remove the exemption:**")
                for d in full["disqualifiers"]:
                    st.markdown(f"- {d}")

            if full.get("section_5_recommended"):
                st.info(
                    "💡 **Section 5 Declaration recommended** — submit a formal request to your "
                    "local planning authority to get a written confirmation of exempt status."
                )

        elif query_type == "ELIGIBILITY" and "permission_required" in full:
            perm = full.get("permission_required")
            if perm is True:
                st.error("❌ Planning permission **is required** for these works.")
            elif perm is False:
                st.success("✅ Planning permission **does not appear to be required** for these works.")
            else:
                st.warning("⚠️ Whether planning permission is required is **uncertain** — a formal determination is needed.")

            if full.get("reasoning"):
                st.markdown("**Reasoning:**")
                for r in full["reasoning"]:
                    st.markdown(f"- {r.get('point', '')} _{r.get('citation', '')}_")

        elif query_type == "PROCESS" and "process_steps" in full:
            st.markdown("**Process steps:**")
            for step in full.get("process_steps", []):
                st.markdown(
                    f"**Step {step.get('step', '')} — {step.get('action', '')}** "
                    f"({step.get('timeline', '')}) _{step.get('citation', '')}_"
                )

            if full.get("key_deadlines"):
                st.markdown("**⏰ Key deadlines:**")
                for d in full["key_deadlines"]:
                    st.warning(
                        f"**{d.get('deadline', '')}** — {d.get('timeframe', '')}. "
                        f"If missed: {d.get('consequence', '')}"
                    )

        # ── Next step ─────────────────────────────
        if full.get("what_to_do_next"):
            st.markdown("**What to do next:**")
            st.success(f"→ {full['what_to_do_next']}")

        # ── Citations ─────────────────────────────
        if answer.get("citations"):
            with st.expander(f"📚 Source citations ({len(answer['citations'])} sources used)"):
                for cite in answer["citations"]:
                    st.markdown(
                        f'<div class="citation-card">'
                        f'<strong>[Chunk {cite["chunk_num"]}]</strong> {cite["source_title"]}<br>'
                        f'<em>Section: {cite["section_ref"] or "N/A"}</em> | '
                        f'Jurisdiction: {cite["jurisdiction"]} | '
                        f'Effective: {cite["effective_date"] or "N/A"}<br>'
                        f'<small>{cite["text_preview"]}</small>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

        # ── Meta ──────────────────────────────────
        meta = data.get("meta", {})
        with st.expander("🔍 Retrieval details"):
            col1, col2, col3 = st.columns(3)
            col1.metric("Chunks retrieved", meta.get("chunks_retrieved", 0))
            col2.metric("Retrieval quality", f"{meta.get('retrieval_quality', 0):.0%}")
            col3.metric("Response time", f"{meta.get('latency_ms', 0)}ms")

        # ── Feedback ──────────────────────────────
        st.markdown("---")
        st.markdown("**Was this answer helpful?**")
        fb_col1, fb_col2 = st.columns(2)
        with fb_col1:
            if st.button("👍 Yes, helpful", use_container_width=True):
                requests.post(f"{API_BASE}/feedback", json={
                    "request_id": data["request_id"],
                    "query": query, "helpful": True,
                }, timeout=5)
                st.success("Thank you for your feedback!")
        with fb_col2:
            if st.button("👎 Needs improvement", use_container_width=True):
                requests.post(f"{API_BASE}/feedback", json={
                    "request_id": data["request_id"],
                    "query": query, "helpful": False,
                }, timeout=5)
                st.info("Thank you — this will help us improve PlanIQ.")

        # ── Mandatory disclaimer ───────────────────
        st.markdown(
            f'<div class="disclaimer-box">⚖️ {answer["disclaimer"]}</div>',
            unsafe_allow_html=True
        )

elif submit and not query:
    st.warning("Please enter a question first.")
