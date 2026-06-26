"""
PlanIQ — Streamlit Cloud Deployment
=====================================
Self-contained Streamlit app that calls the RAG pipeline directly.
No FastAPI server required — runs entirely within Streamlit Cloud.

Deploy to: https://streamlit.io/cloud
"""

import sys
import os
import time
import logging
from pathlib import Path

import streamlit as st

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
for p in [ROOT, ROOT/"ingestion", ROOT/"knowledge_base",
          ROOT/"retrieval", ROOT/"hallucination",
          ROOT/"generation", ROOT/"document_review"]:
    sys.path.insert(0, str(p))

st.set_page_config(
    page_title="PlanIQ -- Irish Planning AI",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 600; color: #1D9E75; margin-bottom: 0.2rem; }
    .sub-header { color: #6B7280; font-size: 1rem; margin-bottom: 2rem; }
    .confidence-high   { color: #065F46; background: #D1FAE5; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 500; }
    .confidence-medium { color: #92400E; background: #FEF3C7; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 500; }
    .confidence-low    { color: #991B1B; background: #FEE2E2; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 500; }
    .disclaimer-box { background: #F9FAFB; border-left: 3px solid #D1D5DB; padding: 0.75rem 1rem; font-size: 0.8rem; color: #6B7280; margin-top: 1.5rem; border-radius: 0 4px 4px 0; }
    .citation-card { background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 0.5rem; }
    .escalation-box { background: #FFF7ED; border: 1px solid #FED7AA; border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)


def _get_secret(key, default=""):
    """Read from st.secrets or os.environ safely."""
    try:
        val = st.secrets.get(key, "")
        if val:
            return val
    except Exception:
        pass
    return os.environ.get(key, default)


@st.cache_resource(show_spinner="Loading PlanIQ knowledge base...")
def load_pipeline():
    from generation.engine import PlanIQGenerationEngine

    qdrant_url = _get_secret("QDRANT_URL")
    qdrant_key = _get_secret("QDRANT_API_KEY")
    anthropic_key = _get_secret("ANTHROPIC_API_KEY")
    if anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key

    if qdrant_url and qdrant_key:
        from retrieval.qdrant_retriever import QdrantRetriever
        retriever = QdrantRetriever(qdrant_url=qdrant_url, qdrant_api_key=qdrant_key)
        stats = {"total_chunks_chroma": 13025, "total_docs_ingested": 101}
    else:
        from knowledge_base.store import PlanIQKnowledgeBase
        from retrieval.hybrid_retriever import HybridRetriever
        kb = PlanIQKnowledgeBase()
        retriever = HybridRetriever(kb)
        stats = kb.get_stats()

    provider = _get_secret("PLANIQ_LLM_PROVIDER", "anthropic")
    engine = PlanIQGenerationEngine(provider=provider)
    return retriever, engine, stats


@st.cache_resource(show_spinner=False)
def load_document_review():
    from document_review.pdf_extractor import PDFFieldExtractor
    from document_review.article22_checker import Article22Checker
    return PDFFieldExtractor(), Article22Checker()


def get_jurisdiction(council):
    from ingestion.schema import Jurisdiction
    if council == "national":
        return None
    try:
        return Jurisdiction(council)
    except ValueError:
        return None


try:
    retriever, engine, kb_stats = load_pipeline()
    pipeline_loaded = True
except Exception as e:
    pipeline_loaded = False
    pipeline_error = str(e)

try:
    extractor, checker = load_document_review()
    doc_review_loaded = True
except Exception:
    doc_review_loaded = False

COUNCIL_OPTIONS = {
    "National (all Ireland)":  "national",
    "Dublin City Council":     "dublin_city",
    "Fingal County Council":   "fingal",
    "South Dublin":            "south_dublin",
    "Dun Laoghaire-Rathdown":  "dun_laoghaire_rathdown",
    "Cork City":               "cork_city",
    "Cork County":             "cork_county",
    "Galway City":             "galway_city",
    "Galway County":           "galway_county",
    "Limerick":                "limerick",
    "Waterford":               "waterford",
    "Kerry":                   "kerry",
    "Kildare":                 "kildare",
    "Meath":                   "meath",
    "Wicklow":                 "wicklow",
    "Wexford":                 "wexford",
    "Kilkenny":                "kilkenny",
    "Tipperary":               "tipperary",
    "Laois":                   "laois",
    "Longford":                "longford",
    "Louth":                   "louth",
    "Monaghan":                "monaghan",
    "Roscommon":               "roscommon",
    "Clare":                   "clare",
    "Cavan":                   "cavan",
    "Offaly":                  "offaly",
}

with st.sidebar:
    st.markdown("## 🏠 PlanIQ")
    st.markdown("---")
    st.markdown("**Select your council**")
    selected_council_name = st.selectbox("Council", list(COUNCIL_OPTIONS.keys()), label_visibility="collapsed")
    council_slug = COUNCIL_OPTIONS[selected_council_name]
    st.markdown("---")
    st.markdown("**About PlanIQ**")
    st.markdown("PlanIQ answers Irish planning questions using real legislation, 8 ministerial guidelines, 27 council development plans, and 37 inspector reports.")
    st.markdown("---")
    if pipeline_loaded:
        st.success(f"KB: {kb_stats['total_chunks_chroma']:,} chunks")
        if doc_review_loaded:
            st.success("Document review enabled")
    else:
        st.error("Pipeline failed")
        st.error(pipeline_error[:300] if 'pipeline_error' in dir() else "Unknown error")
    st.markdown("---")
    st.markdown("[GitHub](https://github.com/parthInAI/planiq) | [Portfolio](https://parthinai.github.io)")
    st.caption("PlanIQ provides guidance only. Always verify with your local planning authority.")

st.markdown('<div class="main-header">🏠 PlanIQ</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered Irish planning permission guidance -- built on Irish planning legislation, 8 ministerial guidelines, and 27 council development plans.</div>', unsafe_allow_html=True)

if not pipeline_loaded:
    st.error(f"PlanIQ pipeline failed to load: {pipeline_error if 'pipeline_error' in dir() else 'Unknown error'}")
    st.info("Check that ANTHROPIC_API_KEY, QDRANT_URL and QDRANT_API_KEY are set in Streamlit secrets.")
    st.stop()

tab1, tab2 = st.tabs(["💬 Planning Query", "📋 Document Review"])

with tab1:
    st.markdown("**Try one of these questions:**")
    examples = [
        "Do I need planning permission to build a rear extension in Dublin?",
        "Is a garden shed exempt from planning permission?",
        "Are solar panels on my roof exempt from planning permission?",
        "How long do I have to appeal a planning decision?",
        "What is genuine local need for rural housing in Ireland?",
    ]
    cols = st.columns(len(examples))
    selected_example = None
    for col, example in zip(cols, examples):
        with col:
            if st.button(example[:35] + "...", use_container_width=True):
                selected_example = example

    query = st.text_area("Your planning question", value=selected_example or "",
                         placeholder="e.g. Do I need planning permission to build a rear extension on my house in Dublin?",
                         height=100)
    col1, col2 = st.columns([1, 4])
    with col1:
        submit = st.button("Ask PlanIQ ->", type="primary", use_container_width=True)

    if submit and query:
        if len(query.strip()) < 10:
            st.warning("Please enter a more detailed question.")
        else:
            with st.spinner("Searching Irish planning law..."):
                try:
                    start = time.time()
                    jurisdiction = get_jurisdiction(council_slug)
                    retrieval = retriever.retrieve(query=query.strip(), jurisdiction=jurisdiction, top_k=5, use_reranker=False)
                    chunks = retrieval.chunks if hasattr(retrieval, "chunks") else []
                    response = engine.generate(query=query.strip(), retrieval_result=retrieval, jurisdiction=council_slug)
                    elapsed = int((time.time() - start) * 1000)
                except Exception as e:
                    st.error(f"Query failed: {e}")
                    import traceback
                    st.error(traceback.format_exc())
                    st.stop()

            if response.is_blocked:
                st.error(f"{response.answer_summary}")
                st.markdown(f'<div class="disclaimer-box">{response.disclaimer}</div>', unsafe_allow_html=True)
                st.stop()

            if response.requires_escalation:
                st.markdown(f'<div class="escalation-box">Professional advice recommended: {response.user_warning}</div>', unsafe_allow_html=True)

            conf = response.confidence
            conf_label = {"high": "High confidence", "medium": "Medium confidence", "low": "Low confidence"}.get(conf, conf)
            st.markdown("---")
            col_a, col_b = st.columns([5, 1])
            with col_a:
                st.markdown("### Answer")
            with col_b:
                st.markdown(f'<span class="confidence-{conf}">{conf_label}</span>', unsafe_allow_html=True)

            st.info(f"**{response.answer_summary}**")

            full = response.full_answer or {}
            query_type = response.query_type or "ELIGIBILITY"

            if query_type == "EXEMPTION" and "is_exempt" in full:
                exempt_val = full.get("is_exempt")
                if exempt_val is True:
                    st.success("Your works appear to qualify as exempted development.")
                elif exempt_val is False:
                    st.error("Your works do not appear to qualify as exempted development.")
                else:
                    st.warning("Exemption status is uncertain for your specific circumstances.")
                if full.get("thresholds"):
                    st.markdown("**Key thresholds:**")
                    for t in full["thresholds"]:
                        st.markdown(f"- {t.get('threshold', '')} _{t.get('citation', '')}_")
                if full.get("conditions"):
                    st.markdown("**Conditions:**")
                    for c in full["conditions"]:
                        condition = c.get("condition", c) if isinstance(c, dict) else c
                        st.markdown(f"- {condition}")
                if full.get("disqualifiers"):
                    st.markdown("**Things that would remove the exemption:**")
                    for d in full["disqualifiers"]:
                        st.markdown(f"- {d}")
                if full.get("section_5_recommended"):
                    st.info("Section 5 Declaration recommended.")

            elif query_type == "ELIGIBILITY" and "permission_required" in full:
                perm = full.get("permission_required")
                if perm is True:
                    st.error("Planning permission is required for these works.")
                elif perm is False:
                    st.success("Planning permission does not appear to be required.")
                else:
                    st.warning("Whether planning permission is required is uncertain.")
                if full.get("reasoning"):
                    st.markdown("**Reasoning:**")
                    for r in full["reasoning"]:
                        st.markdown(f"- {r.get('point', '')} _{r.get('citation', '')}_")

            elif query_type == "PROCESS" and "process_steps" in full:
                st.markdown("**Process steps:**")
                for step in full.get("process_steps", []):
                    st.markdown(f"**Step {step.get('step', '')} -- {step.get('action', '')}** ({step.get('timeline', '')}) _{step.get('citation', '')}_")

            if full.get("what_to_do_next"):
                st.success(f"What to do next: {full['what_to_do_next']}")

            if response.citations:
                with st.expander(f"Source citations ({len(response.citations)} sources)"):
                    for cite in response.citations:
                        st.markdown(f'<div class="citation-card"><strong>[{cite["chunk_num"]}]</strong> {cite["source_title"]}<br><em>{cite["section_ref"] or "N/A"}</em> | {cite["jurisdiction"]}<br><small>{cite["text_preview"]}</small></div>', unsafe_allow_html=True)

            with st.expander("Retrieval details"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Chunks", len(retrieval.chunks))
                c2.metric("Quality", f"{retrieval.retrieval_quality:.0%}")
                c3.metric("Time", f"{elapsed}ms")

            st.markdown(f'<div class="disclaimer-box">{response.disclaimer}</div>', unsafe_allow_html=True)

    elif submit and not query:
        st.warning("Please enter a question first.")


with tab2:
    st.markdown("### Article 22 Pre-Submission Gap Report")
    st.markdown("Upload your draft planning application PDF. PlanIQ will cross-check every required item against Article 22 of PDR 2001.")
    st.info("35% of planning applications are invalidated before assessment due to missing documents or wrong scales.")

    if not doc_review_loaded:
        st.error("Document review tool failed to load.")
    else:
        uploaded_file = st.file_uploader("Upload planning application PDF (max 20MB)", type=["pdf"])
        col1, col2 = st.columns([2, 3])
        with col1:
            review_council = st.selectbox("Council applying to", list(COUNCIL_OPTIONS.keys()),
                                          index=list(COUNCIL_OPTIONS.keys()).index(selected_council_name))
            review_council_slug = COUNCIL_OPTIONS[review_council]

        check_btn = st.button("Run Article 22 Check", type="primary", disabled=uploaded_file is None)

        if check_btn and uploaded_file:
            with st.spinner("Checking Article 22 requirements..."):
                try:
                    pdf_bytes = uploaded_file.read()
                    fields = extractor.extract(pdf_bytes)
                    if review_council_slug != "national" and not fields.get("planning_authority"):
                        fields["planning_authority"] = review_council
                    report = checker.check(fields)
                except Exception as e:
                    st.error(f"Document review failed: {e}")
                    st.stop()

            if report.overall_status == "valid":
                st.success(f"Application appears valid -- {report.passed} checks passed.")
            elif report.overall_status == "likely_invalid":
                st.error(f"Application likely invalid -- {report.failed} critical failures.")
            else:
                st.warning(f"Application needs review -- {report.warnings} warnings, {report.missing} missing.")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Passed", report.passed)
            m2.metric("Failed", report.failed)
            m3.metric("Warnings", report.warnings)
            m4.metric("Missing", report.missing)

            failures = [c for c in report.checks if c.status.value == "fail"]
            if failures:
                st.markdown("#### Critical failures")
                for c in failures:
                    with st.expander(f"FAIL: {c.item}", expanded=True):
                        st.markdown(f"**Finding:** {c.finding}")
                        st.error(f"**Action:** {c.action}")

            missing_items = [c for c in report.checks if c.status.value == "missing"]
            if missing_items:
                st.markdown("#### Missing items")
                for c in missing_items:
                    with st.expander(f"MISSING: {c.item}"):
                        st.markdown(f"**Finding:** {c.finding}")
                        st.warning(f"**Action:** {c.action}")

            warnings_list = [c for c in report.checks if c.status.value == "warning"]
            if warnings_list:
                st.markdown("#### Warnings")
                for c in warnings_list:
                    with st.expander(f"WARNING: {c.item}"):
                        st.info(f"**Action:** {c.action}")

            passed_list = [c for c in report.checks if c.status.value == "pass"]
            if passed_list:
                with st.expander(f"{len(passed_list)} checks passed"):
                    for c in passed_list:
                        st.markdown(f"PASS: **{c.item}**")

            st.markdown(f'<div class="disclaimer-box">{report.disclaimer}</div>', unsafe_allow_html=True)
