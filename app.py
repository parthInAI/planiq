"""
PlanIQ — Streamlit UI v2.0
===========================
Two tabs:
  Tab 1 — Planning Query (existing)
  Tab 2 — Document Review (new Phase 2 feature)

Run: streamlit run app.py
Requires FastAPI on localhost:8000
"""

import sys
import requests
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="PlanIQ — Irish Planning AI",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8000"

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
    .check-pass    { color: #065F46; background: #D1FAE5; padding: 3px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
    .check-fail    { color: #991B1B; background: #FEE2E2; padding: 3px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
    .check-warning { color: #92400E; background: #FEF3C7; padding: 3px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
    .check-missing { color: #1E40AF; background: #DBEAFE; padding: 3px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
    .gap-item { background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://via.placeholder.com/150x50/1D9E75/FFFFFF?text=PlanIQ", width=150)
    st.markdown("---")
    st.markdown("**Select your council**")
    council_options = {
        "National (all Ireland)":  "national",
        "Dublin City Council":     "dublin_city",
        "Fingal County Council":   "fingal",
        "South Dublin":            "south_dublin",
        "Dún Laoghaire-Rathdown":  "dun_laoghaire_rathdown",
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
    }
    selected_council_name = st.selectbox("Council", list(council_options.keys()), label_visibility="collapsed")
    council_slug = council_options[selected_council_name]

    st.markdown("---")
    st.markdown("**About PlanIQ**")
    st.markdown(
        "PlanIQ draws from Irish planning legislation, 8 ministerial guidelines, "
        "25 council development plans, and 37 An Coimisiún Pleanála inspector reports."
    )
    st.markdown("---")
    st.markdown("**System status**")
    try:
        health = requests.get(f"{API_BASE}/health", timeout=3).json()
        st.success(f"API online — {health['kb_chunks']:,} chunks")
        if health.get("document_review"):
            st.success("Document review: enabled")
    except Exception:
        st.error("API offline")
        st.code("uvicorn api.main:app --port 8000")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🏠 PlanIQ</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">AI-powered Irish planning permission guidance — '
    'built on Irish planning legislation, 8 ministerial guidelines, and 25 council development plans.</div>',
    unsafe_allow_html=True
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["💬 Planning Query", "📋 Document Review"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PLANNING QUERY
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
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
                        json={"query": query.strip(), "council": council_slug, "top_k": 5},
                        timeout=60,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to PlanIQ API. Start: `uvicorn api.main:app --port 8000`")
                    st.stop()
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.stop()

            answer = data["answer"]

            if answer["is_blocked"]:
                st.error(f"⚠️ {answer['summary']}")
                st.markdown(f'<div class="disclaimer-box">{answer["disclaimer"]}</div>', unsafe_allow_html=True)
                st.stop()

            if answer["escalation"]:
                st.markdown(
                    f'<div class="escalation-box">⚠️ <strong>Professional advice recommended</strong><br>'
                    f'{answer["warning"]}</div>', unsafe_allow_html=True)

            conf = answer["confidence"]
            conf_label = {"high": "High confidence", "medium": "Medium confidence", "low": "Low confidence"}.get(conf, conf)
            st.markdown("---")
            col_a, col_b = st.columns([5, 1])
            with col_a:
                st.markdown("### Answer")
            with col_b:
                st.markdown(f'<span class="confidence-{conf}">{conf_label}</span>', unsafe_allow_html=True)

            st.info(f"**{answer['summary']}**")

            full = answer.get("full", {})
            query_type = data.get("query_type", "ELIGIBILITY")

            if query_type == "EXEMPTION" and "is_exempt" in full:
                exempt_val = full.get("is_exempt")
                if exempt_val is True:
                    st.success("✅ Your works appear to qualify as **exempted development**.")
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
                    st.info("💡 **Section 5 Declaration recommended.**")

            elif query_type == "ELIGIBILITY" and "permission_required" in full:
                perm = full.get("permission_required")
                if perm is True:
                    st.error("❌ Planning permission **is required** for these works.")
                elif perm is False:
                    st.success("✅ Planning permission **does not appear to be required**.")
                else:
                    st.warning("⚠️ Whether planning permission is required is **uncertain**.")
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
                        st.warning(f"**{d.get('deadline', '')}** — {d.get('timeframe', '')}. If missed: {d.get('consequence', '')}")

            if full.get("what_to_do_next"):
                st.markdown("**What to do next:**")
                st.success(f"→ {full['what_to_do_next']}")

            if answer.get("citations"):
                with st.expander(f"📚 Source citations ({len(answer['citations'])} sources used)"):
                    for cite in answer["citations"]:
                        st.markdown(
                            f'<div class="citation-card">'
                            f'<strong>[Chunk {cite["chunk_num"]}]</strong> {cite["source_title"]}<br>'
                            f'<em>Section: {cite["section_ref"] or "N/A"}</em> | '
                            f'Jurisdiction: {cite["jurisdiction"]} | Effective: {cite["effective_date"] or "N/A"}<br>'
                            f'<small>{cite["text_preview"]}</small></div>',
                            unsafe_allow_html=True)

            meta = data.get("meta", {})
            with st.expander("🔍 Retrieval details"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Chunks retrieved", meta.get("chunks_retrieved", 0))
                c2.metric("Retrieval quality", f"{meta.get('retrieval_quality', 0):.0%}")
                c3.metric("Response time", f"{meta.get('latency_ms', 0)}ms")

            st.markdown("---")
            st.markdown("**Was this answer helpful?**")
            fb1, fb2 = st.columns(2)
            with fb1:
                if st.button("👍 Yes, helpful", use_container_width=True):
                    requests.post(f"{API_BASE}/feedback", json={"request_id": data["request_id"], "query": query, "helpful": True}, timeout=5)
                    st.success("Thank you!")
            with fb2:
                if st.button("👎 Needs improvement", use_container_width=True):
                    requests.post(f"{API_BASE}/feedback", json={"request_id": data["request_id"], "query": query, "helpful": False}, timeout=5)
                    st.info("Thank you — this helps us improve.")

            st.markdown(f'<div class="disclaimer-box">⚖️ {answer["disclaimer"]}</div>', unsafe_allow_html=True)

    elif submit and not query:
        st.warning("Please enter a question first.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DOCUMENT REVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📋 Article 22 Pre-Submission Gap Report")
    st.markdown(
        "Upload your draft planning application PDF. PlanIQ will cross-check every "
        "required item against **Article 22 of the Planning and Development Regulations 2001** "
        "and return a structured gap report — identifying exactly what is missing or incorrect "
        "before you lodge with the council."
    )
    st.info(
        "💡 **Why this matters:** 35% of planning applications are invalidated before assessment "
        "due to missing documents, wrong scales, or late newspaper notices. This check catches "
        "those issues before you submit."
    )

    uploaded_file = st.file_uploader(
        "Upload planning application PDF (max 20MB)",
        type=["pdf"],
        help="Upload your draft planning application Form No. 1 with all attachments as a single PDF.",
    )

    col1, col2 = st.columns([2, 3])
    with col1:
        review_council = st.selectbox(
            "Which council are you applying to?",
            list(council_options.keys()),
            index=list(council_options.keys()).index(selected_council_name),
        )
        review_council_slug = council_options[review_council]

    check_btn = st.button("🔍 Run Article 22 Check", type="primary", disabled=uploaded_file is None)

    if check_btn and uploaded_file:
        with st.spinner("Extracting fields and checking Article 22 requirements..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                    data={"council": review_council_slug},
                    timeout=60,
                )
                resp.raise_for_status()
                result = resp.json()

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to PlanIQ API.")
                st.stop()
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        # ── Extraction warning ────────────────────────────────────────────────
        if result.get("extraction_warning"):
            st.warning(f"⚠️ {result['extraction_warning']}")

        if not result.get("gap_report"):
            st.stop()

        report = result["gap_report"]

        # ── Extracted fields ──────────────────────────────────────────────────
        ef = result.get("extracted_fields", {})
        if any(ef.values()):
            with st.expander("📄 Extracted application details", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Applicant:** {ef.get('applicant_name') or 'Not detected'}")
                    st.markdown(f"**Planning Authority:** {ef.get('planning_authority') or 'Not detected'}")
                    st.markdown(f"**Permission type:** {ef.get('permission_type') or 'Not detected'}")
                with c2:
                    st.markdown(f"**Protected structure:** {'Yes' if ef.get('is_protected_structure') else 'No'}")
                    st.markdown(f"**ACA:** {'Yes' if ef.get('is_aca') else 'No'}")
                    st.markdown(f"**Extraction confidence:** {ef.get('extraction_confidence', 'unknown')}")
                if ef.get("development_description"):
                    st.markdown(f"**Description:** {ef['development_description'][:200]}...")

        # ── Overall status banner ─────────────────────────────────────────────
        status = report["overall_status"]
        if status == "valid":
            st.success(f"✅ Application appears **valid** — {report['passed']} checks passed, {report['warnings']} warnings.")
        elif status == "likely_invalid":
            st.error(f"❌ Application is **likely invalid** — {report['failed']} critical failures detected.")
        else:
            st.warning(f"⚠️ Application needs **review** — {report['warnings']} warnings and {report['missing']} items not confirmed.")

        # ── Summary metrics ───────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("✅ Passed",  report["passed"])
        m2.metric("❌ Failed",  report["failed"])
        m3.metric("⚠️ Warnings", report["warnings"])
        m4.metric("❓ Missing",  report["missing"])

        st.markdown("---")
        st.markdown("### Gap Report — Item by Item")

        # ── Failures first ────────────────────────────────────────────────────
        failures = [c for c in report["checks"] if c["status"] == "fail"]
        if failures:
            st.markdown("#### ❌ Critical failures — must fix before submission")
            for check in failures:
                with st.expander(f"❌ {check['item']} — {check['article']}", expanded=True):
                    st.markdown(f"**Finding:** {check['finding']}")
                    st.markdown(f"**Requirement:** {check['requirement']}")
                    st.error(f"**Action required:** {check['action']}")

        # ── Missing items ─────────────────────────────────────────────────────
        missing = [c for c in report["checks"] if c["status"] == "missing"]
        if missing:
            st.markdown("#### ❓ Missing items — not found in uploaded document")
            for check in missing:
                with st.expander(f"❓ {check['item']} — {check['article']}"):
                    st.markdown(f"**Finding:** {check['finding']}")
                    st.markdown(f"**Requirement:** {check['requirement']}")
                    st.warning(f"**Action required:** {check['action']}")

        # ── Warnings ──────────────────────────────────────────────────────────
        warnings_list = [c for c in report["checks"] if c["status"] == "warning"]
        if warnings_list:
            st.markdown("#### ⚠️ Warnings — verify before submission")
            for check in warnings_list:
                with st.expander(f"⚠️ {check['item']} — {check['article']}"):
                    st.markdown(f"**Finding:** {check['finding']}")
                    st.markdown(f"**Requirement:** {check['requirement']}")
                    st.info(f"**Action:** {check['action']}")

        # ── Passed ────────────────────────────────────────────────────────────
        passed_list = [c for c in report["checks"] if c["status"] == "pass"]
        if passed_list:
            with st.expander(f"✅ {len(passed_list)} checks passed — click to view"):
                for check in passed_list:
                    st.markdown(f"✅ **{check['item']}** — {check['finding']}")

        # ── Meta and disclaimer ───────────────────────────────────────────────
        meta = result.get("meta", {})
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("File size", f"{meta.get('file_size_kb', 0)} KB")
        col2.metric("Text extracted", f"{meta.get('text_chars_extracted', 0):,} chars")
        col3.metric("Processing time", f"{meta.get('latency_ms', 0)}ms")

        st.markdown(
            f'<div class="disclaimer-box">⚖️ {report["disclaimer"]}</div>',
            unsafe_allow_html=True
        )
