# PlanIQ

AI-powered Irish planning permission guidance.

PlanIQ helps property owners, homeowners, and architects understand Irish planning law — whether they need planning permission, whether works are exempted development, and how to navigate the planning application process.

Built on the Planning and Development Acts, Schedule 2 exempted development regulations, and Dublin City Council Development Plan 2022-2028.

---

## What PlanIQ does

- Eligibility checker: do you need planning permission for your project?
- Exemption checker: does your project qualify as exempted development?
- Process guide: how to apply, appeal timelines, Section 5 declarations
- Citation-first answers: every claim cites the exact section of Irish law
- Hallucination prevention: 7-layer detection system blocks ungrounded answers

---

## Tech Stack

- Knowledge base: ChromaDB + BM25 (1,192 chunks of Irish planning law)
- Retrieval: Hybrid dense vector + BM25 + cross-encoder reranker
- Generation: Claude claude-sonnet-4-6 via Anthropic API
- Hallucination detection: Entity grounding, staleness gates, HITL escalation
- API: FastAPI
- UI: Streamlit (MVP)

---

## Project Structure

    PlanIQ/
    ingestion/              Document scraping, chunking, schema
        schema.py           PlanningChunk metadata contract
        chunker.py          Section-aware semantic chunker
        scraper.py          Irish planning document fetcher
    knowledge_base/         Vector store management
        store.py            ChromaDB + BM25 dual index
    retrieval/              Hybrid retrieval engine
        hybrid_retriever.py BM25 + dense + reranker + RRF fusion
    hallucination/          Hallucination detection layer
        detector.py         7-layer hallucination shield
    generation/             LLM generation engine
        engine.py           Generation pipeline
        prompts.py          Citation-first prompt templates
    api/                    FastAPI REST API
        main.py             5 endpoints
    tests/                  113 tests across all layers
    pipeline.py             Knowledge base ingestion orchestrator
    app.py                  Streamlit MVP interface
    requirements.txt        Python dependencies

---

## Setup

### 1. Clone and create virtual environment

    git clone https://github.com/parthInAI/planiq.git
    cd planiq
    python -m venv venv

    Mac/Linux:
    source venv/bin/activate

    Windows PowerShell:
    venv\Scripts\Activate.ps1

### 2. Install dependencies

    pip install -r requirements.txt

### 3. Build the knowledge base

    python pipeline.py

This fetches Irish planning legislation and builds the ChromaDB + BM25 indexes.

For Dublin City Council development plan, download the PDF manually from:
https://www.dublincity.ie/sites/default/files/2023-02/Final%20Vol%201%20Written%20Statement.pdf

Save it as data/raw/dublin_city_devplan_2022.pdf then run:

    python pipeline.py --sources dublin_city_devplan_2022

### 4. Set environment variables

    Mac/Linux:
    export ANTHROPIC_API_KEY=your-key-here
    export PLANIQ_LLM_PROVIDER=anthropic

    Windows PowerShell:
    $env:ANTHROPIC_API_KEY = "your-key-here"
    $env:PLANIQ_LLM_PROVIDER = "anthropic"

### 5. Run the tests

    pytest tests/ -v

Expected: 113 passed

### 6. Start the API

    uvicorn api.main:app --port 8000

### 7. Start the UI

    streamlit run app.py

Open http://localhost:8501

---

## API Endpoints

- POST /query          Main planning query
- GET  /health         System health check
- GET  /stats          Knowledge base statistics
- GET  /councils       List all 31 Irish councils
- POST /feedback       Submit response feedback

### Example query

    curl -X POST http://localhost:8000/query \
      -H "Content-Type: application/json" \
      -d '{"query": "Do I need planning permission for a rear extension?", "council": "dublin_city"}'

---

## Knowledge Base Sources

- Planning and Development Act 2024: 285 chunks
- Dublin City Development Plan 2022-2028: 595 chunks
- National Planning Framework 2040: 178 chunks
- PDR 2001 Schedule 2 Exempted Development: 114 chunks
- Schedule 2 Part 1 Classes 1-7 Seed: 7 chunks
- Solar Panel Exemption S.I. 493/2022: 4 chunks
- Planning and Development Act 2000: 9 chunks
- Total: 1,192 chunks

---

## Responsible AI

- Every response includes mandatory disclaimer
- Hallucination detection runs on every query
- Appeals, enforcement, and Section 5 queries escalate to human review
- Stale regulations are hard-blocked from retrieval
- EU AI Act compliant — not classified as high-risk
- GDPR compliant — data minimisation by design

---

## Built by

Parth Pandya — AI Engineer
Portfolio: https://parthinai.github.io
GitHub: https://github.com/parthInAI
LinkedIn: https://linkedin.com/in/parthpandya-/

---

## Disclaimer

PlanIQ provides guidance only and does not constitute professional planning advice. For formal determinations, engage a registered planning consultant or submit a Section 5 declaration to your local authority.
