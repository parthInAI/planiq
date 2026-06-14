# PlanIQ — Local Setup Guide

## Prerequisites
- Python 3.11 or 3.12
- pip
- Terminal (PowerShell on Windows, Terminal on Mac)

---

## Step 1 — Move downloaded files into your PlanIQ folder

Your desktop PlanIQ folder should look like this when done:

```
PlanIQ/
├── requirements.txt
├── pipeline.py
├── ingestion/
│   ├── __init__.py          ← create this (empty file)
│   ├── schema.py
│   ├── chunker.py
│   └── scraper.py
├── knowledge_base/
│   ├── __init__.py          ← create this (empty file)
│   └── store.py
├── tests/
│   ├── __init__.py          ← create this (empty file)
│   └── test_step1.py
└── data/
    ├── raw/                 ← create empty folder
    └── processed/           ← create empty folder
```

---

## Step 2 — Open terminal in your PlanIQ folder

**Mac:**
Right-click the PlanIQ folder on Desktop → "New Terminal at Folder"

**Windows:**
Open PowerShell → type: `cd C:\Users\YourName\Desktop\PlanIQ`

---

## Step 3 — Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Mac/Linux:**
```bash
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```

You should see `(venv)` appear at the start of your terminal prompt.

---

## Step 4 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will take 3–5 minutes (sentence-transformers downloads a model).

---

## Step 5 — Create the __init__.py files

**Mac/Linux:**
```bash
touch ingestion/__init__.py
touch knowledge_base/__init__.py
touch tests/__init__.py
```

**Windows (PowerShell):**
```powershell
New-Item ingestion\__init__.py -ItemType File
New-Item knowledge_base\__init__.py -ItemType File
New-Item tests\__init__.py -ItemType File
```

---

## Step 6 — Create the data folders

**Mac/Linux:**
```bash
mkdir -p data/raw data/processed
```

**Windows:**
```powershell
mkdir data\raw
mkdir data\processed
```

---

## Step 7 — Run the tests

```bash
pytest tests/test_step1.py -v
```

Expected output:
```
21 passed in 0.10s
```

---

## Step 8 — Run the full ingestion pipeline (optional)

This fetches live Irish planning documents and builds the knowledge base:

```bash
python pipeline.py
```

Takes 2–3 minutes. Downloads and caches documents to data/raw/.

To rebuild from scratch:
```bash
python pipeline.py --rebuild
```

To ingest just one source:
```bash
python pipeline.py --sources citizens_info_exempted
```

---

## Troubleshooting

**"Module not found" errors:**
Make sure your venv is activated (you see `(venv)` in terminal).

**"No space left" errors:**
sentence-transformers needs ~500MB. Free up disk space and retry.

**Windows: "execution policy" error on Activate.ps1:**
Run this first:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Mac: "python not found":**
Try `python3` instead of `python` throughout.
