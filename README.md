# Synapse

**Traditional RAG vs PageIndex RAG — live side-by-side comparison on SEC 10-K filings**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://synapse.streamlit.app)

---

## What is this?

Synapse lets you ask complex financial questions about Apple and Microsoft 10-K annual reports and watch **two RAG systems answer simultaneously**:

| | Traditional RAG | PageIndex RAG |
|---|---|---|
| **How it works** | Fixed-size text chunks → vector embeddings → cosine similarity | Hierarchical JSON tree → LLM tree navigation |
| **Retrieval** | "What text is most similar to the query?" | "Which section of the document would contain this answer?" |
| **Cross-reference support** | ✗ Cannot follow "see Note 7" pointers | ✓ Navigates the tree to the referenced section |
| **Multi-section synthesis** | ✗ Retrieves fragments from one area | ✓ Traverses multiple branches of the document tree |

**Benchmark:** PageIndex scores **98.7%** on FinanceBench (150 financial QA questions) vs ChatGPT+Search at 31%.

---

## Why 10-K filings?

SEC annual reports are specifically designed to break chunked RAG:

- A risk factor 40 pages away explains why a number on page 2 changed
- The MD&A section says "refer to Note 7" — traditional RAG cannot follow this
- "Effective tax rate" and "State Aid Decision" appear in different sections with zero vocabulary overlap — cosine similarity fails to connect them

---

## Demo questions (curated to show the difference)

| Difficulty | Type | Question |
|---|---|---|
| Easy | Simple Lookup | What was Apple's net income for fiscal 2024? |
| Medium | Cross-Section | What was Services revenue AND its gross margin? |
| Hard | Multi-Hop | Why did the effective tax rate increase, and how much did it cost? |
| Hard | Cross-Reference | What does Note 7 say about the European Commission State Aid Decision? |

---

## Tech stack (all free)

| Component | Tool |
|---|---|
| Frontend | Streamlit |
| PageIndex RAG | [PageIndex Python SDK](https://pageindex.ai) (free tier: 1,000 pages) |
| Traditional RAG vector store | ChromaDB (in-memory) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (local, free) |
| LLM | Groq free tier (LLaMA 3.3 70B) |
| Document source | SEC EDGAR (free, public) |

---

## Local setup

### 1. Clone and install

```bash
git clone https://github.com/yourusername/synapse.git
cd synapse
pip install -r requirements.txt
```

### 2. Set API keys

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```
PAGEINDEX_API_KEY=your_pageindex_key_here
GROQ_API_KEY=your_groq_key_here
```

- **PageIndex API key:** Sign up free at [pageindex.ai](https://pageindex.ai)
- **Groq API key:** Sign up free at [console.groq.com](https://console.groq.com)

### 3. Download the 10-K filings

```bash
python download_filings.py
```

This downloads Apple FY2024 and Microsoft FY2024 10-Ks from SEC EDGAR and indexes them with PageIndex (first run takes ~2 minutes; cached after that).

### 4. Run the app

```bash
streamlit run app.py
```

---

## Streamlit Cloud deployment

1. Push to a public GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → select your repo → `app.py`
3. Add secrets in the Streamlit Cloud dashboard:
   ```toml
   PAGEINDEX_API_KEY = "your_key_here"
   GROQ_API_KEY = "your_key_here"
   ```
4. Add the filing data to your repo (the `data/filings/` PDFs and `data/pageindex_cache/` JSONs) so the app does not need to re-download or re-index on cold start

---

## Project structure

```
synapse/
├── app.py                      # Streamlit entry point, dual-panel layout
├── pipelines/
│   ├── traditional_rag.py      # ChromaDB + HuggingFace embeddings pipeline
│   └── pageindex_rag.py        # PageIndexClient wrapper + tree caching
├── data/
│   ├── questions.json          # 10 curated Q&A with ground-truth answers
│   ├── filings/                # 10-K PDFs (download_filings.py generates these)
│   └── pageindex_cache/        # Cached PageIndex doc_ids and tree structures
├── download_filings.py         # One-time SEC EDGAR download + PDF conversion
├── requirements.txt
└── .env.example
```

---

## How PageIndex works

PageIndex parses a document into a **hierarchical JSON tree** that mirrors the document's natural structure:

```
10-K Annual Report
├── Item 1 — Business
│   ├── Overview
│   └── Products & Services
├── Item 1A — Risk Factors
│   ├── Macroeconomic Risks
│   └── Competitive Risks
├── Item 7 — MD&A
│   ├── Revenue Discussion
│   └── Operating Expenses
└── Item 8 — Financial Statements
    ├── Income Statement
    ├── Balance Sheet
    └── Notes to Financial Statements
        ├── Note 1 — Summary of Significant Accounting Policies
        └── Note 7 — Income Taxes
```

When you ask a question, instead of finding the most similar text chunk, PageIndex uses LLM reasoning to **navigate the tree** — deciding which branch to explore based on semantic understanding of the document structure. It can follow cross-references, synthesize from multiple branches, and cite the exact page number of each source.

---

## License

MIT
