"""
Synapse -- Main Streamlit App
Traditional RAG vs PageIndex RAG side-by-side comparison.
"""
import json
import re
import html as html_lib
import markdown as md_lib
import os
import streamlit as st
from pathlib import Path

# -- Inject Streamlit Cloud secrets into os.environ so pipelines can read them
for _key in ("PAGEINDEX_API_KEY", "GROQ_API_KEY"):
    if _key not in os.environ:
        try:
            os.environ[_key] = st.secrets[_key]
        except (KeyError, FileNotFoundError):
            pass

# -- Page config --------------------------------------------------------------
st.set_page_config(
    page_title="Synapse",
    page_icon="\U0001f4ca",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Load question bank -------------------------------------------------------
QUESTIONS = json.loads(
    (Path(__file__).parent / "data" / "questions.json").read_text(encoding="utf-8")
)

FILINGS = {
    "Apple FY2024 (AAPL)": "AAPL_FY2024",
    "Microsoft FY2024 (MSFT)": "MSFT_FY2024",
}

DIFFICULTY_COLOR = {"easy": "\U0001f7e2", "medium": "\U0001f7e1", "hard": "\U0001f534"}
TYPE_LABEL = {
    "simple_lookup": "Simple Lookup",
    "cross_section": "Cross-Section",
    "multi_hop": "Multi-Hop",
    "cross_reference": "Cross-Reference",
}

# -- Custom CSS ---------------------------------------------------------------
st.markdown("""
<style>
    :root { color-scheme: light dark; }

    .main-header { font-size: 2.2rem; font-weight: 700; margin-bottom: 0; }
    .sub-header {
        font-size: 1rem;
        color: color-mix(in srgb, currentColor 55%, transparent);
        margin-top: 0; margin-bottom: 1.5rem;
    }
    .pipeline-header-trad {
        background: color-mix(in srgb, #ffc107 18%, transparent);
        border-left: 4px solid #ffc107;
        padding: 10px 14px; border-radius: 4px; font-weight: 600; font-size: 1.05rem;
    }
    .pipeline-header-pi {
        background: color-mix(in srgb, #17a2b8 18%, transparent);
        border-left: 4px solid #17a2b8;
        padding: 10px 14px; border-radius: 4px; font-weight: 600; font-size: 1.05rem;
    }
    .answer-box {
        background: color-mix(in srgb, currentColor 5%, transparent);
        border-radius: 6px; padding: 14px 18px; font-size: 0.95rem; min-height: 80px;
        border: 1px solid color-mix(in srgb, currentColor 20%, transparent);
        word-break: break-word; line-height: 1.6;
    }
    .answer-box p { margin: 0 0 0.6em 0; }
    .answer-box p:last-child { margin-bottom: 0; }
    .answer-box h1, .answer-box h2, .answer-box h3 {
        font-size: 1rem; font-weight: 700; margin: 0.8em 0 0.3em 0;
    }
    .answer-box ul, .answer-box ol { margin: 0.4em 0 0.6em 1.4em; padding: 0; }
    .answer-box li { margin-bottom: 0.25em; }
    .answer-box strong { font-weight: 700; }
    .answer-box em { font-style: italic; }
    .answer-box table { border-collapse: collapse; width: 100%; font-size: 0.88rem; margin: 0.5em 0; }
    .answer-box th, .answer-box td {
        border: 1px solid color-mix(in srgb, currentColor 25%, transparent);
        padding: 4px 8px; text-align: left;
    }
    .chunk-card {
        background: color-mix(in srgb, currentColor 4%, transparent);
        border: 1px solid color-mix(in srgb, currentColor 15%, transparent);
        border-radius: 4px; padding: 8px 12px; margin-bottom: 6px; font-size: 0.82rem;
    }
    .node-card {
        background: color-mix(in srgb, #17a2b8 10%, transparent);
        border: 1px solid color-mix(in srgb, #17a2b8 35%, transparent);
        border-radius: 4px; padding: 8px 12px; margin-bottom: 6px; font-size: 0.82rem;
        color: color-mix(in srgb, #17a2b8 90%, currentColor);
    }
    .badge-correct  { background:#28a745; color:white!important; padding:3px 10px; border-radius:12px; font-size:0.8rem; font-weight:600; }
    .badge-incorrect{ background:#dc3545; color:white!important; padding:3px 10px; border-radius:12px; font-size:0.8rem; font-weight:600; }
    .badge-partial  { background:#fd7e14; color:white!important; padding:3px 10px; border-radius:12px; font-size:0.8rem; font-weight:600; }
    .citation-pill {
        display: inline-block;
        background: color-mix(in srgb, #17a2b8 20%, transparent);
        border: 1px solid color-mix(in srgb, #17a2b8 50%, transparent);
        color: color-mix(in srgb, #17a2b8 90%, currentColor);
        border-radius: 10px; padding: 1px 8px; font-size: 0.75rem; font-weight: 600;
        margin: 0 2px; white-space: nowrap; vertical-align: middle;
    }
    .latency-box {
        font-size: 0.85rem;
        color: color-mix(in srgb, currentColor 60%, transparent);
        margin-top: 6px;
    }
    .stButton button { width: 100%; font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# -- Cached pipeline loaders --------------------------------------------------
@st.cache_resource(show_spinner="Loading Traditional RAG pipeline...")
def _load_trad_pipeline():
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from pipelines import traditional_rag
    return traditional_rag

@st.cache_resource(show_spinner="Loading PageIndex pipeline...")
def _load_pi_pipeline():
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from pipelines import pageindex_rag
    return pageindex_rag


# -- Helpers ------------------------------------------------------------------
def _accuracy_badge(answer: str, question: dict | None) -> str:
    if not question:
        return ""
    keywords = question.get("ground_truth_keywords", [])
    hits = sum(1 for kw in keywords if kw.lower() in answer.lower())
    if hits == 0:
        return '<span class="badge-incorrect">\u2717 Incorrect</span>'
    elif hits >= len(keywords):
        return '<span class="badge-correct">\u2713 Correct</span>'
    else:
        return '<span class="badge-partial">~ Partial</span>'


def _safe(text: str) -> str:
    return html_lib.escape(str(text))


def _format_citations(text: str) -> str:
    """Replace <doc=FILE;page=N> with a styled pill badge."""
    pattern = r'<doc=([^;]+);page=(\d+)>'
    def replacer(m):
        filename = m.group(1).replace("_", " ").replace(".pdf", "")
        page = m.group(2)
        return (
            f'<span class="citation-pill">'
            f'\U0001f4c4 {html_lib.escape(filename)} \u00b7 p.{page}'
            f'</span>'
        )
    return re.sub(pattern, replacer, text)


def _render_answer(text: str) -> str:
    """Citations -> markdown -> HTML."""
    return md_lib.markdown(_format_citations(text), extensions=["nl2br", "tables"])


def _node_text(node: dict) -> str:
    raw = node.get("text", "")
    if isinstance(raw, dict):
        raw = raw.get("content") or raw.get("summary") or str(raw)
    return str(raw)[:280].replace("\n", " ")


def _badge_label(answer: str, question: dict) -> str:
    keywords = question.get("ground_truth_keywords", [])
    hits = sum(1 for kw in keywords if kw.lower() in answer.lower())
    if hits == 0:
        return "INCORRECT"
    elif hits >= len(keywords):
        return "CORRECT"
    return "PARTIAL"


# -- Header -------------------------------------------------------------------
st.markdown('<p class="main-header">\U0001f4ca Synapse</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Traditional RAG vs PageIndex RAG \u2014 live side-by-side comparison on SEC 10-K filings</p>',
    unsafe_allow_html=True,
)
st.divider()

# -- Sidebar ------------------------------------------------------------------
with st.sidebar:
    st.header("\u2699\ufe0f Settings")

    # ── Document section ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Document")

    uploaded_file = st.file_uploader(
        "Upload your own PDF",
        type=["pdf"],
        help="Upload any PDF to query it with both pipelines.",
    )

    if uploaded_file is not None:
        upload_dir = Path(__file__).parent / "data" / "filings"
        safe_name = re.sub(r"[^A-Za-z0-9_\-]", "_", Path(uploaded_file.name).stem)
        safe_name = re.sub(r"[_\-]+", "_", safe_name).strip("_-")[:38] or "doc"
        upload_label = f"up_{safe_name}"
        pdf_path = upload_dir / f"{upload_label}.pdf"
        txt_path = upload_dir / f"{upload_label}.txt"

        # Save PDF if missing
        if not pdf_path.exists():
            pdf_path.write_bytes(uploaded_file.getvalue())

        # Extract text independently -- needed by traditional RAG chunker
        if not txt_path.exists():
            import PyPDF2
            reader = PyPDF2.PdfReader(pdf_path)
            text = "\n\n".join(p.extract_text() or "" for p in reader.pages)
            txt_path.write_text(text, encoding="utf-8")
            st.success(f"Uploaded: {uploaded_file.name}")

        # Persist in session_state so the entry survives Streamlit reruns
        display_name = f"\U0001f4ce {uploaded_file.name}"
        if "uploaded_filings" not in st.session_state:
            st.session_state["uploaded_filings"] = {}
        st.session_state["uploaded_filings"][display_name] = upload_label
        # Auto-select the uploaded file by writing directly to the widget key
        st.session_state["filing_selector"] = display_name

    # Merge uploaded filings into FILINGS (they live in session_state across reruns)
    if "uploaded_filings" in st.session_state:
        FILINGS.update(st.session_state["uploaded_filings"])

    filing_options = list(FILINGS.keys())
    # Ensure the widget key has a valid value (in case it was set to a name not yet in list)
    if st.session_state.get("filing_selector") not in filing_options:
        st.session_state["filing_selector"] = filing_options[0]

    selected_filing_name = st.selectbox(
        "Select Filing",
        options=filing_options,
        help="Choose which document to query",
        key="filing_selector",
    )
    selected_label = FILINGS[selected_filing_name]

    # ── Question mode ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Question Mode")
    question_mode = st.radio(
        "Input mode",
        options=["Custom Question", "Curated Questions"],
        help="Type your own question or pick from pre-built examples",
    )

    if question_mode == "Curated Questions":
        q_labels = [
            f"{DIFFICULTY_COLOR[q['difficulty']]} [{TYPE_LABEL[q['type']]}] {q['question']}"
            for q in QUESTIONS
        ]
        selected_q_idx = st.selectbox(
            "Select Question",
            options=range(len(QUESTIONS)),
            format_func=lambda i: q_labels[i],
        )
        selected_question = QUESTIONS[selected_q_idx]
        question_text = selected_question["question"]
        st.caption(f"**Why this question?** {selected_question['why_interesting']}")
    else:
        selected_question = None
        question_text = ""   # set below via form

    # ── About ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("About PageIndex")
    st.caption(
        "PageIndex replaces chunk-based vector search with a **hierarchical JSON tree** "
        "built from the document's natural structure (chapters \u2192 sections \u2192 subsections). "
        "Retrieval uses LLM reasoning to *navigate* the tree, not cosine similarity.\n\n"
        "**FinanceBench score:** 98.7% vs ChatGPT+Search at 31%."
    )

    # ── Session score tracker ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Session Score")
    if "scores" not in st.session_state:
        st.session_state.scores = {"trad": [], "pi": []}

    scores = st.session_state.scores
    if scores["trad"] or scores["pi"]:
        def _score_display(name, results):
            correct = results.count("CORRECT")
            partial = results.count("PARTIAL")
            incorrect = results.count("INCORRECT")
            total = len(results)
            pct = int(correct / total * 100) if total else 0
            st.caption(f"**{name}** \u2014 {correct}/{total} correct ({pct}%)")
            cols = st.columns(3)
            cols[0].metric("\u2713 Correct", correct)
            cols[1].metric("~ Partial", partial)
            cols[2].metric("\u2717 Wrong", incorrect)

        _score_display("Traditional RAG", scores["trad"])
        st.markdown("")
        _score_display("PageIndex RAG", scores["pi"])
        if st.button("Reset scores"):
            st.session_state.scores = {"trad": [], "pi": []}
            st.rerun()
    else:
        st.caption("Run some questions to see the score tally.")

# -- Main area ----------------------------------------------------------------
if selected_question:
    st.info(
        f"**{DIFFICULTY_COLOR[selected_question['difficulty']]} "
        f"{selected_question['difficulty'].title()} \u00b7 "
        f"{TYPE_LABEL[selected_question['type']]}** \u2014 {selected_question['explanation']}",
        icon="\u2139\ufe0f",
    )

# Custom question input as a form so Enter submits
if question_mode == "Custom Question":
    with st.form(key="question_form", border=False):
        question_text = st.text_input(
            "Your question",
            placeholder="e.g. What was the effective tax rate and why did it increase?",
        )
        run_clicked = st.form_submit_button(
            "\u25b6 Run Comparison",
            type="primary",
            use_container_width=True,
        )
else:
    st.markdown(f"**Question:** {question_text}")
    st.markdown("")
    run_col, _ = st.columns([1, 3])
    with run_col:
        run_clicked = st.button(
            "\u25b6 Run Comparison",
            type="primary",
            disabled=not question_text,
        )

st.divider()

# -- Results columns ----------------------------------------------------------
col_trad, col_pi = st.columns(2)

with col_trad:
    st.markdown(
        '<div class="pipeline-header-trad">\u26a1 Traditional RAG (Chunked + Vector Search)</div>',
        unsafe_allow_html=True,
    )
    st.caption("Text chunks \u2192 HuggingFace embeddings \u2192 ChromaDB \u2192 Groq LLaMA 3.3 70B")
    st.markdown("")
    trad_answer_ph = st.empty()
    trad_meta_ph = st.empty()
    trad_chunks_ph = st.empty()

with col_pi:
    st.markdown(
        '<div class="pipeline-header-pi">\U0001f9e0 PageIndex RAG (Tree Search + LLM Reasoning)</div>',
        unsafe_allow_html=True,
    )
    st.caption("Hierarchical tree index \u2192 LLM tree navigation \u2192 cited answer")
    st.markdown("")
    pi_answer_ph = st.empty()
    pi_meta_ph = st.empty()
    pi_nodes_ph = st.empty()

# Empty state
if not run_clicked:
    for ph in (trad_answer_ph, pi_answer_ph):
        with ph:
            st.markdown(
                '<div class="answer-box" style="color:#aaa;font-style:italic;">Answer will appear here...</div>',
                unsafe_allow_html=True,
            )

# -- Run pipelines ------------------------------------------------------------
if run_clicked and question_text:
    trad_mod = _load_trad_pipeline()
    pi_mod = _load_pi_pipeline()

    # Traditional RAG
    with trad_answer_ph:
        with st.spinner("Traditional RAG thinking..."):
            trad_result = trad_mod.query(selected_label, question_text)

    with trad_answer_ph:
        st.markdown(
            f'<div class="answer-box">{_render_answer(trad_result["answer"])}</div>',
            unsafe_allow_html=True,
        )
    with trad_meta_ph:
        badge = _accuracy_badge(trad_result["answer"], selected_question)
        st.markdown(
            f'<div class="latency-box">\u23f1 {trad_result["latency_s"]}s &nbsp;|&nbsp; '
            f'{len(trad_result["chunks"])} chunks retrieved &nbsp; {badge}</div>',
            unsafe_allow_html=True,
        )
    with trad_chunks_ph:
        with st.expander("Retrieved chunks", expanded=True):
            for i, chunk in enumerate(trad_result["chunks"], 1):
                preview = _safe(chunk["text"][:250].replace("\n", " "))
                st.markdown(
                    f'<div class="chunk-card"><b>Chunk {i}</b> &nbsp;\u00b7&nbsp; '
                    f'similarity {chunk["similarity"]}<br><br>{preview}\u2026</div>',
                    unsafe_allow_html=True,
                )

    # PageIndex RAG
    pi_result = None
    with pi_answer_ph:
        with st.spinner("PageIndex navigating document tree..."):
            try:
                pi_result = pi_mod.query(selected_label, question_text)
            except RuntimeError as e:
                if "LimitReached" in str(e) or "page limit" in str(e).lower():
                    st.warning(str(e))
                else:
                    st.error(f"PageIndex error: {e}")
            except Exception as e:
                st.error(f"PageIndex error: {e}")

    if pi_result is not None:
        with pi_answer_ph:
            st.markdown(
                f'<div class="answer-box">{_render_answer(pi_result["answer"])}</div>',
                unsafe_allow_html=True,
            )
        with pi_meta_ph:
            badge = _accuracy_badge(pi_result["answer"], selected_question)
            st.markdown(
                f'<div class="latency-box">⏱ {pi_result["latency_s"]}s &nbsp;|&nbsp; '
                f'{len(pi_result["nodes_visited"])} nodes visited &nbsp; {badge}</div>',
                unsafe_allow_html=True,
            )
        with pi_nodes_ph:
            with st.expander("Tree nodes visited", expanded=True):
                for node in pi_result["nodes_visited"]:
                    preview = _safe(_node_text(node))
                    st.markdown(
                        f'<div class="node-card"><b>{_safe(node["title"])}</b>'
                        f'&nbsp;·&nbsp; node {_safe(node["node_id"])}<br><br>{preview}…</div>',
                        unsafe_allow_html=True,
                    )

    # Ground truth reveal
    if selected_question:
        st.divider()
        st.markdown("**Ground Truth Answer:**")
        st.success(selected_question["ground_truth"])

        if "scores" not in st.session_state:
            st.session_state.scores = {"trad": [], "pi": []}
        st.session_state.scores["trad"].append(_badge_label(trad_result["answer"], selected_question))
        if pi_result is not None:
            st.session_state.scores["pi"].append(_badge_label(pi_result["answer"], selected_question))
