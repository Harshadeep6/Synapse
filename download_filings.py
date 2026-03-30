"""
Step 2: Download Apple 10-K from SEC EDGAR and parse it.
Run: python download_filings.py
"""
import os
import re
import requests
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch

DATA_DIR = Path(__file__).parent / "data" / "filings"
DATA_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "EarningsLens research@earningslens.com"}

FILINGS = [
    {
        "company": "Apple",
        "ticker": "AAPL",
        "fiscal_year": "FY2024",
        "cik": "0000320193",
        "accession": "0000320193-24-000123",
        "primary_doc": "aapl-20240928.htm",
    },
    {
        "company": "Microsoft",
        "ticker": "MSFT",
        "fiscal_year": "FY2024",
        "cik": "0000789019",
        "accession": "0000950170-24-087843",
        "primary_doc": "msft-20240630.htm",
    },
]


def fetch_filing_text(filing: dict) -> str:
    accession_nodash = filing["accession"].replace("-", "")
    cik_nodash = filing["cik"].lstrip("0")
    url = (
        f"https://www.sec.gov/Archives/edgar/data/{cik_nodash}/"
        f"{accession_nodash}/{filing['primary_doc']}"
    )
    print(f"  Downloading from: {url}")
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    return resp.text


def html_to_text(html: str) -> str:
    # Remove scripts and styles
    html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    # Replace block elements with newlines
    html = re.sub(r"<(p|div|tr|br|h[1-6])[^>]*>", "\n", html, flags=re.IGNORECASE)
    html = re.sub(r"<td[^>]*>", "\t", html, flags=re.IGNORECASE)
    # Strip all remaining tags
    html = re.sub(r"<[^>]+>", "", html)
    # Decode HTML entities using html module for completeness
    import html as html_module
    html = html_module.unescape(html)
    # Collapse excess whitespace but preserve paragraph breaks
    html = re.sub(r"[ \t]+", " ", html)
    html = re.sub(r"\n{3,}", "\n\n", html)
    # Strip XBRL/metadata header — real content starts at "UNITED STATES"
    marker = "UNITED STATES"
    idx = html.find(marker)
    if idx > 0:
        html = html[idx:]
    return html.strip()


def text_to_pdf(text: str, pdf_path: Path):
    """Convert plain text to a PDF using reportlab."""
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    styles = getSampleStyleSheet()
    normal = styles["Normal"]
    normal.fontSize = 9
    normal.leading = 12

    story = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        # Escape XML special chars for reportlab
        para = para.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        try:
            story.append(Paragraph(para, normal))
            story.append(Spacer(1, 4))
        except Exception:
            pass  # skip any unparseable paragraphs

    doc.build(story)


def download_all():
    for filing in FILINGS:
        label = f"{filing['ticker']}_{filing['fiscal_year']}"
        txt_path = DATA_DIR / f"{label}.txt"
        pdf_path = DATA_DIR / f"{label}.pdf"

        if txt_path.exists():
            print(f"[OK] {label} txt: already downloaded ({txt_path.stat().st_size // 1024} KB)")
        else:
            print(f"\nDownloading {filing['company']} {filing['fiscal_year']} 10-K...")
            html = fetch_filing_text(filing)
            text = html_to_text(html)
            txt_path.write_text(text, encoding="utf-8")
            print(f"[OK] {label}: saved {txt_path.stat().st_size // 1024} KB -> {txt_path}")

        if pdf_path.exists():
            print(f"[OK] {label} pdf: already exists ({pdf_path.stat().st_size // 1024} KB)")
        else:
            print(f"  Converting {label} to PDF...")
            text = txt_path.read_text(encoding="utf-8")
            text_to_pdf(text, pdf_path)
            print(f"[OK] {label}: PDF saved {pdf_path.stat().st_size // 1024} KB -> {pdf_path}")


if __name__ == "__main__":
    print("=== Step 2: Download & Parse SEC 10-K Filings ===\n")
    download_all()

    # Checkpoint: show stats for each file
    print("\n--- Checkpoint ---")
    for path in sorted(DATA_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        non_empty = [l for l in lines if l.strip()]
        print(f"\n{path.name}")
        print(f"  Size: {path.stat().st_size // 1024} KB | Lines: {len(lines)} | Non-empty lines: {len(non_empty)}")
        preview = text[:300].encode('ascii', errors='replace').decode('ascii')
        print(f"  First 300 chars:\n  {preview!r}")
