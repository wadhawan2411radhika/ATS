"""
File Parser Utility.

Extracts clean text from PDF, DOCX, and TXT resume files.
Handles edge cases: multi-column PDFs, table-based DOCX layouts, mixed content.
"""

import logging
from pathlib import Path

import pdfplumber
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}


def _extract_pdf(path: str) -> str:
    """
    Extract text from PDF using pdfplumber.
    Handles multi-column layouts by extracting text with layout preservation.
    """
    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            # Try layout-aware extraction first
            text = page.extract_text(x_tolerance=3, y_tolerance=3)
            if text and text.strip():
                texts.append(text)
            else:
                # Fallback: extract words and reconstruct
                words = page.extract_words()
                if words:
                    texts.append(" ".join(w["text"] for w in words))
    return "\n".join(texts)


def _iter_docx_blocks(doc: Document):
    """
    Yield all content blocks from a DOCX in document order,
    including paragraphs inside tables (handles table-based CV layouts).
    """
    body = doc.element.body
    for child in body:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "p":
            para = Paragraph(child, doc)
            yield "para", para.text
        elif tag == "tbl":
            table = Table(child, doc)
            for row in table.rows:
                row_texts = []
                for cell in row.cells:
                    cell_text = cell.text.strip()
                    if cell_text:
                        row_texts.append(cell_text)
                if row_texts:
                    yield "table_row", " | ".join(row_texts)


def _extract_docx(path: str) -> str:
    """
    Extract text from DOCX, preserving content from both paragraphs and tables.
    Many CV templates use tables for layout — naive paragraph-only extraction misses this.
    """
    doc = Document(path)
    lines = []
    seen = set()  # Deduplicate — tables sometimes repeat cell content

    for block_type, text in _iter_docx_blocks(doc):
        text = text.strip()
        if text and text not in seen:
            lines.append(text)
            seen.add(text)

    return "\n".join(lines)


def _extract_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def parse_file(path: str) -> str:
    """
    Parse a resume file and return clean extracted text.

    Supports: .pdf, .docx, .doc, .txt

    Args:
        path: Absolute or relative path to the file.

    Returns:
        Extracted text as a string.

    Raises:
        ValueError: If file extension is not supported.
        FileNotFoundError: If file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = p.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: '{ext}'. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    logger.debug(f"Parsing {ext} file: {p.name}")

    if ext == ".pdf":
        text = _extract_pdf(path)
    elif ext in {".docx", ".doc"}:
        text = _extract_docx(path)
    elif ext == ".txt":
        text = _extract_txt(path)
    else:
        text = ""

    if not text.strip():
        logger.warning(f"Extracted empty text from {p.name}")

    logger.info(f"Parsed '{p.name}': {len(text)} chars, ~{len(text.split())} words")
    return text.strip()


def load_resumes_from_dir(directory: str) -> dict[str, str]:
    """
    Load all supported resume files from a directory.
    Returns dict of {stem_filename: extracted_text}.
    Files with unsupported extensions are silently skipped.
    """
    resumes = {}
    dir_path = Path(directory)

    for filepath in sorted(dir_path.iterdir()):
        if filepath.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            text = parse_file(str(filepath))
            if text:
                resumes[filepath.stem] = text
            else:
                logger.warning(f"Skipping empty file: {filepath.name}")
        except Exception as e:
            logger.error(f"Failed to parse {filepath.name}: {e}")

    logger.info(f"Loaded {len(resumes)} resumes from {directory}")
    return resumes
