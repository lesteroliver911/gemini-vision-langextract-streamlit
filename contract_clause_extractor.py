from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import textwrap
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Tuple
import csv
import re

from dotenv import load_dotenv
import requests
try:
    from google import genai  # type: ignore[import-not-found]
    from google.genai import types as genai_types  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]

# Third-party deps
import langextract as lx

# LexNLP removed


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def load_env() -> None:
    load_dotenv()


# Provider detection removed (Gemini-only)


def build_legal_prompt() -> str:
    return textwrap.dedent(
        """
        Extract legal entities and clauses from the contract:
        - Parties (name, role), Effective dates, Initial term, Auto-renewal, Notice windows
        - Fees (amount, currency), Payment terms, Price increases
        - Liability cap (amount/basis), Indemnity scope, Warranties
        - Confidentiality, HIPAA/Data Processing (DPA), Security addendum, Governing law, Jurisdiction
        - Termination (for cause/convenience), Non-appropriation (automatic cancellation), SLAs, Service scope
        - SLA metrics: uptime %, response time targets, escalation times, penalties/credits, support hours
        - Records retention and inspection, Audit rights, Loss of data responsibilities

        Important:
        - Use exact text spans from the input for extraction_text (no paraphrasing)
        - Preserve order of appearance and avoid overlapping spans
        - Always include a concise attributes dict for context
        """
    ).strip()


def build_legal_examples() -> List[Any]:  # LangExtract ExampleData list
    return [
        lx.data.ExampleData(
            text=(
                "This Agreement shall commence on January 1, 2025 and continue for an initial term of two years. "
                "The parties agree that liability is capped at fees paid in the preceding twelve months. "
                "This Agreement is governed by the laws of the State of Maryland. "
                "Either party may terminate for convenience upon thirty (30) days' written notice. "
                "Confidentiality obligations survive termination for three (3) years. "
                "If funds are not appropriated for a subsequent fiscal period, this Contract shall be canceled automatically. "
                "Contractor shall hold harmless and indemnify the State against claims arising from Contractor performance. "
                "Contractor shall retain records for five (5) years and make them available for audit and inspection. "
                "Contractor shall comply with HIPAA and the Maryland Confidentiality of Medical Records Act."
            ),
            extractions=[
                lx.data.Extraction(
                    extraction_class="term",
                    extraction_text="commence on January 1, 2025",
                    attributes={"effective_date": "January 1, 2025", "initial_term": "two years"},
                ),
                lx.data.Extraction(
                    extraction_class="liability_cap",
                    extraction_text="liability is capped at fees paid in the preceding twelve months",
                    attributes={"basis": "fees paid in preceding 12 months"},
                ),
                lx.data.Extraction(
                    extraction_class="governing_law",
                    extraction_text="governed by the laws of the State of Maryland",
                    attributes={"jurisdiction": "Maryland"},
                ),
                lx.data.Extraction(
                    extraction_class="termination",
                    extraction_text="terminate for convenience upon thirty (30) days' written notice",
                    attributes={"type": "for convenience", "notice_window": "30 days"},
                ),
                lx.data.Extraction(
                    extraction_class="confidentiality",
                    extraction_text="Confidentiality obligations survive termination for three (3) years",
                    attributes={"survival": "3 years"},
                ),
                lx.data.Extraction(
                    extraction_class="termination",
                    extraction_text="If funds are not appropriated for a subsequent fiscal period, this Contract shall be canceled automatically",
                    attributes={"type": "automatic cancellation", "trigger": "non-appropriation"},
                ),
                lx.data.Extraction(
                    extraction_class="indemnification",
                    extraction_text="hold harmless and indemnify the State against claims arising from Contractor performance",
                    attributes={"scope": "claims arising from performance"},
                ),
                lx.data.Extraction(
                    extraction_class="records",
                    extraction_text="retain records for five (5) years and make them available for audit and inspection",
                    attributes={"retention": "5 years", "rights": "audit and inspection"},
                ),
                lx.data.Extraction(
                    extraction_class="confidentiality",
                    extraction_text="comply with HIPAA and the Maryland Confidentiality of Medical Records Act",
                    attributes={"regimes": ["HIPAA", "Maryland Medical Records Act"]},
                ),
            ],
        )
        ,
        lx.data.ExampleData(
            text=(
                "Provider shall maintain 99.9% uptime measured monthly. "
                "For Severity 1 incidents, initial response within one (1) hour and resolution within four (4) hours. "
                "Service credits: 5% of monthly fee per additional 30 minutes of downtime beyond SLA."
            ),
            extractions=[
                lx.data.Extraction(
                    extraction_class="sla",
                    extraction_text="maintain 99.9% uptime",
                    attributes={"uptime_percent": "99.9"},
                ),
                lx.data.Extraction(
                    extraction_class="sla",
                    extraction_text="initial response within one (1) hour",
                    attributes={"response_time": "1 hour", "severity": "1"},
                ),
                lx.data.Extraction(
                    extraction_class="sla",
                    extraction_text="5% of monthly fee per additional 30 minutes of downtime",
                    attributes={"credit_policy": "5% per 30 minutes overage"},
                ),
            ],
        )
    ]


# LexNLP clause pre-pass removed


def extract_with_langextract(
    text_or_documents: str,
    model_id: str,
    extraction_passes: int,
    max_workers: int,
    max_char_buffer: int,
) -> Any:
    kwargs: Dict[str, Any] = {
        "text_or_documents": text_or_documents,
        "prompt_description": build_legal_prompt(),
        "examples": build_legal_examples(),
        "model_id": model_id,
        "extraction_passes": extraction_passes,
        "max_workers": max_workers,
        "max_char_buffer": max_char_buffer,
    }
    # Gemini-only
    if not os.getenv("LANGEXTRACT_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        raise EnvironmentError("Set LANGEXTRACT_API_KEY or GEMINI_API_KEY for Gemini models")

    try:
        return lx.extract(**kwargs)
    except Exception as exc:  # noqa: BLE001
        logging.error("Extraction failed: %s", exc)
        raise


def save_outputs(result: Any, base_output: str) -> Tuple[Path, Path]:
    jsonl_name = f"{base_output}.jsonl"
    lx.io.save_annotated_documents([result], output_name=jsonl_name)
    jsonl_path = Path("test_output") / jsonl_name
    html = lx.visualize(str(jsonl_path))
    html_path = Path("test_output") / f"{base_output}.html"
    html_path.write_text(html, encoding="utf-8")
    return jsonl_path.resolve(), html_path.resolve()


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract legal clauses and structured entities from contracts",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", type=str, help="Inline text to process")
    src.add_argument("--file", type=str, help="Path to a local file (.txt/.pdf/.docx/.html)")
    src.add_argument("--url", type=str, help="URL to fetch (txt/html)")

    parser.add_argument("--model", type=str, default="gemini-2.5-pro", help="Model ID")
    parser.add_argument("--passes", type=int, default=1, help="Extraction passes (default: 1)")
    parser.add_argument("--max-workers", type=int, default=1, help="Parallel workers (default: 1)")
    parser.add_argument("--buffer", type=int, default=4000, help="Max char buffer (default: 4000)")
    parser.add_argument("--output", type=str, default="contract_extraction", help="Base output name")
    parser.add_argument("--csv", action="store_true", help="Also export a CSV of extractions")
    parser.add_argument("--pdf-password", type=str, default=None, help="Password for protected PDFs")
    parser.add_argument("--vision", action="store_true", help="Use Gemini vision on PDFs (bypass OCR)")
    return parser.parse_args(argv)


def _extract_text_from_pdf(path: Path, password: str | None = None) -> str:
    # Prefer PyMuPDF for robust layout-aware text extraction
    try:
        import fitz  # PyMuPDF  # noqa: WPS433
        text_chunks: List[str] = []
        with fitz.open(path) as doc:  # type: ignore[arg-type]
            if password:
                try:
                    doc.authenticate(password)
                except Exception:
                    pass
            for page in doc:  # type: ignore[attr-defined]
                # "text" mode preserves reading order reasonably well
                text_chunks.append(page.get_text("text"))
        text = "\n\n".join(chunk for chunk in text_chunks if chunk)
        if text.strip():
            return text
    except Exception:
        pass

    # Fallback: pdfplumber (pdfminer.six under the hood)
    try:
        import pdfplumber  # noqa: WPS433
        text_chunks = []
        with pdfplumber.open(path, password=password) as pdf:
            for p in pdf.pages:
                extracted = p.extract_text() or ""
                text_chunks.append(extracted)
        return "\n\n".join(text_chunks)
    except Exception as exc:  # noqa: BLE001
        logging.error("PDF text extraction failed: %s", exc)
        raise


def _extract_text_from_docx(path: Path) -> str:
    try:
        from docx import Document  # python-docx  # noqa: WPS433
    except Exception as exc:  # noqa: BLE001
        logging.error("python-docx not available: %s. Install with: pip install python-docx", exc)
        raise

    doc = Document(str(path))
    parts: List[str] = []
    # Paragraphs
    parts.extend(p.text for p in doc.paragraphs if p.text)
    # Tables
    for table in getattr(doc, "tables", []):
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                parts.append("\t".join(cells))
    return "\n".join(parts)


def _extract_text_from_html_file(path: Path) -> str:
    # Try trafilatura for boilerplate removal
    try:
        import trafilatura  # noqa: WPS433
        raw_html = path.read_text(encoding="utf-8", errors="ignore")
        extracted = trafilatura.extract(raw_html)
        if extracted and extracted.strip():
            return extracted
    except Exception:
        pass

    # Fallback: BeautifulSoup text get
    try:
        from bs4 import BeautifulSoup  # noqa: WPS433
        html = path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "lxml") if html else None
        return soup.get_text(separator="\n") if soup else ""
    except Exception as exc:  # noqa: BLE001
        logging.error("HTML text extraction failed: %s", exc)
        raise


def _extract_text_from_file(path: Path, password: str | None = None) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_text_from_pdf(path, password=password)
    if suffix == ".docx":
        return _extract_text_from_docx(path)
    if suffix in {".html", ".htm"}:
        return _extract_text_from_html_file(path)
    # Default: assume UTF-8 text
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_text_from_url(url: str) -> str:
    lower = url.lower()
    # PDFs: download then extract
    if lower.endswith(".pdf"):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            tmp.write(resp.content)
            tmp_path = Path(tmp.name)
        try:
            return _extract_text_from_pdf(tmp_path)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
    # HTML/text: use trafilatura fetch
    try:
        import trafilatura  # noqa: WPS433
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            extracted = trafilatura.extract(downloaded)
            if extracted and extracted.strip():
                return extracted
    except Exception:
        pass
    # Fallback to requests + bs4
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    content_type = resp.headers.get("content-type", "").lower()
    if "pdf" in content_type:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(resp.content)
            tmp_path = Path(tmp.name)
        try:
            return _extract_text_from_pdf(tmp_path)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
    try:
        from bs4 import BeautifulSoup  # noqa: WPS433
        soup = BeautifulSoup(resp.text, "lxml")
        return soup.get_text(separator="\n")
    except Exception:
        return resp.text


def _gemini_vision_extract_pdf(path: Path, prompt: str) -> str:
    if genai is None or genai_types is None:
        raise RuntimeError("google-genai is not available. Install with: pip install google-genai")
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("LANGEXTRACT_API_KEY")
    if not api_key:
        raise EnvironmentError("Set GEMINI_API_KEY (or LANGEXTRACT_API_KEY) for google-genai")

    client = genai.Client(api_key=api_key)
    pdf_bytes = path.read_bytes()
    contents = [
        genai_types.Content(
            role="user",
            parts=[
                genai_types.Part.from_text(text=prompt),
                genai_types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
            ],
        ),
    ]
    generate_content_config = genai_types.GenerateContentConfig(
        thinking_config=genai_types.ThinkingConfig(thinking_budget=8192),
        media_resolution="MEDIA_RESOLUTION_MEDIUM",
    )

    # Ask for strict JSON to ease parsing
    system_suffix = "\nReturn a JSON array of objects with keys: extraction_class, extraction_text, attributes."
    contents[0].parts[0] = genai_types.Part.from_text(text=prompt + system_suffix)

    buffer: List[str] = []
    for chunk in client.models.generate_content_stream(
        model="gemini-2.5-pro",
        contents=contents,
        config=generate_content_config,
    ):
        if getattr(chunk, "text", None):
            buffer.append(chunk.text)

    return "".join(buffer)


def _parse_gemini_json(text: str) -> List[Dict[str, Any]]:
    # Try to find a JSON array in the response
    try:
        import json as _json  # noqa: WPS433
        # Heuristic: find first '[' and last ']'
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            return _json.loads(text[start : end + 1])
        # Fallback: single JSON object
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return [_json.loads(text[start : end + 1])]
    except Exception:
        pass
    return []


def _save_gemini_results_as_outputs(rows: List[Dict[str, Any]], base_output: str) -> Tuple[Path, Path, Path]:
    # Save as JSONL compatible with our downstream viz minimal expectation
    jsonl_path = Path("test_output") / f"{base_output}.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    doc_obj = {"extractions": rows, "text": ""}
    jsonl_path.write_text(json.dumps(doc_obj), encoding="utf-8")

    # CSV
    csv_path = Path("test_output") / f"{base_output}.csv"
    export_rows = []
    for r in rows:
        export_rows.append({
            "extraction_class": r.get("extraction_class"),
            "extraction_text": r.get("extraction_text"),
            "attributes_json": json.dumps(r.get("attributes"), ensure_ascii=False),
            "start_pos": "",
            "end_pos": "",
            "alignment_status": "",
        })
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "extraction_class", "extraction_text", "attributes_json", "start_pos", "end_pos", "alignment_status"
        ])
        writer.writeheader()
        writer.writerows(export_rows)

    # Minimal HTML using existing visualize if possible, else basic list
    try:
        html = lx.visualize(str(jsonl_path))
    except Exception:
        items = "\n".join(
            f"<li><b>{r.get('extraction_class')}</b>: {r.get('extraction_text')}</li>" for r in rows
        )
        html = f"<html><body><ul>{items}</ul></body></html>"
    html_path = Path("test_output") / f"{base_output}.html"
    html_path.write_text(html, encoding="utf-8")
    return jsonl_path.resolve(), html_path.resolve(), csv_path.resolve()


def read_input_payload(args: argparse.Namespace) -> str:
    if args.text:
        return args.text
    if args.url:
        return _extract_text_from_url(args.url)
    assert args.file, "No input provided"
    path = Path(args.file)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return _extract_text_from_file(path, password=args.pdf_password)


def export_extractions_to_csv(result: Any, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Generic serializer to avoid schema assumptions
    fieldnames = [
        "extraction_class",
        "extraction_text",
        "attributes_json",
        "start_pos",
        "end_pos",
        "alignment_status",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for e in getattr(result, "extractions", []) or []:
            try:
                # Attempt to read nested char_interval if available
                char_interval = getattr(e, "char_interval", None)
                start_pos = getattr(char_interval, "start_pos", None)
                end_pos = getattr(char_interval, "end_pos", None)
                if start_pos is None and isinstance(char_interval, dict):
                    start_pos = char_interval.get("start_pos")
                    end_pos = char_interval.get("end_pos")
                alignment_status = getattr(e, "alignment_status", None)
                writer.writerow({
                    "extraction_class": getattr(e, "extraction_class", ""),
                    "extraction_text": getattr(e, "extraction_text", ""),
                    "attributes_json": json.dumps(getattr(e, "attributes", None), ensure_ascii=False),
                    "start_pos": start_pos,
                    "end_pos": end_pos,
                    "alignment_status": alignment_status,
                })
            except Exception:  # noqa: BLE001
                # Best-effort row; fallback to string repr
                writer.writerow({
                    "extraction_class": str(getattr(e, "extraction_class", "")),
                    "extraction_text": str(getattr(e, "extraction_text", "")),
                    "attributes_json": json.dumps(getattr(e, "attributes", None), ensure_ascii=False),
                    "start_pos": "",
                    "end_pos": "",
                    "alignment_status": "",
                })


def main(argv: List[str]) -> int:
    configure_logging()
    load_env()
    args = parse_args(argv)

    payload = read_input_payload(args)

    # Clause pre-pass removed (LexNLP)

    # If vision mode and input is a PDF file, bypass OCR and use google-genai directly
    used_vision = False
    if args.vision and args.file and Path(args.file).suffix.lower() == ".pdf":
        try:
            vision_text = _gemini_vision_extract_pdf(Path(args.file), build_legal_prompt())
            rows = _parse_gemini_json(vision_text)
            if rows:
                jsonl_path, html_path, csv_path = _save_gemini_results_as_outputs(rows, args.output)
                logging.info("Saved JSONL → %s", jsonl_path)
                logging.info("Saved HTML  → %s", html_path)
                logging.info("Saved CSV   → %s", csv_path)
                used_vision = True
            else:
                logging.warning("Vision returned no parseable JSON; falling back to text pipeline")
        except Exception as exc:  # noqa: BLE001
            logging.warning("Vision path failed (%s); falling back to text pipeline", exc)

    if not used_vision:
        # LangExtract structured pass
        result = extract_with_langextract(
            text_or_documents=payload,
            model_id=args.model,
            extraction_passes=args.passes,
            max_workers=args.max_workers,
            max_char_buffer=args.buffer,
        )
        jsonl_path, html_path = save_outputs(result, args.output)
        logging.info("Saved JSONL → %s", jsonl_path)
        logging.info("Saved HTML  → %s", html_path)
        if args.csv:
            csv_path = Path("test_output") / f"{args.output}.csv"
            export_extractions_to_csv(result, csv_path)
            logging.info("Saved CSV   → %s", csv_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))