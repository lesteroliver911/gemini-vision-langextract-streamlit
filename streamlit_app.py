from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as st_html


TEST_OUTPUT_DIR = Path("test_output")
PROJECT_ROOT = Path(__file__).parent


@dataclass
class ExtractionRow:
    extraction_class: str
    extraction_text: str
    attributes: Dict[str, Any]
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None
    alignment_status: Optional[str] = None


def read_jsonl_result(jsonl_path: Path) -> Tuple[List[ExtractionRow], Dict[str, Any]]:
    data = json.loads(jsonl_path.read_text(encoding="utf-8"))
    rows: List[ExtractionRow] = []
    for item in data.get("extractions", []) or []:
        rows.append(
            ExtractionRow(
                extraction_class=str(item.get("extraction_class", "")),
                extraction_text=str(item.get("extraction_text", "")),
                attributes=item.get("attributes") or {},
                start_pos=(item.get("char_interval", {}) or {}).get("start_pos"),
                end_pos=(item.get("char_interval", {}) or {}).get("end_pos"),
                alignment_status=item.get("alignment_status"),
            )
        )
    return rows, data


def to_dataframe(rows: List[ExtractionRow]) -> pd.DataFrame:
    records = []
    for r in rows:
        records.append(
            {
                "extraction_class": r.extraction_class,
                "extraction_text": r.extraction_text,
                "attributes": json.dumps(r.attributes, ensure_ascii=False),
                "start_pos": r.start_pos,
                "end_pos": r.end_pos,
                "alignment_status": r.alignment_status,
            }
        )
    return pd.DataFrame.from_records(records)


def list_available_results() -> List[Path]:
    # Deprecated in simplified UI; kept for future use
    return sorted(TEST_OUTPUT_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True) if TEST_OUTPUT_DIR.exists() else []


def run_extractor_on_pdf(
    pdf_path: Path,
    output_name: str,
    model: str = "gemini-2.5-pro",
    use_vision: bool = True,
    passes: int = 1,
    max_workers: int = 1,
    buffer: int = 4000,
) -> Tuple[Path, Path, Path]:
    # Ensure env key is present
    env = os.environ.copy()
    if not env.get("GEMINI_API_KEY") and env.get("LANGEXTRACT_API_KEY"):
        env["GEMINI_API_KEY"] = env["LANGEXTRACT_API_KEY"]

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "contract_clause_extractor.py"),
        "--file",
        str(pdf_path),
        "--model",
        model,
        "--passes",
        str(passes),
        "--max-workers",
        str(max_workers),
        "--buffer",
        str(buffer),
        "--output",
        output_name,
        "--csv",
    ]
    if use_vision:
        cmd.append("--vision")

    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Extraction failed: {proc.stderr or proc.stdout}")

    jsonl = TEST_OUTPUT_DIR / f"{output_name}.jsonl"
    html = TEST_OUTPUT_DIR / f"{output_name}.html"
    csv = TEST_OUTPUT_DIR / f"{output_name}.csv"
    return jsonl, html, csv


def main() -> None:
    st.set_page_config(page_title="Contract Intelligence UI", layout="wide")
    st.title("Contract Intelligence UI")
    st.caption("Gemini 2.5 Pro Vision for PDFs + LangExtract")

    with st.sidebar:
        st.subheader("Settings")
        default_model = "gemini-2.5-pro"
        model = st.text_input("Model ID", value=default_model)
        use_vision = st.checkbox("Use Vision for PDFs", value=True)
        st.subheader("Upload")
        uploaded_pdf_bytes: Optional[bytes] = None
        uploaded_pdf_name: Optional[str] = None
        uf = st.file_uploader("Upload contract PDF", type=["pdf"])
        if uf is not None:
            uploaded_pdf_bytes = uf.read()
            uploaded_pdf_name = uf.name
        st.divider()
        run_btn = st.button("Process PDF")

    jsonl_path: Optional[Path] = None
    html_path: Optional[Path] = None
    csv_path: Optional[Path] = None

    if run_btn:
        if not uploaded_pdf_bytes:
            st.error("Upload a PDF.")
            st.stop()
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_name = f"ui_run_{ts}"
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(uploaded_pdf_bytes)
            tmp_path = Path(tmp.name)
        try:
            jsonl_path, html_path, csv_path = run_extractor_on_pdf(
                pdf_path=tmp_path,
                output_name=out_name,
                model=model,
                use_vision=use_vision,
            )
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    if not run_btn:
        st.info("Upload a PDF, then click Process PDF.")
        st.stop()

    if not jsonl_path or not jsonl_path.exists():
        st.error("No JSONL found.")
        st.stop()

    rows, doc = read_jsonl_result(jsonl_path)
    df = to_dataframe(rows)

    st.subheader("Summary")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total extractions", len(rows))
    with c2:
        st.metric("Unique classes", df["extraction_class"].nunique())
    with c3:
        st.metric("Result set", jsonl_path.name)

    st.subheader("Explore")
    classes = sorted(df["extraction_class"].unique().tolist())
    selected_classes = st.multiselect("Filter by class", options=classes, default=classes)
    filtered = df[df["extraction_class"].isin(selected_classes)] if selected_classes else df
    st.dataframe(filtered, use_container_width=True, height=400)

    st.subheader("Details by class")
    for cls in selected_classes:
        with st.expander(f"{cls} ({(filtered['extraction_class'] == cls).sum()})", expanded=False):
            subset = filtered[filtered["extraction_class"] == cls]
            for i, row in subset.iterrows():
                st.markdown(f"**Text**: {row['extraction_text']}")
                try:
                    attrs = json.loads(row["attributes"]) if row["attributes"] else {}
                except Exception:
                    attrs = {"raw": row["attributes"]}
                st.json(attrs)
                st.markdown("---")

    st.subheader("Downloads")
    cdl1, cdl2 = st.columns(2)
    with cdl1:
        filtered_csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered CSV", filtered_csv, file_name="extractions_filtered.csv")
    with cdl2:
        filtered_json = json.dumps(filtered.to_dict(orient="records"), ensure_ascii=False).encode("utf-8")
        st.download_button("Download filtered JSON", filtered_json, file_name="extractions_filtered.json")

    if html_path and html_path.exists():
        st.subheader("Interactive Visualization")
        html_str = html_path.read_text(encoding="utf-8")
        st_html(html_str, height=600)


if __name__ == "__main__":
    main()


