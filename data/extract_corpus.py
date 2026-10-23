"""
Extract text from scraped EIB ESDS PDFs and save as a structured corpus.

Reads PDFs from ../data/pdfs/ and metadata from ../data/esds_metadata.csv,
extracts text, optionally applies year-stratified sampling, splits into
sentences, and saves parquet outputs plus a selected-doc manifest.

Usage:
    python extract_corpus.py --docs-per-year 50 --year-from 2014 --year-to 2025 --output-dir for_tuesday/data_50py
"""

import argparse
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import fitz  # pymupdf
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(__file__).parent.parent
PDF_DIR = ROOT_DIR / "data" / "pdfs"
META_PATH = ROOT_DIR / "data" / "esds_metadata.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "data"


@dataclass(frozen=True)
class ExtractionTask:
    """Metadata for one PDF extraction task."""

    pdf_path: Path
    doc_id: str
    title: str
    year: int
    sector: str
    region: str
    country: str


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    except Exception as e:
        print(f"  Error reading {pdf_path.name}: {e}")
        return ""


def split_into_sentences(text: str) -> list[str]:
    """Simple sentence splitter for structured PDF documents."""
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if 20 < len(s.strip()) < 2000]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Extract EIB corpus to parquet.")
    parser.add_argument(
        "--year-from",
        type=int,
        default=2014,
        help="First year to include (default: 2014).",
    )
    parser.add_argument(
        "--year-to",
        type=int,
        default=2025,
        help="Last year to include (default: 2025).",
    )
    parser.add_argument(
        "--docs-per-year",
        type=int,
        default=None,
        help="If set, sample up to N docs per year (default: use all).",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed for deterministic per-year sampling (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for parquet files (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--manifest-name",
        type=str,
        default="selected_docs.csv",
        help="Name of manifest CSV with selected docs (default: selected_docs.csv).",
    )
    return parser.parse_args()


def load_metadata(meta_path: Path) -> pd.DataFrame:
    """Load metadata and ensure a `year` column exists."""
    meta = pd.read_csv(meta_path)
    if "year" not in meta.columns:
        meta["date"] = pd.to_datetime(meta["publication_date"], format="%d %b %Y")
        meta["year"] = meta["date"].dt.year
    return meta


def build_tasks(meta: pd.DataFrame) -> list[ExtractionTask]:
    """Build extraction tasks for metadata rows with existing PDFs."""
    tasks: list[ExtractionTask] = []
    for _, row in meta.iterrows():
        pdf_path = PDF_DIR / f"{row['doc_id']}.pdf"
        if not pdf_path.exists():
            continue
        tasks.append(
            ExtractionTask(
                pdf_path=pdf_path,
                doc_id=str(row["doc_id"]),
                title=str(row.get("title", "")),
                year=int(row["year"]),
                sector=str(row.get("sector", "")),
                region=str(row.get("region", "")),
                country=str(row.get("country", "")),
            )
        )
    return tasks


def stratified_sample_tasks(
    tasks: list[ExtractionTask],
    docs_per_year: int | None,
    year_from: int,
    year_to: int,
    seed: int,
) -> list[ExtractionTask]:
    """
    Deterministically sample tasks per year.

    If docs_per_year is None, returns all tasks sorted by year/doc_id.
    """
    tasks_sorted = sorted(tasks, key=lambda t: (t.year, t.doc_id))
    if docs_per_year is None:
        return [t for t in tasks_sorted if year_from <= t.year <= year_to]

    by_year: dict[int, list[ExtractionTask]] = {}
    for task in tasks_sorted:
        if not (year_from <= task.year <= year_to):
            continue
        by_year.setdefault(task.year, []).append(task)

    selected: list[ExtractionTask] = []
    for year in range(year_from, year_to + 1):
        candidates = by_year.get(year, [])
        if not candidates:
            print(f"  [WARN] Year {year}: no available PDFs after filtering.")
            continue

        take_n = min(docs_per_year, len(candidates))
        if len(candidates) < docs_per_year:
            print(
                f"  [WARN] Year {year}: requested {docs_per_year}, "
                f"available {len(candidates)}. Taking all."
            )

        year_rng = np.random.default_rng(seed + year * 1009)
        chosen_idx = np.sort(year_rng.choice(len(candidates), size=take_n, replace=False))
        selected.extend(candidates[int(i)] for i in chosen_idx)

    return sorted(selected, key=lambda t: (t.year, t.doc_id))


def write_manifest(tasks: list[ExtractionTask], output_dir: Path, manifest_name: str) -> Path:
    """Write selected docs manifest to CSV."""
    manifest_path = output_dir / manifest_name
    rows = [
        {
            "doc_id": t.doc_id,
            "title": t.title,
            "year": t.year,
            "sector": t.sector,
            "region": t.region,
            "country": t.country,
            "pdf_path": str(t.pdf_path),
        }
        for t in tasks
    ]
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


def extract_documents(tasks: list[ExtractionTask], use_multiprocessing: bool = True) -> list[dict]:
    """Extract PDF text with multiprocessing and fallback to sequential mode."""
    documents: list[dict] = []

    if use_multiprocessing:
        try:
            with ProcessPoolExecutor() as pool:
                futures = {pool.submit(extract_text_from_pdf, t.pdf_path): t for t in tasks}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting PDF text"):
                    task = futures[future]
                    text = future.result()
                    if text.strip():
                        documents.append(
                            {
                                "doc_id": task.doc_id,
                                "title": task.title,
                                "year": task.year,
                                "sector": task.sector,
                                "region": task.region,
                                "country": task.country,
                                "text": text,
                            }
                        )
            return documents
        except PermissionError as e:
            print(f"[WARN] Multiprocessing unavailable ({e}); falling back to sequential extraction.")

    for task in tqdm(tasks, desc="Extracting PDF text (sequential)"):
        text = extract_text_from_pdf(task.pdf_path)
        if text.strip():
            documents.append(
                {
                    "doc_id": task.doc_id,
                    "title": task.title,
                    "year": task.year,
                    "sector": task.sector,
                    "region": task.region,
                    "country": task.country,
                    "text": text,
                }
            )
    return documents


def deterministic_ordered_tasks_by_year(
    tasks: list[ExtractionTask],
    year_from: int,
    year_to: int,
    seed: int,
) -> dict[int, list[ExtractionTask]]:
    """Create deterministic shuffled task order per year."""
    by_year: dict[int, list[ExtractionTask]] = {y: [] for y in range(year_from, year_to + 1)}
    for task in sorted(tasks, key=lambda t: (t.year, t.doc_id)):
        if year_from <= task.year <= year_to:
            by_year.setdefault(task.year, []).append(task)

    ordered: dict[int, list[ExtractionTask]] = {}
    for year in range(year_from, year_to + 1):
        candidates = by_year.get(year, [])
        if not candidates:
            ordered[year] = []
            continue
        year_rng = np.random.default_rng(seed + year * 1009)
        perm = year_rng.permutation(len(candidates))
        ordered[year] = [candidates[int(i)] for i in perm]
    return ordered


def main():
    args = parse_args()

    if args.year_from > args.year_to:
        raise ValueError("--year-from must be <= --year-to")
    if args.docs_per_year is not None and args.docs_per_year <= 0:
        raise ValueError("--docs-per-year must be positive when provided")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata (supports both old format with publication_date and new format with year)
    meta = load_metadata(META_PATH)
    print(f"Metadata: {len(meta)} documents, years {meta['year'].min()}–{meta['year'].max()}")
    meta = meta[(meta["year"] >= args.year_from) & (meta["year"] <= args.year_to)].copy()
    print(
        f"Year filter: {args.year_from}–{args.year_to} -> "
        f"{len(meta)} metadata rows"
    )

    tasks_all = build_tasks(meta)
    task_lookup = {t.doc_id: t for t in tasks_all}
    print(f"Found {len(tasks_all)} PDFs on disk in selected year range")

    tasks = stratified_sample_tasks(
        tasks=tasks_all,
        docs_per_year=args.docs_per_year,
        year_from=args.year_from,
        year_to=args.year_to,
        seed=args.sample_seed,
    )
    if not tasks:
        raise RuntimeError("No tasks selected. Check year range and available PDFs.")

    print(f"Selected {len(tasks)} documents for extraction")
    year_counts = pd.Series([t.year for t in tasks]).value_counts().sort_index()
    print("Selected docs per year:")
    print(year_counts.to_string())

    print("Extracting text...")
    documents = extract_documents(tasks)

    # If requested, backfill years where selected docs had no extractable text.
    if args.docs_per_year is not None:
        ordered_by_year = deterministic_ordered_tasks_by_year(
            tasks=tasks_all,
            year_from=args.year_from,
            year_to=args.year_to,
            seed=args.sample_seed,
        )

        already_used = {t.doc_id for t in tasks}
        current_by_year = pd.Series([d["year"] for d in documents]).value_counts().to_dict()
        target_by_year = {
            y: min(args.docs_per_year, len(ordered_by_year.get(y, [])))
            for y in range(args.year_from, args.year_to + 1)
        }
        next_idx = {y: 0 for y in range(args.year_from, args.year_to + 1)}
        for y in range(args.year_from, args.year_to + 1):
            year_order = ordered_by_year.get(y, [])
            while next_idx[y] < len(year_order) and year_order[next_idx[y]].doc_id in already_used:
                next_idx[y] += 1

        for year in range(args.year_from, args.year_to + 1):
            current_n = int(current_by_year.get(year, 0))
            target_n = int(target_by_year[year])
            if current_n >= target_n:
                continue

            print(
                f"[INFO] Year {year}: extracted {current_n}/{target_n}. "
                "Attempting backfill from reserve candidates."
            )
            year_order = ordered_by_year.get(year, [])
            while current_n < target_n and next_idx[year] < len(year_order):
                candidate = year_order[next_idx[year]]
                next_idx[year] += 1
                if candidate.doc_id in already_used:
                    continue

                already_used.add(candidate.doc_id)
                extra_docs = extract_documents([candidate], use_multiprocessing=False)
                if not extra_docs:
                    continue

                documents.extend(extra_docs)
                current_n += 1
                current_by_year[year] = current_n

            if current_n < target_n:
                print(
                    f"[WARN] Year {year}: unable to reach target {target_n}, "
                    f"final count {current_n}."
                )

    docs_df = pd.DataFrame(documents).sort_values(["year", "doc_id"]).reset_index(drop=True)
    print(f"\nExtracted text from {len(docs_df)} documents")
    print(f"Average text length: {docs_df['text'].str.len().mean():.0f} characters")

    # Manifest reflects final successfully extracted docs.
    final_manifest_tasks = [
        task_lookup[str(doc_id)] for doc_id in docs_df["doc_id"].astype(str).tolist()
        if str(doc_id) in task_lookup
    ]
    manifest_path = write_manifest(final_manifest_tasks, output_dir, args.manifest_name)
    print(f"Manifest saved: {manifest_path}")

    # Split into sentences
    sentence_records = []
    for _, row in docs_df.iterrows():
        for sent in split_into_sentences(row["text"]):
            sentence_records.append({
                "sentence": sent,
                "year": row["year"],
                "doc_id": row["doc_id"],
                "sector": row["sector"],
            })

    sent_df = pd.DataFrame(sentence_records)
    print(f"Total sentences: {len(sent_df)}")
    print(f"\nSentences per year:")
    print(sent_df["year"].value_counts().sort_index().to_string())

    # Save
    docs_path = output_dir / "documents.parquet"
    sents_path = output_dir / "sentences.parquet"

    docs_df.to_parquet(docs_path, index=False)
    sent_df.to_parquet(sents_path, index=False)

    print(f"\nSaved:")
    print(f"  {docs_path} ({len(docs_df)} documents)")
    print(f"  {sents_path} ({len(sent_df)} sentences)")


if __name__ == "__main__":
    main()
