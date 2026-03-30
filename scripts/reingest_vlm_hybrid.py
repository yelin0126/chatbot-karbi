#!/usr/bin/env python3
"""Re-ingest specific table-heavy PDFs with VLM_HYBRID_PDF_ENABLED=true.

Usage:
    .venv/bin/python scripts/reingest_vlm_hybrid.py
"""
import os
import sys

# Force VLM hybrid BEFORE any app imports
os.environ.setdefault("VLM_HYBRID_PDF_ENABLED", "true")

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from app.config import LIBRARY_DIR
from app.core.vectorstore import delete_documents
from app.pipeline.ingest import ingest_single_file

FILES_TO_REINGEST = [
    "◆(20260107)중앙구매 신청 절차 안내(중앙구매팀,2026.1.7.).pdf",
    "◆20260114_중앙구매 학습용 사이트 리스트.pdf",
    "◆중앙구매 학습용 사이트 리스트v2.pdf",
    "(붙임5) 한국연구재단_연구시설 장비의 정의.pdf",
    "붙임 3_제주대학교 RISE사업단 핵심성과지표 정의서.pdf",
    "(붙임2) 장비도입 리스트 (양식).pdf",
]


def main():
    from app.config import VLM_HYBRID_PDF_ENABLED
    print(f"VLM_HYBRID_PDF_ENABLED = {VLM_HYBRID_PDF_ENABLED}")

    for filename in FILES_TO_REINGEST:
        file_path = LIBRARY_DIR / filename
        if not file_path.exists():
            print(f"[SKIP] Not found: {file_path}")
            continue

        # 1. Delete old chunks
        deleted = delete_documents(source=filename)
        print(f"[DELETE] {filename}: removed {deleted} chunks")

        # 2. Re-ingest with VLM hybrid
        result = ingest_single_file(file_path)
        print(f"[INGEST] {filename}: {result['message']} ({result['count']} chunks)")

    print("\nDone.")


if __name__ == "__main__":
    main()
