"""외부 API 없이 BM25·규정용어 검색 품질을 반복 확인하는 평가 스크립트입니다."""

from __future__ import annotations

import json
from pathlib import Path

from rank_bm25 import BM25Okapi

import app


class OfflineVectorStore:
    """평가 중 API 비용을 발생시키지 않기 위한 빈 의미 검색 대체 객체입니다."""

    def similarity_search(self, query: str, k: int) -> list:
        return []


def main() -> None:
    cases = json.loads(Path("eval_cases.json").read_text(encoding="utf-8"))
    parents, skipped = app.load_documents()
    children = app.build_child_chunks(parents)
    bm25 = BM25Okapi([app.tokenize(child.page_content) or ["_"] for child in children])
    vector_store = OfflineVectorStore()

    passed = 0
    for case in cases:
        result = app.retrieve_candidates(vector_store, children, bm25, case["question"])
        actual_pages = {(doc.metadata["source"], doc.metadata.get("page")) for doc in result.candidates[:4]}
        expected_pages = {tuple(page) for page in case["expected_pages"]}
        ok = not result.has_lexical_support if case["should_refuse"] else expected_pages.issubset(actual_pages)
        passed += int(ok)
        print(f"{'PASS' if ok else 'FAIL'} {case['id']}: {sorted(actual_pages)}")

    print(f"\nResult: {passed}/{len(cases)} cases passed")
    if skipped or passed != len(cases):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
