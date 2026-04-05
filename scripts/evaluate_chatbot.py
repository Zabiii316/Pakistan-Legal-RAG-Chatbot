import csv
import json
from pathlib import Path
from typing import Any

import requests


BASE_URL = "http://127.0.0.1:8000"
CHAT_ENDPOINT = f"{BASE_URL}/chat"
CLEAR_ENDPOINT = f"{BASE_URL}/chat/clear"

TEST_SET_PATH = Path("data/eval/legal_qa_test_set.json")
OUTPUT_CSV_PATH = Path("data/eval/auto_evaluation_results.csv")

SESSION_ID = "eval-session-1"


def load_test_set(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def clear_session() -> None:
    payload = {
        "query": "clear",
        "session_id": SESSION_ID,
    }
    try:
        requests.post(CLEAR_ENDPOINT, json=payload, timeout=30)
    except requests.RequestException:
        pass


def call_chatbot(query: str) -> dict[str, Any]:
    payload = {
        "query": query,
        "session_id": SESSION_ID,
    }
    response = requests.post(CHAT_ENDPOINT, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def contains_expected_keywords(answer: str, expected_keywords: list[str]) -> bool:
    if not expected_keywords:
        return True
    answer_lower = answer.lower()
    return all(keyword.lower() in answer_lower for keyword in expected_keywords)


def source_matches(relevant_context: list[Any], expected_source_contains: str) -> bool:
    if not expected_source_contains:
        return True

    source_texts: list[str] = []

    for item in relevant_context:
        if isinstance(item, dict):
            source_texts.append(str(item.get("source", "")))
            source_texts.append(str(item.get("text", "")))
        else:
            source_texts.append(str(item))

    combined = " ".join(source_texts).lower()
    return expected_source_contains.lower() in combined


def refusal_detected(answer: str) -> bool:
    refusal_signals = [
        "insufficient",
        "not enough information",
        "not enough context",
        "does not contain enough information",
        "unable to determine",
        "cannot determine",
        "context is insufficient",
    ]
    answer_lower = answer.lower()
    return any(signal in answer_lower for signal in refusal_signals)


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    query = case["query"]
    expected_keywords = case.get("expected_keywords", [])
    expected_source_contains = case.get("expected_source_contains", "")
    should_refuse = case.get("should_refuse", False)

    try:
        result = call_chatbot(query)
        answer = result.get("answer", "")
        relevant_context = result.get("relevant_context", [])
        metadata = result.get("metadata", {})

        keyword_pass = contains_expected_keywords(answer, expected_keywords)
        source_pass = source_matches(relevant_context, expected_source_contains)
        refusal_pass = refusal_detected(answer) if should_refuse else True

        overall_pass = keyword_pass and source_pass and refusal_pass

        return {
            "id": case["id"],
            "query": query,
            "category": case.get("category", ""),
            "expected_source_contains": expected_source_contains,
            "expected_keywords": "; ".join(expected_keywords),
            "should_refuse": should_refuse,
            "status": "success",
            "answer": answer,
            "relevant_context": json.dumps(relevant_context, ensure_ascii=False),
            "metadata": json.dumps(metadata, ensure_ascii=False),
            "keyword_pass": keyword_pass,
            "source_pass": source_pass,
            "refusal_pass": refusal_pass,
            "overall_pass": overall_pass,
            "notes": case.get("notes", ""),
        }

    except Exception as e:
        return {
            "id": case["id"],
            "query": query,
            "category": case.get("category", ""),
            "expected_source_contains": expected_source_contains,
            "expected_keywords": "; ".join(expected_keywords),
            "should_refuse": should_refuse,
            "status": "error",
            "answer": "",
            "relevant_context": "",
            "metadata": "",
            "keyword_pass": False,
            "source_pass": False,
            "refusal_pass": False,
            "overall_pass": False,
            "notes": f"{case.get('notes', '')} | ERROR: {str(e)}",
        }


def save_results(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "query",
        "category",
        "expected_source_contains",
        "expected_keywords",
        "should_refuse",
        "status",
        "answer",
        "relevant_context",
        "metadata",
        "keyword_pass",
        "source_pass",
        "refusal_pass",
        "overall_pass",
        "notes",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not TEST_SET_PATH.exists():
        raise FileNotFoundError(f"Test set not found: {TEST_SET_PATH}")

    test_cases = load_test_set(TEST_SET_PATH)

    results: list[dict[str, Any]] = []

    clear_session()

    for case in test_cases:
        if case.get("category") != "follow_up_memory":
            clear_session()

        result_row = evaluate_case(case)
        results.append(result_row)

        print(
            f"[{result_row['id']}] {result_row['query']} "
            f"-> {result_row['status']} | pass={result_row['overall_pass']}"
        )

    save_results(OUTPUT_CSV_PATH, results)

    total = len(results)
    passed = sum(1 for r in results if r["overall_pass"])
    errors = sum(1 for r in results if r["status"] == "error")

    print("\nEvaluation complete")
    print(f"Total: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Errors: {errors}")
    print(f"Saved to: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()