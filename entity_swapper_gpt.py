import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict

from openai import OpenAI

SYSTEM_PROMPT = """You are an entity swapper for news sentences.

Task:
- Replace named entities (people, companies, organizations, places, countries, currencies, tickers) with different, plausible alternatives.
- Preserve the meaning and factual structure of the sentence.
- Keep roles and relationships intact (e.g., who did what to whom, dates, quantities, and the overall event).
- Do NOT introduce new entities that are not replacements of existing entities.
- Use replacements consistently within the sentence.
- Identify and keep protected terms that must remain unchanged to preserve meaning (e.g., generic institutional roles such as "central bank", "finance ministry", "supreme court", or role titles like "central bank governor"). Do not replace those protected terms.

Output requirements:
- Return a JSON object:
  {"rewritten_sentence": <string>, "entity_replacements": [{"type": <entity_type>, "original": <original_entity>, "replacement": <new_entity>}], "protected_terms": [<string>, ...]}.
- If there are no named entities, use an empty array for "entity_replacements".
- Output must start with "{" and end with "}" with no extra characters.
"""


def _extract_json(text: str) -> Dict:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output")
    return json.loads(match.group(0))


def _build_user_message(sentence: str) -> str:
    return f"Sentence:\n{sentence}\n"


def _extract_output_text(response_body: Dict) -> str:
    if isinstance(response_body, dict) and response_body.get("output_text"):
        return response_body["output_text"]
    output_items = response_body.get("output", []) if isinstance(response_body, dict) else []
    parts = []
    for item in output_items:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") in ("output_text", "text") and "text" in content:
                parts.append(content["text"])
    return "".join(parts)


def _load_records(records_file: Path) -> Dict[str, Dict]:
    records: Dict[str, Dict] = {}
    with records_file.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            records[str(entry["custom_id"])] = entry["record"]
    return records


def create_batch(client: OpenAI, input_file: Path, output_file: Path, model: str) -> str:
    batch_dir = input_file.parent / f"entity_swap_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    requests_path = batch_dir / "batch_requests.jsonl"
    records_path = batch_dir / "batch_records.jsonl"

    total = 0
    with input_file.open('r', encoding='utf-8') as fin, \
            requests_path.open('w', encoding='utf-8') as req_out, \
            records_path.open('w', encoding='utf-8') as rec_out:
        for idx, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            output_sentence = record.get("output_sentence") or record.get("output") or ""
            if not output_sentence:
                continue
            custom_id = str(idx)

            body = {
                "model": model,
                "input": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_message(output_sentence)},
                ],
                "reasoning": {"effort": "medium"},
            }
            req = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
            }
            req_out.write(json.dumps(req) + "\n")
            rec_out.write(json.dumps({"custom_id": custom_id, "record": record}, ensure_ascii=False) + "\n")
            total += 1

    if total == 0:
        raise SystemExit("No valid records found in input file.")

    input_file_obj = client.files.create(
        file=open(requests_path, 'rb'),
        purpose="batch",
    )

    batch = client.batches.create(
        input_file_id=input_file_obj.id,
        endpoint="/v1/responses",
        completion_window="24h",
    )

    metadata = {
        "batch_id": batch.id,
        "input_file": str(input_file),
        "output_file": str(output_file),
        "requests_file": str(requests_path),
        "records_file": str(records_path),
        "created_at": datetime.now().isoformat(),
        "model": model,
        "total_requests": total,
    }
    with (batch_dir / "batch_metadata.json").open('w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Batch created | ID={batch.id} | REQUESTS={total} | DIR={batch_dir}")
    print(f"Run with --mode status --batch-id {batch.id} --input-file {input_file}")
    return batch.id


def check_status(client: OpenAI, batch_id: str) -> None:
    batch = client.batches.retrieve(batch_id)
    print(f"Batch status | ID={batch_id}")
    print(f"  STATUS={batch.status}")
    print(f"  TOTAL={batch.request_counts.total}")
    print(f"  COMPLETED={batch.request_counts.completed}")
    print(f"  FAILED={batch.request_counts.failed}")
    if batch.status == "completed":
        print("Run with --mode download --batch-id <id> --input-file <path>")


def download_results(client: OpenAI, batch_id: str, input_file: Path) -> None:
    batch = client.batches.retrieve(batch_id)
    if batch.status != "completed":
        raise SystemExit(f"Batch not complete | STATUS={batch.status}")

    # Resolve batch directory based on input file
    batch_dir = None
    for path in input_file.parent.glob("entity_swap_batch_*/batch_metadata.json"):
        try:
            meta = json.loads(path.read_text(encoding='utf-8'))
            if meta.get("batch_id") == batch_id:
                batch_dir = path.parent
                metadata = meta
                break
        except Exception:
            continue
    if batch_dir is None:
        raise SystemExit("Batch metadata not found for this batch id in the input directory.")

    records_file = Path(metadata["records_file"])
    output_file = Path(metadata["output_file"])

    records = _load_records(records_file)

    result_content = client.files.content(batch.output_file_id).text

    written = 0
    with output_file.open('w', encoding='utf-8') as fout:
        for line in result_content.strip().split('\n'):
            if not line.strip():
                continue
            result = json.loads(line)
            custom_id = str(result.get("custom_id"))
            response = result.get("response", {})
            if response.get("status_code") != 200:
                continue
            body = response.get("body", {})
            content = _extract_output_text(body)
            try:
                parsed = _extract_json(content)
                rewritten = parsed.get("rewritten_sentence", "")
                entity_replacements = parsed.get("entity_replacements")
            except Exception:
                continue
            if not rewritten:
                continue
            record = records.get(custom_id)
            if record is None:
                continue
            original = record.get("output_sentence") or record.get("output") or ""
            record["output_sentence_original"] = original
            record["output_sentence"] = rewritten
            if entity_replacements is None:
                entity_replacements = []
            record["entity_replacements"] = entity_replacements
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    xlsx_path = output_file.with_suffix(".xlsx")
    try:
        import pandas as pd
    except Exception as e:
        print(f"Results saved | TOTAL_ENTRIES={written} | OUTPUT_FILE={output_file}")
        raise SystemExit(f"Failed to write XLSX (pandas missing): {e}")

    df_rows = []
    with output_file.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            df_rows.append(json.loads(line))
    df = pd.DataFrame(df_rows)
    df.to_excel(xlsx_path, index=False)

    print(f"Results saved | TOTAL_ENTRIES={written} | OUTPUT_FILE={output_file}")
    print(f"XLSX saved | OUTPUT_FILE={xlsx_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Swap entities in STS database outputs using Batch API")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API key (or set OPENAI_API_KEY)")
    parser.add_argument("--input-file", type=str, required=True, help="Path to sts_database.jsonl")
    parser.add_argument("--mode", type=str, choices=["create", "status", "download"], default="create")
    parser.add_argument("--batch-id", type=str, default=None, help="Batch ID for status/download modes")
    parser.add_argument("--model", type=str, default="gpt-5.2", help="OpenAI model to use")
    parser.add_argument("--output-file", type=str, default=None, help="Output JSONL path (default: same dir, *_swapped.jsonl)")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing API key. Use --api-key or set OPENAI_API_KEY.")

    input_path = Path(args.input_file)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    output_path = Path(args.output_file) if args.output_file else input_path.with_name(input_path.stem + "_swapped.jsonl")

    client = OpenAI(api_key=api_key)

    if args.mode == "create":
        create_batch(client, input_path, output_path, args.model)
    elif args.mode == "status":
        if not args.batch_id:
            raise SystemExit("--batch-id required for status mode")
        check_status(client, args.batch_id)
    elif args.mode == "download":
        if not args.batch_id:
            raise SystemExit("--batch-id required for download mode")
        download_results(client, args.batch_id, input_path)


if __name__ == "__main__":
    main()
