from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SUPPORTED_FORMATS = ("messages", "sharegpt", "alpaca", "prompt_completion", "question_answer")
FORMAT_FIELDS = {
    "messages": frozenset({"messages"}),
    "sharegpt": frozenset({"conversations"}),
    "alpaca": frozenset({"instruction", "input", "output"}),
    "prompt_completion": frozenset({"prompt", "completion"}),
    "question_answer": frozenset({"question", "answer"}),
}
REQUIRED_FORMAT_FIELDS = {
    "messages": frozenset({"messages"}),
    "sharegpt": frozenset({"conversations"}),
    "alpaca": frozenset({"instruction", "output"}),
    "prompt_completion": frozenset({"prompt", "completion"}),
    "question_answer": frozenset({"question", "answer"}),
}


class DatasetError(ValueError):
    pass


@dataclass(frozen=True)
class DatasetProfile:
    path: str
    rows: int
    columns: list[str]
    detected_format: str
    detection_confidence: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetOverview:
    rows: int
    columns: list[str]
    column_types: dict[str, list[str]]
    missing_values: dict[str, int]
    detected_format: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _decode_sequence(value: Any, field: str, row_number: int) -> list[Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"row {row_number}: field '{field}' is not valid JSON") from exc
    if not isinstance(value, list):
        raise DatasetError(f"row {row_number}: field '{field}' must be a list")
    return value


def load_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    suffix = source.suffix.lower()
    try:
        if suffix == ".csv":
            with source.open("r", encoding="utf-8-sig", newline="") as handle:
                return [dict(row) for row in csv.DictReader(handle)]
        if suffix == ".jsonl":
            records = []
            with source.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise DatasetError(f"line {line_number}: invalid JSON") from exc
                    if not isinstance(record, dict):
                        raise DatasetError(f"line {line_number}: each record must be an object")
                    records.append(record)
            return records
        if suffix == ".json":
            with source.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict) and isinstance(payload.get("data"), list):
                payload = payload["data"]
            if isinstance(payload, dict):
                payload = [payload]
            if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
                raise DatasetError("JSON input must be an object, an array of objects, or an object with a data array")
            return payload
    except OSError as exc:
        raise DatasetError(f"could not read input file: {source}") from exc
    except json.JSONDecodeError as exc:
        raise DatasetError("input file contains invalid JSON") from exc
    raise DatasetError("supported input extensions are .csv, .json, and .jsonl")


def _classify(record: dict[str, Any], row_number: int) -> str:
    candidates = []
    if "messages" in record:
        candidates.append("messages")
    if "conversations" in record:
        candidates.append("sharegpt")
    if "instruction" in record and "output" in record:
        candidates.append("alpaca")
    if "prompt" in record and "completion" in record:
        candidates.append("prompt_completion")
    if "question" in record and "answer" in record:
        candidates.append("question_answer")
    if len(candidates) > 1:
        raise DatasetError(f"row {row_number}: multiple supported layouts match; select --format after reviewing the data")
    if not candidates:
        raise DatasetError(f"row {row_number}: no supported supervised fine-tuning layout detected")
    return candidates[0]


def detect_format(records: list[dict[str, Any]], selected_format: str = "auto") -> tuple[str, float]:
    if not records:
        raise DatasetError("input dataset is empty")
    if selected_format != "auto" and selected_format not in SUPPORTED_FORMATS:
        raise DatasetError(f"unsupported format '{selected_format}'")
    if selected_format != "auto":
        return selected_format, 1.0
    formats = [_classify(record, index) for index, record in enumerate(records, 1)]
    if len(set(formats)) != 1:
        raise DatasetError("dataset contains mixed row layouts; normalize it or select a single explicit format")
    return formats[0], 1.0


def _text(value: Any, field: str, row_number: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetError(f"row {row_number}: field '{field}' must contain non-empty text")
    return value.strip()


def _normalize_messages(value: Any, row_number: int) -> list[dict[str, str]]:
    messages = _decode_sequence(value, "messages", row_number)
    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            raise DatasetError(f"row {row_number}: each message must be an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or role not in {"system", "user", "assistant", "tool"}:
            raise DatasetError(f"row {row_number}: message role is unsupported")
        normalized.append({"role": role, "content": _text(content, "content", row_number)})
    if not any(message["role"] == "assistant" for message in normalized):
        raise DatasetError(f"row {row_number}: conversation must include an assistant response")
    if not any(message["role"] == "user" for message in normalized):
        raise DatasetError(f"row {row_number}: conversation must include a user message")
    return normalized


def to_messages(record: dict[str, Any], format_name: str, row_number: int) -> list[dict[str, str]]:
    if format_name == "messages":
        return _normalize_messages(record.get("messages"), row_number)
    if format_name == "sharegpt":
        turns = _decode_sequence(record.get("conversations"), "conversations", row_number)
        role_map = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "system": "system", "tool": "tool"}
        mapped = []
        for turn in turns:
            if not isinstance(turn, dict):
                raise DatasetError(f"row {row_number}: each conversation turn must be an object")
            source_role = turn.get("from")
            if not isinstance(source_role, str) or source_role not in role_map:
                raise DatasetError(f"row {row_number}: ShareGPT role is unsupported")
            mapped.append({"role": role_map[source_role], "content": _text(turn.get("value"), "value", row_number)})
        return _normalize_messages(mapped, row_number)
    if format_name == "alpaca":
        instruction = _text(record.get("instruction"), "instruction", row_number)
        extra_input = record.get("input", "")
        if extra_input is not None and not isinstance(extra_input, str):
            raise DatasetError(f"row {row_number}: field 'input' must be text")
        prompt = f"{instruction}\n\n{extra_input.strip()}" if extra_input and extra_input.strip() else instruction
        return [{"role": "user", "content": prompt}, {"role": "assistant", "content": _text(record.get("output"), "output", row_number)}]
    if format_name == "prompt_completion":
        return [{"role": "user", "content": _text(record.get("prompt"), "prompt", row_number)}, {"role": "assistant", "content": _text(record.get("completion"), "completion", row_number)}]
    if format_name == "question_answer":
        return [{"role": "user", "content": _text(record.get("question"), "question", row_number)}, {"role": "assistant", "content": _text(record.get("answer"), "answer", row_number)}]
    raise DatasetError(f"unsupported format '{format_name}'")


def inspect_dataset(path: str | Path, selected_format: str = "auto") -> DatasetProfile:
    records = load_records(path)
    detected_format, confidence = detect_format(records, selected_format)
    normalized = [to_messages(record, detected_format, index) for index, record in enumerate(records, 1)]
    columns = sorted({key for record in records for key in record})
    return DatasetProfile(str(Path(path)), len(normalized), columns, detected_format, confidence)


def profile_dataset(path: str | Path) -> DatasetOverview:
    records = load_records(path)
    columns = sorted({key for record in records for key in record})
    types = {column: set() for column in columns}
    missing = {column: 0 for column in columns}
    for record in records:
        for column in columns:
            if column not in record or record[column] is None:
                missing[column] += 1
            else:
                value = record[column]
                type_name = "bool" if isinstance(value, bool) else type(value).__name__
                types[column].add(type_name)
    try:
        detected_format = detect_format(records)[0]
    except DatasetError:
        detected_format = None
    return DatasetOverview(
        rows=len(records),
        columns=columns,
        column_types={column: sorted(values) for column, values in types.items()},
        missing_values=missing,
        detected_format=detected_format,
    )


def _map_record_fields(
    record: dict[str, Any],
    format_name: str,
    field_mapping: dict[str, str] | None,
    row_number: int,
) -> dict[str, Any]:
    if not field_mapping:
        return record
    allowed_fields = FORMAT_FIELDS[format_name]
    mapped = dict(record)
    for target, source in field_mapping.items():
        if target not in allowed_fields:
            raise DatasetError(f"field mapping target '{target}' is invalid for format '{format_name}'")
        if not isinstance(source, str) or source not in record:
            raise DatasetError(f"row {row_number}: mapped source field '{source}' does not exist")
        mapped[target] = record[source]
    return mapped


def prepare_dataset(
    input_path: str | Path,
    output_dir: str | Path,
    selected_format: str = "auto",
    validation_ratio: float = 0.1,
    seed: int = 42,
    field_mapping: dict[str, str] | None = None,
) -> dict[str, Any]:
    if not 0 < validation_ratio < 0.5:
        raise DatasetError("validation ratio must be greater than 0 and less than 0.5")
    records = load_records(input_path)
    detected_format, confidence = detect_format(records, selected_format)
    conversations = [
        to_messages(_map_record_fields(record, detected_format, field_mapping, index), detected_format, index)
        for index, record in enumerate(records, 1)
    ]
    unique = []
    seen = set()
    for conversation in conversations:
        key = tuple((message["role"], message["content"]) for message in conversation)
        if key not in seen:
            seen.add(key)
            unique.append({"messages": conversation})
    if len(unique) < 2:
        raise DatasetError("at least two unique conversations are required for a train/validation split")
    indices = list(range(len(unique)))
    random.Random(seed).shuffle(indices)
    validation_count = min(len(unique) - 1, max(1, round(len(unique) * validation_ratio)))
    validation_indices = set(indices[:validation_count])
    train_rows = [row for index, row in enumerate(unique) if index not in validation_indices]
    validation_rows = [row for index, row in enumerate(unique) if index in validation_indices]
    destination = Path(output_dir)
    target_paths = [destination / "train.jsonl", destination / "validation.jsonl", destination / "report.json"]
    if any(path.exists() for path in target_paths):
        raise DatasetError("one or more output files already exist; choose a new output directory")
    destination.mkdir(parents=True, exist_ok=True)
    report = {
        "input_rows": len(records),
        "prepared_rows": len(unique),
        "duplicate_rows_removed": len(records) - len(unique),
        "source_format": detected_format,
        "format_detection_confidence": confidence,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "validation_ratio": validation_ratio,
        "seed": seed,
        "content_inspection": "row contents are not included in this report",
        "limitations": ["personal information is not detected or removed", "labels are not generated"],
    }
    for path, rows in ((target_paths[0], train_rows), (target_paths[1], validation_rows)):
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with target_paths[2].open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return report
