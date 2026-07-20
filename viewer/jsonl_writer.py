"""
viewer/jsonl_writer.py  — V0: structured per-turn session logger
Writes one JSON line per event to logs/sessions/<session_id>.jsonl
"""
from __future__ import annotations
import json, dataclasses
from pathlib import Path
from typing import Any
from shared.models import TurnLog, EpisodeResult


def _serialise(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {fld.name: _serialise(getattr(obj, fld.name)) for fld in dataclasses.fields(obj)}
    if hasattr(obj, "value"):       # Enum
        return obj.value
    if isinstance(obj, (list, tuple)):
        return [_serialise(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    return obj


class JSONLWriter:
    def __init__(self, session_id: str, log_dir: str = "logs/sessions"):
        self.session_id = session_id
        self.path = Path(log_dir) / f"{session_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write_header(self, meta: dict) -> None:
        self._append({"type": "header", "session_id": self.session_id, **meta})

    def write_turn(self, turn_log: TurnLog, graph_snapshot: dict | None = None) -> None:
        row = {"type": "turn", "turn_id": turn_log.turn_id, "data": _serialise(turn_log)}
        if graph_snapshot is not None:
            row["graph"] = graph_snapshot
        self._append(row)

    def write_footer(self, result: EpisodeResult) -> None:
        self._append({"type": "footer", "data": _serialise(result)})

    def _append(self, obj: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, default=str) + "\n")
