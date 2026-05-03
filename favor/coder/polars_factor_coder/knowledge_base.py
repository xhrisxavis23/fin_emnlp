from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _stable_hash(text: str) -> str:
    return md5(text.encode("utf-8")).hexdigest()


def signature_for_factor(
    *,
    factor_name: str,
    factor_definition: str,
    columns: List[str] | None,
    coding_context: str,
) -> str:
    cols = ",".join(columns or [])
    blob = f"name={factor_name}\ndef={factor_definition}\ncols={cols}\nctx={coding_context}"
    return _stable_hash(blob)


@dataclass
class KBSuccessEntry:
    signature: str
    factor_name: str
    factor_definition: str
    columns: List[str]
    coding_context: str
    code_response: Dict[str, Any]


@dataclass
class KBFailureEntry:
    signature: str
    factor_name: str
    factor_definition: str
    implementation: str
    error: str


class PolarsCoSTEERKnowledgeBase:
    """
    Lightweight JSON knowledge base for factor->polars code generation.

    This is intentionally simple (no embeddings / no external deps):
    - cache exact signature success
    - keep last N failures per signature to help prompt retries
    - allow naive similarity retrieval over factor_definition
    """

    def __init__(self, path: Path, max_failures_per_signature: int = 5) -> None:
        self.path = path
        self.max_failures_per_signature = max_failures_per_signature
        self._data: Dict[str, Any] = {"success": [], "failures": []}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self._data = {"success": [], "failures": []}
            return
        try:
            self._data = json.loads(self.path.read_text())
        except Exception:
            self._data = {"success": [], "failures": []}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, ensure_ascii=False))

    def get_exact_success(self, signature: str) -> Optional[KBSuccessEntry]:
        for item in self._data.get("success", []) or []:
            if item.get("signature") == signature:
                return KBSuccessEntry(
                    signature=item["signature"],
                    factor_name=item.get("factor_name", ""),
                    factor_definition=item.get("factor_definition", ""),
                    columns=item.get("columns", []) or [],
                    coding_context=item.get("coding_context", "") or "",
                    code_response=item.get("code_response", {}) or {},
                )
        return None

    def add_success(self, entry: KBSuccessEntry) -> None:
        # overwrite exact signature if exists
        kept = []
        for item in self._data.get("success", []) or []:
            if item.get("signature") != entry.signature:
                kept.append(item)
        kept.append(
            {
                "signature": entry.signature,
                "factor_name": entry.factor_name,
                "factor_definition": entry.factor_definition,
                "columns": entry.columns,
                "coding_context": entry.coding_context,
                "code_response": entry.code_response,
            }
        )
        self._data["success"] = kept

    def add_failure(self, entry: KBFailureEntry) -> None:
        failures = self._data.get("failures", []) or []
        failures.append(
            {
                "signature": entry.signature,
                "factor_name": entry.factor_name,
                "factor_definition": entry.factor_definition,
                "implementation": entry.implementation,
                "error": entry.error,
            }
        )
        # cap per signature (keep last N, drop oldest)
        per_sig_count = sum(1 for f in failures if f.get("signature") == entry.signature)
        overflow = per_sig_count - self.max_failures_per_signature
        if overflow > 0:
            trimmed = []
            for f in failures:
                if f.get("signature") != entry.signature:
                    trimmed.append(f)
                    continue
                if overflow > 0:
                    overflow -= 1
                    continue
                trimmed.append(f)
            failures = trimmed
        self._data["failures"] = failures

    def get_failures(self, signature: str) -> List[KBFailureEntry]:
        out: List[KBFailureEntry] = []
        for item in self._data.get("failures", []) or []:
            if item.get("signature") != signature:
                continue
            out.append(
                KBFailureEntry(
                    signature=item.get("signature", ""),
                    factor_name=item.get("factor_name", ""),
                    factor_definition=item.get("factor_definition", ""),
                    implementation=item.get("implementation", ""),
                    error=item.get("error", ""),
                )
            )
        return out

    def list_success_entries(self) -> List[KBSuccessEntry]:
        out: List[KBSuccessEntry] = []
        for item in self._data.get("success", []) or []:
            out.append(
                KBSuccessEntry(
                    signature=item.get("signature", ""),
                    factor_name=item.get("factor_name", ""),
                    factor_definition=item.get("factor_definition", ""),
                    columns=item.get("columns", []) or [],
                    coding_context=item.get("coding_context", "") or "",
                    code_response=item.get("code_response", {}) or {},
                )
            )
        return out
