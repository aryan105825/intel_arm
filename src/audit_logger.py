""" Signed, hash-chained, append-only compliance ledger for Praxis Guard. """
import hashlib
import json
import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Tuple

GENESIS_CHAIN_HASH = "sha256:" + ("0" * 64)

def _canonicalize(event: Dict[str, Any]) -> str:
    return json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

def sign_event(event: Dict[str, Any]) -> str:
    canonical = _canonicalize(event)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"

def _chain_hash(prev_chain_hash: str, event_signature: str) -> str:
    digest = hashlib.sha256(f"{prev_chain_hash}{event_signature}".encode("utf-8")).hexdigest()
    return f"sha256:{digest}"

class AppendOnlyLedger:
    def __init__(self, path: str = "compliance_ledger.jsonl"):
        self._path = path
        self._lock = threading.Lock()
        directory = os.path.dirname(os.path.abspath(path))
        if directory and not os.path.isdir(directory):
            os.makedirs(directory, exist_ok=True)
        self._last_chain_hash = self._recover_last_chain_hash()

    def _recover_last_chain_hash(self) -> str:
        records = self.read_all()
        if not records:
            return GENESIS_CHAIN_HASH
        return records[-1].get("chain_hash", GENESIS_CHAIN_HASH)

    def append(self, event: Dict[str, Any]) -> str:
        signature = sign_event(event)
        with self._lock:
            chain_hash = _chain_hash(self._last_chain_hash, signature)
            record = {
                "event": event,
                "signature": signature,
                "prev_chain_hash": self._last_chain_hash,
                "chain_hash": chain_hash,
            }
            line = json.dumps(record, sort_keys=True, separators=(",", ":"))
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
            self._last_chain_hash = chain_hash
        return chain_hash

    def read_all(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self._path):
            return []
        records = []
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records