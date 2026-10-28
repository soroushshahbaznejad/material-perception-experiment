from __future__ import annotations

import os

import requests

from experiment_core import utc_now_iso

SCHEMA_ERROR_STATUSES = {400, 404, 406}


def _looks_like_schema_mismatch(response: requests.Response) -> bool:
    if response.status_code not in SCHEMA_ERROR_STATUSES:
        return False

    text = response.text.lower()
    return any(
        marker in text
        for marker in (
            "column",
            "session_data",
            "session_json",
            "observer_id",
            "completed",
        )
    )


def _looks_like_missing_batch_rpc(response: requests.Response) -> bool:
    if response.status_code not in SCHEMA_ERROR_STATUSES:
        return False
    text = response.text.lower()
    return any(marker in text for marker in ("allocate_experiment_batch", "function", "schema cache", "rpc"))


class SupabaseResultStore:
    def __init__(self, url: str, key: str, table: str):
        self.url = url.rstrip("/")
        self.key = key
        self.table = table
        self._read_field_cache: str | None = None
        self._write_field_cache: str | None = None

    @classmethod
    def from_env(cls) -> "SupabaseResultStore | None":
        url = os.getenv("SUPABASE_URL", "").strip()
        key = os.getenv("SUPABASE_KEY", "").strip()
        table = os.getenv("SUPABASE_SESSIONS_TABLE", "experiment_sessions").strip() or "experiment_sessions"
        if not url or not key:
            return None
        return cls(url=url, key=key, table=table)

    @property
    def endpoint(self) -> str:
        return f"{self.url}/rest/v1/{self.table}"

    @property
    def batch_rpc_endpoint(self) -> str:
        return f"{self.url}/rest/v1/rpc/allocate_experiment_batch"

    @property
    def headers(self) -> dict:
        return {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }

    def _read_fields(self) -> list[str]:
        fields = ["session_data", "session_json"]
        if self._read_field_cache in fields:
            fields.remove(self._read_field_cache)
            fields.insert(0, self._read_field_cache)
        return fields

    def _write_variants(self, session: dict) -> list[dict]:
        variants = [
            {
                "json_field": "session_data",
                "payload": {
                    "participant_id": session["participant_storage_id"],
                    "updated_at": utc_now_iso(),
                    "session_data": self._full_session_payload(session),
                },
            },
            {
                "json_field": "session_json",
                "payload": {
                    "participant_id": session["participant_storage_id"],
                    "observer_id": session.get("participant_id", ""),
                    "completed": bool(session.get("completed", False)),
                    "updated_at": utc_now_iso(),
                    "session_json": self._full_session_payload(session),
                },
            },
        ]
        if self._write_field_cache:
            variants.sort(key=lambda variant: variant["json_field"] != self._write_field_cache)
        return variants

    def load_session(self, participant_storage_id: str) -> dict | None:
        last_error: Exception | None = None

        for field in self._read_fields():
            response = requests.get(
                self.endpoint,
                headers=self.headers,
                params={
                    "participant_id": f"eq.{participant_storage_id}",
                    "select": field,
                    "limit": "1",
                },
                timeout=10,
            )
            if response.ok:
                self._read_field_cache = field
                rows = response.json()
                if not rows:
                    return None
                return rows[0].get(field)

            if _looks_like_schema_mismatch(response):
                last_error = requests.HTTPError(response.text, response=response)
                continue

            response.raise_for_status()

        if last_error is not None:
            raise last_error
        return None

    def _allocate_batch_via_rpc(self, batch_count: int) -> int:
        response = requests.post(
            self.batch_rpc_endpoint,
            headers={**self.headers, "Prefer": "return=representation"},
            json={"batch_count": int(batch_count)},
            timeout=10,
        )
        if not response.ok:
            raise requests.HTTPError(response.text, response=response)

        payload = response.json()
        if isinstance(payload, int):
            return payload
        if isinstance(payload, str) and payload.isdigit():
            return int(payload)
        if isinstance(payload, list) and payload:
            first = payload[0]
            if isinstance(first, int):
                return first
            if isinstance(first, dict):
                for value in first.values():
                    if isinstance(value, int):
                        return value
                    if isinstance(value, str) and value.isdigit():
                        return int(value)
        if isinstance(payload, dict):
            for value in payload.values():
                if isinstance(value, int):
                    return value
                if isinstance(value, str) and value.isdigit():
                    return int(value)
        raise ValueError(f"Unexpected batch allocator response: {payload!r}")

    def _count_sessions_for_manifest(self, manifest_id: str, batch_count: int) -> int:
        last_error: Exception | None = None

        for field in self._read_fields():
            response = requests.get(
                self.endpoint,
                headers=self.headers,
                params={
                    "select": field,
                    "limit": "10000",
                },
                timeout=15,
            )
            if response.ok:
                self._read_field_cache = field
                count = 0
                for row in response.json():
                    payload = row.get(field)
                    if not isinstance(payload, dict):
                        continue
                    if (payload.get("stimulus_manifest_id") or "").strip() != manifest_id:
                        continue
                    try:
                        saved_batch_count = int(payload.get("batch_count") or batch_count)
                        saved_batch_number = int(payload.get("batch_number") or 0)
                    except (TypeError, ValueError):
                        continue
                    if saved_batch_count == batch_count and 1 <= saved_batch_number <= batch_count:
                        count += 1
                return count

            if _looks_like_schema_mismatch(response):
                last_error = requests.HTTPError(response.text, response=response)
                continue

            response.raise_for_status()

        if last_error is not None:
            raise last_error
        return 0

    def allocate_batch(self, batch_count: int, manifest_id: str) -> tuple[int, str]:
        if batch_count < 1:
            raise ValueError("batch_count must be at least 1.")

        try:
            assigned = self._allocate_batch_via_rpc(batch_count)
            if not 1 <= assigned <= batch_count:
                raise ValueError(f"Batch allocator returned out-of-range batch {assigned}.")
            return assigned, "Supabase atomic round-robin allocator."
        except requests.HTTPError as exc:
            response = getattr(exc, "response", None)
            if response is None or not _looks_like_missing_batch_rpc(response):
                raise

        # Backward-compatible fallback for deployments where the SQL RPC has
        # not been installed yet. It preserves the 1 -> 2 -> 3 repeat pattern
        # for normal participant starts; the RPC is still preferred for perfect
        # concurrent fairness.
        existing_count = self._count_sessions_for_manifest(manifest_id, batch_count)
        assigned = (existing_count % batch_count) + 1
        return assigned, "Supabase count-based round-robin allocator."

    @staticmethod
    def _full_session_payload(session: dict) -> dict:
        """Compact the session: replace full trial dicts with trial_keys list."""
        compact = dict(session)
        trials = compact.get("trials")
        if trials and isinstance(trials, list) and len(trials) > 0 and isinstance(trials[0], dict):
            compact["trial_keys"] = [t["trial_key"] for t in trials]
            compact.pop("trials", None)
        # Strip bulky cached data
        compact.pop("recent_actions", None)
        return compact

    def save_session(self, session: dict) -> None:
        last_error: Exception | None = None

        for variant in self._write_variants(session):
            response = requests.post(
                self.endpoint,
                headers={**self.headers, "Prefer": "resolution=merge-duplicates,return=minimal"},
                params={"on_conflict": "participant_id"},
                json=[variant["payload"]],
                timeout=15,
            )
            if response.ok:
                self._write_field_cache = variant["json_field"]
                return

            if _looks_like_schema_mismatch(response):
                last_error = requests.HTTPError(response.text, response=response)
                continue

            response.raise_for_status()

        if last_error is not None:
            raise last_error


class ResultStore:
    def __init__(self):
        self.remote = SupabaseResultStore.from_env()

    def is_configured(self) -> bool:
        return self.remote is not None

    def backend_label(self) -> str:
        if self.remote is not None:
            return "Supabase"
        return "Supabase unavailable"

    def load_session(self, participant_storage_id: str) -> tuple[dict | None, str | None]:
        if self.remote is None:
            return None, "Supabase is not configured."

        try:
            return self.remote.load_session(participant_storage_id), None
        except Exception as exc:  # pragma: no cover - depends on remote service
            return None, f"Supabase read failed. ({exc})"

    def allocate_batch(self, batch_count: int, manifest_id: str) -> dict:
        if self.remote is None:
            return {
                "ok": False,
                "message": "Supabase is not configured. Set SUPABASE_URL and SUPABASE_KEY before collecting data.",
            }

        try:
            batch_number, allocator = self.remote.allocate_batch(batch_count, manifest_id)
        except Exception as exc:  # pragma: no cover - depends on remote service
            return {
                "ok": False,
                "message": f"Supabase batch allocation failed. ({exc})",
            }

        return {
            "ok": True,
            "batch_number": batch_number,
            "message": allocator,
        }

    def save_session(self, session: dict) -> dict:
        if self.remote is None:
            return {
                "ok": False,
                "message": "Supabase is not configured. Set SUPABASE_URL and SUPABASE_KEY before collecting data.",
            }

        try:
            self.remote.save_session(session)
        except Exception as exc:  # pragma: no cover - depends on remote service
            return {
                "ok": False,
                "message": f"Supabase save failed. ({exc})",
            }

        return {
            "ok": True,
            "message": "Saved to Supabase.",
        }


class DemoStore:
    """Stand-in store for the public demo: nothing is persisted anywhere."""

    remote = None

    def is_configured(self) -> bool:
        return True

    def backend_label(self) -> str:
        return "Demo (not saved)"

    def load_session(self, participant_storage_id: str) -> tuple[dict | None, str | None]:
        return None, None

    def save_session(self, session: dict) -> dict:
        return {"ok": True, "message": "Demo mode: responses are not saved."}
