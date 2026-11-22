"""Persistent request caps for opt-in, resumable experiment runs."""

from __future__ import annotations

import json
import hashlib
import os
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


@contextmanager
def _interprocess_file_lock(state_path: Path, *, timeout_seconds: float = 60.0) -> Iterator[None]:
    """Serialize ledger read/modify/write across independent Python processes.

    The lock is a sidecar so atomic replacement of the JSON ledger never
    replaces the locked inode/file handle. It is intentionally retained after
    use; deleting a lock file while another process waits can split the lock.
    """
    lock_path = state_path.with_name(state_path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            deadline = time.monotonic() + timeout_seconds
            acquired = False
            while not acquired:
                stream.seek(0)
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    acquired = True
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Timed out locking budget ledger: {lock_path}")
                    time.sleep(0.05)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _ledger_fingerprint(payload: Dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


_PATH_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: dict[str, threading.Lock] = {}


def _thread_lock_for(state_path: Path) -> threading.Lock:
    key = os.path.normcase(str(state_path.resolve()))
    with _PATH_LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(key, threading.Lock())


def _replace_ledger_file(temp_path: Path, state_path: Path, *, attempts: int = 6) -> None:
    """Atomically replace a ledger, retrying only transient Windows sharing locks.

    Windows scanners and short-lived readers can briefly deny ``os.replace``
    even while the sidecar lock correctly serializes ledger writers. Retry a
    small, bounded number of times for those sharing violations; all other
    errors fail closed and prevent the request from being dispatched.
    """
    for attempt in range(attempts):
        try:
            os.replace(temp_path, state_path)
            return
        except PermissionError as exc:
            if os.name != "nt" or getattr(exc, "winerror", None) not in {5, 32, 33}:
                raise
            if attempt + 1 >= attempts:
                raise
            time.sleep(min(0.02 * (2 ** attempt), 0.2))


class RequestBudgetError(RuntimeError):
    """A configured request cap cannot be honored safely."""

    stop_run = True


class RequestBudgetExceeded(RequestBudgetError):
    """The persistent experiment-wide request cap has been exhausted."""


class TokenCostBudgetExceeded(RequestBudgetError):
    """A conservative request-cost reservation would exceed the experiment cap."""


class PersistentTokenCostBudget:
    """Persistent estimated-token cost cap with per-request reservation/settlement.

    Before dispatch, a request reserves its estimated input plus the full
    completion allowance. Once usage is reported, the reservation is settled
    to observed token cost. Missing usage keeps the reservation in place.
    """

    LIMIT_ENV = "TRAIL_LIVE_COST_BUDGET_USD"
    STATE_ENV = "TRAIL_LIVE_COST_BUDGET_STATE_PATH"
    INPUT_RATE_ENV = "TRAIL_LIVE_INPUT_USD_PER_MILLION"
    OUTPUT_RATE_ENV = "TRAIL_LIVE_OUTPUT_USD_PER_MILLION"
    PROTOCOL_ENV = "TRAIL_LIVE_BUDGET_PROTOCOL"

    def __init__(
        self, limit_usd: float, state_path: str, input_rate: float, output_rate: float,
        protocol: int = 1,
    ):
        self.limit_usd = float(limit_usd)
        self.state_path = Path(state_path).expanduser()
        self.input_rate = float(input_rate)
        self.output_rate = float(output_rate)
        self.protocol = int(protocol)
        self.fingerprint = _ledger_fingerprint({
            "protocol": self.protocol,
            "limit_usd": self.limit_usd,
            "input_rate": self.input_rate,
            "output_rate": self.output_rate,
        })
        self._lock = _thread_lock_for(self.state_path)

    @classmethod
    def from_environment(cls) -> Optional["PersistentTokenCostBudget"]:
        raw_limit = os.environ.get(cls.LIMIT_ENV)
        raw_path = os.environ.get(cls.STATE_ENV)
        if not raw_limit and not raw_path:
            if os.environ.get(cls.PROTOCOL_ENV):
                raise TokenCostBudgetExceeded(
                    f"{cls.PROTOCOL_ENV} requires an active token-cost budget."
                )
            return None
        if not raw_limit or not raw_path:
            raise TokenCostBudgetExceeded(
                f"Both {cls.LIMIT_ENV} and {cls.STATE_ENV} must be set together."
            )
        try:
            limit = float(raw_limit)
            input_rate = float(os.environ.get(cls.INPUT_RATE_ENV, "0.25"))
            output_rate = float(os.environ.get(cls.OUTPUT_RATE_ENV, "2.0"))
            protocol = int(os.environ.get(cls.PROTOCOL_ENV, "1"))
        except ValueError as exc:
            raise TokenCostBudgetExceeded("Live token-cost budget configuration is invalid.") from exc
        if limit <= 0 or input_rate < 0 or output_rate < 0 or protocol not in {1, 2}:
            raise TokenCostBudgetExceeded("Live token-cost budget values must be positive.")
        return cls(limit, raw_path, input_rate, output_rate, protocol)

    def _read_state(self) -> Dict[str, Any]:
        path = self.state_path
        if not path.exists():
            state = {
                "version": self.protocol,
                "limit_usd": self.limit_usd,
                "settled_usd": 0.0,
                "pending": {},
            }
            if self.protocol == 2:
                state.update({
                    "input_rate_usd_per_million": self.input_rate,
                    "output_rate_usd_per_million": self.output_rate,
                    "config_fingerprint": self.fingerprint,
                })
            return state
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            if (
                not isinstance(state, dict) or state.get("version") != self.protocol
                or float(state.get("limit_usd")) != self.limit_usd
                or not isinstance(state.get("pending"), dict)
            ):
                raise ValueError("unexpected cost-budget state")
            if self.protocol == 2 and (
                float(state.get("input_rate_usd_per_million")) != self.input_rate
                or float(state.get("output_rate_usd_per_million")) != self.output_rate
                or state.get("config_fingerprint") != self.fingerprint
            ):
                raise ValueError("cost-budget cap/rates fingerprint mismatch")
            return state
        except Exception as exc:
            raise TokenCostBudgetExceeded(
                f"Token-cost budget state is unreadable or mismatched; refusing API calls: {path}"
            ) from exc

    def _write_state(self, state: Dict[str, Any]) -> None:
        path = self.state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temp_path.open("x", encoding="utf-8") as stream:
                json.dump(state, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, path)
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def reserve(self, reservation_id: str, projected_usd: float) -> Dict[str, Any]:
        amount = max(0.0, float(projected_usd))
        with self._lock:
            with _interprocess_file_lock(self.state_path):
                state = self._read_state()
                pending = state["pending"]
                committed = float(state.get("settled_usd", 0.0))
                if reservation_id in pending:
                    raise TokenCostBudgetExceeded("Duplicate live cost reservation ID.")
                reserved_total = sum(float(value) for value in pending.values())
                if committed + reserved_total + amount > self.limit_usd + 1e-12:
                    raise TokenCostBudgetExceeded(
                        "COST_BUDGET_EXCEEDED: conservative request-token reservation would exceed "
                        f"${self.limit_usd:.2f} (settled ${committed:.6f}, pending ${reserved_total:.6f}, "
                        f"requested ${amount:.6f})."
                    )
                pending[reservation_id] = amount
                state["pending"] = pending
                state["last_reservation_id"] = reservation_id
                state["last_reserved_usd"] = amount
                self._write_state(state)
                return {
                    "limit_usd": self.limit_usd,
                    "settled_usd": committed,
                    "pending_usd": reserved_total + amount,
                    "reserved_usd": amount,
                }

    def settle(self, reservation_id: str, actual_usd: float) -> Dict[str, Any]:
        actual = max(0.0, float(actual_usd))
        with self._lock:
            with _interprocess_file_lock(self.state_path):
                state = self._read_state()
                pending = state["pending"]
                reserved = pending.pop(reservation_id, None)
                if reserved is None:
                    return {"settled": False, "reason": "reservation_missing"}
                settled = float(state.get("settled_usd", 0.0)) + actual
                state["pending"] = pending
                state["settled_usd"] = settled
                state["last_settled_usd"] = actual
                state["over_limit_after_settlement"] = settled + sum(float(value) for value in pending.values()) > self.limit_usd + 1e-12
                self._write_state(state)
                return {
                    "settled": True,
                    "reserved_usd": float(reserved),
                    "actual_estimated_usd": actual,
                    "cumulative_settled_usd": settled,
                    "limit_usd": self.limit_usd,
                    "over_limit_after_settlement": state["over_limit_after_settlement"],
                }

    def estimate_actual(self, input_tokens: int, output_tokens: int) -> float:
        return (
            max(0, int(input_tokens)) * self.input_rate
            + max(0, int(output_tokens)) * self.output_rate
        ) / 1_000_000.0


_ACTIVE_CALL: ContextVar[Optional[dict]] = ContextVar("active_request_budget_call", default=None)


def request_budget_environment_configured() -> bool:
    return bool(
        os.environ.get("MAADBENCH_API_REQUEST_BUDGET_LIMIT")
        or os.environ.get("MAADBENCH_API_REQUEST_BUDGET_STATE_PATH")
        or os.environ.get("TRAIL_API_REQUEST_BUDGET_LIMIT")
        or os.environ.get("TRAIL_API_REQUEST_BUDGET_STATE_PATH")
    )


def note_budgeted_http_attempt() -> None:
    """Reserve each actual HTTP attempt before the event-hook transport sends it."""
    active = _ACTIVE_CALL.get()
    if active is None:
        return
    if not active.get("count_at_http_event") and active["http_attempts"] == 0:
        # The enclosing SDK call already reserved the first attempt.
        active["http_attempts"] = 1
        return
    active["budget"].reserve()
    active["http_attempts"] += 1


class PersistentRequestBudget:
    """Persistently counts request dispatches, or actual HTTP attempts in v2."""

    LIMIT_ENV = "MAADBENCH_API_REQUEST_BUDGET_LIMIT"
    STATE_ENV = "MAADBENCH_API_REQUEST_BUDGET_STATE_PATH"
    TRAIL_LIMIT_ENV = "TRAIL_API_REQUEST_BUDGET_LIMIT"
    TRAIL_STATE_ENV = "TRAIL_API_REQUEST_BUDGET_STATE_PATH"
    TRAIL_PROTOCOL_ENV = "TRAIL_API_REQUEST_BUDGET_PROTOCOL"

    def __init__(self, limit: Optional[int], state_path: Optional[str], protocol: int = 1):
        self.limit = limit
        self.state_path = Path(state_path).expanduser() if state_path else None
        self.protocol = int(protocol)
        self.fingerprint = _ledger_fingerprint({"protocol": self.protocol, "limit": self.limit})
        self._lock = _thread_lock_for(self.state_path) if self.state_path else threading.Lock()

    @classmethod
    def from_environment(cls) -> "PersistentRequestBudget":
        maad_limit = os.environ.get(cls.LIMIT_ENV)
        maad_path = os.environ.get(cls.STATE_ENV)
        trail_limit = os.environ.get(cls.TRAIL_LIMIT_ENV)
        trail_path = os.environ.get(cls.TRAIL_STATE_ENV)
        if (maad_limit or maad_path) and (trail_limit or trail_path):
            raise RequestBudgetError(
                "Only one dataset-specific persistent request budget may be active per process."
            )
        if trail_limit or trail_path:
            raw_limit, raw_path = trail_limit, trail_path
            limit_env, state_env = cls.TRAIL_LIMIT_ENV, cls.TRAIL_STATE_ENV
            protocol_raw = os.environ.get(cls.TRAIL_PROTOCOL_ENV, "1")
        else:
            raw_limit, raw_path = maad_limit, maad_path
            limit_env, state_env = cls.LIMIT_ENV, cls.STATE_ENV
            protocol_raw = "1"
        if not raw_limit and not raw_path:
            if trail_limit is None and os.environ.get(cls.TRAIL_PROTOCOL_ENV):
                raise RequestBudgetError(f"{cls.TRAIL_PROTOCOL_ENV} requires an active request budget.")
            return cls(None, None)
        if not raw_limit or not raw_path:
            raise RequestBudgetError(
                f"Both {limit_env} and {state_env} must be set together."
            )
        try:
            limit = int(raw_limit)
            protocol = int(protocol_raw)
        except ValueError as exc:
            raise RequestBudgetError(f"{limit_env} must be a positive integer.") from exc
        if limit < 1 or protocol not in {1, 2}:
            raise RequestBudgetError(f"{limit_env} must be a positive integer.")
        return cls(limit, raw_path, protocol)

    @contextmanager
    def transport_call(self, *, count_at_http_event: bool = False) -> Iterator[None]:
        """Use legacy call-entry counting or opt-in HTTP-event attempt counting."""
        event_accounting = bool(count_at_http_event and self.protocol == 2)
        if not event_accounting:
            self.reserve()
        call_state = {"budget": self, "http_attempts": 0, "count_at_http_event": event_accounting}
        token = _ACTIVE_CALL.set(call_state)
        try:
            yield
        finally:
            _ACTIVE_CALL.reset(token)

    def reserve(self) -> None:
        if self.limit is None or self.state_path is None:
            return
        with self._lock:
            path = self.state_path
            path.parent.mkdir(parents=True, exist_ok=True)
            with _interprocess_file_lock(path):
                if path.exists():
                    try:
                        state = json.loads(path.read_text(encoding="utf-8"))
                        if not isinstance(state, dict) or state.get("version") != self.protocol:
                            raise ValueError("unsupported state format")
                        request_count = int(state.get("request_count", -1))
                        if request_count < 0:
                            raise ValueError("invalid request_count")
                        if self.protocol == 2 and (
                            int(state.get("max_api_requests", -1)) != self.limit
                            or state.get("config_fingerprint") != self.fingerprint
                        ):
                            raise ValueError("request-budget cap fingerprint mismatch")
                    except Exception as exc:
                        raise RequestBudgetError(
                            f"Request-budget state is unreadable or mismatched; refusing API calls: {path}"
                        ) from exc
                else:
                    request_count = 0

                if request_count >= self.limit:
                    raise RequestBudgetExceeded(
                        f"API request budget exhausted ({request_count}/{self.limit}); "
                        "no further request was sent. Increase the cap to resume from checkpoints."
                    )

                state = {
                    "version": self.protocol,
                    "request_count": request_count + 1,
                    "max_api_requests": self.limit,
                    "last_reserved_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                if self.protocol == 2:
                    state["config_fingerprint"] = self.fingerprint
                temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
                try:
                    with temp_path.open("x", encoding="utf-8") as stream:
                        json.dump(state, stream, ensure_ascii=False, indent=2)
                        stream.write("\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    _replace_ledger_file(temp_path, path)
                finally:
                    try:
                        temp_path.unlink(missing_ok=True)
                    except OSError:
                        pass
