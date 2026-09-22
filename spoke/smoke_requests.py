"""Durable interactive-smoke requests; usable without the Spoke application."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import select
import shutil
import subprocess
import tempfile
import threading
from urllib.parse import urlsplit
from uuid import UUID, uuid4

SCHEMA = "spoke.interactive-smoke.v1"


def now():
    return datetime.now(timezone.utc).isoformat()


def request_id(value):
    if not isinstance(value, str) or str(UUID(value)) != value:
        raise ValueError("request id must be a canonical UUID")
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def timestamp(value, field):
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field} must be a timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return value


def validate_request(value):
    if not isinstance(value, dict) or value.get("kind") != "interactive-smoke":
        raise ValueError("only explicit interactive-smoke requests are supported")
    request_id(value.get("id"))
    source = value.get("source", {})
    if not isinstance(source, dict) or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", source.get("diaulos", "")):
        raise ValueError("source must identify the requesting agent")
    request_id(source.get("thread_id"))
    root = source.get("repo_root")
    if not isinstance(root, str) or not Path(root).is_absolute():
        raise ValueError("source repo_root must be absolute")
    for field in ("title", "prompt", "url", "availability_note"):
        if not isinstance(value.get(field), str) or not value[field].strip():
            raise ValueError(f"missing {field}")
    if value.get("availability") not in {"prepared", "preparation-needed", "unavailable"}:
        raise ValueError("invalid reported availability")
    url = urlsplit(value["url"])
    if (url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password
            or any(ord(char) < 32 for char in value["url"])):
        raise ValueError("smoke URL must be HTTP(S), without credentials or control characters")
    return deepcopy(value)


def default_queue():
    return Path(os.environ.get("SPOKE_SMOKE_REQUEST_DIR") or
                Path.home() / ".local/state/spoke/interactive-smokes").expanduser()


class SmokeRequests:
    """One canonical state file per request; locked edits and durable replacement."""

    def __init__(self, directory):
        self.directory = Path(directory).expanduser().absolute()
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)

    def path(self, identity):
        return self.directory / f"{request_id(identity)}.json"

    @contextmanager
    def _locked(self, identity):
        with self.path(identity).with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    @contextmanager
    def _delivery_locked(self, identity):
        with self.path(identity).with_suffix(".delivery.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False
                return
            try:
                yield True
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _write(self, row):
        destination = self.path(row["request"]["id"])
        fd, name = tempfile.mkstemp(prefix=f".{destination.stem}.", dir=self.directory)
        try:
            with os.fdopen(fd, "w") as file:
                json.dump(row, file, ensure_ascii=True, indent=2)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(name, destination)
            directory_fd = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            Path(name).unlink(missing_ok=True)
        return row

    def get(self, identity):
        row = json.loads(self.path(identity).read_text())
        if row.get("schema") != SCHEMA or row.get("request", {}).get("id") != identity:
            raise ValueError("request schema or identity mismatch")
        validate_request(row["request"])
        if row.get("request_digest") != digest(row["request"]):
            raise ValueError("request digest mismatch")
        timestamp(row.get("created_at"), "created_at")
        for field in ("later_at", "opened_at"):
            if row.get(field) is not None:
                timestamp(row[field], field)
        notification = row.get("notification")
        if notification is not None:
            if not isinstance(notification, dict) or notification.get("state") not in {
                    "attempted", "requested", "unavailable"}:
                raise ValueError("invalid notification state")
            timestamp(notification.get("at"), "notification.at")
        response = row.get("response")
        if response is not None and (not isinstance(response, dict)
                                    or response.get("request_digest") != row["request_digest"]
                                    or response.get("source") != "operator-spoke"
                                    or not isinstance(response.get("text"), str)):
            raise ValueError("response digest or source mismatch")
        if response is not None:
            timestamp(response.get("at"), "response.at")
        delivery = row.get("delivery")
        if delivery is not None:
            if not isinstance(delivery, dict) or delivery.get("state") not in {
                    "sending", "delivered", "unconfirmed"}:
                raise ValueError("invalid delivery state")
            request_id(delivery.get("attempt"))
            timestamp(delivery.get("at"), "delivery.at")
            if delivery["state"] != "sending":
                timestamp(delivery.get("finished_at"), "delivery.finished_at")
                if not isinstance(delivery.get("receipt"), dict):
                    raise ValueError("terminal delivery needs a receipt")
        if row.get("status") not in {"pending", "responded", "withdrawn"}:
            raise ValueError("unknown request status")
        if row["status"] == "responded" and response is None:
            raise ValueError("responded state has no response")
        if row["status"] == "pending" and response is not None:
            raise ValueError("pending state cannot have a response")
        if row["status"] == "withdrawn":
            if not isinstance(row.get("withdrawal_reason"), str) or not row["withdrawal_reason"].strip():
                raise ValueError("withdrawn state needs a reason")
            timestamp(row.get("withdrawn_at"), "withdrawn_at")
        return row

    def submit(self, value):
        request = validate_request(value)
        with self._locked(request["id"]):
            if self.path(request["id"]).exists():
                existing = self.get(request["id"])
                if existing["request"] != request:
                    raise ValueError("request identity conflict")
                return existing
            return self._write({"schema": SCHEMA, "request": request,
                "receipt_path": str(self.path(request["id"])),
                "request_digest": digest(request), "created_at": now(),
                "status": "pending", "response": None, "delivery": None,
                "notification": None, "later_at": None, "opened_at": None})

    def scan(self):
        rows, errors = [], []
        for file in sorted(self.directory.glob("*.json")):
            try:
                rows.append(self.get(file.stem))
            except (OSError, ValueError, TypeError, AttributeError, KeyError) as error:
                errors.append(f"{file.name}: {error}")
        rows.sort(key=lambda row: (row["status"] != "pending", row["created_at"]))
        return rows, errors

    def act(self, identity, action, reason=""):
        with self._locked(identity):
            row = self.get(identity)
            if action == "withdraw":
                if not reason.strip():
                    raise ValueError("withdrawal needs a reason")
                row.update(status="withdrawn", withdrawal_reason=reason, withdrawn_at=now())
            elif action in {"later", "opened"}:
                row[f"{action}_at"] = now()
            else:
                raise ValueError("unsupported request action")
            return self._write(row)

    def claim_notifications(self):
        claimed = []
        rows, _ = self.scan()
        for snapshot in rows:
            identity = snapshot["request"]["id"]
            with self._locked(identity):
                try:
                    row = self.get(identity)
                except (OSError, ValueError, TypeError, AttributeError, KeyError):
                    continue
                if row["status"] != "pending" or row["notification"] or row["later_at"]:
                    continue
                row["notification"] = {"state": "attempted", "at": now()}
                claimed.append(self._write(row))
        return claimed

    def notification_result(self, identity, state, error=""):
        with self._locked(identity):
            row = self.get(identity)
            row["notification"] = {"state": state, "at": now(), "error": error}
            return self._write(row)

    def reply(self, identity, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("response is empty")
        with self._locked(identity):
            row = self.get(identity)
            if row["status"] == "withdrawn":
                raise ValueError("request has been withdrawn")
            if row["response"] is not None:
                if row["response"]["text"] != text:
                    raise ValueError("response already saved; conflicting response refused")
                return row
            row.update(status="responded", response={"text": text, "at": now(),
                "source": "operator-spoke", "request_digest": row["request_digest"]})
            return self._write(row)

    def deliver(self, identity, send, *, retry=False):
        # The production sender uses a stable peer request id. Explicit retry
        # therefore recovers a crashed local sender without creating a new turn.
        with self._delivery_locked(identity) as acquired:
            if not acquired:
                return self.get(identity)
            with self._locked(identity):
                row = self.get(identity)
                if row["response"] is None:
                    raise ValueError("no saved response")
                previous = row["delivery"]
                if previous and (previous["state"] == "delivered" or not retry):
                    return row
                attempt = str(uuid4())
                row["delivery"] = {"state": "sending", "attempt": attempt, "at": now(),
                                   "previous": previous}
                self._write(row)
            try:
                receipt = send(row)
            except Exception as error:
                receipt = {"transport_verified": False, "error": str(error)}
            with self._locked(identity):
                row = self.get(identity)
                if row["delivery"].get("attempt") != attempt:
                    return row
                row["delivery"].update(
                    state="delivered" if receipt.get("transport_verified") is True else "unconfirmed",
                    finished_at=now(), receipt=receipt)
                return self._write(row)


def return_response(row, *, executable=None, registry=None, runner=subprocess.run):
    request = row["request"]
    source = request["source"]
    identity = f"smoke-response-{request['id']}"
    executable = executable or shutil.which("epistaxis", path=os.pathsep.join([
        str(Path.home() / ".local/bin"), "/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin"]))
    if not executable:
        raise OSError("agent return command is unavailable")
    text = ("[Spoke interactive smoke response]\n"
            f"Request ID: {request['id']}\nOrigin thread: {source['thread_id']}\n"
            f"Request and response receipt: {row.get('receipt_path', 'not recorded')}\n"
            f"Source repo: {source['repo_root']}\nRequest SHA-256: {row['request_digest']}\n"
            f"Smoke: {request['title']}\nAsked: {request['prompt']}\n"
            f"Exact operator response: {json.dumps(row['response']['text'])}\n"
            "This is a response to the named smoke, not automatic acceptance of its result. "
            "Please reintegrate it at the relevant checkpoint. No acknowledgment-only reply is needed.")
    argv = [str(executable), "pty-broker", "peer-send", "--source-diaulos", source["diaulos"],
            "--target-diaulos", source["diaulos"], "--request-id", identity,
            "--delivery-mode", "immediate", "--immediate-reason", "operator-answer", "--text", text]
    if registry:
        argv += ["--endpoint-registry", str(registry)]
    result = runner(argv, capture_output=True, text=True, check=False)
    try:
        raw = json.loads(result.stdout)
        submit = raw.get("submit_result", {})
        receipt = submit.get("durable_receipt", {})
        valid = (raw.get("schema") == "epistaxis.pty_broker.peer_send_result.v1"
                 and raw.get("source_diaulos") == source["diaulos"]
                 and raw.get("target_diaulos") == source["diaulos"]
                 and submit.get("request_id") == identity
                 and submit.get("target_diaulos") == source["diaulos"]
                 and all(receipt.get(key) is True for key in
                         ("required", "write_verified", "submit_verified")))
        verified = result.returncode == 0 and valid and raw.get("transport_verified") is True
    except (ValueError, AttributeError, TypeError):
        raw, verified = result.stdout, False
    return {"transport_verified": verified, "semantic_receipt": False,
            "raw_receipt": raw, "exit_code": result.returncode, "stderr": result.stderr,
            "error": "" if verified else "Return to requesting agent is unconfirmed"}


def notify_requests(rows, *, runner=subprocess.run):
    """Use the existing source-launch notification route, without interpolating text."""
    title = rows[0]["request"]["title"] if len(rows) == 1 else f"{len(rows)} interactive smokes need you"
    body = "\n".join(f"{row['request']['source']['diaulos']}: {row['request']['prompt']}" for row in rows)
    script = "on run argv\ndisplay notification (item 2 of argv) with title (item 1 of argv)\nend run"
    result = runner(["/usr/bin/osascript", "-e", script, "--", f"Spoke: {title}", body],
                    capture_output=True, text=True, check=False)
    if result.returncode:
        raise OSError(result.stderr.strip() or f"notification command exited {result.returncode}")
    return "requested"


class DirectoryWatch:
    """macOS filesystem events plus an explicit shutdown pipe; no agent polling."""

    def __init__(self, directory, changed, failed):
        self.changed, self.failed = changed, failed
        self._close_lock = threading.Lock()
        self.directory_fd = os.open(directory, os.O_RDONLY)
        self.read_fd, self.write_fd = os.pipe()
        self.events = select.kqueue()
        self.events.control([
            select.kevent(self.directory_fd, filter=select.KQ_FILTER_VNODE,
                          flags=select.KQ_EV_ADD | select.KQ_EV_CLEAR,
                          fflags=select.KQ_NOTE_WRITE | select.KQ_NOTE_RENAME | select.KQ_NOTE_DELETE),
            select.kevent(self.read_fd, filter=select.KQ_FILTER_READ, flags=select.KQ_EV_ADD),
        ], 0)
        self.thread = threading.Thread(target=self._run, daemon=True, name="smoke-inbox-watch")
        self.thread.start()

    def _run(self):
        try:
            self.changed()
            while True:
                events = self.events.control(None, 2)
                if any(event.ident == self.read_fd for event in events):
                    return
                if any(event.fflags & (select.KQ_NOTE_DELETE | select.KQ_NOTE_RENAME) for event in events):
                    raise OSError("request directory moved or disappeared; reopen Spoke after recovery")
                self.changed()
        except Exception as error:
            self.failed(str(error))
        finally:
            self.events.close()
            os.close(self.directory_fd)
            os.close(self.read_fd)

    def close(self):
        with self._close_lock:
            write_fd, self.write_fd = self.write_fd, None
        if write_fd is None:
            return
        try:
            if self.thread.is_alive():
                os.write(write_fd, b"x")
        except OSError:
            pass
        finally:
            os.close(write_fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=default_queue())
    commands = parser.add_subparsers(dest="command", required=True)
    submit = commands.add_parser("submit")
    submit.add_argument("--file", type=Path, required=True)
    commands.add_parser("list")
    for name in ("get", "withdraw", "reply", "deliver"):
        sub = commands.add_parser(name)
        sub.add_argument("--id", required=True)
        if name == "withdraw":
            sub.add_argument("--reason", required=True)
        if name == "reply":
            sub.add_argument("--text-file", type=Path, required=True)
        if name == "deliver":
            sub.add_argument("--retry", action="store_true")
    args = parser.parse_args(argv)
    try:
        queue = SmokeRequests(args.queue)
        if args.command == "submit":
            result = queue.submit(json.loads(args.file.read_text()))
        elif args.command == "list":
            rows, errors = queue.scan()
            result = {"requests": rows, "errors": errors, "queue": str(queue.directory)}
        elif args.command == "withdraw":
            result = queue.act(args.id, "withdraw", args.reason)
        elif args.command == "reply":
            result = queue.reply(args.id, args.text_file.read_text())
        elif args.command == "deliver":
            result = queue.deliver(args.id, return_response, retry=args.retry)
        else:
            result = queue.get(args.id)
        print(json.dumps(result, ensure_ascii=True))
        return 0
    except (OSError, ValueError, TypeError) as error:
        print(json.dumps({"status": "failed", "error": str(error), "queue": str(args.queue)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
