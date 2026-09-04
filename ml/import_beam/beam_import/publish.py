"""Putting a file on the Hugging Face Hub, and proving it came back.

THE RULE: an upload that returns 200 is not evidence the bytes are
retrievable. Every file is uploaded, DOWNLOADED BACK, and sha256-compared
against the local bytes; only then is the local copy deleted. Every prior
pipeline in this repository does this and this one is not the exception.

Two implementations behind one interface:

    hf:<repo_id>     the real Hub
    local:<dir>      a directory that behaves like the Hub — used by the
                     tests, so the whole pipeline can be exercised offline

Commit budget. The Hub enforces an hourly commit quota ("You have exceeded
our hourly quotas for action: commit"). Files are batched `batch_files` at a
time, and each lane additionally waits so that ALL lanes together stay under
`commit_budget_per_hour`: with a budget of 60 and 27 lanes, a lane commits at
most once every 27 * 60 = 1620 s. Fetching takes far longer than that in
practice, so the limiter almost never binds — it is there for the case where
it would.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .hosts import PermanentError, TransientError

# One entry to publish: the local file, where it goes on the Hub.
Entry = Tuple[str, str]


def sha256_of(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class BasePublisher:
    """What a LaneWorker needs from wherever the mirror lives."""

    def __init__(self, commit_min_interval_s: float = 0.0,
                 sleep_fn: Callable[[float], None] = time.sleep,
                 clock_fn: Callable[[], float] = time.monotonic) -> None:
        self.commit_min_interval_s = float(commit_min_interval_s)
        self._sleep = sleep_fn
        self._clock = clock_fn
        self._last_commit_at: Optional[float] = None
        self.commits = 0

    # -- the commit-rate limiter ------------------------------------------
    def _wait_for_commit_slot(self) -> None:
        if self.commit_min_interval_s <= 0 or self._last_commit_at is None:
            return
        wait = self.commit_min_interval_s - (self._clock() - self._last_commit_at)
        if wait > 0:
            self._sleep(wait)

    def _mark_commit(self) -> None:
        self._last_commit_at = self._clock()
        self.commits += 1

    # -- the interface -----------------------------------------------------
    def list_paths(self) -> List[str]:                        # pragma: no cover
        raise NotImplementedError

    def exists(self, hub_path: str) -> bool:                  # pragma: no cover
        raise NotImplementedError

    def _upload(self, entries: Sequence[Entry], message: str) -> None:
        raise NotImplementedError                             # pragma: no cover

    def _download_back(self, hub_path: str, dest_dir: str) -> str:
        raise NotImplementedError                             # pragma: no cover

    # -- the one method the worker calls ----------------------------------
    def publish_verified(self, entries: Sequence[Entry], scratch: str,
                         message: str, note=None) -> Dict[str, str]:
        """Upload a batch, download every file back, compare sha256.

        Returns {hub_path: sha256}. Raises TransientError if anything about
        the round trip disagrees — a mismatch is worth another attempt before
        it is worth a human.
        """
        want = {hub: sha256_of(local) for local, hub in entries}
        self._wait_for_commit_slot()
        self._upload(entries, message)
        self._mark_commit()

        verify_dir = os.path.join(scratch, "_verify")
        os.makedirs(verify_dir, exist_ok=True)
        try:
            for _, hub in entries:
                back = self._download_back(hub, verify_dir)
                got = sha256_of(back)
                if got != want[hub]:
                    raise TransientError(
                        f"restore-verify FAILED for {hub}: uploaded "
                        f"{want[hub][:12]}, got back {got[:12]}")
                os.remove(back)
                if note:
                    note(f"verified {hub} ({want[hub][:12]})")
        finally:
            shutil.rmtree(verify_dir, ignore_errors=True)

        for local, _ in entries:                # only now is it safe to delete
            if os.path.exists(local):
                os.remove(local)
        return want


class LocalPublisher(BasePublisher):
    """A directory pretending to be the Hub. Tests only — but it exercises the
    same batching, the same restore-verify and the same delete-after-verify."""

    def __init__(self, root: str, **kw: Any) -> None:
        super().__init__(**kw)
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

    def list_paths(self) -> List[str]:
        out = []
        for dirpath, _dirs, files in os.walk(self.root):
            for f in files:
                p = os.path.join(dirpath, f)
                out.append(os.path.relpath(p, self.root).replace(os.sep, "/"))
        return sorted(out)

    def exists(self, hub_path: str) -> bool:
        return os.path.exists(os.path.join(self.root, hub_path))

    def _upload(self, entries: Sequence[Entry], message: str) -> None:
        for local, hub in entries:
            dest = os.path.join(self.root, hub)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copyfile(local, dest)

    def _download_back(self, hub_path: str, dest_dir: str) -> str:
        src = os.path.join(self.root, hub_path)
        dest = os.path.join(dest_dir, os.path.basename(hub_path))
        shutil.copyfile(src, dest)
        return dest


class HubPublisher(BasePublisher):
    """The real Hugging Face dataset repository."""

    def __init__(self, repo_id: str, repo_type: str = "dataset",
                 token: Optional[str] = None, **kw: Any) -> None:
        super().__init__(**kw)
        from huggingface_hub import HfApi
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.token = token or read_token()
        self.api = HfApi(token=self.token)
        self._listing: Optional[set] = None
        # The NAMESPACE trap: `blauewelt` is the GitHub organisation and does
        # NOT exist on the Hub. Resolve who we actually are and say so if the
        # registry disagrees, rather than failing later with a 403.
        try:
            me = self.api.whoami().get("name")
            if me and "/" in repo_id and not repo_id.startswith(me + "/"):
                print(f"warning: HF_TOKEN belongs to {me!r} but sources.yaml "
                      f"says {repo_id!r}. If that is not an org you belong to, "
                      "the commit will be refused.")
        except Exception as exc:                              # noqa: BLE001
            raise PermanentError(f"Hub whoami failed: {exc}") from exc

    def list_paths(self) -> List[str]:
        from huggingface_hub import list_repo_files
        return sorted(list_repo_files(self.repo_id, repo_type=self.repo_type,
                                      token=self.token))

    def exists(self, hub_path: str) -> bool:
        try:
            return self.api.file_exists(self.repo_id, hub_path,
                                        repo_type=self.repo_type)
        except Exception as exc:                              # noqa: BLE001
            raise TransientError(f"file_exists {hub_path}: {exc}") from exc

    def _upload(self, entries: Sequence[Entry], message: str) -> None:
        from huggingface_hub import CommitOperationAdd
        ops = [CommitOperationAdd(path_in_repo=hub, path_or_fileobj=local)
               for local, hub in entries]
        try:
            self.api.create_commit(repo_id=self.repo_id,
                                   repo_type=self.repo_type,
                                   operations=ops, commit_message=message)
        except Exception as exc:                              # noqa: BLE001
            raise _hub_error(exc, f"commit of {len(ops)} file(s)") from exc

    def _download_back(self, hub_path: str, dest_dir: str) -> str:
        from huggingface_hub import hf_hub_download
        try:
            got = hf_hub_download(self.repo_id, hub_path,
                                  repo_type=self.repo_type,
                                  local_dir=dest_dir, token=self.token)
        except Exception as exc:                              # noqa: BLE001
            raise _hub_error(exc, f"download-back of {hub_path}") from exc
        # hf_hub_download puts the file at local_dir/<hub_path>; move it flat
        # so the caller can delete it without walking a tree.
        flat = os.path.join(dest_dir, "_back_" + os.path.basename(hub_path))
        shutil.move(got, flat)
        return flat


def _hub_error(exc: Exception, what: str) -> Exception:
    """Turn a Hub exception into our vocabulary, honouring Retry-After."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    text = str(exc)
    if status == 429 or "hourly quota" in text.lower():
        err = TransientError(f"Hub rate limit on {what}: {text}")
        ra = getattr(getattr(exc, "response", None), "headers", {}) or {}
        try:
            err.retry_after = float(ra.get("Retry-After"))     # type: ignore[attr-defined]
        except (TypeError, ValueError):
            err.retry_after = 3600.0                           # type: ignore[attr-defined]
        return err
    if status in (409, 412):
        # Somebody else committed while we were preparing ours. Retrying is
        # correct: our operations are adds at distinct paths.
        return TransientError(f"Hub commit conflict on {what}: {text}")
    if status in (401, 403):
        return PermanentError(f"Hub refused {what} ({status}): {text}")
    return TransientError(f"Hub error on {what}: {text}")


# --------------------------------------------------------------------------
def read_token() -> Optional[str]:
    """HF_TOKEN from the environment, else /home/claude/.hf_token (mode 600).

    Never from argv, never printed. See CREDENTIALS.md.
    """
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok.strip()
    for path in (os.path.expanduser("~/.hf_token"), "/home/claude/.hf_token"):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                return fh.read().strip()
    return None


def commit_min_interval_s(total_lanes: int, budget_per_hour: int) -> float:
    """Seconds a single lane must leave between two commits."""
    if budget_per_hour <= 0:
        return 0.0
    return 3600.0 * max(1, total_lanes) / float(budget_per_hour)


def make_publisher(spec: str, repo_id: str, repo_type: str = "dataset",
                   commit_interval_s: float = 0.0, **kw: Any) -> BasePublisher:
    """`spec` is 'hub', 'hub:<repo_id>' or 'local:<dir>'."""
    if spec.startswith("local:"):
        return LocalPublisher(spec.split(":", 1)[1],
                              commit_min_interval_s=commit_interval_s, **kw)
    if spec.startswith("hub:"):
        repo_id = spec.split(":", 1)[1]
    return HubPublisher(repo_id, repo_type,
                        commit_min_interval_s=commit_interval_s, **kw)


PREFLIGHT_BYTES = b"beam_import hub round-trip preflight\n"    # 29+ bytes


def roundtrip_preflight(pub: BasePublisher, scratch: str) -> str:
    """Upload a tiny file, download it back, compare, delete the local copy.

    Run at the START of every pipeline, before a single upstream byte is
    fetched: a Hub that cannot round-trip 37 bytes will not round-trip 257 MB
    either, and finding that out after an hour of downloading is the expensive
    order to find it out in.
    """
    os.makedirs(scratch, exist_ok=True)
    local = os.path.join(scratch, "_preflight.txt")
    with open(local, "wb") as fh:
        fh.write(PREFLIGHT_BYTES)
    hub_path = "sources/_preflight/roundtrip.txt"
    got = pub.publish_verified([(local, hub_path)], scratch,
                               "beam_import: Hub round-trip preflight")
    return got[hub_path]
