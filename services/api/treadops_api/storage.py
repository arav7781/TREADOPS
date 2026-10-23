"""S3-compatible object storage (SeaweedFS locally). Keys are content-addressed wherever possible."""

from __future__ import annotations

import builtins
import threading
from functools import lru_cache
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from .settings import get_settings


class StorageUnavailable(RuntimeError):
    """Transient storage failure (retryable)."""


class Storage:
    def __init__(self) -> None:
        s = get_settings()
        self.bucket = s.s3_bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=s.s3_endpoint_url,
            aws_access_key_id=s.s3_access_key_id,
            aws_secret_access_key=s.s3_secret_access_key,
            region_name="us-east-1",
            config=Config(retries={"max_attempts": 3, "mode": "standard"}, connect_timeout=5, read_timeout=60,
                          s3={"addressing_style": "path"}),
        )
        self._lock = threading.Lock()

    def ensure_bucket(self) -> None:
        try:
            self._client.head_bucket(Bucket=self.bucket)
        except ClientError:
            self._client.create_bucket(Bucket=self.bucket)

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        try:
            self._client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)
        except Exception as exc:
            raise StorageUnavailable(f"put {key}: {exc.__class__.__name__}") from exc
        return key

    def put_file(self, key: str, path: Path, content_type: str = "application/octet-stream") -> str:
        return self.put(key, path.read_bytes(), content_type)

    def get(self, key: str) -> bytes:
        try:
            return self._client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                raise FileNotFoundError(key) from exc
            raise StorageUnavailable(f"get {key}: {exc}") from exc
        except Exception as exc:
            raise StorageUnavailable(f"get {key}: {exc.__class__.__name__}") from exc

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise StorageUnavailable(str(exc)) from exc

    def list(self, prefix: str) -> list[dict]:
        out: list[dict] = []
        token = None
        while True:
            kw = {"Bucket": self.bucket, "Prefix": prefix}
            if token:
                kw["ContinuationToken"] = token
            resp = self._client.list_objects_v2(**kw)
            out.extend({"key": o["Key"], "size": o["Size"], "last_modified": o["LastModified"]}
                       for o in resp.get("Contents", []))
            if not resp.get("IsTruncated"):
                return out
            token = resp["NextContinuationToken"]

    def delete(self, key: str) -> None:
        self._client.delete_object(Bucket=self.bucket, Key=key)

    def download_prefix(self, prefix: str, dest: Path) -> builtins.list[Path]:
        paths = []
        for obj in self.list(prefix):
            rel = obj["key"][len(prefix):].lstrip("/")
            p = dest / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(self.get(obj["key"]))
            paths.append(p)
        return paths

    def ping(self) -> bool:
        try:
            self._client.head_bucket(Bucket=self.bucket)
            return True
        except Exception:
            return False


@lru_cache
def get_storage() -> Storage:
    return Storage()


def content_type_for(key: str) -> str:
    k = key.lower()
    if k.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if k.endswith(".png"):
        return "image/png"
    if k.endswith(".json"):
        return "application/json"
    if k.endswith(".html"):
        return "text/html"
    return "application/octet-stream"
