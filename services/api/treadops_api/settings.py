"""Typed runtime configuration from the environment (see .env.example)."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = Field(alias="DATABASE_URL")
    redis_url: str = Field("redis://127.0.0.1:56379/0", alias="REDIS_URL")
    s3_endpoint_url: str = Field(alias="S3_ENDPOINT_URL")
    s3_bucket: str = Field("treadops", alias="S3_BUCKET")
    s3_access_key_id: str = Field(alias="S3_ACCESS_KEY_ID")
    s3_secret_access_key: str = Field(alias="S3_SECRET_ACCESS_KEY")
    mlflow_tracking_uri: str | None = Field(None, alias="MLFLOW_TRACKING_URI")

    jwt_secret: str = Field(alias="JWT_SECRET")
    jwt_issuer: str = Field("treadops-api", alias="JWT_ISSUER")
    jwt_audience: str = Field("treadops-web", alias="JWT_AUDIENCE")
    session_ttl_minutes: int = Field(480, alias="SESSION_TTL_MINUTES")
    media_url_ttl_seconds: int = Field(600, alias="MEDIA_URL_TTL_SECONDS")
    app_origin: str = Field("http://localhost:3000", alias="APP_ORIGIN")

    max_upload_bytes: int = Field(8 * 1024 * 1024, alias="MAX_UPLOAD_BYTES")
    max_decoded_pixels: int = Field(25_000_000, alias="MAX_DECODED_PIXELS")
    inference_queue_limit: int = Field(100, alias="INFERENCE_QUEUE_LIMIT")
    train_device: str = Field("cpu", alias="TRAIN_DEVICE")
    inference_device: str = Field("cpu", alias="INFERENCE_DEVICE")
    bundle_cache_dir: str = Field("models/cache", alias="BUNDLE_CACHE_DIR")
    worker_name: str | None = Field(None, alias="WORKER_NAME")

    llm_base_url: str | None = Field(None, alias="LLM_BASE_URL")
    llm_api_key: str | None = Field(None, alias="LLM_API_KEY")
    llm_model: str | None = Field(None, alias="LLM_MODEL")


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
