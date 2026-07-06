from __future__ import annotations

import logging
import re
import time
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urljoin, urlsplit

import httpx

from .settings import Settings


RawFiles = dict[str, str | bytes]
MineruApiResult = dict[str, object]
logger = logging.getLogger(__name__)


class MineruApiError(RuntimeError):
    """Base exception for MinerU API client failures."""


class MineruApiTokenMissingError(MineruApiError):
    pass


class MineruApiUploadUrlError(MineruApiError):
    pass


class MineruApiUploadError(MineruApiError):
    pass


class MineruApiParseError(MineruApiError):
    pass


class MineruApiTimeoutError(MineruApiError):
    pass


class MineruApiNoMarkdownError(MineruApiError):
    pass


@dataclass(frozen=True)
class MineruApiClientConfig:
    api_base: str = "https://mineru.net/api/v4"
    api_token: str = ""
    poll_interval_seconds: float = 3.0
    timeout_seconds: float = 600.0
    enable_formula: bool = True
    enable_table: bool = True
    is_ocr: bool = True

    @classmethod
    def from_settings(cls, settings: Settings) -> "MineruApiClientConfig":
        return cls(
            api_base=settings.mineru_api_base,
            api_token=settings.mineru_api_token,
            poll_interval_seconds=settings.mineru_api_poll_interval_seconds,
            timeout_seconds=settings.mineru_api_timeout_seconds,
            enable_formula=settings.mineru_api_enable_formula,
            enable_table=settings.mineru_api_enable_table,
            is_ocr=settings.mineru_api_is_ocr,
        )


class MineruApiClient:
    def __init__(
        self,
        config: MineruApiClientConfig,
        *,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.client = client or httpx.Client(timeout=config.timeout_seconds)
        self._sleep = sleep
        self._clock = clock

    def parse_pdf(self, path: Path) -> MineruApiResult:
        self._ensure_token()
        upload_info = self._request_upload_url(path)
        self._upload_pdf(upload_info["upload_url"], path)
        zip_url = self._poll_zip_url(upload_info["batch_id"])
        raw_files = self._download_zip(zip_url)
        markdown = self._extract_markdown(raw_files)
        return {
            "success": True,
            "markdown": markdown,
            "raw_files": raw_files,
            "metadata": {
                "parser": "mineru_api",
                "enable_formula": self.config.enable_formula,
                "enable_table": self.config.enable_table,
                "is_ocr": self.config.is_ocr,
            },
        }

    def _request_upload_url(self, path: Path) -> dict[str, str]:
        response = self.client.post(
            self._api_url("/file-urls/batch"),
            headers=self._auth_headers(),
            json={
                "files": [
                    {
                        "name": path.name,
                        "enable_formula": self.config.enable_formula,
                        "enable_table": self.config.enable_table,
                        "is_ocr": self.config.is_ocr,
                    }
                ],
                "enable_formula": self.config.enable_formula,
                "enable_table": self.config.enable_table,
                "is_ocr": self.config.is_ocr,
            },
        )
        if response.status_code < 200 or response.status_code >= 300:
            raise MineruApiUploadUrlError(f"Failed to get upload URL: HTTP {response.status_code}")
        payload = self._json_response(response, MineruApiUploadUrlError)
        self._ensure_business_success(payload, MineruApiUploadUrlError)
        data = _payload_data(payload)
        batch_id = _string_value(data, "batch_id", "batchId", "id")
        upload_url = _find_upload_url(data)
        if not batch_id or not upload_url:
            raise MineruApiUploadUrlError("MinerU upload URL response is missing batch_id or upload_url.")
        return {"batch_id": batch_id, "upload_url": upload_url}

    def _upload_pdf(self, upload_url: str, path: Path) -> None:
        response = self.client.put(
            upload_url,
            content=path.read_bytes(),
            headers={},
        )
        _log_upload_response(upload_url, response)
        if response.status_code < 200 or response.status_code >= 300:
            raise MineruApiUploadError(f"Failed to upload PDF: HTTP {response.status_code}")

    def _poll_zip_url(self, batch_id: str) -> str:
        deadline = self._clock() + self.config.timeout_seconds
        while True:
            response = self.client.get(
                self._api_url(f"/extract-results/batch/{batch_id}"),
                headers=self._auth_headers(),
            )
            if response.status_code < 200 or response.status_code >= 300:
                raise MineruApiParseError(f"Failed to poll parse result: HTTP {response.status_code}")
            payload = self._json_response(response, MineruApiParseError)
            self._ensure_business_success(payload, MineruApiParseError)
            status = (_find_status(payload) or "").lower()
            if status in {"failed", "fail", "error", "canceled", "cancelled"}:
                message = _find_message(payload) or "MinerU parsing failed."
                raise MineruApiParseError(message)
            zip_url = _find_zip_url(payload)
            if status in {"success", "succeeded", "done", "finished", "completed", "complete"} or zip_url:
                if not zip_url:
                    raise MineruApiParseError("MinerU parse result is complete but missing ZIP download URL.")
                return zip_url
            if self._clock() >= deadline:
                raise MineruApiTimeoutError(f"MinerU parsing timed out after {self.config.timeout_seconds:g} seconds.")
            remaining = max(0.0, deadline - self._clock())
            self._sleep(min(self.config.poll_interval_seconds, remaining))

    def _download_zip(self, zip_url: str) -> RawFiles:
        response = self.client.get(self._absolute_url(zip_url))
        if response.status_code < 200 or response.status_code >= 300:
            raise MineruApiParseError(f"Failed to download MinerU ZIP: HTTP {response.status_code}")
        try:
            with zipfile.ZipFile(BytesIO(response.content)) as archive:
                raw_files: RawFiles = {}
                for name in sorted(archive.namelist()):
                    if name.endswith("/"):
                        continue
                    content = archive.read(name)
                    raw_files[name] = _decode_text(content) if _is_text_file(name) else content
                return raw_files
        except zipfile.BadZipFile as exc:
            raise MineruApiParseError("MinerU result download is not a valid ZIP file.") from exc

    @staticmethod
    def _extract_markdown(raw_files: RawFiles) -> str:
        markdown_parts = [
            str(raw_files[name]).strip()
            for name in sorted(raw_files)
            if _is_markdown_file(name) and str(raw_files[name]).strip()
        ]
        if not markdown_parts:
            raise MineruApiNoMarkdownError("MinerU result ZIP does not contain Markdown content.")
        return "\n\n".join(markdown_parts)

    def _ensure_token(self) -> None:
        if not self.config.api_token.strip():
            raise MineruApiTokenMissingError("MINERU_API_TOKEN is required for MinerU API parsing.")

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.config.api_token.strip()}"}

    def _api_url(self, path: str) -> str:
        return f"{self.config.api_base.rstrip('/')}/{path.lstrip('/')}"

    def _absolute_url(self, url: str) -> str:
        if url.startswith(("http://", "https://")):
            return url
        return urljoin(f"{self.config.api_base.rstrip('/')}/", url.lstrip("/"))

    @staticmethod
    def _json_response(response: httpx.Response, error_type: type[MineruApiError]) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise error_type("MinerU API returned invalid JSON.") from exc
        if not isinstance(payload, dict):
            raise error_type("MinerU API returned an unexpected JSON payload.")
        return payload

    @staticmethod
    def _ensure_business_success(payload: dict[str, Any], error_type: type[MineruApiError]) -> None:
        code = payload.get("code")
        success = payload.get("success")
        if success is False:
            raise error_type(_find_message(payload) or "MinerU API request failed.")
        if code is None:
            return
        if code in (0, 200, "0", "200"):
            return
        raise error_type(_find_message(payload) or f"MinerU API request failed with code {code}.")


def _payload_data(payload: dict[str, Any]) -> Any:
    return payload.get("data", payload)


def _string_value(payload: Any, *keys: str) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, int):
            return str(value)
    return None


def _find_upload_url(payload: Any) -> str | None:
    if isinstance(payload, str) and payload.strip():
        return payload.strip()
    if isinstance(payload, dict):
        direct = _string_value(payload, "upload_url", "uploadUrl", "url")
        if direct:
            return direct
        for key in ("file_urls", "fileUrls", "upload_urls", "uploadUrls", "urls", "files"):
            value = payload.get(key)
            found = _find_upload_url(value)
            if found:
                return found
    if isinstance(payload, list):
        for item in payload:
            found = _find_upload_url(item)
            if found:
                return found
    return None


def _find_zip_url(payload: Any) -> str | None:
    if isinstance(payload, dict):
        direct = _string_value(payload, "full_zip_url", "zip_url", "download_url", "fullZipUrl", "zipUrl", "downloadUrl")
        if direct:
            return direct
        for value in payload.values():
            found = _find_zip_url(value)
            if found:
                return found
    if isinstance(payload, list):
        for item in payload:
            found = _find_zip_url(item)
            if found:
                return found
    return None


def _find_status(payload: Any) -> str | None:
    if isinstance(payload, dict):
        direct = _string_value(payload, "status", "state")
        if direct:
            return direct
        data = payload.get("data")
        if data is not payload:
            found = _find_status(data)
            if found:
                return found
    if isinstance(payload, list):
        for item in payload:
            found = _find_status(item)
            if found:
                return found
    return None


def _find_message(payload: Any) -> str | None:
    if isinstance(payload, dict):
        direct = _string_value(payload, "message", "msg", "error")
        if direct:
            return direct
        data = payload.get("data")
        if data is not payload:
            found = _find_message(data)
            if found:
                return found
    return None


def _is_markdown_file(name: str) -> bool:
    return Path(name).suffix.lower() in {".md", ".markdown"}


def _is_text_file(name: str) -> bool:
    return Path(name).suffix.lower() in {".md", ".markdown", ".txt"}


def _decode_text(content: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def _log_upload_response(upload_url: str, response: httpx.Response) -> None:
    message = (
        "MinerU upload PUT response status_code=%s response_body_prefix=%r "
        "file_url_has_query=%s file_url_length=%d"
    )
    args = (
        response.status_code,
        _safe_response_body_prefix(response),
        bool(urlsplit(upload_url).query),
        len(upload_url),
    )
    if response.status_code < 200 or response.status_code >= 300:
        logger.warning(message, *args)
    else:
        logger.debug(message, *args)


def _safe_response_body_prefix(response: httpx.Response) -> str:
    text = response.content[:2000].decode("utf-8", errors="replace")
    return _redact_sensitive_debug_text(text)[:1000]


def _redact_sensitive_debug_text(text: str) -> str:
    text = re.sub(r"(?im)(Authorization\s*:\s*).*", r"\1<redacted>", text)
    for key in (
        "Authorization",
        "Signature",
        "OSSAccessKeyId",
        "AccessKeyId",
        "SecurityToken",
        "x-oss-security-token",
        "X-Amz-Signature",
        "X-Amz-Credential",
    ):
        escaped = re.escape(key)
        text = re.sub(rf"(?i)({escaped}\s*[:=]\s*)(\"[^\"]*\"|'[^']*'|[^&\s<]+)", r"\1<redacted>", text)
        text = re.sub(rf"(?is)(<{escaped}>).*?(</{escaped}>)", r"\1<redacted>\2", text)
    return text
