from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import urlsplit

import requests


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload a PDF directly to a MinerU presigned file_url.")
    parser.add_argument("--file-url", required=True, help="MinerU presigned upload URL returned by the create task API.")
    parser.add_argument("--pdf", required=True, type=Path, help="Local PDF file to upload.")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    with args.pdf.open("rb") as stream:
        response = requests.put(args.file_url, data=stream, headers={}, timeout=args.timeout)

    print(f"PUT status_code={response.status_code}")
    print(f"file_url_has_query={bool(urlsplit(args.file_url).query)}")
    print(f"file_url_length={len(args.file_url)}")
    print(f"response_body_prefix={_safe_body_prefix(response)!r}")
    return 0 if 200 <= response.status_code < 300 else 1


def _safe_body_prefix(response: requests.Response) -> str:
    return _redact_sensitive_debug_text(response.content[:2000].decode("utf-8", errors="replace"))[:1000]


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


if __name__ == "__main__":
    raise SystemExit(main())
