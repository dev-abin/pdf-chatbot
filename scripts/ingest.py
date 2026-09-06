"""Upload one document through the running API, preserving normal validation."""

from __future__ import annotations

import argparse
import json
import mimetypes
from pathlib import Path
from urllib.request import Request, urlopen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path)
    parser.add_argument("--thread-id", required=True)
    parser.add_argument("--token", required=True, help="Bearer access token")
    parser.add_argument("--api-url", default="http://localhost:8000")
    args = parser.parse_args()

    boundary = "----documind-upload-boundary"
    content_type = mimetypes.guess_type(args.file.name)[0] or "application/octet-stream"
    file_bytes = args.file.read_bytes()
    body = b"\r\n".join(
        [
            f"--{boundary}".encode(),
            b'Content-Disposition: form-data; name="thread_id"',
            b"",
            args.thread_id.encode(),
            f"--{boundary}".encode(),
            (
                f'Content-Disposition: form-data; name="file"; '
                f'filename="{args.file.name}"'
            ).encode(),
            f"Content-Type: {content_type}".encode(),
            b"",
            file_bytes,
            f"--{boundary}--".encode(),
            b"",
        ]
    )
    request = Request(
        f"{args.api_url.rstrip('/')}/upload-files/",
        data=body,
        headers={
            "Authorization": f"Bearer {args.token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    with urlopen(request, timeout=120) as response:  # nosec B310: user-selected local API
        print(json.loads(response.read().decode("utf-8")))


if __name__ == "__main__":
    main()
