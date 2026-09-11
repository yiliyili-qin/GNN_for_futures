"""Download authorized physical-market indicators from SMM Data Pro.

SMM indicators are daily or lower-frequency physical-market series, not
15-minute futures bars. This module therefore stores them separately as raw
API responses plus a normalized long table suitable for later as-of joins.

Credentials are read from ``SMM_USER_NAME`` and ``SMM_PASSWORD``. The password
is MD5-hashed as required by SMM before transmission and is never persisted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests


BASE_URL = "https://datapro-api.smm.cn/dapi"
AUTH_PATH = "/user/auth"
QUOTA_LIST_PATH = "/quota/quota_list"
EXT_INFO_PATH = "/quota/ext_info"
DATA_ORIGIN_PATH = "/quota/data_origin"

DEFAULT_START_DATE = "2016-01-01"
DEFAULT_END_DATE = date.today().isoformat()
MAX_QUOTA_IDS_PER_REQUEST = 100
QUOTA_LIST_INTERVAL = 1.0


class SmmError(RuntimeError):
    """An HTTP, transport, or business error returned by SMM Data Pro."""


@dataclass(frozen=True)
class SmmDownloadResult:
    normalized: pd.DataFrame
    raw_batches: list[dict[str, Any]]


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


class SmmClient:
    """Small client for the documented SMM Data Pro API."""

    def __init__(
        self,
        user_name: str | None = None,
        password: str | None = None,
        token: str | None = None,
        token_cache: Path | None = None,
        use_proxy: bool = False,
        timeout: float = 30.0,
        retries: int = 3,
        session: requests.Session | None = None,
    ) -> None:
        self.user_name = user_name
        self.password = password
        self.token = token
        self.token_cache = token_cache
        self.timeout = timeout
        self.retries = retries
        self.session = session or requests.Session()
        self.session.trust_env = use_proxy

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        last_error: requests.RequestException | None = None
        for attempt in range(self.retries):
            try:
                response = self.session.request(
                    method,
                    f"{BASE_URL}{path}",
                    timeout=self.timeout,
                    **kwargs,
                )
                response.raise_for_status()
                return response
            except requests.RequestException as error:
                last_error = error
                if attempt + 1 < self.retries:
                    time.sleep(2**attempt)
        raise SmmError(f"SMM request failed after {self.retries} attempts: {last_error}")

    @staticmethod
    def _payload(response: requests.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except requests.JSONDecodeError as error:
            raise SmmError("SMM returned a non-JSON response") from error
        if not isinstance(payload, dict):
            raise SmmError("SMM returned an unexpected response shape")
        if payload.get("code") != 0:
            message = payload.get("msg") or payload.get("message") or "unknown error"
            raise SmmError(f"SMM business error {payload.get('code')}: {message}")
        return payload

    def _load_cached_token(self) -> str | None:
        if not self.token_cache or not self.token_cache.is_file():
            return None
        try:
            payload = json.loads(self.token_cache.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        token = payload.get("token")
        return token if isinstance(token, str) and token else None

    def _save_cached_token(self, token: str) -> None:
        if not self.token_cache:
            return
        self.token_cache.parent.mkdir(parents=True, exist_ok=True)
        self.token_cache.write_text(
            json.dumps({"token": token}, ensure_ascii=True), encoding="utf-8"
        )

    def auth(self) -> str:
        if self.token:
            return self.token
        cached_token = self._load_cached_token()
        if cached_token:
            self.token = cached_token
            return cached_token
        if not self.user_name or not self.password:
            raise SmmError(
                "Set SMM_USER_NAME and SMM_PASSWORD, pass --token, or provide a token cache"
            )
        response = self._request(
            "POST",
            AUTH_PATH,
            data={
                "user_name": self.user_name,
                "password": _md5(self.password),
                "source": "datapro",
            },
        )
        payload = self._payload(response)
        token = (payload.get("data") or {}).get("token")
        if not isinstance(token, str) or not token:
            raise SmmError("SMM authentication response did not contain a token")
        self.token = token
        self._save_cached_token(token)
        return token

    def _require_token(self) -> str:
        return self.token or self.auth()

    def list_quotas(self, page: int = 1, page_size: int = 500) -> dict[str, Any]:
        response = self._request(
            "GET",
            QUOTA_LIST_PATH,
            params={
                "token": self._require_token(),
                "page": page,
                "page_size": page_size,
            },
        )
        return self._payload(response).get("data") or {}

    def get_ext_info(self, quota_ids: list[str]) -> list[dict[str, Any]]:
        response = self._request(
            "GET",
            EXT_INFO_PATH,
            params={
                "token": self._require_token(),
                "quota_ids": ",".join(quota_ids),
            },
        )
        data = self._payload(response).get("data") or []
        return data if isinstance(data, list) else [data]

    def get_data_origin(
        self, quota_ids: list[str], start_date: str, end_date: str
    ) -> dict[str, Any]:
        if not quota_ids:
            raise ValueError("At least one quota ID is required")
        if len(quota_ids) > MAX_QUOTA_IDS_PER_REQUEST:
            raise ValueError("SMM accepts at most 100 quota IDs per request")
        response = self._request(
            "POST",
            DATA_ORIGIN_PATH,
            data={
                "token": self._require_token(),
                "request_source": "API",
                "quota_ids": ",".join(quota_ids),
                "start_date": start_date,
                "end_date": end_date,
                "sort_type": "1",
                "more_data": "0000",
            },
        )
        return self._payload(response)


def normalize_data(payloads: list[dict[str, Any]]) -> pd.DataFrame:
    """Normalize SMM's per-indicator response into one typed long table."""
    records: list[dict[str, Any]] = []
    for payload in payloads:
        series_list = payload.get("data") or []
        if isinstance(series_list, dict):
            series_list = [series_list]
        for series in series_list:
            metadata = {
                "quota_id": series.get("quota_id"),
                "quota_name": series.get("quota_name"),
                "unit": series.get("unit"),
                "frequency": series.get("frequency"),
            }
            points = series.get("data") or []
            for point in points:
                records.append(
                    {
                        **metadata,
                        "date": point.get("date"),
                        "value": point.get("value"),
                    }
                )

    columns = ["quota_id", "quota_name", "unit", "frequency", "date", "value"]
    frame = pd.DataFrame.from_records(records, columns=columns)
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["quota_id", "date"])
    return frame.sort_values(["quota_id", "date"]).drop_duplicates(
        ["quota_id", "date"], keep="last"
    )


def validate_permissions(payloads: list[dict[str, Any]]) -> None:
    unauthorized: list[str] = []
    for payload in payloads:
        series_list = payload.get("data") or []
        if isinstance(series_list, dict):
            series_list = [series_list]
        for series in series_list:
            if series.get("perm_state") == -1 or series.get("state_desc") == "no authorize":
                unauthorized.append(str(series.get("quota_id") or "unknown"))
    if unauthorized:
        raise SmmError(
            "SMM account is not authorized for quota IDs: "
            + ", ".join(unauthorized)
        )


def download_quotas(
    client: SmmClient,
    quota_ids: list[str],
    start_date: str,
    end_date: str,
) -> SmmDownloadResult:
    unique_ids = list(dict.fromkeys(quota_ids))
    payloads = [
        client.get_data_origin(batch, start_date, end_date)
        for batch in _chunks(unique_ids, MAX_QUOTA_IDS_PER_REQUEST)
    ]
    validate_permissions(payloads)
    return SmmDownloadResult(normalize_data(payloads), payloads)


def download_quota_catalog(
    client: SmmClient,
    keywords: list[str] | None = None,
    max_pages: int | None = None,
    page_size: int = 500,
) -> pd.DataFrame:
    """Download the quota catalog and optionally filter names/categories locally."""
    records: list[dict[str, Any]] = []
    page = 1
    page_count = 1
    normalized_keywords = [keyword.casefold() for keyword in keywords or []]

    while page <= page_count and (max_pages is None or page <= max_pages):
        data = client.list_quotas(page=page, page_size=page_size)
        page_count = int(data.get("page_count") or 1)
        if page == 1 or page % 25 == 0 or page == page_count:
            print(f"SMM catalog: page {page}/{page_count}", flush=True)
        items = data.get("data") or []
        for item in items:
            haystack = " ".join(
                str(item.get(field) or "")
                for field in ("quota_id", "quota_name", "type_all", "tags", "mark")
            ).casefold()
            if normalized_keywords and not any(
                keyword in haystack for keyword in normalized_keywords
            ):
                continue
            update_rule = item.get("update_rule") or {}
            records.append(
                {
                    "quota_id": item.get("quota_id"),
                    "quota_name": item.get("quota_name"),
                    "unit": item.get("unit"),
                    "frequency": item.get("frequency"),
                    "source": item.get("source"),
                    "type_all": item.get("type_all"),
                    "tags": item.get("tags"),
                    "data_start": item.get("data_start"),
                    "data_end": item.get("data_end"),
                    "data_state": item.get("data_state"),
                    "auth_start": item.get("auth_start"),
                    "auth_finish": item.get("auth_finish"),
                    "release_time": update_rule.get("release_time"),
                    "update_rule": update_rule.get("rule_desc"),
                    "mark": item.get("mark"),
                }
            )
        if page < page_count and (max_pages is None or page < max_pages):
            time.sleep(QUOTA_LIST_INTERVAL)
        page += 1

    columns = [
        "quota_id", "quota_name", "unit", "frequency", "source", "type_all",
        "tags", "data_start", "data_end", "data_state", "auth_start",
        "auth_finish", "release_time", "update_rule", "mark",
    ]
    return pd.DataFrame.from_records(records, columns=columns)


def save_result(
    result: SmmDownloadResult,
    output_dir: Path,
    quota_ids: list[str],
    start_date: str,
    end_date: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "smm_raw.json").write_text(
        json.dumps(result.raw_batches, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result.normalized.to_csv(output_dir / "smm_normalized.csv", index=False)
    metadata = {
        "source": "SMM Data Pro API",
        "quota_ids": list(dict.fromkeys(quota_ids)),
        "start_date": start_date,
        "end_date": end_date,
        "row_count": len(result.normalized),
        "columns": result.normalized.columns.tolist(),
        "created_on": date.today().isoformat(),
    }
    (output_dir / "smm_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download SMM Data Pro indicators")
    parser.add_argument("--quota-ids", nargs="+")
    parser.add_argument(
        "--list-quotas",
        action="store_true",
        help="Export quota IDs and metadata from the account's quota catalog",
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        help="Keep catalog rows matching any keyword in name, category, tags, or notes",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        help="Limit quota catalog pages for a quick preview; omit to scan all pages",
    )
    parser.add_argument(
        "--catalog-output",
        type=Path,
        default=Path("raw_data/smm/smm_quota_catalog.csv"),
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--output-dir", type=Path, default=Path("raw_data/smm"))
    parser.add_argument("--token", help="Use a token directly instead of environment credentials")
    parser.add_argument(
        "--token-cache",
        type=Path,
        default=Path("raw_data/smm/.smm_token.json"),
    )
    parser.add_argument(
        "--use-proxy",
        action="store_true",
        help="Honor HTTP(S)_PROXY environment variables; direct connections are the default",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.list_quotas and not args.quota_ids:
        raise SystemExit("Specify --list-quotas or at least one --quota-ids value")
    client = SmmClient(
        user_name=os.environ.get("SMM_USER_NAME"),
        password=os.environ.get("SMM_PASSWORD"),
        token=args.token,
        token_cache=args.token_cache,
        use_proxy=args.use_proxy,
    )
    if args.list_quotas:
        catalog = download_quota_catalog(
            client,
            keywords=args.keywords,
            max_pages=args.max_pages,
        )
        args.catalog_output.parent.mkdir(parents=True, exist_ok=True)
        catalog.to_csv(args.catalog_output, index=False, encoding="utf-8-sig")
        print(f"Saved {len(catalog):,} matching quota records to {args.catalog_output}")
        return

    result = download_quotas(
        client, args.quota_ids, args.start_date, args.end_date
    )
    save_result(
        result, args.output_dir, args.quota_ids, args.start_date, args.end_date
    )
    print(
        f"Saved {len(result.normalized):,} rows for "
        f"{result.normalized['quota_id'].nunique() if not result.normalized.empty else 0} "
        f"indicators to {args.output_dir}"
    )


if __name__ == "__main__":
    main()