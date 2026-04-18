#!/usr/bin/env python3
"""
Download Stack Overflow CSV bundles (math or DBA), fetch the shared schema SQL,
and convert every table to Parquet using DuckDB.

The dumps come from https://db.in.tum.de/~schmidt/data/ and already contain
headerless CSV files, so the whole workflow is:

  uv run --with duckdb python benchmark/stackoverflow/setup_stackoverflow.py --mode math 
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import duckdb

DEFAULT_SITE = "math.stackexchange.com"
CI_SITE = "dba.stackexchange.com"
DEFAULT_SCHEMA_URL = "https://db.in.tum.de/~schmidt/data/stackoverflow_schema.sql"

DATASETS = {
    DEFAULT_SITE: {
        "url": "https://db.in.tum.de/~schmidt/data/stackoverflow_math.tar.gz",
        "archive": "stackoverflow_math.tar.gz",
    },
    CI_SITE: {
        "url": "https://db.in.tum.de/~schmidt/data/stackoverflow_dba.tar.gz",
        "archive": "stackoverflow_dba.tar.gz",
    },
}

TABLE_ORDER = [
    "PostTypes",
    "PostHistoryTypes",
    "VoteTypes",
    "CloseReasonTypes",
    "LinkTypes",
    "Badges",
    "Tags",
    "Posts",
    "Comments",
    "Votes",
    "PostHistory",
    "PostLinks",
    "Users",
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Stack Overflow CSV dumps and convert them to Parquet."
    )
    parser.add_argument(
        "--site",
        help="StackExchange site (defaults to math, or dba for --mode dba)",
    )
    parser.add_argument(
        "--mode",
        choices=("math", "dba"),
        default="math",
        help="math = entire history, dba = DBA slice",
    )
    parser.add_argument(
        "--downloads-dir",
        default="benchmark/stackoverflow/downloads",
        help="Directory to store the downloaded tarballs",
    )
    parser.add_argument(
        "--raw-dir",
        default="benchmark/stackoverflow/raw",
        help="Directory to unpack the raw CSV files",
    )
    parser.add_argument(
        "--output-dir",
        help="Where to write Parquet files (defaults to benchmark/stackoverflow/data/<site>)",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the tarball even if it already exists",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-extract the tarball even if CSV files already exist",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep the extracted CSV files after conversion",
    )
    parser.add_argument(
        "--schema-path",
        default="benchmark/stackoverflow/schema.sql",
        help="Path to the schema SQL file (downloaded automatically if missing)",
    )
    parser.add_argument(
        "--schema-url",
        default=DEFAULT_SCHEMA_URL,
        help="URL to fetch the schema SQL from",
    )
    parser.add_argument(
        "--force-schema-download",
        action="store_true",
        help="Re-download the schema SQL even if it already exists locally",
    )
    return parser.parse_args()


def dataset_for(site: str) -> Dict[str, str]:
    try:
        return DATASETS[site]
    except KeyError:
        known = ", ".join(DATASETS.keys())
        raise SystemExit(f"Unsupported site '{site}'. Known datasets: {known}")


def ensure_schema_file(path: Path, url: str, force: bool) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[schema] Downloading {url}")
    with urllib.request.urlopen(url) as response, open(path, "wb") as fout:
        shutil.copyfileobj(response, fout)
    print(f"[schema] Saved to {path}")
    return path


def build_table_schemas(
    schema_path: Path,
    tables: Sequence[str],
) -> Dict[str, Tuple[List[str], List[str]]]:
    sql_text = strip_sql_comments(schema_path.read_text())
    conn = duckdb.connect(database=":memory:")
    # conn.execute("PRAGMA foreign_keys=off")
    try:
        conn.execute(sql_text)
    except duckdb.Error as err:
        raise SystemExit(f"Failed to apply schema SQL: {err}") from err
    schemas: Dict[str, Tuple[List[str], List[str]]] = {}
    for table in tables:
        info = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
        if not info:
            print(f"[schema] Warning: table {table} missing from schema SQL")
            continue
        columns = [row[1] for row in info]
        types = [row[2] for row in info]
        schemas[table] = (columns, types)
    conn.close()
    return schemas


def strip_sql_comments(sql_text: str) -> str:
    without_block = re.sub(r"/\*.*?\*/", "", sql_text, flags=re.DOTALL)
    return re.sub(r"--.*?$", "", without_block, flags=re.MULTILINE)


def slugify_site(site: str) -> str:
    if site.endswith(".stackexchange.com"):
        return site.replace(".stackexchange.com", "")
    if site.endswith(".com"):
        return site.replace(".com", "")
    return site.replace(".", "_")


def download_archive(site: str, downloads_dir: Path, force: bool) -> Path:
    dataset = dataset_for(site)
    url = dataset["url"]
    archive_name = dataset.get("archive") or Path(url).name

    downloads_dir.mkdir(parents=True, exist_ok=True)
    archive_path = downloads_dir / archive_name
    if archive_path.exists() and not force:
        print(f"[download] Reusing {archive_path}")
        return archive_path

    print(f"[download] Fetching {url}")
    with urllib.request.urlopen(url) as response, open(archive_path, "wb") as fout:
        total_size = int(response.headers.get("content-length", "0"))
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            fout.write(chunk)
            downloaded += len(chunk)
            if total_size:
                pct = downloaded / total_size * 100
                sys.stderr.write(
                    f"[download] {downloaded/1_048_576:.1f} / {total_size/1_048_576:.1f} MB ({pct:.1f}%)\r"
                )
    sys.stderr.write("\n")
    print(f"[download] Saved to {archive_path}")
    return archive_path


def extract_archive(
    site: str,
    archive_path: Path,
    raw_root: Path,
    force: bool,
) -> Path:
    dest = raw_root / slugify_site(site)
    if dest.exists() and not force:
        if any(dest.rglob("*.csv")):
            print(f"[extract] Reusing {dest}")
            return dest
        print(f"[extract] {dest} exists but no CSV files found, re-extracting")
        shutil.rmtree(dest)
    elif dest.exists() and force:
        shutil.rmtree(dest)

    dest.mkdir(parents=True, exist_ok=True)
    print(f"[extract] Extracting {archive_path} -> {dest}")
    with tarfile.open(archive_path, mode="r:gz") as archive:
        safe_extract(archive, dest)
    return dest


def safe_extract(archive: tarfile.TarFile, dest: Path) -> None:
    dest = dest.resolve()
    for member in archive.getmembers():
        member_path = (dest / member.name).resolve()
        if not str(member_path).startswith(str(dest)):
            raise RuntimeError(f"Blocked path traversal in tar file: {member.name}")
    archive.extractall(path=dest)


def convert_all(
    raw_dir: Path,
    output_dir: Path,
    table_schemas: Dict[str, Tuple[Sequence[str], Sequence[str]]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(database=":memory:")
    try:
        for table_name in TABLE_ORDER:
            schema = table_schemas.get(table_name)
            if not schema:
                print(f"[schema] Warning: {table_name} not found in schema SQL, skipping")
                continue
            convert_single_table(conn, raw_dir, output_dir, table_name, schema)
    finally:
        conn.close()


def convert_single_table(
    conn: duckdb.DuckDBPyConnection,
    raw_dir: Path,
    output_dir: Path,
    table_name: str,
    schema: Tuple[Sequence[str], Sequence[str]],
) -> None:
    columns, types = schema
    csv_path = locate_table_csv(raw_dir, table_name)
    parquet_path = output_dir / f"{table_name}.parquet"

    if parquet_path.exists():
        print(f"[convert] {table_name}: already exists, skipping")
        return

    if csv_path is None:
        print(f"[convert] {table_name}: missing CSV, writing empty parquet")
        create_empty_parquet(conn, table_name, columns, types, parquet_path)
        return

    if csv_path.parent != raw_dir:
        rel = csv_path.relative_to(raw_dir)
        print(f"[convert] {table_name}: found CSV at {rel}")

    conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    col_defs = ", ".join(f'"{col}" {dtype}' for col, dtype in zip(columns, types))
    conn.execute(f'CREATE TABLE "{table_name}" ({col_defs})')

    copy_sql = f"""
    COPY "{table_name}" FROM '{csv_path}'
    (HEADER false, DELIMITER ',', QUOTE '"', ESCAPE '"', NULL '', SAMPLE_SIZE -1, STRICT_MODE false, PARALLEL false)
    """
    conn.execute(copy_sql)

    conn.execute(
        f"COPY \"{table_name}\" TO '{parquet_path}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    conn.execute(f'DROP TABLE "{table_name}"')
    print(f"[convert] {table_name}: wrote {parquet_path}")


def locate_table_csv(raw_dir: Path, table_name: str) -> Optional[Path]:
    direct = raw_dir / f"{table_name}.csv"
    if direct.exists():
        return direct
    target = table_name.lower()
    for candidate in raw_dir.rglob("*.csv"):
        if candidate.stem.lower() == target:
            return candidate
    return None


def create_empty_parquet(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    columns: Sequence[str],
    types: Sequence[str],
    destination: Path,
) -> None:
    conn.execute('DROP TABLE IF EXISTS "__empty"')
    col_defs = ", ".join(f'"{col}" {dtype}' for col, dtype in zip(columns, types))
    conn.execute(f'CREATE TABLE "__empty" ({col_defs})')
    conn.execute(
        f"COPY \"__empty\" TO '{destination}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    conn.execute('DROP TABLE "__empty"')


def main() -> None:
    args = parse_args()
    site = args.site or (CI_SITE if args.mode == "dba" else DEFAULT_SITE)
    if args.site is None:
        print(f"[config] Mode {args.mode}: using default site '{site}'")
    downloads_dir = Path(args.downloads_dir)
    raw_root = Path(args.raw_dir)
    site_slug = slugify_site(site)
    output_dir = Path(args.output_dir) if args.output_dir else Path(
        "benchmark/stackoverflow/data"
    ) / site_slug

    schema_path = ensure_schema_file(
        Path(args.schema_path),
        args.schema_url,
        args.force_schema_download,
    )
    table_schemas = build_table_schemas(schema_path, TABLE_ORDER)

    archive_path = download_archive(site, downloads_dir, args.force_download)
    raw_dir = extract_archive(site, archive_path, raw_root, args.force_extract)
    convert_all(raw_dir, output_dir, table_schemas)

    if not args.keep_raw:
        print(f"[cleanup] Removing extracted CSV directory {raw_dir}")
        shutil.rmtree(raw_dir)

    print(f"[done] Parquet files ready under {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Aborted", file=sys.stderr)
        sys.exit(1)
