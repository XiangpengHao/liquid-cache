#!/usr/bin/env python3
"""
Fetch the 99 TPC-DS queries from DuckDB's tpcds extension and write qNN.sql files.

Usage:
  uvx --from duckdb python benchmark/tpcds/fetch_queries.py
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import duckdb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch TPC-DS queries from DuckDB and write qNN.sql files")
    parser.add_argument("--queries-dir", type=Path, default=Path("benchmark/tpcds/queries"), help="Directory to write qNN.sql files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing qNN.sql files")
    return parser.parse_args()


def fetch_queries(conn: duckdb.DuckDBPyConnection) -> List[Tuple[int, str]]:
    rows = conn.execute("SELECT query_nr, query FROM tpcds_queries() ORDER BY query_nr").fetchall()
    return [(int(q), sql) for q, sql in rows]


def write_queries(queries: List[Tuple[int, str]], out_dir: Path, overwrite: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    for qnum, sql in queries:
        fname = out_dir / f"q{qnum}.sql"
        if fname.exists() and not overwrite:
            print(f"Skipping existing {fname} (use --overwrite to refresh)")
            continue
        fname.write_text(sql)
        print(f"Wrote {fname}")


def main():
    args = parse_args()
    conn = duckdb.connect(database=":memory:")
    try:
        conn.execute("INSTALL tpcds")
        conn.execute("LOAD tpcds")
        queries = fetch_queries(conn)
        if len(queries) != 99:
            print(f"Warning: expected 99 queries, got {len(queries)}")
        write_queries(queries, args.queries_dir, args.overwrite)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
