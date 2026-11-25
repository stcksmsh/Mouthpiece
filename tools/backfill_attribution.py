"""Apply a per-source attribution table (CSV) to a Mouthpiece DB.

    python tools/backfill_attribution.py --db db/mouthpiece.db --csv corpus/attribution.csv [--dry-run]

Rows are matched by filename. Every CSV column other than `filename` is written to
the source row; an empty cell means NULL. Prints one line per changed row. A CSV
cell containing "@" (an email) stops the run before anything is written.
"""

import argparse
import csv
import re
import sqlite3
import sys
from pathlib import Path

# Run as a script from any directory: make the repo's `mouthpiece` package importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mouthpiece.index import get_connection

FIELDS = ("speaker", "author", "reader", "uploader", "source_label", "licence_url", "lang")
_EMAIL = re.compile(r"[^\s@]+@[^\s@]+")


def _show(value: str | None) -> str:
    return "NULL" if value is None else repr(_EMAIL.sub("<email>", value))


def read_table(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != ("filename", *FIELDS):
            raise ValueError(f"{path}: header must be {','.join(('filename', *FIELDS))}")
        table = {}
        for line, row in enumerate(reader, start=2):
            if any("@" in (v or "") for v in row.values()):
                raise ValueError(f"{path}:{line}: cell contains '@'")
            table[row["filename"]] = {k: row[k] or None for k in FIELDS}
    return table


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true", help="print the changes, write nothing")
    args = ap.parse_args(argv)

    try:
        table = read_table(args.csv)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not args.db.is_file():
        print(f"error: no DB at {args.db}", file=sys.stderr)
        return 1

    if args.dry_run:
        # Read-only, so not even the schema migration is written; columns the
        # DB lacks yet read as NULL, which is what the migration would add.
        conn = sqlite3.connect(f"file:{args.db.resolve()}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    else:
        conn = get_connection(args.db)

    cur = conn.execute("SELECT * FROM sources ORDER BY id")
    cols = {d[0] for d in cur.description}
    rows = cur.fetchall()
    in_db = {r["filename"] for r in rows}
    for filename in sorted(table.keys() - in_db):
        print(f"warning: {filename}: not in db, skipped", file=sys.stderr)

    updates = []
    for row in rows:
        new = table.get(row["filename"])
        if new is None:
            print(f"{row['filename']}: not in csv")
            continue
        old = {k: row[k] if k in cols else None for k in FIELDS}
        changed = [k for k in FIELDS if old[k] != new[k]]
        if changed:
            print(f"{row['filename']}: " + "; ".join(f"{k}: {_show(old[k])} → {_show(new[k])}" for k in changed))
            updates.append((*(new[k] for k in FIELDS), row["id"]))

    if not args.dry_run and updates:
        with conn:
            conn.executemany(
                f"UPDATE sources SET {', '.join(f'{k}=?' for k in FIELDS)} WHERE id=?", updates
            )
    conn.close()
    print(f"{len(updates)} row(s) {'would change' if args.dry_run else 'changed'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
