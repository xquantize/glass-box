"""SQLite query tool — stdlib sqlite3 only, results returned as text."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from pydantic import BaseModel, Field

from glassbox.tools.base import FunctionTool

DEFAULT_DB_PATH = Path("data/glassbox.db")
MAX_ROWS = 100


class SqlArgs(BaseModel):
    query: str = Field(
        description=(
            "One or more SQL statements separated by semicolons "
            "(e.g. CREATE …; INSERT …; SELECT …). "
            "SELECT results are capped."
        )
    )


def _format_rows(columns: list[str], rows: list[tuple]) -> str:
    if not columns:
        return "(no columns)"
    payload = [dict(zip(columns, row, strict=True)) for row in rows]
    return json.dumps(payload, indent=2, default=str)


def _split_statements(sql: str) -> list[str]:
    """Split on semicolons. Good enough for agent SQL without nested scripts."""
    return [part.strip() for part in sql.split(";") if part.strip()]


def _run_one(conn: sqlite3.Connection, statement: str, *, max_rows: int) -> str:
    cursor = conn.execute(statement)
    if cursor.description is None:
        conn.commit()
        return f"ok — {cursor.rowcount} row(s) affected"
    columns = [col[0] for col in cursor.description]
    rows = cursor.fetchmany(max_rows + 1)
    truncated = len(rows) > max_rows
    rows = rows[:max_rows]
    body = _format_rows(columns, [tuple(r) for r in rows])
    if truncated:
        return f"{body}\n\n(truncated to {max_rows} rows)"
    if not rows:
        return "[]"
    return body


def run_sql(db_path: Path, query: str, *, max_rows: int = MAX_ROWS) -> str:
    """Execute one or more SQL statements; return a readable string observation."""
    statements = _split_statements(query)
    if not statements:
        raise ValueError("query must not be empty")

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    parts: list[str] = []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        for statement in statements:
            parts.append(_run_one(conn, statement, max_rows=max_rows))

    if len(parts) == 1:
        return parts[0]
    return "\n---\n".join(parts)


def build_sql(db_path: str | Path = DEFAULT_DB_PATH) -> FunctionTool[SqlArgs]:
    """Construct a run_sql tool bound to one SQLite database file."""
    path = Path(db_path).resolve()

    def _run(args: SqlArgs) -> str:
        return run_sql(path, args.query)

    return FunctionTool(
        name="run_sql",
        description=(
            f"Execute SQL against the SQLite database at {path}. "
            "Accepts one statement or several separated by semicolons. "
            f"SELECT results return up to {MAX_ROWS} rows as JSON."
        ),
        parameters=SqlArgs,
        handler=_run,
    )
