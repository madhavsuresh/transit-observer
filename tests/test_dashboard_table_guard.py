"""Regression test for the ``_table_exists`` guard the bus_v3 dashboard
helpers rely on.

When the v3 schema migration hasn't landed yet (e.g. right after merging
the PR, before the collector has restarted), the read replica may not
contain ``bus_v3_*`` tables. The dashboard helpers must return an empty
DataFrame instead of raising a ``CatalogException``.

We can't import ``transit_observer.dashboard`` directly inside a test
because its module-level ``_render()`` call assumes a live Streamlit
context. Instead, we exercise the helper that the v3 callers gate on
via an AST-driven extraction.
"""

from __future__ import annotations

import ast
import os
import tempfile
from pathlib import Path

import duckdb
import pytest


def _load_table_exists():
    """Import just the ``_table_exists`` helper out of dashboard.py
    without executing the module's top-level ``_render()`` call.

    We parse the module with ``ast``, pick the one ``_table_exists``
    function definition, ``ast.unparse`` it, and exec it in an empty
    namespace. The function is pure (only uses its ``conn`` argument
    and standard SQL), so this isolation is safe.
    """
    src_path = Path(__file__).resolve().parents[1] / "src" / "transit_observer" / "dashboard.py"
    tree = ast.parse(src_path.read_text())
    fn = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_table_exists"
    )
    ns: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(src_path), "exec"), ns)  # noqa: S102
    return ns["_table_exists"]


@pytest.fixture()
def conn():
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test.duckdb")
    conn = duckdb.connect(path)
    try:
        yield conn
    finally:
        conn.close()
        import shutil

        shutil.rmtree(tmpdir)


def test_table_exists_false_for_missing_table(conn):
    """Fresh DuckDB with no tables — guard returns False."""
    _table_exists = _load_table_exists()
    assert _table_exists(conn, "bus_v3_api_poll") is False
    assert _table_exists(conn, "bus_v3_arrival_event") is False
    assert _table_exists(conn, "bus_v3_residual_quantile") is False


def test_table_exists_true_after_create(conn):
    """After ``CREATE TABLE`` the guard returns True."""
    _table_exists = _load_table_exists()
    conn.execute("CREATE TABLE bus_v3_api_poll (x INTEGER)")
    assert _table_exists(conn, "bus_v3_api_poll") is True


def test_table_exists_after_init_schema():
    """The repo's ``init_schema`` creates all 17 bus_v3 tables."""
    from transit_observer.db import init_schema

    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test.duckdb")
    conn = duckdb.connect(path)
    init_schema(conn)
    _table_exists = _load_table_exists()
    try:
        assert _table_exists(conn, "bus_v3_api_poll") is True
        assert _table_exists(conn, "bus_v3_arrival_event") is True
        assert _table_exists(conn, "bus_v3_residual_quantile") is True
        # Negative case stays negative.
        assert _table_exists(conn, "this_table_does_not_exist") is False
    finally:
        conn.close()
        import shutil

        shutil.rmtree(tmpdir)
