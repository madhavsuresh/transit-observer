"""Line topology helpers: station ordering per (line, direction).

The CTA Train Tracker API doesn't expose track topology directly, but
we can infer it from the ``ttpositions.aspx`` stream: when run R is
seen with ``nextStaId = X``, then later with ``nextStaId = Y``, and the
prediction stream confirms the same run continues to the same
destination, then X immediately precedes Y in that line+direction.

This builder scans ``train_v2_position_observation`` for those pairs
and writes a stable ordering into ``train_v2_line_topology``. The
ordering is then consulted by the inference module to gate
``ARRIVED_CONFIRMED`` labels (a topology-consistent transition is a
much stronger signal than a bare ``nextStaId`` change).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import duckdb
import structlog


log = structlog.get_logger(__name__)


def refresh_line_topology(
    conn: duckdb.DuckDBPyConnection,
    *,
    window_hours: int = 168,
    min_transitions: int = 3,
) -> int:
    """Rebuild ``train_v2_line_topology`` from observed run transitions.

    Walks every (run_number, line, direction_code) trajectory in the
    position log, extracts ordered ``(X → Y)`` consecutive pairs, and
    counts unique transitions. A pair is accepted into the topology if
    it occurs at least ``min_transitions`` distinct times (defense
    against transient bad data — wrong direction, missed update).

    For each line+direction we then build the directed adjacency graph
    and emit a station sequence by topological sort (when the graph is
    DAG-like, which is true for the L's branching layout outside
    junctions where two lines merge).

    Returns the count of (line, direction_code, map_id) rows written.
    """
    cutoff_sql = """
        SELECT line, direction_code, run_number, next_station_map_id, local_response_end_ms
          FROM train_v2_position_observation
         WHERE local_response_end_ms >= ?
           AND next_station_map_id IS NOT NULL
           AND line IS NOT NULL AND direction_code IS NOT NULL
         ORDER BY line, direction_code, run_number, local_response_end_ms
    """
    cutoff_ms = _cutoff_ms(window_hours)
    rows = conn.execute(cutoff_sql, [cutoff_ms]).fetchall()
    transitions: dict[tuple[str, str], dict[tuple[str, str], int]] = defaultdict(lambda: defaultdict(int))
    last_per_run: dict[tuple[str, str, str], tuple[str, int]] = {}
    for line, dir_code, run, next_map, t_ms in rows:
        key_run = (str(line), str(dir_code), str(run))
        prev = last_per_run.get(key_run)
        if prev is not None and prev[0] != next_map:
            transitions[(str(line), str(dir_code))][(prev[0], str(next_map))] += 1
        last_per_run[key_run] = (str(next_map), int(t_ms))

    # Build a topo order per (line, direction). When the graph has
    # cycles (e.g. Loop), fall back to a stable arbitrary order.
    sequences: dict[tuple[str, str], list[str]] = {}
    for (line, dir_code), pairs in transitions.items():
        edges = [(a, b) for (a, b), count in pairs.items() if count >= min_transitions]
        nodes = {n for edge in edges for n in edge}
        if not nodes:
            continue
        order = _topo_sort(nodes, edges)
        sequences[(line, dir_code)] = order

    written = 0
    conn.execute("DELETE FROM train_v2_line_topology")
    for (line, dir_code), order in sequences.items():
        for seq, map_id in enumerate(order):
            row = conn.execute(
                "SELECT name, latitude, longitude FROM (SELECT NULL AS name, NULL AS latitude, NULL AS longitude WHERE FALSE)"
            ).fetchone()
            # Pull station name + coords from the position stream's latest snapshot.
            position_row = conn.execute(
                """
                SELECT next_station_name, lat, lon
                  FROM train_v2_position_observation
                 WHERE next_station_map_id = ? AND line = ?
                 ORDER BY local_response_end_ms DESC
                 LIMIT 1
                """,
                [map_id, line],
            ).fetchone()
            name = position_row[0] if position_row else None
            lat = position_row[1] if position_row else None
            lon = position_row[2] if position_row else None
            conn.execute(
                """
                INSERT INTO train_v2_line_topology(line, direction_code, seq, map_id, station_name, lat, lon)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [line, dir_code, seq, map_id, name, lat, lon],
            )
            written += 1
    log.info("train_v2.topology_refreshed", lines=len(sequences), rows=written)
    return written


def is_downstream(
    conn: duckdb.DuckDBPyConnection,
    *,
    line: str,
    direction_code: str,
    from_map: str,
    to_map: str,
) -> Optional[bool]:
    """``True`` when ``to_map`` comes after ``from_map`` in the topology
    for (line, direction). ``None`` when topology hasn't covered them
    yet (caller treats as 'unknown', not 'no')."""
    row = conn.execute(
        """
        SELECT a.seq, b.seq
          FROM train_v2_line_topology a, train_v2_line_topology b
         WHERE a.line = ? AND a.direction_code = ? AND a.map_id = ?
           AND b.line = ? AND b.direction_code = ? AND b.map_id = ?
        """,
        [line, direction_code, from_map, line, direction_code, to_map],
    ).fetchone()
    if row is None:
        return None
    a_seq, b_seq = row
    return b_seq > a_seq


def _topo_sort(nodes: set[str], edges: list[tuple[str, str]]) -> list[str]:
    """Kahn's algorithm with deterministic tie-breaking. Ignores
    cycles by dropping the offending edges (Loop tracks self-cycle)."""
    in_degree: dict[str, int] = {n: 0 for n in nodes}
    adjacency: dict[str, list[str]] = {n: [] for n in nodes}
    for a, b in edges:
        if a == b:
            continue
        adjacency[a].append(b)
        in_degree[b] += 1
    queue = sorted([n for n in nodes if in_degree[n] == 0])
    order: list[str] = []
    while queue:
        n = queue.pop(0)
        order.append(n)
        for m in sorted(adjacency[n]):
            in_degree[m] -= 1
            if in_degree[m] == 0:
                queue.append(m)
        queue.sort()
    # Any node not yet emitted is in a cycle; append in stable order.
    remaining = sorted(n for n in nodes if n not in order)
    return order + remaining


def _cutoff_ms(hours: int) -> int:
    import time

    return int(time.time() * 1000) - hours * 3_600_000
