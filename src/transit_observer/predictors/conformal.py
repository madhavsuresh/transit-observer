"""Online conformal prediction wrapper — adaptive coverage under drift.

Implements an ACI-style (Gibbs & Candès, NeurIPS 2021) online step
update with a DtACI-style (JMLR 2024) candidate-bank step-size selector.
Goal: pin the empirical coverage of each predicted quantile to its
nominal target even when the underlying residual distribution drifts
(DST transitions, snow, service disruptions).

The update is one scalar per (predictor_version, line, direction_code,
leg, quantile). State lives in the ``predictor_state`` table; the
resolver calls :func:`update` after each scored outcome.

Algorithm (online ACI):
    given target quantile q ∈ (0, 1) and offset o_t,
        adjusted_q_t = raw_q_t + o_t
        covered_t = 1 if observed ≤ adjusted_q_t else 0
        o_{t+1} = o_t + γ * (q − covered_t)

When the model under-covers (covered=0 too often), o grows and widens
the interval. When it over-covers, o shrinks. Marginal coverage
converges to q under mild conditions, and the convergence holds under
distribution shift if γ adapts to the shift rate.

DtACI adds: a bank of candidate γs, exponentially weighted by recent
regret. The chosen γ is the weight-averaged combination. Step sizes in
*seconds* (not raw probabilities) so the values are interpretable
("widen the 90th percentile by 12 s when we miss it").
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable

import duckdb


# DtACI candidate step-size bank (in seconds). The smallest reacts slowly
# (good for stationary regime); the largest reacts fast (good for
# disruptive shift). Weighted online by regret.
DEFAULT_STEP_BANK: tuple[float, ...] = (1.0, 5.0, 20.0, 60.0)

# Forgetting factor for the exponential weights — η in DtACI. Higher
# means faster reaction to which γ is currently winning.
ETA: float = 0.05


@dataclass
class DtACIState:
    """One row of online state per (line, dir, leg, quantile)."""

    predictor_version: str
    line: str
    direction_code: str
    leg: str
    quantile: float
    offset_seconds: float = 0.0
    step_size: float = 5.0
    coverage_target: float = 0.0      # equal to ``quantile`` for one-sided upper bounds
    n: int = 0
    miscoverage_count: int = 0
    log_weights: list[float] = field(default_factory=lambda: [0.0] * len(DEFAULT_STEP_BANK))
    step_bank: tuple[float, ...] = DEFAULT_STEP_BANK

    @property
    def coverage_observed(self) -> float | None:
        if self.n == 0:
            return None
        return 1.0 - self.miscoverage_count / self.n

    def adjust(self, raw_quantile_seconds: float) -> float:
        return max(0.0, raw_quantile_seconds + self.offset_seconds)

    def step(self, raw_quantile_seconds: float, observed_seconds: float) -> None:
        """ACI step on one observation.

        ``raw_quantile_seconds`` is the model's raw output (before
        conformal adjustment). ``observed_seconds`` is the realized
        outcome. The state is updated in place.
        """
        adjusted = self.adjust(raw_quantile_seconds)
        covered = 1.0 if observed_seconds <= adjusted else 0.0
        self.n += 1
        self.miscoverage_count += int(1.0 - covered)

        # Update each candidate step's notional offset and incur its
        # miscoverage loss. The DtACI loss is the pinball/quantile loss
        # of the candidate's predicted offset, summarized as
        # (target - covered) shape.
        per_step_loss = []
        residual = observed_seconds - raw_quantile_seconds   # what offset *would* have covered
        for gamma in self.step_bank:
            candidate_offset = self.offset_seconds + gamma * (self.coverage_target - covered)
            # pinball-like loss at target quantile for this candidate
            candidate_adj = raw_quantile_seconds + candidate_offset
            err = observed_seconds - candidate_adj
            # one-sided quantile loss: q*err if err>=0, (q-1)*err otherwise
            loss = max(self.coverage_target * err, (self.coverage_target - 1.0) * err)
            per_step_loss.append(loss)

        # Update the log-weights via exponential weighting on regret.
        # Normalize so they remain bounded.
        new_log_weights = []
        for lw, loss in zip(self.log_weights, per_step_loss):
            new_log_weights.append(lw - ETA * loss)
        m = max(new_log_weights)
        new_log_weights = [w - m for w in new_log_weights]
        self.log_weights = new_log_weights

        # Effective step is the weighted geometric average of the bank.
        weights = [math.exp(w) for w in self.log_weights]
        norm = sum(weights) or 1.0
        weights = [w / norm for w in weights]
        eff_gamma = sum(w * g for w, g in zip(weights, self.step_bank))
        self.step_size = eff_gamma

        # Standard ACI update with the effective γ.
        self.offset_seconds += eff_gamma * (self.coverage_target - covered)
        # Clamp the offset to a sane range so a bad-data burst can't move
        # it past the credible window (20 minutes either way for transit
        # is more than enough).
        self.offset_seconds = max(-1200.0, min(1200.0, self.offset_seconds))


def load_state(
    conn: duckdb.DuckDBPyConnection,
    *,
    predictor_version: str,
    line: str,
    direction_code: str,
    leg: str,
    quantile: float,
) -> DtACIState:
    """Hydrate one DtACI state row from ``predictor_state``.

    Returns a fresh-default state if the row is missing — the first
    ``update`` will INSERT it.
    """
    row = conn.execute(
        """
        SELECT offset_seconds, step_size, coverage_target, coverage_observed,
               n_observations
          FROM predictor_state
         WHERE predictor_version = ?
           AND line = ? AND direction_code = ?
           AND leg = ? AND quantile = ?
        """,
        [predictor_version, line, direction_code, leg, quantile],
    ).fetchone()
    if row is None:
        return DtACIState(
            predictor_version=predictor_version,
            line=line, direction_code=direction_code,
            leg=leg, quantile=quantile, coverage_target=quantile,
        )
    offset, step, target, observed, n_obs = row
    state = DtACIState(
        predictor_version=predictor_version,
        line=line, direction_code=direction_code,
        leg=leg, quantile=quantile,
        offset_seconds=float(offset or 0.0),
        step_size=float(step or 5.0),
        coverage_target=float(target or quantile),
        n=int(n_obs or 0),
    )
    obs = observed
    if obs is not None and state.n > 0:
        state.miscoverage_count = int(round((1.0 - float(obs)) * state.n))
    return state


def persist_state(
    conn: duckdb.DuckDBPyConnection,
    state: DtACIState,
    *,
    now: datetime,
) -> None:
    """UPSERT one row back into ``predictor_state``."""
    conn.execute(
        """
        INSERT INTO predictor_state
            (predictor_version, line, direction_code, leg, quantile,
             offset_seconds, step_size, coverage_target, coverage_observed,
             n_observations, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (predictor_version, line, direction_code, leg, quantile)
        DO UPDATE SET
            offset_seconds = excluded.offset_seconds,
            step_size = excluded.step_size,
            coverage_observed = excluded.coverage_observed,
            n_observations = excluded.n_observations,
            updated_at = excluded.updated_at
        """,
        [
            state.predictor_version, state.line, state.direction_code,
            state.leg, state.quantile,
            state.offset_seconds, state.step_size, state.coverage_target,
            state.coverage_observed, state.n, now,
        ],
    )


def update(
    conn: duckdb.DuckDBPyConnection,
    *,
    predictor_version: str,
    line: str,
    direction_code: str,
    leg: str,
    quantile: float,
    raw_quantile_seconds: float,
    observed_seconds: float,
    now: datetime,
) -> DtACIState:
    """Load → step → persist, in one shot. Returns the updated state."""
    state = load_state(
        conn,
        predictor_version=predictor_version,
        line=line, direction_code=direction_code,
        leg=leg, quantile=quantile,
    )
    state.step(raw_quantile_seconds, observed_seconds)
    persist_state(conn, state, now=now)
    return state


def offsets_for(
    conn: duckdb.DuckDBPyConnection,
    *,
    predictor_version: str,
    line: str,
    direction_code: str,
    leg: str,
    quantiles: Iterable[float],
) -> dict[float, float]:
    """Bulk lookup of all conformal offsets for a (line, dir, leg).

    Returns a {quantile: offset_seconds} dict; quantiles missing from
    ``predictor_state`` map to 0.0.
    """
    qs = list(quantiles)
    if not qs:
        return {}
    placeholders = ",".join("?" * len(qs))
    rows = conn.execute(
        f"""
        SELECT quantile, offset_seconds
          FROM predictor_state
         WHERE predictor_version = ?
           AND line = ? AND direction_code = ? AND leg = ?
           AND quantile IN ({placeholders})
        """,
        [predictor_version, line, direction_code, leg, *qs],
    ).fetchall()
    out: dict[float, float] = {q: 0.0 for q in qs}
    for q, off in rows:
        if off is not None:
            out[float(q)] = float(off)
    return out


def reset_predictor_state(
    conn: duckdb.DuckDBPyConnection,
    *,
    predictor_version: str | None = None,
) -> int:
    """Wipe DtACI state (e.g., after a DST transition or a model retrain).

    Returns the number of rows deleted. With ``predictor_version=None``
    this wipes everything — caller's responsibility to be sure.
    """
    if predictor_version is None:
        cur = conn.execute("DELETE FROM predictor_state")
    else:
        cur = conn.execute(
            "DELETE FROM predictor_state WHERE predictor_version = ?",
            [predictor_version],
        )
    return cur.fetchone()[0] if cur.description else 0
