#!/usr/bin/env python3
"""
dethloff_directed_info_cuts_pilot_v2.py

Small multi-route kill test for the directed information-relaxed P-cuts on
Dethloff-derived geometry and demand heterogeneity.

IMPORTANT SCOPE
---------------
This is NOT the original Dethloff VRPSPD model.  The theory being tested is for
single-commodity VRPSD with optimal restocking.  This script therefore uses:

  * a Dethloff distance matrix;
  * one selected Dethloff demand column only (delivery, pickup, or their sum);
  * a synthetic two-state common-factor stochastic overlay with the selected
    Dethloff values preserving relative customer mean-demand heterogeneity.

The script extracts a SMALL customer subset and compares two exact outer-loop
integer-L-shaped prototypes:

  BASELINE : exact full-route cuts generated on integer incumbents only.
  HYBRID   : the same exact full-route cuts, plus dynamically separated
             directed information-relaxed proper-subpath cuts at violated
             integer incumbents. Optional root separation remains available.

SciPy/HiGHS has no callback interface, so "lazy" and "user" cuts are emulated
with solve--separate--resolve outer loops.  This is a go/no-go pilot, not a
production branch-and-cut implementation.

Windows CMD example
-------------------
python dethloff_directed_info_cuts_pilot_v2.py ^
  file=Dethloff\\CON3-0.vrpspd n=8 routes=2 select=random seed=0 ^
  demand=delivery mean_level=4 load=0.90 delta=1 ^
  p=0.50,0.80,0.95,0.99 max_path=4 batch=25 time=120 ^
  out=dethloff_directed_info_pilot.csv

Recommended first smoke test
----------------------------
python dethloff_directed_info_cuts_pilot.py file=Dethloff\\CON3-0.vrpspd ^
  n=7 routes=2 p=0.80,0.95 max_path=4 time=60

Dependencies
------------
Python 3.10+, NumPy, SciPy (with scipy.optimize.milp).
"""
from __future__ import annotations

import csv
import math
import re
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, permutations
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix, vstack


# =============================================================================
# CLI
# =============================================================================

def parse_kv(argv: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for token in argv:
        if "=" in token:
            k, v = token.split("=", 1)
            out[k.strip().lower()] = v.strip()
        elif token in {"-h", "--help", "help"}:
            out["help"] = "1"
        else:
            raise ValueError(f"Unknown argument {token!r}; use key=value")
    return out


def as_int(cfg: dict[str, str], key: str, default: int) -> int:
    return int(cfg.get(key, default))


def as_float(cfg: dict[str, str], key: str, default: float) -> float:
    return float(cfg.get(key, default))


def as_list_float(cfg: dict[str, str], key: str, default: str) -> list[float]:
    return [float(x) for x in cfg.get(key, default).split(",") if x.strip()]


# =============================================================================
# Dethloff parser
# =============================================================================

def _lines_after(txt: str, tag: str) -> list[str]:
    out: list[str] = []
    capture = False
    for line in txt.splitlines():
        s = line.strip()
        if not s:
            continue
        if not capture:
            if s.upper().startswith(tag):
                capture = True
            continue
        if any(ch.isalpha() for ch in s):
            break
        out.append(s)
    return out


def _header_number(txt: str, key: str, cast=float):
    for line in txt.splitlines():
        if line.strip().upper().startswith(key):
            found = re.findall(r"-?\d+(?:\.\d+)?", line)
            if found:
                return cast(found[-1])
    return None


@dataclass(frozen=True)
class DethloffData:
    name: str
    distance: np.ndarray
    pickup: np.ndarray
    delivery: np.ndarray
    capacity: float
    vehicles: int | None


def parse_dethloff(path: str | Path, dist_scale: float = 10_000.0) -> DethloffData:
    p = Path(path)
    txt = p.read_text(errors="ignore")
    n = _header_number(txt, "DIMENSION", int)
    cap = _header_number(txt, "CAPACITY", float)
    veh = _header_number(txt, "VEHICLES", int)
    if n is None or cap is None:
        raise ValueError("Missing DIMENSION or CAPACITY")

    tokens: list[str] = []
    for row in _lines_after(txt, "EDGE_WEIGHT_SECTION"):
        tokens.extend(row.split())
    vals = [float(x) for x in tokens]
    if len(vals) != n * n:
        raise ValueError(f"Expected {n*n} edge weights, found {len(vals)}")
    dist = np.asarray(vals, dtype=float).reshape(n, n) / dist_scale

    pd = np.zeros((n, 2), dtype=float)
    for row in _lines_after(txt, "PICKUP_AND_DELIVERY_SECTION"):
        t = row.split()
        if len(t) < 3:
            continue
        idx = int(float(t[0])) - 1
        if 0 <= idx < n:
            # Dethloff files place pickup and delivery in the final two columns.
            pd[idx, 0] = float(t[-2])
            pd[idx, 1] = float(t[-1])

    return DethloffData(
        name=p.stem,
        distance=dist,
        pickup=pd[:, 0],
        delivery=pd[:, 1],
        capacity=float(cap),
        vehicles=veh,
    )


# =============================================================================
# Dethloff-derived small instance
# =============================================================================

@dataclass(frozen=True)
class SmallInstance:
    name: str
    c: np.ndarray                 # depot = 0, selected customers = 1..n
    original_ids: tuple[int, ...] # 1-based Dethloff node IDs
    mean: tuple[int, ...]
    low: tuple[int, ...]
    high: tuple[int, ...]
    Q: int
    routes: int
    raw_demand: tuple[float, ...]


def choose_customers(
    data: DethloffData,
    n: int,
    mode: str,
    seed: int,
    raw: np.ndarray,
) -> np.ndarray:
    candidates = np.arange(1, len(raw))  # exclude depot
    positive = candidates[raw[candidates] > 0]
    if len(positive) < n:
        raise ValueError(f"Only {len(positive)} positive-demand customers, need {n}")
    mode = mode.lower()
    if mode == "first":
        return positive[:n]
    if mode == "nearest":
        return positive[np.argsort(data.distance[0, positive])[:n]]
    if mode == "farthest":
        return positive[np.argsort(data.distance[0, positive])[-n:]]
    if mode == "largest":
        return positive[np.argsort(raw[positive])[-n:]]
    if mode == "random":
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(positive, size=n, replace=False))
    raise ValueError("select must be first|nearest|farthest|largest|random")


def build_small_instance(
    data: DethloffData,
    *,
    n: int,
    routes: int,
    select: str,
    seed: int,
    demand_mode: str,
    mean_level: float,
    delta: int,
    load: float,
) -> SmallInstance:
    if not (1 <= routes <= n):
        raise ValueError("routes must be between 1 and n")
    if not (0 < load <= 1.0):
        raise ValueError("load must be in (0,1]")
    if delta < 1:
        raise ValueError("delta must be >= 1")

    dm = demand_mode.lower()
    if dm == "pickup":
        raw_all = data.pickup.copy()
    elif dm == "delivery":
        raw_all = data.delivery.copy()
    elif dm in {"sum", "pickup+delivery", "both"}:
        raw_all = data.pickup + data.delivery
    else:
        raise ValueError("demand must be pickup|delivery|sum")

    ids0 = choose_customers(data, n, select, seed, raw_all)
    raw = raw_all[ids0]
    med = float(np.median(raw[raw > 0]))
    if med <= 0:
        raise ValueError("Selected demand median is non-positive")

    # Preserve relative Dethloff heterogeneity while keeping a small integer DP.
    means = np.maximum(delta + 1, np.rint(mean_level * raw / med).astype(int))
    low = np.maximum(0, means - delta)
    high = means + delta

    total_mean = int(means.sum())
    Q = max(int(high.max()), int(math.ceil(total_mean / (routes * load))))

    # Selected matrix: depot, then chosen customers.
    keep = np.concatenate(([0], ids0))
    c = data.distance[np.ix_(keep, keep)].copy()

    return SmallInstance(
        name=data.name,
        c=c,
        original_ids=tuple(int(i + 1) for i in ids0),
        mean=tuple(int(x) for x in means),
        low=tuple(int(x) for x in low),
        high=tuple(int(x) for x in high),
        Q=Q,
        routes=routes,
        raw_demand=tuple(float(x) for x in raw),
    )


# =============================================================================
# Correlated-demand OR recourse evaluator
# =============================================================================

@dataclass
class RecourseEvaluator:
    inst: SmallInstance
    p: float
    bP: float
    bF: float

    def __post_init__(self) -> None:
        if not (0.5 <= self.p < 1.0):
            raise ValueError("p must satisfy 0.5 <= p < 1")
        if self.bP < 0 or self.bF < 0 or self.bP > self.bF:
            raise ValueError("Require 0 <= bP <= bF")
        self._qbar = self._build_qbar()

    @staticmethod
    def _post_F1(k: int, s: int, p: float) -> float:
        if k == 0:
            return 0.5
        l1 = (p**s) * ((1.0 - p) ** (k - s))
        l0 = ((1.0 - p) ** s) * (p ** (k - s))
        den = 0.5 * l1 + 0.5 * l0
        if den <= 0:
            return 0.5
        return 0.5 * l1 / den

    def _build_qbar(self):
        c, Q, p, bP, bF = self.inst.c, self.inst.Q, self.p, self.bP, self.bF
        lows = (0,) + self.inst.low
        highs = (0,) + self.inst.high

        @lru_cache(maxsize=None)
        def qbar(path: tuple[int, ...], Fcode: int = -1) -> float:
            L = len(path)

            @lru_cache(maxsize=None)
            def V(stage: int, q: int, k: int, s: int) -> float:
                if stage == L:
                    return 0.0
                j = path[stage]
                prev = 0 if stage == 0 else path[stage - 1]
                cP = bP + c[prev, 0] + c[0, j] - c[prev, j]
                cF = bF + 2.0 * c[0, j]

                def after(qq: int) -> float:
                    if Fcode >= 0:
                        phigh = p if Fcode == 1 else 1.0 - p
                    else:
                        pf = self._post_F1(k, s, p)
                        phigh = pf * p + (1.0 - pf) * (1.0 - p)
                    total = 0.0
                    for z, prob in ((1, phigh), (0, 1.0 - phigh)):
                        if prob <= 0:
                            continue
                        d = highs[j] if z else lows[j]
                        if d <= qq:
                            total += prob * V(stage + 1, qq - d, k + 1, s + z)
                        else:
                            # high[j] <= Q by construction, hence at most one refill.
                            rem = d - qq
                            total += prob * (cF + V(stage + 1, Q - rem, k + 1, s + z))
                    return total

                return min(after(q), cP + after(Q))

            return V(0, Q, 0, 0)

        return qbar

    def qbar(self, path: tuple[int, ...]) -> float:
        return float(self._qbar(tuple(path), -1))

    def qf(self, path: tuple[int, ...]) -> float:
        path = tuple(path)
        return 0.5 * float(self._qbar(path, 0)) + 0.5 * float(self._qbar(path, 1))


# =============================================================================
# Directed multi-route master
# =============================================================================

SparseRow = tuple[list[int], list[float]]


class DirectedMaster:
    def __init__(self, inst: SmallInstance):
        self.inst = inst
        self.n = len(inst.mean)
        self.nodes = tuple(range(self.n + 1))
        self.customers = tuple(range(1, self.n + 1))
        self.arcs = tuple((i, j) for i in self.nodes for j in self.nodes if i != j)
        self.arc_idx = {a: k for k, a in enumerate(self.arcs)}
        self.theta_idx = {i: len(self.arcs) + i - 1 for i in self.customers}
        self.nvar = len(self.arcs) + self.n

        self.obj = np.zeros(self.nvar)
        for arc, idx in self.arc_idx.items():
            self.obj[idx] = inst.c[arc]
        for i in self.customers:
            self.obj[self.theta_idx[i]] = 1.0

        self.bounds_lp = [(0.0, 1.0)] * len(self.arcs) + [(0.0, None)] * self.n
        self.Aeq, self.beq = self._degree_equalities()
        self.base_rows, self.base_rhs = self._rounded_capacity_rows()

    def _degree_equalities(self):
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        rhs: list[float] = []
        r = 0

        for i in self.customers:
            for j in self.nodes:
                if i != j:
                    rows.append(r); cols.append(self.arc_idx[(i, j)]); vals.append(1.0)
            rhs.append(1.0); r += 1
        for j in self.customers:
            for i in self.nodes:
                if i != j:
                    rows.append(r); cols.append(self.arc_idx[(i, j)]); vals.append(1.0)
            rhs.append(1.0); r += 1

        for j in self.customers:
            rows.append(r); cols.append(self.arc_idx[(0, j)]); vals.append(1.0)
        rhs.append(float(self.inst.routes)); r += 1
        for i in self.customers:
            rows.append(r); cols.append(self.arc_idx[(i, 0)]); vals.append(1.0)
        rhs.append(float(self.inst.routes)); r += 1

        A = coo_matrix((vals, (rows, cols)), shape=(r, self.nvar)).tocsr()
        return A, np.asarray(rhs)

    def _rounded_capacity_rows(self):
        rows: list[SparseRow] = []
        rhs: list[float] = []
        means = (0,) + self.inst.mean
        Q = self.inst.Q
        for k in range(1, self.n + 1):
            for S in combinations(self.customers, k):
                need = int(math.ceil(sum(means[i] for i in S) / Q - 1e-12))
                need = max(1, need)
                bound = k - need
                idx: list[int] = []
                val: list[float] = []
                for i in S:
                    for j in S:
                        if i != j:
                            idx.append(self.arc_idx[(i, j)])
                            val.append(1.0)
                rows.append((idx, val))
                rhs.append(float(bound))
        return rows, rhs

    def info_cut(self, path: tuple[int, ...], q: float) -> tuple[SparseRow, float]:
        # sum theta(path) >= q * (sum internal arcs - |p| + 2)
        idx: list[int] = []
        val: list[float] = []
        for i, j in zip(path[:-1], path[1:]):
            idx.append(self.arc_idx[(i, j)])
            val.append(q)
        for i in path:
            idx.append(self.theta_idx[i])
            val.append(-1.0)
        return (idx, val), q * (len(path) - 2)

    def exact_route_cut(self, route: tuple[int, ...], q: float) -> tuple[SparseRow, float]:
        # Exact cut only when the complete directed depot cycle is selected:
        # sum theta(route) >= q * (sum cycle arcs - |route|).
        idx: list[int] = []
        val: list[float] = []
        cycle = ((0, route[0]),) + tuple(zip(route[:-1], route[1:])) + ((route[-1], 0),)
        for arc in cycle:
            idx.append(self.arc_idx[arc])
            val.append(q)
        for i in route:
            idx.append(self.theta_idx[i])
            val.append(-1.0)
        return (idx, val), q * len(route)

    def _matrix(self, extra_rows: list[SparseRow]):
        rows_all = self.base_rows + extra_rows
        rr: list[int] = []
        cc: list[int] = []
        dd: list[float] = []
        for r, (idx, val) in enumerate(rows_all):
            rr.extend([r] * len(idx)); cc.extend(idx); dd.extend(val)
        return coo_matrix((dd, (rr, cc)), shape=(len(rows_all), self.nvar)).tocsr()

    def solve_lp(self, extra_rows: list[SparseRow], extra_rhs: list[float]):
        Aub = self._matrix(extra_rows)
        rhs = np.asarray(self.base_rhs + extra_rhs)
        t0 = time.perf_counter()
        res = linprog(
            self.obj,
            A_ub=Aub,
            b_ub=rhs,
            A_eq=self.Aeq,
            b_eq=self.beq,
            bounds=self.bounds_lp,
            method="highs",
        )
        dt = time.perf_counter() - t0
        if not res.success:
            raise RuntimeError(f"LP failed: {res.message}")
        return res, dt

    def solve_mip(
        self,
        extra_rows: list[SparseRow],
        extra_rhs: list[float],
        time_limit: float,
    ):
        Aub = self._matrix(extra_rows)
        rhs = np.asarray(self.base_rhs + extra_rhs)
        A = vstack([self.Aeq, Aub], format="csr")
        lb = np.concatenate([self.beq, np.full(len(rhs), -np.inf)])
        ub = np.concatenate([self.beq, rhs])
        integrality = np.zeros(self.nvar, dtype=int)
        integrality[: len(self.arcs)] = 1
        lo = np.zeros(self.nvar)
        hi = np.full(self.nvar, np.inf)
        hi[: len(self.arcs)] = 1.0
        t0 = time.perf_counter()
        res = milp(
            self.obj,
            integrality=integrality,
            bounds=Bounds(lo, hi),
            constraints=LinearConstraint(A, lb, ub),
            options={
                "time_limit": float(time_limit),
                "mip_rel_gap": 0.0,
                "presolve": True,
            },
        )
        dt = time.perf_counter() - t0
        if res.x is None:
            raise RuntimeError(f"MIP failed without incumbent: {res.message}")
        return res, dt

    def cut_violation(self, row: SparseRow, rhs: float, x: np.ndarray) -> float:
        idx, val = row
        return float(sum(v * x[j] for j, v in zip(idx, val)) - rhs)

    def extract_routes(self, x: np.ndarray, tol: float = 0.5) -> list[tuple[int, ...]]:
        succ: dict[int, int] = {}
        for (i, j), idx in self.arc_idx.items():
            if i == 0:
                continue
            if x[idx] > tol:
                if i in succ:
                    raise RuntimeError(f"Multiple selected successors for node {i}")
                succ[i] = j

        starts = [j for j in self.customers if x[self.arc_idx[(0, j)]] > tol]
        routes: list[tuple[int, ...]] = []
        seen: set[int] = set()
        for start in starts:
            route: list[int] = []
            cur = start
            while cur != 0:
                if cur in seen:
                    raise RuntimeError("Cycle extraction encountered a repeated customer")
                seen.add(cur)
                route.append(cur)
                if cur not in succ:
                    raise RuntimeError(f"Missing successor for customer {cur}")
                cur = succ[cur]
            routes.append(tuple(route))

        if seen != set(self.customers):
            raise RuntimeError(f"Route extraction missed customers {set(self.customers)-seen}")
        if len(routes) != self.inst.routes:
            raise RuntimeError(f"Expected {self.inst.routes} routes, extracted {len(routes)}")
        return routes


# =============================================================================
# Algorithms
# =============================================================================

def all_candidate_paths(n: int, max_len: int) -> Iterable[tuple[int, ...]]:
    customers = tuple(range(1, n + 1))
    for k in range(1, min(max_len, n) + 1):
        for path in permutations(customers, k):
            yield path


@dataclass
class RootSeparationResult:
    rows: list[SparseRow]
    rhs: list[float]
    rounds: int
    cuts: int
    candidates: int
    candidate_time: float
    lp_time: float
    initial_bound: float
    final_bound: float


def separate_information_cuts(
    master: DirectedMaster,
    ev: RecourseEvaluator,
    *,
    max_path: int,
    batch: int,
    max_rounds: int,
    tol: float,
) -> RootSeparationResult:
    t0 = time.perf_counter()
    candidates: list[tuple[tuple[int, ...], float, SparseRow, float]] = []
    for path in all_candidate_paths(master.n, max_path):
        q = ev.qf(path)
        if q > 1e-12:
            row, rhs = master.info_cut(path, q)
            candidates.append((path, q, row, rhs))
    candidate_time = time.perf_counter() - t0

    active_rows: list[SparseRow] = []
    active_rhs: list[float] = []
    used: set[int] = set()
    lp_time = 0.0
    initial_bound = math.nan
    final_bound = math.nan
    rounds = 0

    for rnd in range(max_rounds):
        res, dt = master.solve_lp(active_rows, active_rhs)
        lp_time += dt
        rounds = rnd + 1
        if rnd == 0:
            initial_bound = float(res.fun)
        final_bound = float(res.fun)

        violations: list[tuple[float, int]] = []
        for idx, (_, _, row, rhs) in enumerate(candidates):
            if idx in used:
                continue
            v = master.cut_violation(row, rhs, res.x)
            if v > tol:
                violations.append((v, idx))
        if not violations:
            break
        violations.sort(reverse=True)
        for _, idx in violations[:batch]:
            used.add(idx)
            active_rows.append(candidates[idx][2])
            active_rhs.append(candidates[idx][3])
    else:
        print(f"WARNING: root separation hit max_rounds={max_rounds}")

    return RootSeparationResult(
        rows=active_rows,
        rhs=active_rhs,
        rounds=rounds,
        cuts=len(active_rows),
        candidates=len(candidates),
        candidate_time=candidate_time,
        lp_time=lp_time,
        initial_bound=initial_bound,
        final_bound=final_bound,
    )


@dataclass
class ExactSolveResult:
    method: str
    objective: float
    actual_objective: float
    travel: float
    recourse: float
    routes: list[tuple[int, ...]]
    outer_rounds: int
    exact_cuts: int
    mip_nodes: float
    mip_time: float
    total_time: float
    root_info_cuts: int
    incumbent_info_candidates: int
    incumbent_info_violated: int
    incumbent_info_cuts: int
    total_info_cuts: int
    info_retention_median: float
    final_info_gap: float
    final_info_gap_ratio: float
    root_info_rounds: int
    root_bound_initial: float
    root_bound_final: float
    info_candidate_time: float
    info_lp_time: float
    status: int


def solve_exact_outer_loop(
    master: DirectedMaster,
    ev: RecourseEvaluator,
    *,
    method: str,
    info: RootSeparationResult | None,
    time_limit: float,
    max_lazy_rounds: int,
    tol: float,
    add_incumbent_info: bool = False,
    max_path: int = 4,
    incumbent_batch: int = 25,
) -> ExactSolveResult:
    info_rows = [] if info is None else list(info.rows)
    info_rhs = [] if info is None else list(info.rhs)
    exact_rows: list[SparseRow] = []
    exact_rhs: list[float] = []
    exact_keys: set[tuple[int, ...]] = set()
    info_keys: set[tuple[int, ...]] = set()
    incumbent_info_count = 0
    incumbent_info_candidates = 0
    incumbent_info_violated = 0
    added_retentions: list[float] = []

    total_mip_time = 0.0
    total_nodes = 0.0
    last_res = None
    last_routes: list[tuple[int, ...]] = []
    start = time.perf_counter()

    for outer in range(1, max_lazy_rounds + 1):
        elapsed = time.perf_counter() - start
        remaining = max(1.0, time_limit - elapsed)
        res, dt = master.solve_mip(
            info_rows + exact_rows,
            info_rhs + exact_rhs,
            remaining,
        )
        total_mip_time += dt
        nodes = getattr(res, "mip_node_count", 0.0)
        if nodes is not None and np.isfinite(nodes):
            total_nodes += float(nodes)
        last_res = res
        last_routes = master.extract_routes(res.x)

        violated = 0
        violated_routes: list[tuple[int, ...]] = []
        for route in last_routes:
            q = ev.qbar(route)
            theta_sum = sum(res.x[master.theta_idx[i]] for i in route)
            if theta_sum + tol < q and route not in exact_keys:
                row, rhs = master.exact_route_cut(route, q)
                exact_rows.append(row)
                exact_rhs.append(rhs)
                exact_keys.add(route)
                violated += 1
                violated_routes.append(route)

        # Proper incumbent separation: among contiguous proper subpaths of the
        # exact-route-violated incumbents, add only genuinely violated
        # information cuts, ranked by current violation. `incumbent_batch`
        # therefore controls the cut count (unlike the v1 pilot).
        if add_incumbent_info and violated_routes:
            candidates: list[tuple[float, tuple[int, ...], SparseRow, float, float, float]] = []
            for route in violated_routes:
                L = len(route)
                for a in range(L):
                    for b in range(a + 1, min(L, a + max_path)):
                        sub = route[a:b + 1]
                        if sub in info_keys or sub in exact_keys:
                            continue
                        # The full route already receives an exact cut.
                        if len(sub) == L:
                            continue
                        qsub = ev.qf(sub)
                        if qsub <= 1e-12:
                            continue
                        row, rhs = master.info_cut(sub, qsub)
                        incumbent_info_candidates += 1
                        cut_v = master.cut_violation(row, rhs, res.x)
                        if cut_v <= tol:
                            continue
                        incumbent_info_violated += 1
                        qbar_sub = ev.qbar(sub)
                        candidates.append((cut_v, sub, row, rhs, qsub, qbar_sub))

            candidates.sort(key=lambda z: z[0], reverse=True)
            for _, sub, row, rhs, qsub, qbar_sub in candidates[:max(0, incumbent_batch)]:
                info_rows.append(row)
                info_rhs.append(rhs)
                info_keys.add(sub)
                incumbent_info_count += 1
                if qbar_sub > 1e-12:
                    added_retentions.append(qsub / qbar_sub)
        if violated == 0:
            break
    else:
        raise RuntimeError(f"{method}: exceeded max_lazy_rounds={max_lazy_rounds}")

    assert last_res is not None
    travel = 0.0
    for (i, j), idx in master.arc_idx.items():
        travel += master.inst.c[i, j] * last_res.x[idx]
    recourse = sum(ev.qbar(r) for r in last_routes)
    actual = travel + recourse
    if abs(float(last_res.fun) - actual) > 1e-6 * max(1.0, abs(actual)):
        raise AssertionError(
            f"{method}: master objective {last_res.fun} != evaluated objective {actual}"
        )

    info_total = 0.0 if info is None else info.candidate_time + info.lp_time
    final_qf = sum(ev.qf(r) for r in last_routes)
    final_info_gap = recourse - final_qf
    final_info_gap_ratio = final_info_gap / max(recourse, 1e-12)
    retention_median = (
        float(np.median(np.asarray(added_retentions, dtype=float)))
        if added_retentions else math.nan
    )
    return ExactSolveResult(
        method=method,
        objective=float(last_res.fun),
        actual_objective=actual,
        travel=travel,
        recourse=recourse,
        routes=last_routes,
        outer_rounds=outer,
        exact_cuts=len(exact_rows),
        mip_nodes=total_nodes,
        mip_time=total_mip_time,
        total_time=total_mip_time + info_total,
        root_info_cuts=0 if info is None else info.cuts,
        incumbent_info_candidates=incumbent_info_candidates,
        incumbent_info_violated=incumbent_info_violated,
        incumbent_info_cuts=incumbent_info_count,
        total_info_cuts=(0 if info is None else info.cuts) + incumbent_info_count,
        info_retention_median=retention_median,
        final_info_gap=final_info_gap,
        final_info_gap_ratio=final_info_gap_ratio,
        root_info_rounds=0 if info is None else info.rounds,
        root_bound_initial=math.nan if info is None else info.initial_bound,
        root_bound_final=math.nan if info is None else info.final_bound,
        info_candidate_time=0.0 if info is None else info.candidate_time,
        info_lp_time=0.0 if info is None else info.lp_time,
        status=int(last_res.status),
    )


# =============================================================================
# Main
# =============================================================================

def format_routes(routes: list[tuple[int, ...]], original_ids: tuple[int, ...]) -> str:
    mapped = []
    for r in routes:
        mapped.append("-".join(str(original_ids[i - 1]) for i in r))
    return "|".join(mapped)


def main(argv: list[str]) -> int:
    cfg = parse_kv(argv)
    if "help" in cfg or "file" not in cfg:
        print(__doc__)
        return 0 if "help" in cfg else 2

    path = cfg["file"]
    n = as_int(cfg, "n", 8)
    routes = as_int(cfg, "routes", 2)
    select = cfg.get("select", "random")
    seed = as_int(cfg, "seed", 0)
    demand = cfg.get("demand", "delivery")
    mean_level = as_float(cfg, "mean_level", 4.0)
    delta = as_int(cfg, "delta", 1)
    load = as_float(cfg, "load", 0.90)
    ps = as_list_float(cfg, "p", "0.50,0.80,0.95,0.99")
    max_path = as_int(cfg, "max_path", 4)
    batch = as_int(cfg, "batch", 25)
    max_sep_rounds = as_int(cfg, "sep_rounds", 30)
    root_sep = as_int(cfg, "root", 0) != 0
    max_lazy_rounds = as_int(cfg, "lazy_rounds", 30)
    time_limit = as_float(cfg, "time", 120.0)
    bP = as_float(cfg, "bp", 0.0)
    bF = as_float(cfg, "bf", 0.5)
    dist_scale = as_float(cfg, "dist_scale", 10_000.0)
    out = cfg.get("out", "dethloff_directed_info_pilot.csv")
    tol = as_float(cfg, "tol", 1e-8)

    data = parse_dethloff(path, dist_scale=dist_scale)
    inst = build_small_instance(
        data,
        n=n,
        routes=routes,
        select=select,
        seed=seed,
        demand_mode=demand,
        mean_level=mean_level,
        delta=delta,
        load=load,
    )
    master = DirectedMaster(inst)

    print("=" * 116)
    print(" DETHLOFF-DERIVED DIRECTED INFORMATION-CUT PILOT")
    print("=" * 116)
    print(f" source={path}")
    print(f" instance={inst.name}  selected original node IDs={inst.original_ids}")
    print(f" n={n} routes={routes} select={select} seed={seed} demand={demand}")
    print(f" mean={inst.mean}")
    print(f" low ={inst.low}")
    print(f" high={inst.high}")
    print(f" Q={inst.Q}  expected load ratio total/(routes*Q)={sum(inst.mean)/(routes*inst.Q):.3f}")
    print(f" max_path={max_path} batch={batch} root_sep={int(root_sep)} time_limit={time_limit:.1f}s per method/p")
    print(" NOTE: Dethloff-derived VRPSD pilot; pickups/deliveries are not jointly modeled.")
    print("=" * 116)

    records: list[dict[str, object]] = []
    print(
        f"{'p':>5} {'method':>9} {'obj':>11} {'travel':>10} {'rec':>9} "
        f"{'root':>11} {'info':>6} {'lazy':>5} {'nodes':>8} {'mip_s':>8} {'total_s':>8} {'gap%':>7}"
    )

    for pval in ps:
        ev = RecourseEvaluator(inst, pval, bP, bF)

        # Independence control: posterior and full-information coefficients coincide.
        if abs(pval - 0.5) <= 1e-12:
            sample = tuple(range(1, min(n, max_path) + 1))
            if abs(ev.qbar(sample) - ev.qf(sample)) > 1e-9:
                raise AssertionError("At p=0.5, qbar must equal qF")

        # Baseline exact outer loop.
        base = solve_exact_outer_loop(
            master,
            ev,
            method="BASELINE",
            info=None,
            time_limit=time_limit,
            max_lazy_rounds=max_lazy_rounds,
            tol=tol,
            add_incumbent_info=False,
            max_path=max_path,
            incumbent_batch=batch,
        )

        # Optional root user-cut separation. Incumbent subpath cuts are always
        # generated in HYBRID when a violated exact route is encountered.
        info = (
            separate_information_cuts(
                master,
                ev,
                max_path=max_path,
                batch=batch,
                max_rounds=max_sep_rounds,
                tol=tol,
            )
            if root_sep
            else RootSeparationResult([], [], 0, 0, 0, 0.0, 0.0, math.nan, math.nan)
        )
        hybrid = solve_exact_outer_loop(
            master,
            ev,
            method="HYBRID",
            info=info,
            time_limit=time_limit,
            max_lazy_rounds=max_lazy_rounds,
            tol=tol,
            add_incumbent_info=True,
            max_path=max_path,
            incumbent_batch=batch,
        )

        if abs(base.actual_objective - hybrid.actual_objective) > 1e-6 * max(
            1.0, abs(base.actual_objective)
        ):
            raise AssertionError(
                f"Exact objective mismatch at p={pval}: "
                f"baseline={base.actual_objective}, hybrid={hybrid.actual_objective}"
            )

        for result in (base, hybrid):
            root = (
                math.nan
                if result.method == "BASELINE"
                else result.root_bound_final
            )
            print(
                f"{pval:5.2f} {result.method:>9} {result.actual_objective:11.4f} "
                f"{result.travel:10.4f} {result.recourse:9.4f} "
                f"{root:11.4f} {result.total_info_cuts:6d} {result.exact_cuts:5d} "
                f"{result.mip_nodes:8.0f} {result.mip_time:8.3f} {result.total_time:8.3f} "
                f"{100.0*result.final_info_gap_ratio:7.2f}"
            )
            records.append(
                {
                    "instance": inst.name,
                    "source_file": str(path),
                    "p": pval,
                    "method": result.method,
                    "n": n,
                    "routes": routes,
                    "selection": select,
                    "seed": seed,
                    "demand_mode": demand,
                    "mean_level": mean_level,
                    "delta": delta,
                    "load_target": load,
                    "Q": inst.Q,
                    "expected_load_ratio": sum(inst.mean) / (routes * inst.Q),
                    "selected_original_ids": ";".join(map(str, inst.original_ids)),
                    "scaled_means": ";".join(map(str, inst.mean)),
                    "objective": result.actual_objective,
                    "travel": result.travel,
                    "recourse": result.recourse,
                    "routes_solution": format_routes(result.routes, inst.original_ids),
                    "root_info_candidates": 0 if result.method == "BASELINE" else info.candidates,
                    "root_info_cuts": result.root_info_cuts,
                    "incumbent_info_candidates": result.incumbent_info_candidates,
                    "incumbent_info_violated": result.incumbent_info_violated,
                    "incumbent_info_cuts": result.incumbent_info_cuts,
                    "total_info_cuts": result.total_info_cuts,
                    "info_retention_median": result.info_retention_median,
                    "final_info_gap": result.final_info_gap,
                    "final_info_gap_ratio": result.final_info_gap_ratio,
                    "root_info_rounds": result.root_info_rounds,
                    "root_bound_initial": result.root_bound_initial,
                    "root_bound_final": result.root_bound_final,
                    "info_candidate_time": result.info_candidate_time,
                    "info_lp_time": result.info_lp_time,
                    "exact_lazy_rounds": result.outer_rounds,
                    "exact_route_cuts": result.exact_cuts,
                    "mip_nodes_sum": result.mip_nodes,
                    "mip_time": result.mip_time,
                    "total_time_including_info": result.total_time,
                    "status": result.status,
                }
            )

        node_reduction = (
            100.0 * (base.mip_nodes - hybrid.mip_nodes) / base.mip_nodes
            if base.mip_nodes > 0
            else math.nan
        )
        print(
            f"      delta: nodes={node_reduction:+.1f}% reduction; "
            f"MIP speed={base.mip_time/max(hybrid.mip_time,1e-12):.2f}x; "
            f"end-to-end speed={base.total_time/max(hybrid.total_time,1e-12):.2f}x; "
            f"info cuts={hybrid.total_info_cuts} "
            f"(root {info.cuts}/{info.candidates}, incumbent "
            f"{hybrid.incumbent_info_cuts}/{hybrid.incumbent_info_violated}/"
            f"{hybrid.incumbent_info_candidates}; "
            f"median retention={hybrid.info_retention_median:.3f})"
        )

    if records:
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        print(f"\nWrote {out}")

    print("\nGO signal: identical exact objective and HYBRID reduces total wall-clock or solved-node count.")
    print("NO-GO: root cuts are weak, exact rounds/nodes do not fall, or separation overhead dominates.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
