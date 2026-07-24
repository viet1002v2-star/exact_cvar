#!/usr/bin/env python3
"""
dethloff_affine_certified_alns_diag_v6.py

DIAGNOSTIC build of dethloff_affine_certified_alns.py.

The solver, parser, route kernels, and ALNS operators are a verbatim copy of the
original benchmark script (neither imported nor modified). Added on top:

  * a route-level geometry probe (chi, kappa, zeta, R/delta, nu, D_align, residual coherence, and
    exact gamma under SLOPE / per-route delivery baseline / best-found direction),
  * spectral accumulators A = sum K_r D_r D_r^T and M = sum b_t b_t^T reported
    on BOTH the training route pool and the actual ALNS search stream, with
    exact leading-eigenspace transfer diagnostics for M (lambda_2/lambda_1,
    angle to the leading eigenspace, q^T M q, and bottom-alignment beta_q),
  * an optional tier-2 screened Certificate II measurement on ambiguous routes
    (candidate ratio rho_cand vs the (1-alpha) floor, and tier2/full cost),
  * a residual-direction diagnostic for C_e(u)=CVaR(||(I-uu^T)(X-mu)||),
    including a numerical best-found directional range, a rigorous global lower
    bound, and the full certificate-width factors that multiply the M-coherence
    factor by rho_e=C_e(u_T)/min_u C_e(u).

Design decisions baked in (see the accompanying discussion):
  - spectral summary uses EVERY check; route-level percentiles use a SAMPLED
    subset; the two populations are reported separately and never merged.
  - kappa is taken at t* = argmax_t ||q_t|| (the cut that DEFINES R), which is
    what the proven bound  1 <= chi <= min(3, B_align)  requires. Do not "fix"
    it to argmax ||d+q_t||: that breaks the bound derivation.
  - gamma_r(u) is the EXACT max_t ||(I-uu^T) b_t|| (O(Kd)), already what the
    certificate computes; the delta+R decomposition is only an interpretation
    (proven to sit in [1,3] relative to exact gamma).
  - benchmark mode (diagnostic=0) leaves the hot path untouched for clean timing.

Diagnostic CLI (added):
  diagnostic=1                 turn on diagnostics (default 0 = clean benchmark)
  diag_sample_rate=0.25        fraction of route checks logged at route level
  diag_tier2=0/1               run screened Certificate II on ambiguous routes
  diag_best_sample=200       sampled routes that get multistart best-found search
  diag_best_iters=80          projected-subgradient iterations per start
  diag_ce=0/1                  measure residual-direction variation (default 1)
  diag_ce_random=2048          random directions in the shared C_e search
  diag_ce_batch=64             batch size for vectorized C_e evaluation
  diag_ce_local_starts=6       multistart local refinements for min/max C_e
  diag_ce_local_steps=7        tangent-search scales per start
  diag_ce_local_trials=24      proposals per scale and start
  diag_out_prefix=diagnostics  CSV filename prefix

Outputs (diagnostic mode):
  <prefix>_route_checks.csv
  <prefix>_spectral_summary.csv
  <prefix>_tier2.csv

Quick diagnostic run (Windows CMD), 4 instances, short:
  python dethloff_affine_certified_alns_diag.py ^
    dir=Dethloff regex="(CON3-0|CON8-0|SCA3-0|SCA8-0)$" ^
    profiles=moderate d=10 N=10000 iters=5 reps=1 ^
    policy=SAA direction=SLOPE ^
    diagnostic=1 diag_sample_rate=0.5 diag_tier2=1 diag_best_sample=150

This run is NOT for speed-up.  In v6 the final spectral-only run additionally checks:
  1. whether the leading eigenspace of M_search is well separated,
  2. the exact angle from u_train to that eigenspace (not an arbitrary eigenvector),
  3. the Rayleigh quotient of the off-leading component q, and
  4. beta_q=(q^T M q-lambda_d)/(lambda_1-lambda_d), which distinguishes
     genuine bottom-eigenspace alignment from a merely small transfer angle, and
  5. the missing residual-direction factor C_e(u), so the code reports both the
     slope-radius factor sqrt(mu_max R_M) and the full half-width factor after
     multiplying by rho_e.  The best-found rho_e is descriptive; the version
     based on the global lower bound is conservative.
"""

from __future__ import annotations

import bisect
import csv
import glob
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

ALPHA = 0.90
CV = 0.25
EPS_FRAC = 0.15
N_DATA = 50_000
FACTOR_DIMENSION = 10
SUPPORT = 2.5
SEED = 2026

DEFAULT_ITERS = 50
DEFAULT_REPS = 3
DEFAULT_TRAIN_ROUTES = 200
DEFAULT_DIRECTION = "SLOPE"
DEFAULT_POLICY = "both"
DEFAULT_DATA_DIR = "Dethloff"
DEFAULT_OUTPUT = "results_affine_certified_alns.csv"


@dataclass(frozen=True)
class OverlayProfile:
    name: str
    spatial_decay: float
    common_weight: float


PROFILES = {
    "concentrated": OverlayProfile("concentrated", 2.0, 1.50),
    "moderate": OverlayProfile("moderate", 1.0, 1.00),
    "diffuse": OverlayProfile("diffuse", 0.50, 0.65),
}


def _lines_after(txt, tag):
    out, cap = [], False
    for line in txt.splitlines():
        s = line.strip()
        if not s:
            continue
        if not cap:
            if s.upper().startswith(tag):
                cap = True
            continue
        if any(ch.isalpha() for ch in s):
            break
        out.append(s)
    return out


def _header_int(txt, key):
    for line in txt.splitlines():
        if line.strip().upper().startswith(key):
            match = re.findall(r"-?\d+", line)
            if match:
                return int(match[-1])
    return None


def _header_float(txt, key):
    for line in txt.splitlines():
        if line.strip().upper().startswith(key):
            match = re.findall(r"-?\d+\.?\d*", line)
            if match:
                return float(match[-1])
    return None


def _parse_pd(txt, n):
    demands = np.zeros((n, 2), dtype=float)
    for row in _lines_after(txt, "PICKUP_AND_DELIVERY_SECTION"):
        tokens = row.split()
        if len(tokens) < 3:
            continue
        index = int(float(tokens[0])) - 1
        if 0 <= index < n:
            demands[index, 0] = float(tokens[-2])
            demands[index, 1] = float(tokens[-1])
    return demands


def parse_dethloff(path):
    txt = Path(path).read_text(errors="ignore")
    n = _header_int(txt, "DIMENSION")
    capacity = _header_float(txt, "CAPACITY")
    tokens = []
    for row in _lines_after(txt, "EDGE_WEIGHT_SECTION"):
        tokens.extend(row.split())
    values = [int(float(token)) for token in tokens]
    if n is None or len(values) != n * n:
        raise ValueError(
            "EDGE_WEIGHT_SECTION: got %d tokens, expected n*n=%s (need the FULL matrix)"
            % (len(values), None if n is None else n * n))
    distance = np.asarray(values, dtype=np.int64).reshape(n, n)
    demands = _parse_pd(txt, n)
    return distance, demands, capacity, n, 10000


@dataclass(frozen=True)
class AffineInstance:
    name: str
    distance: np.ndarray
    capacity: float
    n: int
    scale: float
    X: np.ndarray
    d0: np.ndarray
    p0: np.ndarray
    dvec: np.ndarray
    pvec: np.ndarray
    delivery_values: np.ndarray
    pickup_values: np.ndarray


def bounded_factor_samples(N, dimension, support, seed):
    rng = np.random.default_rng(seed)
    X = np.clip(rng.normal(size=(N, dimension)), -support, support)
    X -= X.mean(axis=0, keepdims=True)
    maximum = np.max(np.abs(X), axis=0)
    correction = np.maximum(1.0, maximum / support)
    X /= correction[None, :]
    return X


def spatial_basis_from_distance(customer_distance, components, decay):
    positive = customer_distance[customer_distance > 1e-12]
    length = float(np.median(positive)) if positive.size else 1.0
    length = max(length, 1e-12)
    kernel = np.exp(-customer_distance / length)
    kernel = 0.5 * (kernel + kernel.T)
    values, vectors = np.linalg.eigh(kernel)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    vectors = vectors[:, order]
    take = min(components, vectors.shape[1])
    weights = np.power(values[:take] + 1e-12, decay / 2.0)
    basis = vectors[:, :take] * weights[None, :]
    if take < components:
        basis = np.pad(basis, ((0, 0), (0, components - take)))
    return basis


def row_l1_normalize(matrix):
    norm = np.sum(np.abs(matrix), axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    return matrix / norm


def make_affine_instance(path, profile, N, dimension, cv, support, seed):
    distance, demands, capacity, n, scale = parse_dethloff(path)
    if dimension < 2:
        raise ValueError("dimension must be at least 2")
    if not (0.0 < cv < 0.95):
        raise ValueError("cv must lie in (0,0.95)")
    customer_count = n - 1
    X = bounded_factor_samples(N, dimension, support, seed)
    customer_distance = distance[1:, 1:].astype(float)
    basis = spatial_basis_from_distance(
        customer_distance, components=dimension - 1, decay=profile.spatial_decay)
    delivery_raw = np.column_stack(
        (np.full(customer_count, profile.common_weight), basis))
    rng = np.random.default_rng(seed + 1)
    rotation, _ = np.linalg.qr(rng.normal(size=(dimension - 1, dimension - 1)))
    pickup_spatial = basis @ rotation
    pickup_raw = np.column_stack(
        (np.full(customer_count, 0.85 * profile.common_weight), pickup_spatial))
    delivery_loading = row_l1_normalize(delivery_raw)
    pickup_loading = row_l1_normalize(pickup_raw)
    delivery_cv = np.minimum(
        cv * rng.uniform(0.80, 1.20, size=customer_count), 0.90)
    pickup_cv = np.minimum(
        cv * rng.uniform(0.80, 1.20, size=customer_count), 0.90)
    d0 = demands[:, 0].astype(float).copy()
    p0 = demands[:, 1].astype(float).copy()
    dvec = np.zeros((n, dimension))
    pvec = np.zeros((n, dimension))
    dvec[1:] = d0[1:, None] * delivery_cv[:, None] * delivery_loading / support
    pvec[1:] = p0[1:, None] * pickup_cv[:, None] * pickup_loading / support
    delivery_values = d0[:, None] + dvec @ X.T
    pickup_values = p0[:, None] + pvec @ X.T
    if delivery_values.min() < -1e-9 or pickup_values.min() < -1e-9:
        raise AssertionError("Bounded affine positivity construction failed.")
    return AffineInstance(
        name=Path(path).stem, distance=distance, capacity=float(capacity),
        n=n, scale=float(scale), X=X, d0=d0, p0=p0, dvec=dvec, pvec=pvec,
        delivery_values=delivery_values, pickup_values=pickup_values)


def route_cost(route, distance):
    if not route:
        return 0.0
    cost = distance[0, route[0]] + distance[route[-1], 0]
    for index in range(len(route) - 1):
        cost += distance[route[index], route[index + 1]]
    return float(cost)


def route_peaks(route, delivery_values, pickup_values):
    if not route:
        return np.zeros(delivery_values.shape[1])
    delivery = delivery_values[route].T
    pickup = pickup_values[route].T
    total_delivery = delivery.sum(axis=1)
    middle = (total_delivery[:, None] - np.cumsum(delivery, axis=1)
              + np.cumsum(pickup, axis=1))
    return np.maximum(total_delivery, middle.max(axis=1))


def tail_parameters(alpha, n):
    if not (0.0 <= alpha < 1.0):
        raise ValueError("alpha must lie in [0,1).")
    if n <= 0:
        raise ValueError("sample must be nonempty")
    tail_mass = (1.0 - alpha) * n
    nearest = float(round(tail_mass))
    tolerance = 16.0 * max(math.ulp(tail_mass), math.ulp(nearest))
    if abs(tail_mass - nearest) <= tolerance:
        tail_mass = nearest
    count = min(n, max(1, int(math.ceil(tail_mass))))
    boundary_weight = min(1.0, max(0.0, tail_mass - (count - 1)))
    return tail_mass, count, boundary_weight


def empirical_cvar(values, alpha):
    array = np.asarray(values, dtype=float)
    n = array.size
    tail_mass, count, boundary_weight = tail_parameters(alpha, n)
    tail = np.partition(array, n - count)[n - count:]
    boundary = float(tail.min())
    return (float(tail.sum()) - (1.0 - boundary_weight) * boundary) / tail_mass


def empirical_cvar_columns(values, alpha):
    """Column-wise empirical CVaR with the same fractional tail convention.

    Parameters
    ----------
    values : array_like, shape (N, m)
        Each column is one empirical loss sample.
    alpha : float
        CVaR confidence level in (0,1).
    """
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("values must be a two-dimensional array")
    n = array.shape[0]
    tail_mass, count, boundary_weight = tail_parameters(alpha, n)
    partitioned = np.partition(array, n - count, axis=0)
    tail = partitioned[n - count:, :]
    boundary = np.min(tail, axis=0)
    return (np.sum(tail, axis=0) - (1.0 - boundary_weight) * boundary) / tail_mass


def _normalize_direction_columns(U):
    U = np.asarray(U, dtype=float)
    if U.ndim == 1:
        U = U[:, None]
    norms = np.linalg.norm(U, axis=0)
    keep = norms > 1e-14
    if not np.any(keep):
        return np.empty((U.shape[0], 0), dtype=float)
    return U[:, keep] / norms[keep][None, :]


@dataclass
class ResidualDirectionDiagnostic:
    """Direction sensitivity of C_e(u)=CVaR(||(I-uu^T)(X-mu)||).

    The global search is numerical and therefore reports a *best-found* minimum.
    Separately, ``global_lower_bound`` is a rigorous direction-uniform lower bound:

        C_e(u) >= E||e(u)|| >= E||e(u)||^2 / max_i ||X_i-mu||
               >= (tr(Sigma)-lambda_1(Sigma)) / max_i ||X_i-mu||.

    Thus ``rho_certified_upper = C_e(u_T)/global_lower_bound`` is conservative,
    while ``rho_best_found`` is descriptive rather than certified.
    """

    centered: np.ndarray
    alpha: float
    random_count: int
    batch_size: int
    local_starts: int
    local_steps: int
    local_trials: int
    seed: int

    row_sq: np.ndarray = field(init=False)
    covariance: np.ndarray = field(init=False)
    covariance_eigenvalues: np.ndarray = field(init=False)
    covariance_eigenvectors: np.ndarray = field(init=False)
    global_lower_bound: float = field(init=False)
    best_found_min: float = field(init=False)
    best_found_max: float = field(init=False)
    best_found_min_direction: np.ndarray = field(init=False)
    best_found_max_direction: np.ndarray = field(init=False)
    random_p05: float = field(init=False)
    random_p50: float = field(init=False)
    random_p95: float = field(init=False)
    pca_top_cvar: float = field(init=False)
    pca_bottom_cvar: float = field(init=False)
    search_seconds: float = field(init=False)

    def __post_init__(self):
        t0 = time.perf_counter()
        X = np.asarray(self.centered, dtype=float)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] < 2:
            raise ValueError("centered factor sample must have shape (N,d), d>=2")
        X = X - X.mean(axis=0, keepdims=True)
        self.centered = X
        self.row_sq = np.einsum("ij,ij->i", X, X)
        self.covariance = X.T @ X / X.shape[0]
        vals, vecs = np.linalg.eigh(0.5 * (self.covariance + self.covariance.T))
        order = np.argsort(vals)[::-1]
        self.covariance_eigenvalues = np.maximum(vals[order], 0.0)
        self.covariance_eigenvectors = vecs[:, order]

        residual_second_moment_floor = max(
            float(np.sum(self.covariance_eigenvalues) - self.covariance_eigenvalues[0]), 0.0)
        max_centered_norm = float(np.sqrt(np.max(self.row_sq))) if self.row_sq.size else 0.0
        self.global_lower_bound = (
            residual_second_moment_floor / max_centered_norm
            if max_centered_norm > 1e-300 else 0.0)

        rng = np.random.default_rng(self.seed)
        d = X.shape[1]
        random_U = rng.normal(size=(d, max(0, self.random_count)))
        random_U = _normalize_direction_columns(random_U)
        random_values = self.evaluate_many(random_U)
        if random_values.size:
            self.random_p05 = float(np.percentile(random_values, 5))
            self.random_p50 = float(np.percentile(random_values, 50))
            self.random_p95 = float(np.percentile(random_values, 95))
        else:
            self.random_p05 = self.random_p50 = self.random_p95 = float("nan")

        canonical = np.eye(d)
        base_U = np.concatenate(
            [self.covariance_eigenvectors, canonical, random_U], axis=1)
        base_U = _normalize_direction_columns(base_U)
        base_values = self.evaluate_many(base_U)
        if base_values.size == 0:
            raise RuntimeError("residual-direction diagnostic generated no directions")

        min_order = np.argsort(base_values)[:max(1, self.local_starts)]
        max_order = np.argsort(base_values)[-max(1, self.local_starts):]
        min_U, min_values = self._local_refine(
            base_U[:, min_order], base_values[min_order], minimize=True, rng=rng)
        max_U, max_values = self._local_refine(
            base_U[:, max_order], base_values[max_order], minimize=False, rng=rng)

        i_min = int(np.argmin(min_values))
        i_max = int(np.argmax(max_values))
        self.best_found_min = float(min_values[i_min])
        self.best_found_max = float(max_values[i_max])
        self.best_found_min_direction = min_U[:, i_min].copy()
        self.best_found_max_direction = max_U[:, i_max].copy()
        self.pca_top_cvar = float(self.evaluate_one(self.covariance_eigenvectors[:, 0]))
        self.pca_bottom_cvar = float(self.evaluate_one(self.covariance_eigenvectors[:, -1]))
        self.search_seconds = time.perf_counter() - t0

        tol = 1e-10 * max(1.0, self.best_found_min)
        if self.global_lower_bound > self.best_found_min + tol:
            raise AssertionError(
                "rigorous C_e lower bound exceeds best-found value: "
                f"{self.global_lower_bound} > {self.best_found_min}")

    def evaluate_many(self, U):
        U = _normalize_direction_columns(U)
        m = U.shape[1]
        if m == 0:
            return np.empty(0, dtype=float)
        out = np.empty(m, dtype=float)
        batch = max(1, int(self.batch_size))
        for start in range(0, m, batch):
            stop = min(m, start + batch)
            z = self.centered @ U[:, start:stop]
            residual_sq = np.maximum(self.row_sq[:, None] - z * z, 0.0)
            out[start:stop] = empirical_cvar_columns(np.sqrt(residual_sq), self.alpha)
        return out

    def evaluate_one(self, u):
        u = np.asarray(u, dtype=float)
        norm = float(np.linalg.norm(u))
        if norm <= 1e-14:
            raise ValueError("direction must be nonzero")
        u = u / norm
        z = self.centered @ u
        residual = np.sqrt(np.maximum(self.row_sq - z * z, 0.0))
        return float(empirical_cvar(residual, self.alpha))

    def _local_refine(self, starts, values, minimize, rng):
        U = _normalize_direction_columns(starts)
        vals = np.asarray(values, dtype=float).copy()
        if U.shape[1] != vals.size:
            raise ValueError("local-refinement direction/value mismatch")
        if self.local_steps <= 0 or self.local_trials <= 0:
            return U, vals
        d, n_starts = U.shape
        for step in range(self.local_steps):
            scale = 0.40 * (0.48 ** step)
            proposals = []
            owner = []
            for j in range(n_starts):
                noise = rng.normal(size=(d, self.local_trials))
                noise -= U[:, [j]] * (U[:, [j]].T @ noise)
                cand = U[:, [j]] + scale * noise
                cand = _normalize_direction_columns(cand)
                proposals.append(cand)
                owner.extend([j] * cand.shape[1])
            if not proposals:
                break
            P = np.concatenate(proposals, axis=1)
            pv = self.evaluate_many(P)
            owner = np.asarray(owner, dtype=int)
            for j in range(n_starts):
                idx = np.flatnonzero(owner == j)
                if idx.size == 0:
                    continue
                local = idx[int(np.argmin(pv[idx]) if minimize else np.argmax(pv[idx]))]
                better = pv[local] < vals[j] if minimize else pv[local] > vals[j]
                if better:
                    U[:, j] = P[:, local]
                    vals[j] = pv[local]
        return U, vals

    def summary_for(self, u):
        ce_u = self.evaluate_one(u)
        min_with_u = min(self.best_found_min, ce_u)
        max_with_u = max(self.best_found_max, ce_u)
        rho_best = ce_u / min_with_u if min_with_u > 1e-300 else float("inf")
        spread_best = max_with_u / min_with_u if min_with_u > 1e-300 else float("inf")
        rho_upper = (
            ce_u / self.global_lower_bound
            if self.global_lower_bound > 1e-300 else float("inf"))
        return {
            "Ce_u": ce_u,
            "Ce_best_found_min": min_with_u,
            "Ce_best_found_max": max_with_u,
            "Ce_best_found_spread": spread_best,
            "Ce_rho_best_found": rho_best,
            "Ce_global_lower_bound": self.global_lower_bound,
            "Ce_rho_certified_upper": rho_upper,
            "Ce_random_p05": self.random_p05,
            "Ce_random_p50": self.random_p50,
            "Ce_random_p95": self.random_p95,
            "Ce_pca_top": self.pca_top_cvar,
            "Ce_pca_bottom": self.pca_bottom_cvar,
            "Ce_cov_l1": float(self.covariance_eigenvalues[0]),
            "Ce_cov_l2": float(self.covariance_eigenvalues[1]),
            "Ce_cov_ld": float(self.covariance_eigenvalues[-1]),
            "Ce_cov_l2_over_l1": (
                float(self.covariance_eigenvalues[1] / self.covariance_eigenvalues[0])
                if self.covariance_eigenvalues[0] > 1e-300 else float("nan")),
            "Ce_search_seconds": self.search_seconds,
        }


def upper_envelope(lines):
    records = sorted(lines, key=lambda item: (item[1], item[0]))
    unique = []
    for intercept, slope in records:
        intercept = float(intercept)
        slope = float(slope)
        if unique and slope == unique[-1][1]:
            if intercept > unique[-1][0]:
                unique[-1] = (intercept, slope)
            continue
        unique.append((intercept, slope))
    if not unique:
        raise ValueError("At least one line is required.")

    def crossing(left, right):
        return (left[0] - right[0]) / (right[1] - left[1])

    stack = []
    starts = []
    for line in unique:
        while stack:
            x = crossing(stack[-1], line)
            if starts and x <= starts[-1]:
                stack.pop()
                starts.pop()
            else:
                break
        if not stack:
            stack.append(line)
            starts.append(-math.inf)
        else:
            starts.append(crossing(stack[-1], line))
            stack.append(line)
    A = np.asarray([line[0] for line in stack], dtype=float)
    B = np.asarray([line[1] for line in stack], dtype=float)
    xbr = np.asarray(starts + [math.inf], dtype=float)
    return A, B, xbr


def cvar_os(envelope, z_sorted, prefix, alpha):
    A, B, xbr = envelope
    n = len(z_sorted)
    pieces = len(A)
    tail_mass, count, boundary_weight = tail_parameters(alpha, n)

    def value(index):
        z = z_sorted[index]
        piece = bisect.bisect_right(xbr, z) - 1
        piece = min(max(piece, 0), pieces - 1)
        return float(A[piece] + B[piece] * z)

    sign_change = bisect.bisect_left(B, 0.0)
    if sign_change == 0:
        valley = 0
    elif sign_change >= pieces:
        valley = n - 1
    else:
        position = bisect.bisect_left(z_sorted, xbr[sign_change])
        candidates = [index for index in (position - 1, position, position + 1)
                      if 0 <= index < n]
        valley = min(candidates, key=value)

    left_size = valley + 1
    right_size = n - 1 - valley

    def left_arm(k):
        if k <= 0:
            return math.inf
        if k > left_size:
            return -math.inf
        return value(k - 1)

    def right_arm(k):
        if k <= 0:
            return math.inf
        if k > right_size:
            return -math.inf
        return value(n - k)

    low = max(0, count - right_size)
    high = min(count, left_size)
    lo, hi, best = low, high, low - 1
    while lo <= hi:
        middle = (lo + hi) // 2
        if left_arm(middle) >= right_arm(count - middle + 1):
            best = middle
            lo = middle + 1
        else:
            hi = middle - 1
    selected_left = max(best, low)
    selected_right = count - selected_left

    def range_sum(first, last):
        if first > last:
            return 0.0
        total = 0.0
        index = first
        while index <= last:
            z = z_sorted[index]
            piece = bisect.bisect_right(xbr, z) - 1
            piece = min(max(piece, 0), pieces - 1)
            end = bisect.bisect_left(z_sorted, xbr[piece + 1]) - 1
            end = min(max(end, index), last)
            total += (A[piece] * (end - index + 1)
                      + B[piece] * (prefix[end + 1] - prefix[index]))
            index = end + 1
        return float(total)

    selected_sum = (range_sum(0, selected_left - 1)
                    + range_sum(n - selected_right, n - 1))
    boundary = min(left_arm(selected_left), right_arm(selected_right))
    return (selected_sum - (1.0 - boundary_weight) * boundary) / tail_mass


@dataclass(frozen=True)
class CertificateCache:
    alpha: float
    u: np.ndarray
    mu: np.ndarray
    z_sorted: np.ndarray
    z_prefix: np.ndarray
    residual_cvar: float
    explained_variance: float
    effective_rank: float
    entropy_rank: float
    top_share: float
    direction: str
    prep_seconds: float


@dataclass(frozen=True)
class RouteCertificate:
    projected_cvar: float
    gamma: float
    radius: float
    lower: float
    upper: float
    pieces: int


def route_affine_lines(instance, route):
    route = list(route)
    length = len(route)
    intercepts = np.empty(length + 1, dtype=float)
    slopes = np.empty((length + 1, instance.X.shape[1]), dtype=float)
    intercept = float(instance.d0[route].sum())
    slope = instance.dvec[route].sum(axis=0)
    intercepts[0] = intercept
    slopes[0] = slope
    for index, customer in enumerate(route, 1):
        intercept += instance.p0[customer] - instance.d0[customer]
        slope = slope + instance.pvec[customer] - instance.dvec[customer]
        intercepts[index] = intercept
        slopes[index] = slope
    return intercepts, slopes


def training_route_slopes(instance, route_count, seed):
    rng = np.random.default_rng(seed)
    customers = np.arange(1, instance.n)
    positive = instance.d0[1:][instance.d0[1:] > 0]
    mean_delivery = float(np.mean(positive)) if positive.size else 1.0
    target_length = max(2, int(round(instance.capacity / mean_delivery)))
    maximum_length = min(instance.n - 1,
                         max(8, int(math.ceil(1.7 * target_length))))
    routes = []
    for _ in range(route_count):
        length = int(rng.integers(2, maximum_length + 1))
        routes.append(tuple(int(customer) for customer in
                            rng.choice(customers, size=length, replace=False)))
    return routes


def slope_second_moment(instance, routes):
    dimension = instance.X.shape[1]
    moment = np.zeros((dimension, dimension), dtype=float)
    count = 0
    for route in routes:
        _, slopes = route_affine_lines(instance, route)
        moment += slopes.T @ slopes
        count += slopes.shape[0]
    if count:
        moment /= count
    return moment


def matrix_geometry(matrix):
    values = np.maximum(np.linalg.eigvalsh(matrix), 0.0)[::-1]
    total = float(values.sum())
    if total <= 1e-15:
        return 0.0, 0.0, 0.0
    weights = values / total
    positive = weights[weights > 1e-15]
    effective_rank = float(1.0 / np.sum(weights * weights))
    entropy_rank = float(np.exp(-np.sum(positive * np.log(positive))))
    top_share = float(weights[0])
    return effective_rank, entropy_rank, top_share


def build_certificate_cache(instance, alpha, direction, train_routes, seed):
    t0 = time.perf_counter()
    X = np.asarray(instance.X, dtype=float)
    mu = X.mean(axis=0)
    centered = X - mu[None, :]
    routes = training_route_slopes(instance, train_routes, seed)
    slope_moment = slope_second_moment(instance, routes)
    geometry = matrix_geometry(slope_moment)
    key = direction.upper()
    if key == "SLOPE":
        values, vectors = np.linalg.eigh(slope_moment)
        u = vectors[:, int(np.argmax(values))]
    elif key == "PCA":
        covariance = centered.T @ centered / max(1, centered.shape[0])
        values, vectors = np.linalg.eigh(covariance)
        u = vectors[:, int(np.argmax(values))]
    elif key == "DELIVERY":
        delivery_sums = []
        for route in routes:
            delivery_sums.append(instance.dvec[list(route)].sum(axis=0))
        matrix = np.asarray(delivery_sums)
        delivery_moment = matrix.T @ matrix / max(1, len(matrix))
        values, vectors = np.linalg.eigh(delivery_moment)
        u = vectors[:, int(np.argmax(values))]
    else:
        raise ValueError("direction must be SLOPE, PCA, or DELIVERY")
    norm = float(np.linalg.norm(u))
    if norm <= 1e-15:
        raise ValueError("Direction construction returned zero.")
    u = u / norm
    z = centered @ u
    residual_sq = np.maximum(
        np.einsum("ij,ij->i", centered, centered) - z * z, 0.0)
    residual_norm = np.sqrt(residual_sq)
    z_sorted = np.sort(z)
    z_prefix = np.concatenate(([0.0], np.cumsum(z_sorted)))
    residual_cvar = empirical_cvar(residual_norm, alpha)
    total_variance = float(np.sum(centered * centered))
    explained = (float(np.dot(z, z)) / total_variance
                 if total_variance > 0 else 1.0)
    return CertificateCache(
        alpha=float(alpha), u=u, mu=mu, z_sorted=z_sorted, z_prefix=z_prefix,
        residual_cvar=float(residual_cvar), explained_variance=float(explained),
        effective_rank=geometry[0], entropy_rank=geometry[1],
        top_share=geometry[2], direction=key,
        prep_seconds=(time.perf_counter() - t0))


def route_certificate(instance, route, cache):
    intercepts, slopes = route_affine_lines(instance, route)
    projected_intercepts = intercepts + slopes @ cache.mu
    projected_slopes = slopes @ cache.u
    envelope = upper_envelope(zip(projected_intercepts, projected_slopes))
    projected_cvar = cvar_os(envelope, cache.z_sorted, cache.z_prefix, cache.alpha)
    orthogonal_sq = np.maximum(
        np.einsum("ij,ij->i", slopes, slopes)
        - projected_slopes * projected_slopes, 0.0)
    gamma = float(np.sqrt(orthogonal_sq).max())
    radius = gamma * cache.residual_cvar
    return RouteCertificate(
        projected_cvar=float(projected_cvar), gamma=gamma, radius=float(radius),
        lower=float(projected_cvar - radius), upper=float(projected_cvar + radius),
        pieces=len(envelope[0]))


# ============================================================
# DIAGNOSTIC CORE (geometry / spectral / percentile helpers)
# ============================================================

# ---------------------------------------------------------------------------
# Per-route geometry record
# ---------------------------------------------------------------------------

@dataclass
class RouteDiag:
    K: int
    decision: str
    margin_sign: float

    gamma_slope: float
    gamma_delivery_route: float
    gamma_best_found: float
    gamma_lower_bound_l2: float

    # L-infinity / L2 residual coherence under the running global direction.
    # S = sum_t ||(I-uu^T)b_t||^2 and mu = K*gamma^2/S in [1,K].
    residual_l2_sq: float
    residual_rms: float
    residual_coherence_mu: float
    residual_coherence_sqrt: float

    delta: float
    R: float
    chi: float
    kappa: float
    R_over_delta: float
    zeta: float
    B_align: float
    B_kappa: float

    nu: float
    decision_slack_norm: float
    D_align: float

    best_search_starts: float = float("nan")
    best_search_spread: float = float("nan")


# ---------------------------------------------------------------------------
# Core geometry computation from the already-computed (intercepts, slopes)
# ---------------------------------------------------------------------------

def _gamma_for_direction(slopes: np.ndarray, u: np.ndarray) -> float:
    """max_t || (I - u u^T) b_t ||, exact, O(K d)."""
    proj = slopes @ u                       # (K,)
    orth_sq = np.maximum(
        np.einsum("ij,ij->i", slopes, slopes) - proj * proj, 0.0
    )
    return float(np.sqrt(orth_sq).max())


def _residual_coherence(slopes: np.ndarray, u: np.ndarray):
    """Return (S, RMS, mu, sqrt(mu)) for route residual slopes.

    S = sum_t ||(I-uu^T)b_t||^2, gamma = max_t ||(I-uu^T)b_t||, and
    mu = K*gamma^2/S.  For S>0, 1 <= mu <= K.  In the exact rank-one
    case S=gamma=0, we use the benign convention mu=1.
    """
    slopes = np.asarray(slopes, dtype=float)
    u = np.asarray(u, dtype=float)
    K = int(slopes.shape[0])
    proj = slopes @ u
    orth_sq = np.maximum(
        np.einsum("ij,ij->i", slopes, slopes) - proj * proj, 0.0
    )
    S = float(orth_sq.sum())
    gamma_sq = float(orth_sq.max()) if orth_sq.size else 0.0
    rms = math.sqrt(S / max(1, K))
    if S <= 1e-300:
        mu = 1.0
    else:
        mu = K * gamma_sq / S
        if mu < 1.0 - 1e-8 or mu > K + 1e-8:
            raise AssertionError(f"residual coherence outside [1,K]: mu={mu}, K={K}")
        mu = float(np.clip(mu, 1.0, float(K)))
    return S, rms, mu, math.sqrt(mu)



def _best_found_gamma(
    slopes: np.ndarray,
    seeds: list[np.ndarray],
    n_random: int,
    rng: np.random.Generator,
    iters: int = 60,
    step0: float = 0.5,
):
    """Return the best gamma found by multistart projected subgradient search.

    This is an upper bound on the unknown route optimum, not a certified oracle.
    The search always includes all supplied seed directions in its incumbent.
    """
    d = slopes.shape[1]
    bnorm_sq = np.einsum("ij,ij->i", slopes, slopes)

    def f_and_active(u):
        proj = slopes @ u
        vals = bnorm_sq - proj * proj
        fmax = float(vals.max())
        tol = 1e-9 * (1.0 + abs(fmax))
        active = np.where(vals >= fmax - tol - 1e-12)[0]
        return fmax, proj, active

    starts = []
    for seed in seeds:
        v = np.asarray(seed, dtype=float)
        nv = float(np.linalg.norm(v))
        if nv > 1e-15:
            starts.append(v / nv)
    for _ in range(n_random):
        v = rng.normal(size=d)
        nv = float(np.linalg.norm(v))
        if nv > 1e-15:
            starts.append(v / nv)

    results = []
    for u0 in starts:
        u = u0.copy()
        step = float(step0)
        fbest_local, _, _ = f_and_active(u)
        for _ in range(iters):
            fval, proj, active = f_and_active(u)
            fbest_local = min(fbest_local, fval)
            # Honest active-gradient averaging; this is not a solved bundle QP.
            G = -2.0 * (proj[active, None] * slopes[active, :])
            g = G.mean(axis=0)
            g = g - float(g @ u) * u
            gn = float(np.linalg.norm(g))
            if gn <= 1e-14:
                break
            trial = u - step * (g / gn)
            nn = float(np.linalg.norm(trial))
            if nn <= 1e-15:
                break
            trial /= nn
            f_new, _, _ = f_and_active(trial)
            if f_new < fval:
                u = trial
                step = min(step * 1.15, step0)
            else:
                step *= 0.6
                if step < 1e-6:
                    break
        results.append(math.sqrt(max(fbest_local, 0.0)))

    if not results:
        return float("nan"), 0.0, float("nan")
    arr = np.asarray(results, dtype=float)
    return float(arr.min()), float(arr.size), float(arr.max() - arr.min())



def build_route_diag(
    *,
    slopes: np.ndarray,
    D_direct: np.ndarray,
    u_slope: np.ndarray,
    projected_cvar: float,
    gamma_slope: float,
    radius: float,
    capacity: float,
    C_e: float,
    padding: float,
    decision: str,
    run_best_search: bool,
    best_rng: np.random.Generator | None,
    best_iters: int,
    tol: float = 1e-7,
) -> RouteDiag:
    """Build one sampled route diagnostic without changing the solver decision."""
    slopes = np.asarray(slopes, dtype=float)
    u_slope = np.asarray(u_slope, dtype=float)
    D_direct = np.asarray(D_direct, dtype=float)
    K, d = slopes.shape

    un = float(np.linalg.norm(u_slope))
    if abs(un - 1.0) > tol:
        raise AssertionError(f"u_slope is not unit: ||u||={un}")
    if np.max(np.abs(slopes[0] - D_direct)) > tol * (1.0 + np.max(np.abs(D_direct))):
        raise AssertionError("route convention failed: b_0 != direct delivery-slope sum")

    gamma_check = _gamma_for_direction(slopes, u_slope)
    if abs(float(gamma_slope) - gamma_check) > tol * (1.0 + gamma_check):
        raise AssertionError(f"gamma mismatch: passed={gamma_slope}, recomputed={gamma_check}")
    radius_check = gamma_check * float(C_e)
    if abs(float(radius) - radius_check) > tol * (1.0 + abs(radius_check)):
        raise AssertionError(f"radius mismatch: passed={radius}, recomputed={radius_check}")

    projected = slopes @ u_slope
    perp = slopes - np.outer(projected, u_slope)
    d_vec = perp[0]
    delta = float(np.linalg.norm(d_vec))

    q = perp - perp[0][None, :]
    q_norm = np.sqrt(np.maximum(np.einsum("ij,ij->i", q, q), 0.0))
    tstar = int(np.argmax(q_norm))
    R = float(q_norm[tstar])
    gamma = gamma_check
    residual_l2_sq, residual_rms, residual_coherence_mu, residual_coherence_sqrt = \
        _residual_coherence(slopes, u_slope)
    chi = 1.0 if gamma <= 1e-300 else (delta + R) / gamma

    if delta <= 1e-300 or R <= 1e-300:
        kappa = float("nan")
    else:
        raw = float(np.dot(d_vec, q[tstar]) / (delta * R))
        if raw < -1.0 - 1e-10 or raw > 1.0 + 1e-10:
            raise AssertionError(f"kappa outside [-1,1]: {raw}")
        kappa = float(np.clip(raw, -1.0, 1.0))

    if delta <= 1e-300:
        R_over_delta = float("inf") if R > 1e-300 else float("nan")
    else:
        R_over_delta = R / delta

    step_perp = perp[1:] - perp[:-1]
    step_len = np.sqrt(np.maximum(np.einsum("ij,ij->i", step_perp, step_perp), 0.0))
    walk_length = float(step_len.sum())
    zeta = float("nan") if walk_length <= 1e-300 else R / walk_length
    if math.isfinite(zeta) and (zeta < -1e-8 or zeta > 1.0 + 1e-8):
        raise AssertionError(f"zeta outside [0,1]: {zeta}")
    if math.isfinite(zeta):
        zeta = float(np.clip(zeta, 0.0, 1.0))

    if math.isnan(kappa):
        B_align = float("nan")
        B_kappa = float("nan")
    else:
        disc = delta * delta + R * R + 2.0 * delta * R * kappa
        scale = delta * delta + R * R + 1.0
        if disc < -1e-10 * scale:
            raise AssertionError(f"negative alignment discriminant: {disc}")
        disc = max(disc, 0.0)
        B_align = float("inf") if disc <= 1e-300 else (delta + R) / math.sqrt(disc)
        B_kappa = float("inf") if kappa <= -1.0 + 1e-15 else math.sqrt(2.0 / (1.0 + kappa))

    denom = gamma * float(C_e)
    margin = float(capacity) - float(projected_cvar)
    margin_sign = 1.0 if margin >= 0 else -1.0
    if denom <= 1e-300:
        nu = float("inf") if abs(margin) > 1e-300 else float("nan")
        raw_slack = abs(margin) - gamma * float(C_e) - float(padding)
        decision_slack_norm = (
            float("inf") if raw_slack > 0
            else float("-inf") if raw_slack < 0
            else float("nan")
        )
    else:
        nu = abs(margin) / denom
        decision_slack_norm = (abs(margin) - denom - float(padding)) / denom

    Dn = float(np.linalg.norm(D_direct))
    D_align = float("nan") if Dn <= 1e-300 else float(abs(np.dot(D_direct / Dn, u_slope)))
    gamma_delivery = float("nan") if Dn <= 1e-300 else _gamma_for_direction(slopes, D_direct / Dn)

    # Cheap certified lower bound on the unknown min-gamma objective:
    # max residual^2 >= average residual^2, minimized by the leading eigenvector.
    M = slopes.T @ slopes
    evals = np.linalg.eigvalsh(M)
    gamma_lb_sq = max((float(np.trace(M)) - float(evals[-1])) / max(1, K), 0.0)
    gamma_lb = math.sqrt(gamma_lb_sq)

    gamma_best = float("nan")
    n_starts = float("nan")
    spread = float("nan")
    if run_best_search and best_rng is not None:
        seeds = [u_slope]
        if Dn > 1e-300:
            seeds.append(D_direct / Dn)
        try:
            evals_m, evecs_m = np.linalg.eigh(M)
            order = np.argsort(evals_m)[::-1]
            for j in range(min(3, d)):
                seeds.append(evecs_m[:, order[j]])
        except np.linalg.LinAlgError:
            pass
        gamma_best, n_starts, spread = _best_found_gamma(
            slopes,
            seeds,
            max(1, int(math.ceil(math.sqrt(d)))),
            best_rng,
            iters=best_iters,
        )

    if gamma > 1e-12:
        if chi < 1.0 - 1e-6 or chi > 3.0 + 1e-6:
            raise AssertionError(f"chi outside [1,3]: {chi}")
        if math.isfinite(B_align) and chi > B_align + 1e-6:
            raise AssertionError(f"chi > B_align: {chi} > {B_align}")
    if math.isfinite(gamma_best) and gamma_best > gamma + 1e-6 * (1.0 + gamma):
        raise AssertionError("best-found gamma exceeds seeded SLOPE gamma")
    if gamma_lb > gamma + 1e-6 * (1.0 + gamma):
        raise AssertionError("lower bound exceeds feasible SLOPE gamma")

    return RouteDiag(
        K=K,
        decision=decision,
        margin_sign=margin_sign,
        gamma_slope=gamma,
        gamma_delivery_route=gamma_delivery,
        gamma_best_found=gamma_best,
        gamma_lower_bound_l2=gamma_lb,
        residual_l2_sq=residual_l2_sq,
        residual_rms=residual_rms,
        residual_coherence_mu=residual_coherence_mu,
        residual_coherence_sqrt=residual_coherence_sqrt,
        delta=delta,
        R=R,
        chi=chi,
        kappa=kappa,
        R_over_delta=R_over_delta,
        zeta=zeta,
        B_align=B_align,
        B_kappa=B_kappa,
        nu=nu,
        decision_slack_norm=decision_slack_norm,
        D_align=D_align,
        best_search_starts=n_starts,
        best_search_spread=spread,
    )


# ---------------------------------------------------------------------------
# Spectral accumulator: A = sum K_r D_r D_r^T,  M = sum_{r,t} b_t b_t^T
# accumulated over EVERY check (rank-1 / rank-K updates are cheap, reuse slopes)
# ---------------------------------------------------------------------------

@dataclass
class SpectralAccumulator:
    d: int
    A: np.ndarray = field(default=None)
    M: np.ndarray = field(default=None)
    n_routes: int = 0

    def __post_init__(self):
        if self.A is None:
            self.A = np.zeros((self.d, self.d), dtype=float)
        if self.M is None:
            self.M = np.zeros((self.d, self.d), dtype=float)

    def update(self, slopes: np.ndarray):
        slopes = np.asarray(slopes, dtype=float)
        K = slopes.shape[0]
        D_r = slopes[0]
        self.A += K * np.outer(D_r, D_r)
        self.M += slopes.T @ slopes
        self.n_routes += 1

    def summary(self, u_slope: np.ndarray, label: str) -> dict:
        out = {
            "label": label,
            "population": label,
            "n_routes": self.n_routes,
            "A_l1": float("nan"),
            "A_l2": float("nan"),
            "A_l2_over_l1": float("nan"),
            "A_l1_over_tr": float("nan"),
            "Delta_A": float("nan"),
            "E_spec_norm": float("nan"),
            "eta": float("nan"),
            "angle_vM_vA_deg": float("nan"),
            "angle_uslope_vM_deg": float("nan"),
            "angle_uslope_E1_deg": float("nan"),
            "angle_uslope_vA_deg": float("nan"),
            "M_l1": float("nan"),
            "M_l2": float("nan"),
            "M_ld": float("nan"),
            "M_l2_over_l1": float("nan"),
            "M_gap12": float("nan"),
            "M_gap12_rel": float("nan"),
            "M_leading_multiplicity": 0,
            "M_trace": float("nan"),
            "M_residual_opt": float("nan"),
            "M_residual_uslope": float("nan"),
            "M_transfer_ratio": float("nan"),
            "M_transfer_bound_ratio": float("nan"),
            "M_transfer_excess": float("nan"),
            "M_transfer_bound_excess": float("nan"),
            "M_q_rayleigh": float("nan"),
            "M_beta_q": float("nan"),
            "M_transfer_identity_excess": float("nan"),
            "M_bound_slack_over_actual_excess": float("nan"),
            "M_bound_slack_fraction": float("nan"),
            "M_gap12_degenerate": 0,
            "eigengap_degenerate": 0,
        }
        if self.n_routes == 0:
            return out

        def leading(mat):
            vals, vecs = np.linalg.eigh(0.5 * (mat + mat.T))
            order = np.argsort(vals)[::-1]
            return vals[order], vecs[:, order]

        try:
            evA, vecA = leading(self.A)
            evM, vecM = leading(self.M)
        except np.linalg.LinAlgError:
            out["eigengap_degenerate"] = 1
            return out

        evA_nonneg = np.maximum(evA, 0.0)
        trA = float(evA_nonneg.sum())
        l1 = float(evA_nonneg[0])
        l2 = float(evA_nonneg[1]) if evA_nonneg.size > 1 else 0.0
        gap = l1 - l2
        E = self.M - self.A
        try:
            E_norm = float(np.linalg.norm(E, 2))
        except np.linalg.LinAlgError:
            E_norm = float("nan")

        out.update({
            "A_l1": l1,
            "A_l2": l2,
            "A_l2_over_l1": l2 / l1 if l1 > 1e-300 else float("nan"),
            "A_l1_over_tr": l1 / trA if trA > 1e-300 else float("nan"),
            "Delta_A": gap,
            "E_spec_norm": E_norm,
        })
        rel_gap = gap / l1 if l1 > 1e-300 else 0.0
        if gap <= 1e-12 * max(1.0, l1) or rel_gap < 1e-9:
            out["eigengap_degenerate"] = 1
        elif math.isfinite(E_norm):
            out["eta"] = E_norm / gap

        def angle(a, b):
            na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
            if na <= 1e-300 or nb <= 1e-300:
                return float("nan")
            c = abs(float(np.dot(a / na, b / nb)))
            return float(math.degrees(math.acos(float(np.clip(c, 0.0, 1.0)))))

        vA = vecA[:, 0]
        vM = vecM[:, 0]
        out["angle_vM_vA_deg"] = angle(vM, vA)
        out["angle_uslope_vM_deg"] = angle(u_slope, vM)  # legacy, basis-dependent under multiplicity
        out["angle_uslope_vA_deg"] = angle(u_slope, vA)

        # Exact train-to-search transfer quantities for
        # J_S(u)=tr(M)-u^T M u.  v6 measures the angle to the full leading
        # eigenspace, not to an arbitrary single eigenvector.  Near-degeneracy
        # is reported separately by lambda_2/lambda_1 and the relative gap.
        evM_nonneg = np.maximum(evM, 0.0)
        M_trace = float(evM_nonneg.sum())
        M_l1 = float(evM_nonneg[0]) if evM_nonneg.size else 0.0
        M_l2 = float(evM_nonneg[1]) if evM_nonneg.size > 1 else 0.0
        M_ld = float(evM_nonneg[-1]) if evM_nonneg.size else 0.0
        M_l2_over_l1 = M_l2 / M_l1 if M_l1 > 1e-300 else float("nan")
        M_gap12 = max(M_l1 - M_l2, 0.0)
        M_gap12_rel = M_gap12 / M_l1 if M_l1 > 1e-300 else float("nan")

        # Numerical equality tolerance only.  Do NOT fold merely near-leading
        # eigenvectors into E1: Proposition 2 requires an exact eigenspace.
        eps = np.finfo(float).eps
        eig_tol = 100.0 * eps * max(1.0, abs(float(evM[0]))) * max(1, self.d)
        lead_idx = np.flatnonzero(np.abs(evM - evM[0]) <= eig_tol)
        if lead_idx.size == 0:  # defensive; index 0 should always qualify
            lead_idx = np.array([0], dtype=int)
        E1 = vecM[:, lead_idx]
        M_leading_multiplicity = int(lead_idx.size)
        M_gap12_degenerate = int(evM.size > 1 and abs(float(evM[0] - evM[1])) <= eig_tol)

        un = float(np.linalg.norm(u_slope))
        if un > 1e-300:
            uu = np.asarray(u_slope, dtype=float) / un
            symM = 0.5 * (self.M + self.M.T)
            rayleigh = float(uu @ symM @ uu)
            # Rayleigh cannot exceed lambda_1 mathematically; trim only tiny roundoff.
            if rayleigh > M_l1 and rayleigh <= M_l1 + 1e-10 * max(1.0, M_l1):
                rayleigh = M_l1

            coeff = E1.T @ uu
            proj = E1 @ coeff
            c2 = float(np.dot(proj, proj))
            c2 = float(np.clip(c2, 0.0, 1.0))
            sin2 = max(0.0, 1.0 - c2)
            angle_E1 = float(math.degrees(math.asin(math.sqrt(float(np.clip(sin2, 0.0, 1.0))))))
            out["angle_uslope_E1_deg"] = angle_E1

            M_residual_opt = max(M_trace - M_l1, 0.0)
            M_residual_uslope = max(M_trace - rayleigh, 0.0)
            M_transfer_excess = max(M_residual_uslope - M_residual_opt, 0.0)
            M_transfer_bound_excess = max((M_l1 - M_ld) * sin2, 0.0)

            q_comp = uu - proj
            q_norm = float(np.linalg.norm(q_comp))
            q_tol = 1e-12
            if q_norm > q_tol:
                q = q_comp / q_norm
                M_q_rayleigh = float(q @ symM @ q)
                M_transfer_identity_excess = max(sin2 * (M_l1 - M_q_rayleigh), 0.0)
                spread = M_l1 - M_ld
                if spread > 1e-14 * max(1.0, M_l1):
                    raw_beta = (M_q_rayleigh - M_ld) / spread
                    if raw_beta < -1e-8 or raw_beta > 1.0 + 1e-8:
                        raise AssertionError(f"M beta_q outside [0,1]: {raw_beta}")
                    M_beta_q = float(np.clip(raw_beta, 0.0, 1.0))
                else:
                    M_beta_q = float("nan")
            else:
                # theta=0 (or full leading eigenspace): q is not uniquely defined,
                # but the exact transfer excess is identically zero.
                M_q_rayleigh = float("nan")
                M_beta_q = float("nan")
                M_transfer_identity_excess = 0.0

            identity_scale = max(1.0, M_trace, abs(M_transfer_excess), abs(M_transfer_identity_excess))
            if abs(M_transfer_excess - M_transfer_identity_excess) > 1e-8 * identity_scale:
                raise AssertionError(
                    "spectral transfer identity mismatch: "
                    f"actual={M_transfer_excess}, identity={M_transfer_identity_excess}")

            bound_slack = max(M_transfer_bound_excess - M_transfer_excess, 0.0)
            M_bound_slack_over_actual_excess = (
                bound_slack / M_transfer_excess
                if M_transfer_excess > 1e-14 * max(1.0, M_trace)
                else float("nan")
            )
            M_bound_slack_fraction = (
                bound_slack / M_transfer_bound_excess
                if M_transfer_bound_excess > 1e-14 * max(1.0, M_trace)
                else float("nan")
            )
            if math.isfinite(M_beta_q) and math.isfinite(M_bound_slack_fraction):
                if abs(M_beta_q - M_bound_slack_fraction) > 1e-7:
                    raise AssertionError(
                        "beta_q does not match normalized spectral-bound slack: "
                        f"beta={M_beta_q}, slack_fraction={M_bound_slack_fraction}")

            scale = max(1.0, M_trace)
            if M_residual_opt > 1e-14 * scale:
                M_transfer_ratio = M_residual_uslope / M_residual_opt
                M_transfer_bound_ratio = 1.0 + M_transfer_bound_excess / M_residual_opt
            elif M_residual_uslope <= 1e-14 * scale:
                M_transfer_ratio = 1.0
                M_transfer_bound_ratio = 1.0
            else:
                M_transfer_ratio = float("inf")
                M_transfer_bound_ratio = float("inf")
            if math.isfinite(M_transfer_ratio) and M_transfer_ratio < 1.0 - 1e-8:
                raise AssertionError(f"M transfer ratio below one: {M_transfer_ratio}")
            if (math.isfinite(M_transfer_ratio) and math.isfinite(M_transfer_bound_ratio)
                    and M_transfer_ratio > M_transfer_bound_ratio + 1e-7):
                raise AssertionError(
                    f"exact M transfer ratio exceeds spectral bound: "
                    f"{M_transfer_ratio} > {M_transfer_bound_ratio}")
        else:
            M_residual_opt = M_residual_uslope = float("nan")
            M_transfer_ratio = M_transfer_bound_ratio = float("nan")
            M_transfer_excess = M_transfer_bound_excess = float("nan")
            M_q_rayleigh = M_beta_q = float("nan")
            M_transfer_identity_excess = float("nan")
            M_bound_slack_over_actual_excess = M_bound_slack_fraction = float("nan")

        out.update({
            "M_l1": M_l1,
            "M_l2": M_l2,
            "M_ld": M_ld,
            "M_l2_over_l1": M_l2_over_l1,
            "M_gap12": M_gap12,
            "M_gap12_rel": M_gap12_rel,
            "M_leading_multiplicity": M_leading_multiplicity,
            "M_trace": M_trace,
            "M_residual_opt": M_residual_opt,
            "M_residual_uslope": M_residual_uslope,
            "M_transfer_ratio": M_transfer_ratio,
            "M_transfer_bound_ratio": M_transfer_bound_ratio,
            "M_transfer_excess": M_transfer_excess,
            "M_transfer_bound_excess": M_transfer_bound_excess,
            "M_q_rayleigh": M_q_rayleigh,
            "M_beta_q": M_beta_q,
            "M_transfer_identity_excess": M_transfer_identity_excess,
            "M_bound_slack_over_actual_excess": M_bound_slack_over_actual_excess,
            "M_bound_slack_fraction": M_bound_slack_fraction,
            "M_gap12_degenerate": M_gap12_degenerate,
        })
        return out



@dataclass
class AAccumulator:
    d: int
    A: np.ndarray = field(default=None)
    n_routes: int = 0

    def __post_init__(self):
        if self.A is None:
            self.A = np.zeros((self.d, self.d), dtype=float)

    def update(self, slopes: np.ndarray):
        K = slopes.shape[0]
        D_r = slopes[0]
        self.A += K * np.outer(D_r, D_r)
        self.n_routes += 1

    def summary(self, u_slope: np.ndarray, label: str) -> dict:
        temp = SpectralAccumulator(self.d)
        temp.A = self.A.copy()
        temp.M = self.A.copy()
        temp.n_routes = self.n_routes
        out = temp.summary(u_slope, label)
        out["E_spec_norm"] = float("nan")
        out["eta"] = float("nan")
        out["angle_vM_vA_deg"] = 0.0 if self.n_routes else float("nan")
        out["angle_uslope_vM_deg"] = out["angle_uslope_vA_deg"]
        out["angle_uslope_E1_deg"] = float("nan")
        for key in (
            "M_l1", "M_l2", "M_ld", "M_l2_over_l1", "M_gap12",
            "M_gap12_rel", "M_trace", "M_residual_opt",
            "M_residual_uslope", "M_transfer_ratio",
            "M_transfer_bound_ratio", "M_transfer_excess",
            "M_transfer_bound_excess", "M_q_rayleigh", "M_beta_q",
            "M_transfer_identity_excess", "M_bound_slack_over_actual_excess",
            "M_bound_slack_fraction",
        ):
            out[key] = float("nan")
        out["M_leading_multiplicity"] = 0
        out["M_gap12_degenerate"] = 0
        return out

def _coherence_summary(values):
    """Compact all-route coherence summary for a population."""
    arr = np.asarray([x for x in values if x is not None and math.isfinite(x)], dtype=float)
    keys = {
        "mu_n": int(arr.size), "mu_mean": float("nan"),
        "mu_p50": float("nan"), "mu_p90": float("nan"),
        "mu_p95": float("nan"), "mu_max": float("nan"),
        "sqrt_mu_p50": float("nan"), "sqrt_mu_p90": float("nan"),
        "sqrt_mu_p95": float("nan"), "sqrt_mu_max": float("nan"),
    }
    if arr.size:
        root = np.sqrt(arr)
        keys.update({
            "mu_mean": float(arr.mean()),
            "mu_p50": float(np.percentile(arr, 50)),
            "mu_p90": float(np.percentile(arr, 90)),
            "mu_p95": float(np.percentile(arr, 95)),
            "mu_max": float(arr.max()),
            "sqrt_mu_p50": float(np.percentile(root, 50)),
            "sqrt_mu_p90": float(np.percentile(root, 90)),
            "sqrt_mu_p95": float(np.percentile(root, 95)),
            "sqrt_mu_max": float(root.max()),
        })
    return keys


def _attach_transfer_theorem_metrics(out):
    """Attach the observed multiplicative factors in the M-coherence theorem.

    J_gamma(u_train) <= mu_max * M_transfer_ratio * J_gamma^*, hence the
    weighted-RMS factor is sqrt(mu_max * M_transfer_ratio).  The bound variant
    replaces the exact transfer ratio by its eigenspread-angle upper bound.
    """
    mu = out.get("mu_max", float("nan"))
    exact = out.get("M_transfer_ratio", float("nan"))
    bound = out.get("M_transfer_bound_ratio", float("nan"))
    out["theorem_factor_exact_observed"] = (
        math.sqrt(mu * exact)
        if math.isfinite(mu) and math.isfinite(exact) and mu >= 0 and exact >= 0
        else float("nan")
    )
    out["theorem_factor_spectral_bound"] = (
        math.sqrt(mu * bound)
        if math.isfinite(mu) and math.isfinite(bound) and mu >= 0 and bound >= 0
        else float("nan")
    )


def _attach_certificate_width_metrics(out, ce_metrics):
    """Attach the missing residual-direction factor in the full bracket width.

    The route-radius theorem controls J_gamma.  The actual certificate half-width
    is gamma_r(u) C_e(u), so at population level Psi(u)=C_e(u)^2 J_gamma(u).
    ``Ce_rho_best_found`` is descriptive because the denominator is numerically
    minimized. ``Ce_rho_certified_upper`` uses a rigorous global lower bound.
    """
    if ce_metrics is None:
        ce_metrics = {}
    for key in (
        "Ce_u", "Ce_best_found_min", "Ce_best_found_max",
        "Ce_best_found_spread", "Ce_rho_best_found",
        "Ce_global_lower_bound", "Ce_rho_certified_upper",
        "Ce_random_p05", "Ce_random_p50", "Ce_random_p95",
        "Ce_pca_top", "Ce_pca_bottom",
        "Ce_cov_l1", "Ce_cov_l2", "Ce_cov_ld",
        "Ce_cov_l2_over_l1", "Ce_search_seconds",
    ):
        out[key] = ce_metrics.get(key, float("nan"))

    exact = out.get("theorem_factor_exact_observed", float("nan"))
    spectral = out.get("theorem_factor_spectral_bound", float("nan"))
    rho_best = out.get("Ce_rho_best_found", float("nan"))
    rho_upper = out.get("Ce_rho_certified_upper", float("nan"))
    out["certificate_factor_best_found"] = (
        exact * rho_best
        if math.isfinite(exact) and math.isfinite(rho_best) and exact >= 0 and rho_best >= 0
        else float("nan")
    )
    out["certificate_factor_spectral_best_found"] = (
        spectral * rho_best
        if math.isfinite(spectral) and math.isfinite(rho_best) and spectral >= 0 and rho_best >= 0
        else float("nan")
    )
    out["certificate_factor_certified_upper"] = (
        exact * rho_upper
        if math.isfinite(exact) and math.isfinite(rho_upper) and exact >= 0 and rho_upper >= 0
        else float("nan")
    )
    return out


# ---------------------------------------------------------------------------
# Percentile summary helper
# ---------------------------------------------------------------------------

def percentile_summary(values, name, pcts=(10, 25, 50, 75, 90, 95)):
    arr = np.asarray([v for v in values
                      if v is not None and math.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"name": name, "n": 0}
    out = {"name": name, "n": int(arr.size),
           "mean": float(arr.mean()), "min": float(arr.min()),
           "max": float(arr.max())}
    for p in pcts:
        out[f"p{p}"] = float(np.percentile(arr, p))
    return out


# ==== DIAGNOSTIC-AWARE GATES + verbatim ALNS operators ====

class FullGate:
    mode = "FULL"

    def __init__(self, instance, capacity, alpha):
        self.instance = instance
        self.capacity = float(capacity)
        self.alpha = float(alpha)
        self.calls = 0
        self.route_length_sum = 0
        self.gate_seconds = 0.0

    def feasible(self, route):
        if not route:
            return True
        self.calls += 1
        self.route_length_sum += len(route)
        t0 = time.perf_counter()
        value = empirical_cvar(
            route_peaks(route, self.instance.delivery_values,
                        self.instance.pickup_values), self.alpha)
        self.gate_seconds += time.perf_counter() - t0
        return value <= self.capacity + 1e-9


class CertifiedGate:
    mode = "CERT"

    def __init__(self, instance, capacity, alpha, cache, audit=False,
                 diag=None):
        self.instance = instance
        self.capacity = float(capacity)
        self.alpha = float(alpha)
        self.cache = cache
        self.audit = bool(audit)
        self.diag = diag                      # DiagContext or None
        self.calls = 0
        self.route_length_sum = 0
        self.certified_feasible = 0
        self.certified_infeasible = 0
        self.fallbacks = 0
        self.pieces_sum = 0
        self.certificate_seconds = 0.0
        self.fallback_seconds = 0.0
        self.audit_seconds = 0.0

    @property
    def coverage(self):
        if not self.calls:
            return 0.0
        return (self.certified_feasible + self.certified_infeasible) / self.calls

    @property
    def gate_seconds(self):
        return self.certificate_seconds + self.fallback_seconds

    def full_decision(self, route):
        value = empirical_cvar(
            route_peaks(route, self.instance.delivery_values,
                        self.instance.pickup_values), self.alpha)
        return value <= self.capacity + 1e-9

    def feasible(self, route):
        if not route:
            return True
        self.calls += 1
        self.route_length_sum += len(route)

        t0 = time.perf_counter()
        # Reuse intercepts/slopes: recompute here but hand them to diag so the
        # diagnostic never calls route_affine_lines a second time.
        intercepts, slopes = route_affine_lines(self.instance, route)
        projected_intercepts = intercepts + slopes @ self.cache.mu
        projected_slopes = slopes @ self.cache.u
        envelope = upper_envelope(zip(projected_intercepts, projected_slopes))
        projected_cvar = cvar_os(envelope, self.cache.z_sorted,
                                 self.cache.z_prefix, self.cache.alpha)
        orthogonal_sq = np.maximum(
            np.einsum("ij,ij->i", slopes, slopes)
            - projected_slopes * projected_slopes, 0.0)
        gamma = float(np.sqrt(orthogonal_sq).max())
        radius = gamma * self.cache.residual_cvar
        lower = projected_cvar - radius
        upper = projected_cvar + radius
        pieces = len(envelope[0])
        self.certificate_seconds += time.perf_counter() - t0
        self.pieces_sum += pieces

        padding = (64.0 * np.finfo(float).eps
                   * (1.0 + abs(self.capacity) + abs(projected_cvar) + abs(radius)))

        if upper + padding <= self.capacity:
            decision = True
            self.certified_feasible += 1
            decision_label = "certified_feasible"
            fell_back = False
        elif lower - padding > self.capacity:
            decision = False
            self.certified_infeasible += 1
            decision_label = "certified_infeasible"
            fell_back = False
        else:
            self.fallbacks += 1
            decision_label = "ambiguous"
            fell_back = True
            t0 = time.perf_counter()
            decision = self.full_decision(route)
            self.fallback_seconds += time.perf_counter() - t0

        # -------- diagnostic hook (sampled; does not affect decision) --------
        if self.diag is not None:
            self.diag.observe(
                route=route, slopes=slopes, projected_cvar=projected_cvar,
                gamma_slope=gamma, radius=radius, capacity=self.capacity,
                padding=padding, decision_label=decision_label,
                envelope_pieces=pieces, cache=self.cache)

        if self.audit and not fell_back:
            t0 = time.perf_counter()
            truth = self.full_decision(route)
            self.audit_seconds += time.perf_counter() - t0
            if decision != truth:
                raise AssertionError(
                    "Certificate mismatch: "
                    f"route={route}, lower={lower}, upper={upper}, "
                    f"capacity={self.capacity}, cert={decision}, full={truth}")

        return decision


def cw_init(distance, gate, n):
    routes = [[customer] for customer in range(1, n)
              if gate.feasible([customer])]
    placed = {customer for route in routes for customer in route}
    leftovers = [customer for customer in range(1, n) if customer not in placed]
    savings = []
    for a in range(1, n):
        for b in range(a + 1, n):
            savings.append((distance[0, a] + distance[0, b] - distance[a, b], a, b))
    savings.sort(reverse=True)
    routes_by_id = {index: route for index, route in enumerate(routes)}
    where = {route[0]: index for index, route in enumerate(routes)}
    for saving, a, b in savings:
        if saving <= 0:
            break
        ra, rb = where.get(a), where.get(b)
        if ra is None or rb is None or ra == rb:
            continue
        route_a = routes_by_id[ra]
        route_b = routes_by_id[rb]
        if route_a[-1] == a and route_b[0] == b:
            merged = route_a + route_b
        elif route_a[0] == a and route_b[-1] == b:
            merged = route_b + route_a
        elif route_a[-1] == a and route_b[-1] == b:
            merged = route_a + route_b[::-1]
        elif route_a[0] == a and route_b[0] == b:
            merged = route_a[::-1] + route_b
        else:
            continue
        if not gate.feasible(merged):
            continue
        routes_by_id[ra] = merged
        for customer in route_b:
            where[customer] = ra
        del routes_by_id[rb]
    solution = list(routes_by_id.values())
    for customer in leftovers:
        solution.append([customer])
    return solution


def two_opt_gate(route, distance, gate):
    if len(route) < 4:
        return route
    current = route[:]
    improved = True
    while improved:
        improved = False
        for i in range(len(current) - 1):
            for k in range(i + 1, len(current)):
                a = current[i - 1] if i > 0 else 0
                b = current[i]
                c = current[k]
                d = current[k + 1] if k + 1 < len(current) else 0
                if a == c or b == d:
                    continue
                delta = (distance[a, c] + distance[b, d]
                         - distance[a, b] - distance[c, d])
                if delta < -1e-9:
                    candidate = (current[:i] + current[i:k + 1][::-1]
                                 + current[k + 1:])
                    if gate.feasible(candidate):
                        current = candidate
                        improved = True
                        break
            if improved:
                break
    return current


def relocate_gate(solution, distance, gate):
    improved = True
    while improved:
        improved = False
        for route_index in range(len(solution)):
            route = solution[route_index]
            for position in range(len(route)):
                customer = route[position]
                a = route[position - 1] if position > 0 else 0
                b = route[position + 1] if position + 1 < len(route) else 0
                gain = distance[a, customer] + distance[customer, b] - distance[a, b]
                for target_index in range(len(solution)):
                    if target_index == route_index:
                        continue
                    target = solution[target_index]
                    for insertion_position in range(len(target) + 1):
                        u = target[insertion_position - 1] if insertion_position > 0 else 0
                        v = target[insertion_position] if insertion_position < len(target) else 0
                        delta = (distance[u, customer] + distance[customer, v]
                                 - distance[u, v] - gain)
                        if delta < -1e-9:
                            new_route = route[:position] + route[position + 1:]
                            new_target = (target[:insertion_position] + [customer]
                                          + target[insertion_position:])
                            if gate.feasible(new_route) and gate.feasible(new_target):
                                solution[route_index] = new_route
                                solution[target_index] = new_target
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        solution = [route for route in solution if route]
    return solution


def greedy_insert(solution, customer, distance, gate):
    best = (None, None, math.inf)
    for route_index, route in enumerate(solution):
        for position in range(len(route) + 1):
            candidate = route[:position] + [customer] + route[position:]
            if not gate.feasible(candidate):
                continue
            u = route[position - 1] if position > 0 else 0
            v = route[position] if position < len(route) else 0
            delta = distance[u, customer] + distance[customer, v] - distance[u, v]
            if delta < best[2]:
                best = (route_index, position, delta)
    return best


def ruin_recreate(solution, distance, gate, rng, q_frac=0.2):
    solution = [route[:] for route in solution]
    customers = [customer for route in solution for customer in route]
    q = max(1, int(q_frac * len(customers)))
    removed = rng.sample(customers, min(q, len(customers)))
    removed_set = set(removed)
    solution = [[customer for customer in route if customer not in removed_set]
                for route in solution]
    solution = [route for route in solution if route]
    rng.shuffle(removed)
    for customer in removed:
        route_index, position, _ = greedy_insert(solution, customer, distance, gate)
        if route_index is None:
            solution.append([customer])
        else:
            solution[route_index].insert(position, customer)
    return solution


def local_search(solution, distance, gate):
    solution = [two_opt_gate(route, distance, gate)
                for route in solution if route]
    solution = relocate_gate(solution, distance, gate)
    return [route for route in solution if route]


def econ_cost(solution, distance, omega_vehicle):
    return (sum(route_cost(route, distance) for route in solution)
            + omega_vehicle * sum(1 for route in solution if route))


def solve_fixed_iterations(distance, gate, n, omega_vehicle, iterations, seed):
    rng = random.Random(seed)
    current = local_search(cw_init(distance, gate, n), distance, gate)
    best = [route[:] for route in current]
    best_cost = econ_cost(best, distance, omega_vehicle)
    for _ in range(iterations):
        candidate = local_search(
            ruin_recreate(best, distance, gate, rng,
                          rng.choice([0.1, 0.15, 0.2, 0.3])),
            distance, gate)
        candidate_cost = econ_cost(candidate, distance, omega_vehicle)
        if candidate_cost < best_cost - 1e-9:
            best = [route[:] for route in candidate]
            best_cost = candidate_cost
    return [route for route in best if route]



# ==== DIAGNOSTIC CONTEXT: sampling, spectral accumulation, tier-2 ====


class DiagContext:
    """Diagnostics for one CERT search stream.

    Call-weighted moments explain evaluator workload. Unique-route moments are
    additionally reported to expose duplication artifacts in the ALNS stream.
    Route-level rows are sampled with an RNG independent of the solver RNG.
    """

    def __init__(
        self,
        *,
        instance,
        capacity,
        alpha,
        cache,
        sample_rate,
        best_sample,
        best_iters,
        tier2,
        seed,
        stream_label,
    ):
        self.instance = instance
        self.capacity = float(capacity)
        self.alpha = float(alpha)
        self.cache = cache
        self.sample_rate = float(sample_rate)
        self.best_sample = int(best_sample)
        self.best_iters = int(best_iters)
        self.tier2_enabled = bool(tier2)
        self.stream_label = stream_label

        # Independent diagnostic RNGs: never shared with random.Random used by ALNS.
        self.sample_rng = np.random.default_rng(seed)
        self.best_rng = np.random.default_rng(seed + 777)

        d = instance.X.shape[1]
        self.spectral_call = SpectralAccumulator(d=d)
        self.spectral_unique_ordered = SpectralAccumulator(d=d)
        self.A_unique_customer_set = AAccumulator(d=d)
        self.seen_ordered = set()
        self.seen_customer_sets = set()

        self.route_rows = []
        self.best_done = 0
        self.tier2_rows = []
        self.n_observed = 0
        self.n_sampled = 0

        # Exact coherence is cheap and accumulated on the full search stream.
        self.mu_call_weighted = []
        self.mu_unique_ordered = []

    def observe(
        self,
        *,
        route,
        slopes,
        projected_cvar,
        gamma_slope,
        radius,
        capacity,
        padding,
        decision_label,
        envelope_pieces,
        cache,
    ):
        self.n_observed += 1
        route_tuple = tuple(int(x) for x in route)
        set_key = tuple(sorted(route_tuple))

        self.spectral_call.update(slopes)
        _, _, mu_now, _ = _residual_coherence(slopes, cache.u)
        self.mu_call_weighted.append(mu_now)
        if route_tuple not in self.seen_ordered:
            self.seen_ordered.add(route_tuple)
            self.spectral_unique_ordered.update(slopes)
            self.mu_unique_ordered.append(mu_now)
        if set_key not in self.seen_customer_sets:
            self.seen_customer_sets.add(set_key)
            self.A_unique_customer_set.update(slopes)

        if self.sample_rng.random() >= self.sample_rate:
            return
        self.n_sampled += 1

        run_best = self.best_done < self.best_sample
        D_direct = self.instance.dvec[list(route_tuple)].sum(axis=0)
        rd = build_route_diag(
            slopes=slopes,
            D_direct=D_direct,
            u_slope=cache.u,
            projected_cvar=projected_cvar,
            gamma_slope=gamma_slope,
            radius=radius,
            capacity=capacity,
            C_e=cache.residual_cvar,
            padding=padding,
            decision=decision_label,
            run_best_search=run_best,
            best_rng=self.best_rng if run_best else None,
            best_iters=self.best_iters,
        )
        if run_best:
            self.best_done += 1
        self.route_rows.append(rd)

        if self.tier2_enabled and decision_label == "ambiguous":
            self._tier2(route_tuple, slopes, cache)

    def _tier2(self, route, slopes, cache):
        """Measure scenario-wise screened bounds on sampled tier-1 fallbacks."""
        t0 = time.perf_counter()
        X = self.instance.X
        mu = cache.mu
        u = cache.u
        N = X.shape[0]
        centered = X - mu[None, :]
        z = centered @ u
        e = centered - np.outer(z, u)
        e_norm = np.sqrt(np.maximum(np.einsum("ij,ij->i", e, e), 0.0))

        pint = self._intercepts(route) + slopes @ mu
        pslope = slopes @ u
        c = slopes - np.outer(pslope, u)
        c_norm = np.sqrt(np.maximum(np.einsum("ij,ij->i", c, c), 0.0))

        G = pint[None, :] + np.outer(z, pslope)
        Ptilde = G.max(axis=1)
        active = G.argmax(axis=1)
        Delta = Ptilde[:, None] - G
        L = Ptilde - c_norm[active] * e_norm
        U = Ptilde + (c_norm[None, :] * e_norm[:, None] - Delta).max(axis=1)

        cvar_L = empirical_cvar(L, self.alpha)
        cvar_U = empirical_cvar(U, self.alpha)
        pad2 = 64.0 * np.finfo(float).eps * (
            1.0 + abs(self.capacity) + abs(cvar_L) + abs(cvar_U)
        )
        if cvar_U + pad2 <= self.capacity:
            tier2_decision = "certified_feasible"
        elif cvar_L - pad2 > self.capacity:
            tier2_decision = "certified_infeasible"
        else:
            tier2_decision = "ambiguous"

        _, m, _ = tail_parameters(self.alpha, N)
        theta = np.partition(L, N - m)[N - m]
        cand_mask = U >= theta
        cand = np.nonzero(cand_mask)[0]
        rho_cand = cand.size / N

        t_partial = time.perf_counter()
        if cand.size:
            dvals = self.instance.delivery_values[list(route)][:, cand]
            pvals = self.instance.pickup_values[list(route)][:, cand]
            total_delivery = dvals.sum(axis=0)
            middle = total_delivery[None, :] - np.cumsum(dvals, axis=0) + np.cumsum(pvals, axis=0)
            peaks_cand = np.maximum(total_delivery, middle.max(axis=0))
            tail_mass, count, boundary_weight = tail_parameters(self.alpha, N)
            tail = np.partition(peaks_cand, peaks_cand.size - count)[peaks_cand.size - count:]
            boundary = float(tail.min())
            cvar_candidate = (float(tail.sum()) - (1.0 - boundary_weight) * boundary) / tail_mass
        else:
            peaks_cand = np.array([], dtype=float)
            cvar_candidate = float("nan")
        partial_seconds = time.perf_counter() - t_partial
        tier2_seconds = time.perf_counter() - t0

        t_full = time.perf_counter()
        peaks_full = route_peaks(list(route), self.instance.delivery_values, self.instance.pickup_values)
        cvar_full = empirical_cvar(peaks_full, self.alpha)
        full_seconds = time.perf_counter() - t_full
        full_feasible = cvar_full <= self.capacity + 1e-9

        # Ties can make an arbitrary argpartition top-m set unsuitable as a proof;
        # equality of candidate and full CVaR is the robust correctness check.
        cvar_match = bool(
            math.isfinite(cvar_candidate)
            and abs(cvar_candidate - cvar_full) <= 1e-8 * (1.0 + abs(cvar_full))
        )
        if not cvar_match:
            raise AssertionError(
                f"tier-2 candidate CVaR mismatch: cand={cvar_candidate}, full={cvar_full}"
            )
        if tier2_decision == "certified_feasible" and not full_feasible:
            raise AssertionError("tier-2 false feasible certificate")
        if tier2_decision == "certified_infeasible" and full_feasible:
            raise AssertionError("tier-2 false infeasible certificate")

        self.tier2_rows.append({
            "stream": self.stream_label,
            "route_len": len(route),
            "rho_cand": rho_cand,
            "cand_count": int(cand.size),
            "N": N,
            "candidate_cvar_matches_full": int(cvar_match),
            "tier2_decision": tier2_decision,
            "tier2_incremental_certified": int(tier2_decision != "ambiguous"),
            "full_feasible": int(full_feasible),
            "t_tier2_total": tier2_seconds,
            "t_tier2_exact_partial": partial_seconds,
            "t_full": full_seconds,
            "tier2_over_full": tier2_seconds / full_seconds if full_seconds > 0 else float("nan"),
            "cvar_L": float(cvar_L),
            "cvar_U": float(cvar_U),
            "cvar_candidate": float(cvar_candidate),
            "cvar_full": float(cvar_full),
        })

    def _intercepts(self, route):
        intercept = float(self.instance.d0[list(route)].sum())
        out = np.empty(len(route) + 1, dtype=float)
        out[0] = intercept
        for i, customer in enumerate(route, 1):
            intercept += self.instance.p0[customer] - self.instance.d0[customer]
            out[i] = intercept
        return out

    def spectral_summaries(self):
        call = self.spectral_call.summary(self.cache.u, "search_call_weighted")
        call.update(_coherence_summary(self.mu_call_weighted))
        _attach_transfer_theorem_metrics(call)
        unique = self.spectral_unique_ordered.summary(
            self.cache.u, "search_unique_ordered_route"
        )
        unique.update(_coherence_summary(self.mu_unique_ordered))
        _attach_transfer_theorem_metrics(unique)
        a_only = self.A_unique_customer_set.summary(
            self.cache.u, "search_unique_customer_set_A_only"
        )
        a_only.update(_coherence_summary([]))
        _attach_transfer_theorem_metrics(a_only)
        return [call, unique, a_only]




def median_iqr(values):
    array = np.asarray(list(values), dtype=float)
    return float(np.median(array)), float(np.quantile(array, 0.75) - np.quantile(array, 0.25))


def benchmark_cell(
    instance, profile, policy, capacity, alpha, cache,
    iterations, repetitions, seed, audit,
):
    warm_iterations = min(2, iterations)
    if warm_iterations > 0:
        wf = run_solver_once(instance, capacity, alpha, cache, warm_iterations, seed, "FULL", False)
        wc = run_solver_once(instance, capacity, alpha, cache, warm_iterations, seed, "CERT", False)
        if wf["plan_key"] != wc["plan_key"]:
            raise AssertionError("Warm-up FULL/CERT plans differ")

    rng = random.Random(seed + sum(map(ord, policy)) + sum(map(ord, profile)))
    full_runs, cert_runs, gate_speedups, solver_speedups = [], [], [], []
    for repetition in range(repetitions):
        order = ["FULL", "CERT"]
        rng.shuffle(order)
        current = {}
        for method in order:
            current[method] = run_solver_once(
                instance, capacity, alpha, cache, iterations, seed,
                method, audit if method == "CERT" else False,
            )
        full, cert = current["FULL"], current["CERT"]
        if full["plan_key"] != cert["plan_key"]:
            raise AssertionError(f"{instance.name}/{profile}/{policy}: final plans differ")
        if full["calls"] != cert["calls"]:
            raise AssertionError(f"{instance.name}/{profile}/{policy}: check counts differ")
        if abs(full["objective_raw"] - cert["objective_raw"]) > 1e-8:
            raise AssertionError(f"{instance.name}/{profile}/{policy}: objectives differ")
        full_runs.append(full); cert_runs.append(cert)
        gate_speedups.append(full["gate_seconds"] / cert["gate_seconds"])
        solver_speedups.append(full["total_seconds"] / cert["total_seconds"])
        print(
            f"      rep {repetition + 1}/{repetitions}: "
            f"gate={gate_speedups[-1]:.2f}x solver={solver_speedups[-1]:.2f}x "
            f"cover={cert['coverage']:.3f} calls={cert['calls']} assert=PASS"
        )

    full_gate, full_gate_iqr = median_iqr(r["gate_seconds"] for r in full_runs)
    cert_gate, cert_gate_iqr = median_iqr(r["gate_seconds"] for r in cert_runs)
    full_solver, full_solver_iqr = median_iqr(r["total_seconds"] for r in full_runs)
    cert_solver, cert_solver_iqr = median_iqr(r["total_seconds"] for r in cert_runs)
    gate_speed, gate_speed_iqr = median_iqr(gate_speedups)
    solver_speed, solver_speed_iqr = median_iqr(solver_speedups)
    ref = cert_runs[0]
    return {
        "instance": instance.name, "profile": profile, "policy": policy,
        "N": instance.X.shape[0], "dimension": instance.X.shape[1],
        "alpha": alpha, "capacity": capacity, "iterations": iterations,
        "repetitions": repetitions, "direction": cache.direction,
        "explained_variance": cache.explained_variance,
        "effective_rank": cache.effective_rank,
        "entropy_rank": cache.entropy_rank,
        "top_share": cache.top_share,
        "residual_cvar": cache.residual_cvar,
        "prep_seconds": cache.prep_seconds,
        "calls": ref["calls"], "coverage": ref["coverage"],
        "certified_feasible": ref["certified_feasible"],
        "certified_infeasible": ref["certified_infeasible"],
        "fallbacks": ref["fallbacks"], "mean_pieces": ref["mean_pieces"],
        "mean_route_length_checked": ref["mean_route_length_checked"],
        "K": ref["K"], "distance": ref["distance_raw"] / instance.scale,
        "objective_raw": ref["objective_raw"],
        "full_gate_seconds": full_gate, "full_gate_iqr": full_gate_iqr,
        "cert_gate_seconds": cert_gate, "cert_gate_iqr": cert_gate_iqr,
        "gate_speedup": gate_speed, "gate_speedup_iqr": gate_speed_iqr,
        "full_solver_seconds": full_solver, "full_solver_iqr": full_solver_iqr,
        "cert_solver_seconds": cert_solver, "cert_solver_iqr": cert_solver_iqr,
        "solver_speedup": solver_speed, "solver_speedup_iqr": solver_speed_iqr,
        "solver_speedup_with_prep": full_solver / (cert_solver + cache.prep_seconds),
        "audit": audit,
    }


BENCHMARK_CSV_FIELDS = [
    "instance", "profile", "policy", "N", "dimension", "alpha", "capacity",
    "iterations", "repetitions", "direction", "explained_variance",
    "effective_rank", "entropy_rank", "top_share", "residual_cvar",
    "prep_seconds", "calls", "coverage", "certified_feasible",
    "certified_infeasible", "fallbacks", "mean_pieces",
    "mean_route_length_checked", "K", "distance", "objective_raw",
    "full_gate_seconds", "full_gate_iqr", "cert_gate_seconds", "cert_gate_iqr",
    "gate_speedup", "gate_speedup_iqr", "full_solver_seconds",
    "full_solver_iqr", "cert_solver_seconds", "cert_solver_iqr",
    "solver_speedup", "solver_speedup_iqr", "solver_speedup_with_prep", "audit",
]

# ==== BENCHMARK (diagnostic-aware) + CLI + REPORTING ====

def canonical_plan(plan):
    return tuple(sorted(tuple(route) for route in plan))



def run_solver_once(
    instance,
    capacity,
    alpha,
    cache,
    iterations,
    seed,
    method,
    audit,
    diag=None,
):
    if method == "FULL":
        gate = FullGate(instance, capacity, alpha)
    elif method == "CERT":
        gate = CertifiedGate(instance, capacity, alpha, cache, audit=audit, diag=diag)
    else:
        raise ValueError(method)

    omega_vehicle = float(np.mean(instance.distance[instance.distance > 0]))
    t0 = time.perf_counter()
    plan = solve_fixed_iterations(
        instance.distance, gate, instance.n, omega_vehicle, iterations, seed
    )
    total_seconds = time.perf_counter() - t0
    return {
        "method": method,
        "plan": plan,
        "plan_key": canonical_plan(plan),
        "K": len(plan),
        "distance_raw": sum(route_cost(route, instance.distance) for route in plan),
        "objective_raw": econ_cost(plan, instance.distance, omega_vehicle),
        "calls": gate.calls,
        "gate_seconds": gate.gate_seconds,
        "total_seconds": total_seconds,
        "mean_route_length_checked": gate.route_length_sum / gate.calls if gate.calls else 0.0,
        "coverage": gate.coverage if method == "CERT" else 0.0,
        "certified_feasible": gate.certified_feasible if method == "CERT" else 0,
        "certified_infeasible": gate.certified_infeasible if method == "CERT" else 0,
        "fallbacks": gate.fallbacks if method == "CERT" else gate.calls,
        "certificate_seconds": gate.certificate_seconds if method == "CERT" else 0.0,
        "fallback_seconds": gate.fallback_seconds if method == "CERT" else gate.gate_seconds,
        "audit_seconds": gate.audit_seconds if method == "CERT" else 0.0,
        "mean_pieces": gate.pieces_sum / gate.calls if method == "CERT" and gate.calls else 0.0,
    }


def write_csv_rows(path, rows, fields):
    if not rows:
        return
    p = Path(path)
    exists = p.exists()
    with p.open("a", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


ROUTE_CSV_FIELDS = [
    "instance", "profile", "policy", "stream", "K", "decision", "margin_sign",
    "gamma_slope", "gamma_delivery_route", "gamma_best_found", "gamma_lower_bound_l2",
    "gamma_slope_over_delivery", "gamma_slope_over_best", "gamma_best_over_lower_bound",
    "residual_l2_sq", "residual_rms", "residual_coherence_mu",
    "residual_coherence_sqrt",
    "delta", "R", "chi", "kappa", "R_over_delta", "zeta",
    "B_align", "B_kappa", "nu", "decision_slack_norm", "D_align",
    "best_search_starts", "best_search_spread",
]

SPECTRAL_CSV_FIELDS = [
    "instance", "profile", "policy", "label", "population", "n_routes",
    "A_l1", "A_l2", "A_l2_over_l1", "A_l1_over_tr", "Delta_A",
    "E_spec_norm", "eta", "angle_vM_vA_deg", "angle_uslope_vM_deg",
    "angle_uslope_E1_deg", "angle_uslope_vA_deg",
    "M_l1", "M_l2", "M_ld", "M_l2_over_l1", "M_gap12", "M_gap12_rel",
    "M_leading_multiplicity", "M_trace",
    "M_residual_opt", "M_residual_uslope", "M_transfer_ratio",
    "M_transfer_bound_ratio", "M_transfer_excess",
    "M_transfer_bound_excess", "M_q_rayleigh", "M_beta_q",
    "M_transfer_identity_excess", "M_bound_slack_over_actual_excess",
    "M_bound_slack_fraction", "M_gap12_degenerate", "eigengap_degenerate",
    "mu_n", "mu_mean", "mu_p50", "mu_p90", "mu_p95", "mu_max",
    "sqrt_mu_p50", "sqrt_mu_p90", "sqrt_mu_p95", "sqrt_mu_max",
    "theorem_factor_exact_observed", "theorem_factor_spectral_bound",
    "Ce_u", "Ce_best_found_min", "Ce_best_found_max",
    "Ce_best_found_spread", "Ce_rho_best_found",
    "Ce_global_lower_bound", "Ce_rho_certified_upper",
    "Ce_random_p05", "Ce_random_p50", "Ce_random_p95",
    "Ce_pca_top", "Ce_pca_bottom",
    "Ce_cov_l1", "Ce_cov_l2", "Ce_cov_ld", "Ce_cov_l2_over_l1",
    "Ce_search_seconds",
    "certificate_factor_best_found",
    "certificate_factor_spectral_best_found",
    "certificate_factor_certified_upper",
]

TIER2_CSV_FIELDS = [
    "instance", "profile", "policy", "stream", "route_len", "rho_cand",
    "cand_count", "N", "candidate_cvar_matches_full", "tier2_decision",
    "tier2_incremental_certified", "full_feasible", "t_tier2_total",
    "t_tier2_exact_partial", "t_full", "tier2_over_full", "cvar_L", "cvar_U",
    "cvar_candidate", "cvar_full",
]



def route_diag_to_row(rd, instance_name, profile, policy):
    def ratio(a, b):
        if b is None or not math.isfinite(b) or b <= 1e-300:
            return float("nan")
        if a is None or not math.isfinite(a):
            return float("nan")
        return a / b

    return {
        "instance": instance_name,
        "profile": profile,
        "policy": policy,
        "stream": "search_sampled",
        "K": rd.K,
        "decision": rd.decision,
        "margin_sign": rd.margin_sign,
        "gamma_slope": rd.gamma_slope,
        "gamma_delivery_route": rd.gamma_delivery_route,
        "gamma_best_found": rd.gamma_best_found,
        "gamma_lower_bound_l2": rd.gamma_lower_bound_l2,
        "gamma_slope_over_delivery": ratio(rd.gamma_slope, rd.gamma_delivery_route),
        "gamma_slope_over_best": ratio(rd.gamma_slope, rd.gamma_best_found),
        "gamma_best_over_lower_bound": ratio(rd.gamma_best_found, rd.gamma_lower_bound_l2),
        "residual_l2_sq": rd.residual_l2_sq,
        "residual_rms": rd.residual_rms,
        "residual_coherence_mu": rd.residual_coherence_mu,
        "residual_coherence_sqrt": rd.residual_coherence_sqrt,
        "delta": rd.delta,
        "R": rd.R,
        "chi": rd.chi,
        "kappa": rd.kappa,
        "R_over_delta": rd.R_over_delta,
        "zeta": rd.zeta,
        "B_align": rd.B_align,
        "B_kappa": rd.B_kappa,
        "nu": rd.nu,
        "decision_slack_norm": rd.decision_slack_norm,
        "D_align": rd.D_align,
        "best_search_starts": rd.best_search_starts,
        "best_search_spread": rd.best_search_spread,
    }


def print_percentiles(rows, name, key_getter):
    vals = [key_getter(r) for r in rows]
    ps = percentile_summary(vals, name)
    if ps.get("n", 0) == 0:
        print(f"    {name:24s}: (no finite values)")
        return
    order = ["p10", "p25", "p50", "p75", "p90", "p95"]
    body = " ".join(f"{k}={ps[k]:.4g}" for k in order if k in ps)
    print(f"    {name:24s}: n={ps['n']:5d}  {body}  [min={ps['min']:.4g} max={ps['max']:.4g}]")



def run_diagnostic_cell(
    instance,
    profile,
    policy,
    capacity,
    alpha,
    cache,
    iterations,
    seed,
    cfg,
    out_paths,
    ce_metrics=None,
):
    diag = DiagContext(
        instance=instance,
        capacity=capacity,
        alpha=alpha,
        cache=cache,
        sample_rate=cfg["diag_sample_rate"],
        best_sample=cfg["diag_best_sample"],
        best_iters=cfg["diag_best_iters"],
        tier2=bool(cfg["diag_tier2"]),
        seed=seed + 4242,
        stream_label="search",
    )
    cert = run_solver_once(
        instance, capacity, alpha, cache, iterations, seed,
        "CERT", audit=cfg["audit"], diag=diag,
    )
    full = run_solver_once(
        instance, capacity, alpha, cache, iterations, seed,
        "FULL", audit=False, diag=None,
    )
    if cert["plan_key"] != full["plan_key"]:
        raise AssertionError("diagnostic CERT/FULL plans differ")
    if cert["calls"] != full["calls"]:
        raise AssertionError(f"diagnostic CERT/FULL calls differ ({cert['calls']} vs {full['calls']})")
    if abs(cert["objective_raw"] - full["objective_raw"]) > 1e-8:
        raise AssertionError("diagnostic CERT/FULL objectives differ")

    route_rows = [route_diag_to_row(rd, instance.name, profile, policy) for rd in diag.route_rows]
    write_csv_rows(out_paths["route"], route_rows, ROUTE_CSV_FIELDS)

    specs = diag.spectral_summaries()
    spec_rows = []
    for spec in specs:
        _attach_certificate_width_metrics(spec, ce_metrics)
        row = {"instance": instance.name, "profile": profile, "policy": policy}
        row.update(spec)
        spec_rows.append(row)
    write_csv_rows(out_paths["spectral"], spec_rows, SPECTRAL_CSV_FIELDS)

    if diag.tier2_rows:
        for row in diag.tier2_rows:
            row["instance"] = instance.name
            row["profile"] = profile
            row["policy"] = policy
        write_csv_rows(out_paths["tier2"], diag.tier2_rows, TIER2_CSV_FIELDS)

    return diag, cert, full, specs



def build_train_spectral(instance, cache, cfg, ce_metrics=None):
    routes = training_route_slopes(instance, cfg["train"], cfg["seed"] + 1000)
    acc = SpectralAccumulator(d=instance.X.shape[1])
    mu_values = []
    seen = set()
    for route in routes:
        key = tuple(route)
        if key in seen:
            continue
        seen.add(key)
        _, slopes = route_affine_lines(instance, route)
        acc.update(slopes)
        _, _, mu, _ = _residual_coherence(slopes, cache.u)
        mu_values.append(mu)
    out = acc.summary(cache.u, "train_random_unique")
    out.update(_coherence_summary(mu_values))
    _attach_transfer_theorem_metrics(out)
    _attach_certificate_width_metrics(out, ce_metrics)
    return out


def parse_list(text):
    return [v.strip() for v in text.split(",") if v.strip()]



def parse_args(argv):
    config = {
        "dir": DEFAULT_DATA_DIR, "regex": "", "max": None,
        "profiles": ["concentrated", "moderate", "diffuse"],
        "N": N_DATA, "d": FACTOR_DIMENSION, "iters": DEFAULT_ITERS,
        "reps": DEFAULT_REPS, "policy": DEFAULT_POLICY,
        "direction": DEFAULT_DIRECTION, "train": DEFAULT_TRAIN_ROUTES,
        "seed": SEED, "cv": CV, "alpha": ALPHA, "eps": EPS_FRAC,
        "support": SUPPORT, "audit": False, "out": DEFAULT_OUTPUT,
        "diagnostic": 0, "diag_sample_rate": 0.25, "diag_tier2": 0,
        "diag_best_sample": 200, "diag_best_iters": 80,
        "diag_ce": 1, "diag_ce_random": 2048, "diag_ce_batch": 64,
        "diag_ce_local_starts": 6, "diag_ce_local_steps": 7,
        "diag_ce_local_trials": 24,
        "diag_out_prefix": "diagnostics",
    }
    aliases = {
        "diag_oracle_sample": "diag_best_sample",
        "diag_oracle_iters": "diag_best_iters",
    }
    for argument in argv:
        if argument == "audit":
            config["audit"] = True
            continue
        if "=" not in argument:
            raise ValueError(f"Unknown argument: {argument}")
        key, value = argument.split("=", 1)
        key = aliases.get(key, key)
        if key in {"dir", "regex", "policy", "direction", "out", "diag_out_prefix"}:
            config[key] = value
        elif key == "profiles":
            config[key] = parse_list(value)
        elif key in {
            "max", "N", "d", "iters", "reps", "train", "seed", "diagnostic",
            "diag_tier2", "diag_best_sample", "diag_best_iters",
            "diag_ce", "diag_ce_random", "diag_ce_batch",
            "diag_ce_local_starts", "diag_ce_local_steps",
            "diag_ce_local_trials",
        }:
            config[key] = int(value)
        elif key in {"cv", "alpha", "eps", "support", "diag_sample_rate"}:
            config[key] = float(value)
        else:
            raise ValueError(f"Unknown argument: {argument}")

    config["policy"] = config["policy"].upper()
    config["direction"] = config["direction"].upper()
    config["profiles"] = [p.lower() for p in config["profiles"]]
    if config["policy"] not in {"SAA", "WDRO", "BOTH"}:
        raise ValueError("policy must be SAA, WDRO, or both")
    if config["direction"] not in {"SLOPE", "PCA", "DELIVERY"}:
        raise ValueError("direction must be SLOPE, PCA, or DELIVERY")
    for profile in config["profiles"]:
        if profile not in PROFILES:
            raise ValueError(f"Unknown profile: {profile}")
    if config["N"] <= 0 or config["d"] < 2 or config["iters"] < 0 or config["reps"] <= 0:
        raise ValueError("invalid N/d/iters/reps")
    if not (0.0 < config["alpha"] < 1.0):
        raise ValueError("alpha must lie in (0,1)")
    if not (0.0 <= config["eps"] < 1.0):
        raise ValueError("eps must lie in [0,1)")
    if not (0.0 <= config["diag_sample_rate"] <= 1.0):
        raise ValueError("diag_sample_rate must lie in [0,1]")
    if config["diag_best_sample"] < 0 or config["diag_best_iters"] < 0:
        raise ValueError("diagnostic best-search settings must be nonnegative")
    if config["diag_ce"] not in {0, 1}:
        raise ValueError("diag_ce must be 0 or 1")
    if min(config["diag_ce_random"], config["diag_ce_local_starts"],
           config["diag_ce_local_steps"], config["diag_ce_local_trials"]) < 0:
        raise ValueError("C_e diagnostic counts must be nonnegative")
    if config["diag_ce_batch"] <= 0:
        raise ValueError("diag_ce_batch must be positive")
    return config



def main():
    try:
        config = parse_args(sys.argv[1:])
    except Exception as error:
        print("ARGUMENT ERROR:", error)
        raise SystemExit(2)

    files = sorted(glob.glob(str(Path(config["dir"]) / "*.vrpspd")))
    if config["regex"]:
        pattern = re.compile(config["regex"], re.IGNORECASE)
        files = [f for f in files if pattern.search(Path(f).stem)]
    if config["max"]:
        files = files[:config["max"]]
    if not files:
        print(f"ERROR: no matching .vrpspd files in '{config['dir']}'.")
        return

    policies = ["SAA", "WDRO"] if config["policy"] == "BOTH" else [config["policy"]]
    diagnostic = bool(config["diagnostic"])
    out_paths = {
        "route": f"{config['diag_out_prefix']}_route_checks.csv",
        "spectral": f"{config['diag_out_prefix']}_spectral_summary.csv",
        "tier2": f"{config['diag_out_prefix']}_tier2.csv",
    }

    print("=" * 104)
    print(f" EXACT MULTIVARIATE CVaR CERTIFICATE — {'DIAGNOSTIC' if diagnostic else 'BENCHMARK'} MODE")
    print("=" * 104)
    print(
        f"instances={len(files)} profiles={','.join(config['profiles'])} N={config['N']:,} "
        f"d={config['d']} iters={config['iters']} reps={config['reps']} "
        f"policies={','.join(policies)} direction={config['direction']}"
    )
    print(
        f"alpha={config['alpha']} cv={config['cv']} eps={config['eps']} "
        f"train={config['train']} audit={config['audit']}"
    )
    if diagnostic:
        print(
            f"diag: sample_rate={config['diag_sample_rate']} tier2={config['diag_tier2']} "
            f"best_sample={config['diag_best_sample']} best_iters={config['diag_best_iters']} "
            f"Ce={config['diag_ce']} Ce_random={config['diag_ce_random']}"
        )
        print("NOTE: diagnostic mode is not a timing benchmark.")
        print("Spectra: call-weighted + unique ordered routes + unique customer sets (A only).")
    else:
        print("Protocol: fixed iterations, randomized FULL/CERT order, paired assertions.")
    print("-" * 104)

    benchmark_rows = []
    sampled_rows = []
    tier2_rows_all = []
    failures = 0
    ce_diag_shared = None

    for file_index, file in enumerate(files, 1):
        name = Path(file).stem
        print(f"\n[{file_index}/{len(files)}] {name}")
        for profile_name in config["profiles"]:
            print(f"   profile={profile_name}")
            try:
                t0 = time.perf_counter()
                instance = make_affine_instance(
                    file, PROFILES[profile_name], config["N"], config["d"],
                    config["cv"], config["support"], config["seed"],
                )
                scenario_seconds = time.perf_counter() - t0
                cache = build_certificate_cache(
                    instance, config["alpha"], config["direction"],
                    config["train"], config["seed"] + 1000,
                )
                ce_metrics = None
                if diagnostic and config["diag_ce"]:
                    if ce_diag_shared is None:
                        ce_diag_shared = ResidualDirectionDiagnostic(
                            centered=np.asarray(instance.X, dtype=float),
                            alpha=config["alpha"],
                            random_count=config["diag_ce_random"],
                            batch_size=config["diag_ce_batch"],
                            local_starts=config["diag_ce_local_starts"],
                            local_steps=config["diag_ce_local_steps"],
                            local_trials=config["diag_ce_local_trials"],
                            seed=config["seed"] + 8080,
                        )
                        print(
                            f"      C_e direction search: {ce_diag_shared.search_seconds:.2f}s "
                            f"best_min={ce_diag_shared.best_found_min:.6g} "
                            f"best_max={ce_diag_shared.best_found_max:.6g} "
                            f"spread={ce_diag_shared.best_found_max / max(ce_diag_shared.best_found_min, 1e-300):.6g} "
                            f"rigorous_LB={ce_diag_shared.global_lower_bound:.6g}"
                        )
                    ce_metrics = ce_diag_shared.summary_for(cache.u)
                print(
                    f"      scenario={scenario_seconds:.2f}s prep={cache.prep_seconds:.2f}s "
                    f"participation_rank={cache.effective_rank:.3f} "
                    f"entropy_rank={cache.entropy_rank:.3f} top={cache.top_share:.3f} "
                    f"EV1={cache.explained_variance:.3f}"
                )

                if diagnostic:
                    train_spec = build_train_spectral(instance, cache, config, ce_metrics=ce_metrics)
                    train_row = {"instance": name, "profile": profile_name, "policy": "-", **train_spec}
                    write_csv_rows(out_paths["spectral"], [train_row], SPECTRAL_CSV_FIELDS)
                    print(
                        f"      [train_random_unique] l2/l1={_fmt(train_spec['A_l2_over_l1'])} "
                        f"eta={_fmt(train_spec['eta'])} "
                        f"angle(vM,vA)={_fmt(train_spec['angle_vM_vA_deg'])}deg "
                        f"angle(u,E1_M)={_fmt(train_spec['angle_uslope_E1_deg'])}deg "
                        f"M_l2/l1={_fmt(train_spec['M_l2_over_l1'])} "
                        f"mu50={_fmt(train_spec['mu_p50'])} mu95={_fmt(train_spec['mu_p95'])} "
                        f"transfer={_fmt(train_spec['M_transfer_ratio'])} "
                        f"beta_q={_fmt(train_spec['M_beta_q'])} "
                        f"factor={_fmt(train_spec['theorem_factor_exact_observed'])} "
                        f"Ce={_fmt(train_spec['Ce_u'])} rho_e(best)={_fmt(train_spec['Ce_rho_best_found'])}"
                    )

                for policy in policies:
                    capacity = instance.capacity if policy == "SAA" else instance.capacity * (1.0 - config["eps"])
                    print(f"      {policy}: cap={capacity:.2f}")
                    if diagnostic:
                        diag, cert, full, specs = run_diagnostic_cell(
                            instance, profile_name, policy, capacity, config["alpha"], cache,
                            config["iters"], config["seed"], config, out_paths,
                            ce_metrics=ce_metrics,
                        )
                        rows = [route_diag_to_row(rd, name, profile_name, policy) for rd in diag.route_rows]
                        sampled_rows.extend(rows)
                        tier2_rows_all.extend(diag.tier2_rows)
                        print(
                            f"        calls={cert['calls']} coverage={cert['coverage']:.3f} "
                            f"sampled={diag.n_sampled}/{diag.n_observed} "
                            f"best_search={diag.best_done} tier2_rows={len(diag.tier2_rows)} assert=PASS"
                        )
                        for spec in specs:
                            print(
                                f"        [{spec['label']}] n={spec['n_routes']} "
                                f"l2/l1={_fmt(spec['A_l2_over_l1'])} eta={_fmt(spec['eta'])} "
                                f"a(vM,vA)={_fmt(spec['angle_vM_vA_deg'])} "
                                f"a(u,E1_M)={_fmt(spec['angle_uslope_E1_deg'])} "
                                f"M_l2/l1={_fmt(spec['M_l2_over_l1'])} "
                                f"mult={spec['M_leading_multiplicity']} "
                                f"mu50={_fmt(spec['mu_p50'])} mu95={_fmt(spec['mu_p95'])} "
                                f"transfer={_fmt(spec['M_transfer_ratio'])} "
                                f"beta_q={_fmt(spec['M_beta_q'])} "
                                f"factor={_fmt(spec['theorem_factor_exact_observed'])} "
                                f"Ce={_fmt(spec['Ce_u'])} rho_e(best)={_fmt(spec['Ce_rho_best_found'])} "
                                f"full_factor(best)={_fmt(spec['certificate_factor_best_found'])}"
                            )
                    else:
                        row = benchmark_cell(
                            instance, profile_name, policy, capacity, config["alpha"], cache,
                            config["iters"], config["reps"], config["seed"], config["audit"],
                        )
                        write_csv_rows(config["out"], [row], BENCHMARK_CSV_FIELDS)
                        benchmark_rows.append(row)
                        print(
                            f"      => calls={row['calls']} cover={row['coverage']:.3f} "
                            f"pieces={row['mean_pieces']:.2f} gate={row['gate_speedup']:.2f}x "
                            f"solver={row['solver_speedup']:.2f}x prep={row['solver_speedup_with_prep']:.2f}x "
                            f"K={row['K']} dist={row['distance']:.2f} assert=PASS"
                        )
            except Exception as error:
                failures += 1
                import traceback
                print(f"      ERROR: {type(error).__name__}: {error}")
                traceback.print_exc()

    if diagnostic and sampled_rows:
        print("\n" + "=" * 104)
        print("DIAGNOSTIC PERCENTILES — SAMPLED SEARCH CHECKS")
        print("=" * 104)
        for label, key in [
            ("chi", "chi"), ("kappa", "kappa"), ("zeta", "zeta"),
            ("R_over_delta", "R_over_delta"), ("nu (identity metric)", "nu"),
            ("decision_slack_norm", "decision_slack_norm"), ("D_align", "D_align"),
            ("coherence mu", "residual_coherence_mu"),
            ("sqrt(coherence mu)", "residual_coherence_sqrt"),
            ("gamma_slope/best", "gamma_slope_over_best"),
            ("best/lower_bound", "gamma_best_over_lower_bound"),
            ("gamma_slope/delivery", "gamma_slope_over_delivery"), ("K", "K"),
        ]:
            print_percentiles(sampled_rows, label, lambda r, k=key: r[k])

        if tier2_rows_all:
            print("\n  Tier-2 on sampled tier-1 ambiguous routes:")
            print_percentiles(tier2_rows_all, "rho_cand", lambda r: r["rho_cand"])
            print_percentiles(tier2_rows_all, "tier2/full", lambda r: r["tier2_over_full"])
            inc = np.mean([r["tier2_incremental_certified"] for r in tier2_rows_all])
            ok = all(r["candidate_cvar_matches_full"] for r in tier2_rows_all)
            print(f"    incremental certification={inc:.3f} candidate_CVaR_exact={ok}")

        print(f"\n  wrote {out_paths['route']}")
        print(f"  wrote {out_paths['spectral']}")
        if tier2_rows_all:
            print(f"  wrote {out_paths['tier2']}")

    if benchmark_rows:
        gate = np.asarray([r["gate_speedup"] for r in benchmark_rows])
        solver = np.asarray([r["solver_speedup"] for r in benchmark_rows])
        cov = np.asarray([r["coverage"] for r in benchmark_rows])
        print("\n" + "=" * 104)
        print("SUMMARY")
        print(
            f"cells={len(benchmark_rows)} median gate={np.median(gate):.2f}x min={gate.min():.2f}x "
            f"median solver={np.median(solver):.2f}x min={solver.min():.2f}x "
            f"median coverage={np.median(cov):.3f}"
        )
        if np.std(cov) > 1e-12:
            print(f"corr(coverage,solver-speed)={np.corrcoef(cov, solver)[0,1]:.3f}")
        print(f"wrote {config['out']}")

    if failures:
        raise SystemExit(f"Completed with {failures} failed cell(s).")
    print("\nDONE.")


def _fmt(x):
    if x is None:
        return "nan"
    try:
        if math.isnan(x):
            return "nan"
    except (TypeError, ValueError):
        return str(x)
    return f"{x:.4g}"


if __name__ == "__main__":
    main()