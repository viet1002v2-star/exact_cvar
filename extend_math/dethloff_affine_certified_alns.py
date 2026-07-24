#!/usr/bin/env python3
"""
dethloff_affine_certified_alns.py

Self-contained solver-level benchmark for the exact multivariate projection
certificate inside the user's existing Dethloff ALNS/ILS.

The parser, route kernels, and ALNS operators are copied from
`dethloff_runner.py`; the original file is neither imported nor modified.

Why the uncertainty model is affine
-----------------------------------
The certificate is exact for max-affine route load under a multivariate affine
factor model.  Therefore this script uses the same bounded spatial affine
overlay as the preceding full-instance Dethloff experiments, rather than the
runner's nonlinear gamma-copula marginal transform.  The deterministic
Dethloff matrix, customer means, and vehicle capacity remain unchanged.

Compared gates
--------------
FULL
    Exact empirical CVaR from all N affine scenarios.

CERT
    Exact one-dimensional projected CVaR
        -> safe route-specific lower/upper certificate
        -> exact FULL fallback only when ambiguous.

Fair solver protocol
--------------------
The original ALNS stops by wall-clock time.  A faster evaluator would then get
more iterations, so the two methods would not follow the same route stream.
Here the copied ALNS uses a fixed outer-iteration budget.  FULL and CERT receive
the same instance, scenarios, seed, and number of iterations.  The script
asserts equal capacity-check counts, final plan, distance, and objective.

Examples (Windows CMD)
----------------------
Audit one instance:
python dethloff_affine_certified_alns.py ^
  dir=Dethloff max=1 profiles=moderate ^
  N=5000 iters=5 reps=1 audit

Four-instance gate:
python dethloff_affine_certified_alns.py ^
  dir=Dethloff ^
  regex="(CON3-0|CON8-0|SCA3-0|SCA8-0)$" ^
  profiles=concentrated,moderate,diffuse ^
  N=50000 iters=50 reps=3 policy=both

Full 40-instance run:
python dethloff_affine_certified_alns.py ^
  dir=Dethloff ^
  profiles=concentrated,moderate,diffuse ^
  N=50000 iters=50 reps=5 policy=both

Arguments
---------
dir=<folder>
regex=<regular expression>
max=<integer>
profiles=concentrated,moderate,diffuse
N=<scenario count>
d=<factor dimension>
iters=<fixed outer iterations>
reps=<paired timing repetitions>
policy=SAA|WDRO|both
direction=SLOPE|PCA|DELIVERY
train=<number of direction-training routes>
seed=<integer>
cv=<coefficient of variation>
alpha=<CVaR level>
eps=<WDRO capacity tightening fraction>
support=<bounded factor support>
audit
out=<CSV path>
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
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np


# =============================================================================
# Defaults
# =============================================================================

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
    "concentrated": OverlayProfile(
        "concentrated", 2.0, 1.50
    ),
    "moderate": OverlayProfile(
        "moderate", 1.0, 1.00
    ),
    "diffuse": OverlayProfile(
        "diffuse", 0.50, 0.65
    ),
}


# =============================================================================
# Dethloff parser -- copied from dethloff_runner.py
# =============================================================================

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
    for row in _lines_after(
        txt,
        "PICKUP_AND_DELIVERY_SECTION",
    ):
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
    for row in _lines_after(
        txt,
        "EDGE_WEIGHT_SECTION",
    ):
        tokens.extend(row.split())

    values = [int(float(token)) for token in tokens]
    if n is None or len(values) != n * n:
        raise ValueError(
            "EDGE_WEIGHT_SECTION: got %d tokens, expected n*n=%s "
            "(need the FULL matrix)"
            % (
                len(values),
                None if n is None else n * n,
            )
        )

    distance = np.asarray(
        values,
        dtype=np.int64,
    ).reshape(n, n)
    demands = _parse_pd(txt, n)
    return distance, demands, capacity, n, 10000


# =============================================================================
# Bounded spatial affine uncertainty overlay
# =============================================================================

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


def bounded_factor_samples(
    N: int,
    dimension: int,
    support: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = np.clip(
        rng.normal(size=(N, dimension)),
        -support,
        support,
    )
    X -= X.mean(axis=0, keepdims=True)

    maximum = np.max(np.abs(X), axis=0)
    correction = np.maximum(
        1.0,
        maximum / support,
    )
    X /= correction[None, :]
    return X


def spatial_basis_from_distance(
    customer_distance: np.ndarray,
    components: int,
    decay: float,
) -> np.ndarray:
    positive = customer_distance[
        customer_distance > 1e-12
    ]
    length = (
        float(np.median(positive))
        if positive.size
        else 1.0
    )
    length = max(length, 1e-12)

    kernel = np.exp(
        -customer_distance / length
    )
    kernel = 0.5 * (kernel + kernel.T)

    values, vectors = np.linalg.eigh(kernel)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    vectors = vectors[:, order]

    take = min(components, vectors.shape[1])
    weights = np.power(
        values[:take] + 1e-12,
        decay / 2.0,
    )
    basis = (
        vectors[:, :take]
        * weights[None, :]
    )

    if take < components:
        basis = np.pad(
            basis,
            (
                (0, 0),
                (0, components - take),
            ),
        )
    return basis


def row_l1_normalize(matrix: np.ndarray):
    norm = np.sum(
        np.abs(matrix),
        axis=1,
        keepdims=True,
    )
    norm[norm < 1e-12] = 1.0
    return matrix / norm


def make_affine_instance(
    path,
    profile: OverlayProfile,
    N: int,
    dimension: int,
    cv: float,
    support: float,
    seed: int,
) -> AffineInstance:
    distance, demands, capacity, n, scale = (
        parse_dethloff(path)
    )
    if dimension < 2:
        raise ValueError("dimension must be at least 2")
    if not (0.0 < cv < 0.95):
        raise ValueError("cv must lie in (0,0.95)")

    customer_count = n - 1
    X = bounded_factor_samples(
        N,
        dimension,
        support,
        seed,
    )

    customer_distance = distance[1:, 1:].astype(
        float
    )
    basis = spatial_basis_from_distance(
        customer_distance,
        components=dimension - 1,
        decay=profile.spatial_decay,
    )

    delivery_raw = np.column_stack(
        (
            np.full(
                customer_count,
                profile.common_weight,
            ),
            basis,
        )
    )

    rng = np.random.default_rng(seed + 1)
    rotation, _ = np.linalg.qr(
        rng.normal(
            size=(
                dimension - 1,
                dimension - 1,
            )
        )
    )
    pickup_spatial = basis @ rotation
    pickup_raw = np.column_stack(
        (
            np.full(
                customer_count,
                0.85 * profile.common_weight,
            ),
            pickup_spatial,
        )
    )

    delivery_loading = row_l1_normalize(
        delivery_raw
    )
    pickup_loading = row_l1_normalize(
        pickup_raw
    )

    delivery_cv = np.minimum(
        cv
        * rng.uniform(
            0.80,
            1.20,
            size=customer_count,
        ),
        0.90,
    )
    pickup_cv = np.minimum(
        cv
        * rng.uniform(
            0.80,
            1.20,
            size=customer_count,
        ),
        0.90,
    )

    d0 = demands[:, 0].astype(float).copy()
    p0 = demands[:, 1].astype(float).copy()
    dvec = np.zeros((n, dimension))
    pvec = np.zeros((n, dimension))

    dvec[1:] = (
        d0[1:, None]
        * delivery_cv[:, None]
        * delivery_loading
        / support
    )
    pvec[1:] = (
        p0[1:, None]
        * pickup_cv[:, None]
        * pickup_loading
        / support
    )

    delivery_values = (
        d0[:, None] + dvec @ X.T
    )
    pickup_values = (
        p0[:, None] + pvec @ X.T
    )

    if (
        delivery_values.min() < -1e-9
        or pickup_values.min() < -1e-9
    ):
        raise AssertionError(
            "Bounded affine positivity construction failed."
        )

    return AffineInstance(
        name=Path(path).stem,
        distance=distance,
        capacity=float(capacity),
        n=n,
        scale=float(scale),
        X=X,
        d0=d0,
        p0=p0,
        dvec=dvec,
        pvec=pvec,
        delivery_values=delivery_values,
        pickup_values=pickup_values,
    )


# =============================================================================
# Route cost, exact scenario peak, and empirical CVaR
# =============================================================================

def route_cost(route, distance):
    if not route:
        return 0.0

    cost = (
        distance[0, route[0]]
        + distance[route[-1], 0]
    )
    for index in range(len(route) - 1):
        cost += distance[
            route[index],
            route[index + 1],
        ]
    return float(cost)


def route_peaks(
    route,
    delivery_values,
    pickup_values,
):
    if not route:
        return np.zeros(
            delivery_values.shape[1]
        )

    delivery = delivery_values[route].T
    pickup = pickup_values[route].T
    total_delivery = delivery.sum(axis=1)
    middle = (
        total_delivery[:, None]
        - np.cumsum(delivery, axis=1)
        + np.cumsum(pickup, axis=1)
    )
    return np.maximum(
        total_delivery,
        middle.max(axis=1),
    )


def tail_parameters(alpha: float, n: int):
    if not (0.0 <= alpha < 1.0):
        raise ValueError(
            "alpha must lie in [0,1)."
        )
    if n <= 0:
        raise ValueError(
            "sample must be nonempty"
        )

    tail_mass = (1.0 - alpha) * n
    nearest = float(round(tail_mass))
    tolerance = 16.0 * max(
        math.ulp(tail_mass),
        math.ulp(nearest),
    )
    if abs(tail_mass - nearest) <= tolerance:
        tail_mass = nearest

    count = min(
        n,
        max(1, int(math.ceil(tail_mass))),
    )
    boundary_weight = min(
        1.0,
        max(
            0.0,
            tail_mass - (count - 1),
        ),
    )
    return tail_mass, count, boundary_weight


def empirical_cvar(values, alpha):
    array = np.asarray(values, dtype=float)
    n = array.size
    tail_mass, count, boundary_weight = (
        tail_parameters(alpha, n)
    )
    tail = np.partition(
        array,
        n - count,
    )[n - count:]
    boundary = float(tail.min())
    return (
        float(tail.sum())
        - (1.0 - boundary_weight) * boundary
    ) / tail_mass


# =============================================================================
# Exact O(p log N) projected kernel
# =============================================================================

def upper_envelope(
    lines: Iterable[tuple[float, float]],
):
    records = sorted(
        lines,
        key=lambda item: (item[1], item[0]),
    )
    unique = []

    for intercept, slope in records:
        intercept = float(intercept)
        slope = float(slope)
        if unique and slope == unique[-1][1]:
            if intercept > unique[-1][0]:
                unique[-1] = (
                    intercept,
                    slope,
                )
            continue
        unique.append((intercept, slope))

    if not unique:
        raise ValueError(
            "At least one line is required."
        )

    def crossing(left, right):
        return (
            (left[0] - right[0])
            / (right[1] - left[1])
        )

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
            starts.append(
                crossing(stack[-1], line)
            )
            stack.append(line)

    A = np.asarray(
        [line[0] for line in stack],
        dtype=float,
    )
    B = np.asarray(
        [line[1] for line in stack],
        dtype=float,
    )
    xbr = np.asarray(
        starts + [math.inf],
        dtype=float,
    )
    return A, B, xbr


def cvar_os(
    envelope,
    z_sorted,
    prefix,
    alpha,
):
    A, B, xbr = envelope
    n = len(z_sorted)
    pieces = len(A)
    tail_mass, count, boundary_weight = (
        tail_parameters(alpha, n)
    )

    def value(index):
        z = z_sorted[index]
        piece = (
            bisect.bisect_right(xbr, z) - 1
        )
        piece = min(
            max(piece, 0),
            pieces - 1,
        )
        return float(
            A[piece] + B[piece] * z
        )

    sign_change = bisect.bisect_left(B, 0.0)
    if sign_change == 0:
        valley = 0
    elif sign_change >= pieces:
        valley = n - 1
    else:
        position = bisect.bisect_left(
            z_sorted,
            xbr[sign_change],
        )
        candidates = [
            index
            for index in (
                position - 1,
                position,
                position + 1,
            )
            if 0 <= index < n
        ]
        valley = min(
            candidates,
            key=value,
        )

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
        if (
            left_arm(middle)
            >= right_arm(count - middle + 1)
        ):
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
            piece = (
                bisect.bisect_right(xbr, z)
                - 1
            )
            piece = min(
                max(piece, 0),
                pieces - 1,
            )
            end = (
                bisect.bisect_left(
                    z_sorted,
                    xbr[piece + 1],
                )
                - 1
            )
            end = min(
                max(end, index),
                last,
            )
            total += (
                A[piece] * (end - index + 1)
                + B[piece]
                * (
                    prefix[end + 1]
                    - prefix[index]
                )
            )
            index = end + 1

        return float(total)

    selected_sum = (
        range_sum(0, selected_left - 1)
        + range_sum(
            n - selected_right,
            n - 1,
        )
    )
    boundary = min(
        left_arm(selected_left),
        right_arm(selected_right),
    )
    return (
        selected_sum
        - (1.0 - boundary_weight) * boundary
    ) / tail_mass


# =============================================================================
# Projection cache and route certificate
# =============================================================================

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


def route_affine_lines(
    instance: AffineInstance,
    route,
):
    route = list(route)
    length = len(route)

    intercepts = np.empty(
        length + 1,
        dtype=float,
    )
    slopes = np.empty(
        (length + 1, instance.X.shape[1]),
        dtype=float,
    )

    intercept = float(
        instance.d0[route].sum()
    )
    slope = instance.dvec[route].sum(
        axis=0
    )

    intercepts[0] = intercept
    slopes[0] = slope

    for index, customer in enumerate(
        route,
        1,
    ):
        intercept += (
            instance.p0[customer]
            - instance.d0[customer]
        )
        slope = (
            slope
            + instance.pvec[customer]
            - instance.dvec[customer]
        )
        intercepts[index] = intercept
        slopes[index] = slope

    return intercepts, slopes


def training_route_slopes(
    instance: AffineInstance,
    route_count: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    customers = np.arange(1, instance.n)

    positive = instance.d0[1:][
        instance.d0[1:] > 0
    ]
    mean_delivery = (
        float(np.mean(positive))
        if positive.size
        else 1.0
    )
    target_length = max(
        2,
        int(round(
            instance.capacity / mean_delivery
        )),
    )
    maximum_length = min(
        instance.n - 1,
        max(
            8,
            int(math.ceil(
                1.7 * target_length
            )),
        ),
    )

    routes = []
    for _ in range(route_count):
        length = int(
            rng.integers(
                2,
                maximum_length + 1,
            )
        )
        routes.append(
            tuple(
                int(customer)
                for customer in rng.choice(
                    customers,
                    size=length,
                    replace=False,
                )
            )
        )
    return routes


def slope_second_moment(
    instance: AffineInstance,
    routes,
):
    dimension = instance.X.shape[1]
    moment = np.zeros(
        (dimension, dimension),
        dtype=float,
    )
    count = 0

    for route in routes:
        _, slopes = route_affine_lines(
            instance,
            route,
        )
        moment += slopes.T @ slopes
        count += slopes.shape[0]

    if count:
        moment /= count
    return moment


def matrix_geometry(matrix):
    values = np.maximum(
        np.linalg.eigvalsh(matrix),
        0.0,
    )[::-1]
    total = float(values.sum())
    if total <= 1e-15:
        return 0.0, 0.0, 0.0

    weights = values / total
    positive = weights[
        weights > 1e-15
    ]
    effective_rank = float(
        1.0 / np.sum(weights * weights)
    )
    entropy_rank = float(
        np.exp(
            -np.sum(
                positive
                * np.log(positive)
            )
        )
    )
    top_share = float(weights[0])
    return (
        effective_rank,
        entropy_rank,
        top_share,
    )


def build_certificate_cache(
    instance: AffineInstance,
    alpha: float,
    direction: str,
    train_routes: int,
    seed: int,
):
    t0 = time.perf_counter()

    X = np.asarray(instance.X, dtype=float)
    mu = X.mean(axis=0)
    centered = X - mu[None, :]

    routes = training_route_slopes(
        instance,
        train_routes,
        seed,
    )
    slope_moment = slope_second_moment(
        instance,
        routes,
    )
    geometry = matrix_geometry(slope_moment)

    key = direction.upper()
    if key == "SLOPE":
        values, vectors = np.linalg.eigh(
            slope_moment
        )
        u = vectors[:, int(np.argmax(values))]
    elif key == "PCA":
        covariance = (
            centered.T @ centered
            / max(1, centered.shape[0])
        )
        values, vectors = np.linalg.eigh(
            covariance
        )
        u = vectors[:, int(np.argmax(values))]
    elif key == "DELIVERY":
        delivery_sums = []
        for route in routes:
            delivery_sums.append(
                instance.dvec[list(route)].sum(
                    axis=0
                )
            )
        matrix = np.asarray(delivery_sums)
        delivery_moment = (
            matrix.T @ matrix
            / max(1, len(matrix))
        )
        values, vectors = np.linalg.eigh(
            delivery_moment
        )
        u = vectors[:, int(np.argmax(values))]
    else:
        raise ValueError(
            "direction must be SLOPE, PCA, or DELIVERY"
        )

    norm = float(np.linalg.norm(u))
    if norm <= 1e-15:
        raise ValueError(
            "Direction construction returned zero."
        )
    u = u / norm

    z = centered @ u
    residual_sq = np.maximum(
        np.einsum(
            "ij,ij->i",
            centered,
            centered,
        )
        - z * z,
        0.0,
    )
    residual_norm = np.sqrt(residual_sq)

    z_sorted = np.sort(z)
    z_prefix = np.concatenate(
        ([0.0], np.cumsum(z_sorted))
    )
    residual_cvar = empirical_cvar(
        residual_norm,
        alpha,
    )

    total_variance = float(
        np.sum(centered * centered)
    )
    explained = (
        float(np.dot(z, z))
        / total_variance
        if total_variance > 0
        else 1.0
    )

    return CertificateCache(
        alpha=float(alpha),
        u=u,
        mu=mu,
        z_sorted=z_sorted,
        z_prefix=z_prefix,
        residual_cvar=float(residual_cvar),
        explained_variance=float(explained),
        effective_rank=geometry[0],
        entropy_rank=geometry[1],
        top_share=geometry[2],
        direction=key,
        prep_seconds=(
            time.perf_counter() - t0
        ),
    )


def route_certificate(
    instance: AffineInstance,
    route,
    cache: CertificateCache,
):
    intercepts, slopes = route_affine_lines(
        instance,
        route,
    )

    projected_intercepts = (
        intercepts + slopes @ cache.mu
    )
    projected_slopes = slopes @ cache.u
    envelope = upper_envelope(
        zip(
            projected_intercepts,
            projected_slopes,
        )
    )
    projected_cvar = cvar_os(
        envelope,
        cache.z_sorted,
        cache.z_prefix,
        cache.alpha,
    )

    orthogonal_sq = np.maximum(
        np.einsum(
            "ij,ij->i",
            slopes,
            slopes,
        )
        - projected_slopes
        * projected_slopes,
        0.0,
    )
    gamma = float(
        np.sqrt(orthogonal_sq).max()
    )
    radius = (
        gamma * cache.residual_cvar
    )

    return RouteCertificate(
        projected_cvar=float(projected_cvar),
        gamma=gamma,
        radius=float(radius),
        lower=float(
            projected_cvar - radius
        ),
        upper=float(
            projected_cvar + radius
        ),
        pieces=len(envelope[0]),
    )


# =============================================================================
# FULL and CERT gates
# =============================================================================

class FullGate:
    mode = "FULL"

    def __init__(
        self,
        instance,
        capacity,
        alpha,
    ):
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
            route_peaks(
                route,
                self.instance.delivery_values,
                self.instance.pickup_values,
            ),
            self.alpha,
        )
        self.gate_seconds += (
            time.perf_counter() - t0
        )
        return (
            value
            <= self.capacity + 1e-9
        )


class CertifiedGate:
    mode = "CERT"

    def __init__(
        self,
        instance,
        capacity,
        alpha,
        cache,
        audit=False,
    ):
        self.instance = instance
        self.capacity = float(capacity)
        self.alpha = float(alpha)
        self.cache = cache
        self.audit = bool(audit)

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
        return (
            self.certified_feasible
            + self.certified_infeasible
        ) / self.calls

    @property
    def gate_seconds(self):
        return (
            self.certificate_seconds
            + self.fallback_seconds
        )

    def full_decision(self, route):
        value = empirical_cvar(
            route_peaks(
                route,
                self.instance.delivery_values,
                self.instance.pickup_values,
            ),
            self.alpha,
        )
        return (
            value
            <= self.capacity + 1e-9
        )

    def feasible(self, route):
        if not route:
            return True

        self.calls += 1
        self.route_length_sum += len(route)

        t0 = time.perf_counter()
        cert = route_certificate(
            self.instance,
            route,
            self.cache,
        )
        self.certificate_seconds += (
            time.perf_counter() - t0
        )
        self.pieces_sum += cert.pieces

        padding = (
            64.0
            * np.finfo(float).eps
            * (
                1.0
                + abs(self.capacity)
                + abs(cert.projected_cvar)
                + abs(cert.radius)
            )
        )

        if (
            cert.upper + padding
            <= self.capacity
        ):
            decision = True
            self.certified_feasible += 1
        elif (
            cert.lower - padding
            > self.capacity
        ):
            decision = False
            self.certified_infeasible += 1
        else:
            self.fallbacks += 1
            t0 = time.perf_counter()
            decision = self.full_decision(route)
            self.fallback_seconds += (
                time.perf_counter() - t0
            )
            return decision

        if self.audit:
            t0 = time.perf_counter()
            truth = self.full_decision(route)
            self.audit_seconds += (
                time.perf_counter() - t0
            )
            if decision != truth:
                raise AssertionError(
                    "Certificate mismatch: "
                    f"route={route}, "
                    f"lower={cert.lower}, "
                    f"upper={cert.upper}, "
                    f"capacity={self.capacity}, "
                    f"cert={decision}, full={truth}"
                )

        return decision


# =============================================================================
# ALNS/ILS core -- copied from dethloff_runner.py
# =============================================================================

def cw_init(distance, gate, n):
    routes = [
        [customer]
        for customer in range(1, n)
        if gate.feasible([customer])
    ]
    placed = {
        customer
        for route in routes
        for customer in route
    }
    leftovers = [
        customer
        for customer in range(1, n)
        if customer not in placed
    ]

    savings = []
    for a in range(1, n):
        for b in range(a + 1, n):
            savings.append(
                (
                    distance[0, a]
                    + distance[0, b]
                    - distance[a, b],
                    a,
                    b,
                )
            )
    savings.sort(reverse=True)

    routes_by_id = {
        index: route
        for index, route in enumerate(routes)
    }
    where = {
        route[0]: index
        for index, route in enumerate(routes)
    }

    for saving, a, b in savings:
        if saving <= 0:
            break

        ra, rb = where.get(a), where.get(b)
        if (
            ra is None
            or rb is None
            or ra == rb
        ):
            continue

        route_a = routes_by_id[ra]
        route_b = routes_by_id[rb]

        if (
            route_a[-1] == a
            and route_b[0] == b
        ):
            merged = route_a + route_b
        elif (
            route_a[0] == a
            and route_b[-1] == b
        ):
            merged = route_b + route_a
        elif (
            route_a[-1] == a
            and route_b[-1] == b
        ):
            merged = (
                route_a + route_b[::-1]
            )
        elif (
            route_a[0] == a
            and route_b[0] == b
        ):
            merged = (
                route_a[::-1] + route_b
            )
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
                a = (
                    current[i - 1]
                    if i > 0
                    else 0
                )
                b = current[i]
                c = current[k]
                d = (
                    current[k + 1]
                    if k + 1 < len(current)
                    else 0
                )

                if a == c or b == d:
                    continue

                delta = (
                    distance[a, c]
                    + distance[b, d]
                    - distance[a, b]
                    - distance[c, d]
                )
                if delta < -1e-9:
                    candidate = (
                        current[:i]
                        + current[
                            i:k + 1
                        ][::-1]
                        + current[k + 1:]
                    )
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
        for route_index in range(
            len(solution)
        ):
            route = solution[route_index]
            for position in range(
                len(route)
            ):
                customer = route[position]
                a = (
                    route[position - 1]
                    if position > 0
                    else 0
                )
                b = (
                    route[position + 1]
                    if position + 1 < len(route)
                    else 0
                )
                gain = (
                    distance[a, customer]
                    + distance[customer, b]
                    - distance[a, b]
                )

                for target_index in range(
                    len(solution)
                ):
                    if (
                        target_index
                        == route_index
                    ):
                        continue

                    target = solution[
                        target_index
                    ]
                    for insertion_position in range(
                        len(target) + 1
                    ):
                        u = (
                            target[
                                insertion_position - 1
                            ]
                            if insertion_position > 0
                            else 0
                        )
                        v = (
                            target[
                                insertion_position
                            ]
                            if insertion_position < len(target)
                            else 0
                        )
                        delta = (
                            distance[u, customer]
                            + distance[customer, v]
                            - distance[u, v]
                            - gain
                        )

                        if delta < -1e-9:
                            new_route = (
                                route[:position]
                                + route[
                                    position + 1:
                                ]
                            )
                            new_target = (
                                target[
                                    :insertion_position
                                ]
                                + [customer]
                                + target[
                                    insertion_position:
                                ]
                            )

                            if (
                                gate.feasible(
                                    new_route
                                )
                                and gate.feasible(
                                    new_target
                                )
                            ):
                                solution[
                                    route_index
                                ] = new_route
                                solution[
                                    target_index
                                ] = new_target
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

        solution = [
            route
            for route in solution
            if route
        ]

    return solution


def greedy_insert(
    solution,
    customer,
    distance,
    gate,
):
    best = (None, None, math.inf)
    for route_index, route in enumerate(
        solution
    ):
        for position in range(
            len(route) + 1
        ):
            candidate = (
                route[:position]
                + [customer]
                + route[position:]
            )
            if not gate.feasible(candidate):
                continue

            u = (
                route[position - 1]
                if position > 0
                else 0
            )
            v = (
                route[position]
                if position < len(route)
                else 0
            )
            delta = (
                distance[u, customer]
                + distance[customer, v]
                - distance[u, v]
            )
            if delta < best[2]:
                best = (
                    route_index,
                    position,
                    delta,
                )

    return best


def ruin_recreate(
    solution,
    distance,
    gate,
    rng,
    q_frac=0.2,
):
    solution = [
        route[:] for route in solution
    ]
    customers = [
        customer
        for route in solution
        for customer in route
    ]
    q = max(
        1,
        int(q_frac * len(customers)),
    )
    removed = rng.sample(
        customers,
        min(q, len(customers)),
    )
    removed_set = set(removed)

    solution = [
        [
            customer
            for customer in route
            if customer not in removed_set
        ]
        for route in solution
    ]
    solution = [
        route
        for route in solution
        if route
    ]

    rng.shuffle(removed)
    for customer in removed:
        (
            route_index,
            position,
            _,
        ) = greedy_insert(
            solution,
            customer,
            distance,
            gate,
        )

        if route_index is None:
            solution.append([customer])
        else:
            solution[
                route_index
            ].insert(
                position,
                customer,
            )

    return solution


def local_search(solution, distance, gate):
    solution = [
        two_opt_gate(
            route,
            distance,
            gate,
        )
        for route in solution
        if route
    ]
    solution = relocate_gate(
        solution,
        distance,
        gate,
    )
    return [
        route
        for route in solution
        if route
    ]


def econ_cost(
    solution,
    distance,
    omega_vehicle,
):
    return (
        sum(
            route_cost(route, distance)
            for route in solution
        )
        + omega_vehicle
        * sum(
            1
            for route in solution
            if route
        )
    )


def solve_fixed_iterations(
    distance,
    gate,
    n,
    omega_vehicle,
    iterations,
    seed,
):
    rng = random.Random(seed)
    current = local_search(
        cw_init(distance, gate, n),
        distance,
        gate,
    )

    best = [
        route[:] for route in current
    ]
    best_cost = econ_cost(
        best,
        distance,
        omega_vehicle,
    )

    for _ in range(iterations):
        candidate = local_search(
            ruin_recreate(
                best,
                distance,
                gate,
                rng,
                rng.choice(
                    [0.1, 0.15, 0.2, 0.3]
                ),
            ),
            distance,
            gate,
        )
        candidate_cost = econ_cost(
            candidate,
            distance,
            omega_vehicle,
        )

        if (
            candidate_cost
            < best_cost - 1e-9
        ):
            best = [
                route[:]
                for route in candidate
            ]
            best_cost = candidate_cost

    return [
        route for route in best if route
    ]


# =============================================================================
# Paired benchmark
# =============================================================================

def canonical_plan(plan):
    return tuple(
        sorted(
            tuple(route)
            for route in plan
        )
    )


def median_iqr(values):
    array = np.asarray(
        list(values),
        dtype=float,
    )
    return (
        float(np.median(array)),
        float(
            np.quantile(array, 0.75)
            - np.quantile(array, 0.25)
        ),
    )


def run_solver_once(
    instance,
    capacity,
    alpha,
    cache,
    iterations,
    seed,
    method,
    audit,
):
    if method == "FULL":
        gate = FullGate(
            instance,
            capacity,
            alpha,
        )
    elif method == "CERT":
        gate = CertifiedGate(
            instance,
            capacity,
            alpha,
            cache,
            audit=audit,
        )
    else:
        raise ValueError(method)

    omega_vehicle = float(
        np.mean(
            instance.distance[
                instance.distance > 0
            ]
        )
    )

    t0 = time.perf_counter()
    plan = solve_fixed_iterations(
        instance.distance,
        gate,
        instance.n,
        omega_vehicle,
        iterations,
        seed,
    )
    total_seconds = (
        time.perf_counter() - t0
    )

    return {
        "method": method,
        "plan": plan,
        "plan_key": canonical_plan(plan),
        "K": len(plan),
        "distance_raw": sum(
            route_cost(
                route,
                instance.distance,
            )
            for route in plan
        ),
        "objective_raw": econ_cost(
            plan,
            instance.distance,
            omega_vehicle,
        ),
        "calls": gate.calls,
        "gate_seconds": gate.gate_seconds,
        "total_seconds": total_seconds,
        "mean_route_length_checked": (
            gate.route_length_sum / gate.calls
            if gate.calls
            else 0.0
        ),
        "coverage": (
            gate.coverage
            if method == "CERT"
            else 0.0
        ),
        "certified_feasible": (
            gate.certified_feasible
            if method == "CERT"
            else 0
        ),
        "certified_infeasible": (
            gate.certified_infeasible
            if method == "CERT"
            else 0
        ),
        "fallbacks": (
            gate.fallbacks
            if method == "CERT"
            else gate.calls
        ),
        "certificate_seconds": (
            gate.certificate_seconds
            if method == "CERT"
            else 0.0
        ),
        "fallback_seconds": (
            gate.fallback_seconds
            if method == "CERT"
            else gate.gate_seconds
        ),
        "audit_seconds": (
            gate.audit_seconds
            if method == "CERT"
            else 0.0
        ),
        "mean_pieces": (
            gate.pieces_sum / gate.calls
            if method == "CERT"
            and gate.calls
            else 0.0
        ),
    }


def benchmark_cell(
    instance,
    profile,
    policy,
    capacity,
    alpha,
    cache,
    iterations,
    repetitions,
    seed,
    audit,
):
    warm_iterations = min(2, iterations)
    if warm_iterations > 0:
        warm_full = run_solver_once(
            instance,
            capacity,
            alpha,
            cache,
            warm_iterations,
            seed,
            "FULL",
            False,
        )
        warm_cert = run_solver_once(
            instance,
            capacity,
            alpha,
            cache,
            warm_iterations,
            seed,
            "CERT",
            False,
        )
        if (
            warm_full["plan_key"]
            != warm_cert["plan_key"]
        ):
            raise AssertionError(
                "Warm-up FULL/CERT plans differ."
            )

    rng = random.Random(
        seed
        + sum(ord(character) for character in policy)
        + sum(ord(character) for character in profile)
    )

    full_runs = []
    cert_runs = []
    gate_speedups = []
    solver_speedups = []

    for repetition in range(repetitions):
        order = ["FULL", "CERT"]
        rng.shuffle(order)
        current = {}

        for method in order:
            current[method] = run_solver_once(
                instance,
                capacity,
                alpha,
                cache,
                iterations,
                seed,
                method,
                audit if method == "CERT" else False,
            )

        full = current["FULL"]
        cert = current["CERT"]

        if (
            full["plan_key"]
            != cert["plan_key"]
        ):
            raise AssertionError(
                f"{instance.name}/{profile}/{policy}: "
                "final plans differ."
            )
        if full["calls"] != cert["calls"]:
            raise AssertionError(
                f"{instance.name}/{profile}/{policy}: "
                f"check counts differ "
                f"({full['calls']} vs {cert['calls']})."
            )
        if abs(
            full["objective_raw"]
            - cert["objective_raw"]
        ) > 1e-8:
            raise AssertionError(
                f"{instance.name}/{profile}/{policy}: "
                "objectives differ."
            )

        full_runs.append(full)
        cert_runs.append(cert)
        gate_speedups.append(
            full["gate_seconds"]
            / cert["gate_seconds"]
        )
        solver_speedups.append(
            full["total_seconds"]
            / cert["total_seconds"]
        )

        print(
            f"      rep {repetition + 1}/{repetitions}: "
            f"gate={gate_speedups[-1]:.2f}x "
            f"solver={solver_speedups[-1]:.2f}x "
            f"cover={cert['coverage']:.3f} "
            f"calls={cert['calls']} "
            f"assert=PASS"
        )

    full_gate, full_gate_iqr = median_iqr(
        run["gate_seconds"]
        for run in full_runs
    )
    cert_gate, cert_gate_iqr = median_iqr(
        run["gate_seconds"]
        for run in cert_runs
    )
    full_solver, full_solver_iqr = median_iqr(
        run["total_seconds"]
        for run in full_runs
    )
    cert_solver, cert_solver_iqr = median_iqr(
        run["total_seconds"]
        for run in cert_runs
    )
    gate_speed, gate_speed_iqr = median_iqr(
        gate_speedups
    )
    solver_speed, solver_speed_iqr = median_iqr(
        solver_speedups
    )

    reference = cert_runs[0]
    with_prep = (
        full_solver
        / (
            cert_solver
            + cache.prep_seconds
        )
    )

    return {
        "instance": instance.name,
        "profile": profile,
        "policy": policy,
        "N": instance.X.shape[0],
        "dimension": instance.X.shape[1],
        "alpha": alpha,
        "capacity": capacity,
        "iterations": iterations,
        "repetitions": repetitions,
        "direction": cache.direction,
        "explained_variance": (
            cache.explained_variance
        ),
        "effective_rank": cache.effective_rank,
        "entropy_rank": cache.entropy_rank,
        "top_share": cache.top_share,
        "residual_cvar": cache.residual_cvar,
        "prep_seconds": cache.prep_seconds,
        "calls": reference["calls"],
        "coverage": reference["coverage"],
        "certified_feasible": reference[
            "certified_feasible"
        ],
        "certified_infeasible": reference[
            "certified_infeasible"
        ],
        "fallbacks": reference["fallbacks"],
        "mean_pieces": reference["mean_pieces"],
        "mean_route_length_checked": reference[
            "mean_route_length_checked"
        ],
        "K": reference["K"],
        "distance": (
            reference["distance_raw"]
            / instance.scale
        ),
        "objective_raw": reference[
            "objective_raw"
        ],
        "full_gate_seconds": full_gate,
        "full_gate_iqr": full_gate_iqr,
        "cert_gate_seconds": cert_gate,
        "cert_gate_iqr": cert_gate_iqr,
        "gate_speedup": gate_speed,
        "gate_speedup_iqr": gate_speed_iqr,
        "full_solver_seconds": full_solver,
        "full_solver_iqr": full_solver_iqr,
        "cert_solver_seconds": cert_solver,
        "cert_solver_iqr": cert_solver_iqr,
        "solver_speedup": solver_speed,
        "solver_speedup_iqr": solver_speed_iqr,
        "solver_speedup_with_prep": with_prep,
        "audit": audit,
    }


CSV_FIELDS = [
    "instance",
    "profile",
    "policy",
    "N",
    "dimension",
    "alpha",
    "capacity",
    "iterations",
    "repetitions",
    "direction",
    "explained_variance",
    "effective_rank",
    "entropy_rank",
    "top_share",
    "residual_cvar",
    "prep_seconds",
    "calls",
    "coverage",
    "certified_feasible",
    "certified_infeasible",
    "fallbacks",
    "mean_pieces",
    "mean_route_length_checked",
    "K",
    "distance",
    "objective_raw",
    "full_gate_seconds",
    "full_gate_iqr",
    "cert_gate_seconds",
    "cert_gate_iqr",
    "gate_speedup",
    "gate_speedup_iqr",
    "full_solver_seconds",
    "full_solver_iqr",
    "cert_solver_seconds",
    "cert_solver_iqr",
    "solver_speedup",
    "solver_speedup_iqr",
    "solver_speedup_with_prep",
    "audit",
]


def append_csv(path, row):
    path = Path(path)
    exists = path.exists()
    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CSV_FIELDS,
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# =============================================================================
# CLI
# =============================================================================

def parse_list(text):
    return [
        value.strip()
        for value in text.split(",")
        if value.strip()
    ]


def parse_args(argv):
    config = {
        "dir": DEFAULT_DATA_DIR,
        "regex": "",
        "max": None,
        "profiles": [
            "concentrated",
            "moderate",
            "diffuse",
        ],
        "N": N_DATA,
        "d": FACTOR_DIMENSION,
        "iters": DEFAULT_ITERS,
        "reps": DEFAULT_REPS,
        "policy": DEFAULT_POLICY,
        "direction": DEFAULT_DIRECTION,
        "train": DEFAULT_TRAIN_ROUTES,
        "seed": SEED,
        "cv": CV,
        "alpha": ALPHA,
        "eps": EPS_FRAC,
        "support": SUPPORT,
        "audit": False,
        "out": DEFAULT_OUTPUT,
    }

    for argument in argv:
        if argument == "audit":
            config["audit"] = True
            continue
        if "=" not in argument:
            raise ValueError(
                f"Unknown argument: {argument}"
            )

        key, value = argument.split("=", 1)
        if key in {
            "dir",
            "regex",
            "policy",
            "direction",
            "out",
        }:
            config[key] = value
        elif key == "profiles":
            config[key] = parse_list(value)
        elif key in {
            "max",
            "N",
            "d",
            "iters",
            "reps",
            "train",
            "seed",
        }:
            config[key] = int(value)
        elif key in {
            "cv",
            "alpha",
            "eps",
            "support",
        }:
            config[key] = float(value)
        else:
            raise ValueError(
                f"Unknown argument: {argument}"
            )

    config["policy"] = config["policy"].upper()
    config["direction"] = config[
        "direction"
    ].upper()
    config["profiles"] = [
        profile.lower()
        for profile in config["profiles"]
    ]

    if config["policy"] not in {
        "SAA",
        "WDRO",
        "BOTH",
    }:
        raise ValueError(
            "policy must be SAA, WDRO, or both"
        )
    if config["direction"] not in {
        "SLOPE",
        "PCA",
        "DELIVERY",
    }:
        raise ValueError(
            "direction must be SLOPE, PCA, or DELIVERY"
        )
    for profile in config["profiles"]:
        if profile not in PROFILES:
            raise ValueError(
                f"Unknown profile: {profile}"
            )

    if config["N"] <= 0:
        raise ValueError("N must be positive")
    if config["d"] < 2:
        raise ValueError(
            "factor dimension must be >= 2"
        )
    if config["iters"] < 0:
        raise ValueError(
            "iters must be nonnegative"
        )
    if config["reps"] <= 0:
        raise ValueError(
            "reps must be positive"
        )
    if not (
        0.0 < config["alpha"] < 1.0
    ):
        raise ValueError(
            "alpha must lie in (0,1)"
        )
    if not (
        0.0 <= config["eps"] < 1.0
    ):
        raise ValueError(
            "eps must lie in [0,1)"
        )

    return config


def main():
    try:
        config = parse_args(sys.argv[1:])
    except Exception as error:
        print("ARGUMENT ERROR:", error)
        raise SystemExit(2)

    files = sorted(
        glob.glob(
            str(
                Path(config["dir"])
                / "*.vrpspd"
            )
        )
    )
    if config["regex"]:
        pattern = re.compile(
            config["regex"],
            re.IGNORECASE,
        )
        files = [
            file
            for file in files
            if pattern.search(
                Path(file).stem
            )
        ]
    if config["max"]:
        files = files[:config["max"]]

    if not files:
        print(
            f"ERROR: no matching .vrpspd files "
            f"in '{config['dir']}'."
        )
        return

    policies = (
        ["SAA", "WDRO"]
        if config["policy"] == "BOTH"
        else [config["policy"]]
    )

    print("=" * 104)
    print(
        " EXACT MULTIVARIATE CVaR CERTIFICATE INSIDE THE EXISTING DETHLOFF ALNS"
    )
    print("=" * 104)
    print(
        f"instances={len(files)} "
        f"profiles={','.join(config['profiles'])} "
        f"N={config['N']:,} d={config['d']} "
        f"iters={config['iters']} "
        f"reps={config['reps']} "
        f"policies={','.join(policies)}"
    )
    print(
        f"alpha={config['alpha']} "
        f"cv={config['cv']} "
        f"eps={config['eps']} "
        f"direction={config['direction']} "
        f"train={config['train']} "
        f"audit={config['audit']}"
    )
    print(
        "Protocol: copied ALNS, fixed iterations, randomized FULL/CERT order, "
        "equal calls/plan/objective asserted."
    )
    print("-" * 104)

    rows = []
    for file_index, file in enumerate(
        files,
        1,
    ):
        name = Path(file).stem
        print(
            f"\n[{file_index}/{len(files)}] {name}"
        )

        for profile_name in config[
            "profiles"
        ]:
            print(
                f"   profile={profile_name}"
            )
            try:
                t0 = time.perf_counter()
                instance = make_affine_instance(
                    file,
                    PROFILES[profile_name],
                    config["N"],
                    config["d"],
                    config["cv"],
                    config["support"],
                    config["seed"],
                )
                scenario_seconds = (
                    time.perf_counter() - t0
                )

                cache = build_certificate_cache(
                    instance,
                    config["alpha"],
                    config["direction"],
                    config["train"],
                    config["seed"] + 1000,
                )
                print(
                    f"      scenario={scenario_seconds:.2f}s "
                    f"prep={cache.prep_seconds:.2f}s "
                    f"erank={cache.effective_rank:.3f} "
                    f"top={cache.top_share:.3f} "
                    f"EV1={cache.explained_variance:.3f}"
                )

                for policy in policies:
                    capacity = (
                        instance.capacity
                        if policy == "SAA"
                        else instance.capacity
                        * (1.0 - config["eps"])
                    )
                    print(
                        f"      {policy}: "
                        f"cap={capacity:.2f}"
                    )

                    row = benchmark_cell(
                        instance,
                        profile_name,
                        policy,
                        capacity,
                        config["alpha"],
                        cache,
                        config["iters"],
                        config["reps"],
                        config["seed"],
                        config["audit"],
                    )
                    append_csv(
                        config["out"],
                        row,
                    )
                    rows.append(row)

                    print(
                        f"      => calls={row['calls']} "
                        f"cover={row['coverage']:.3f} "
                        f"pieces={row['mean_pieces']:.2f} "
                        f"gate={row['gate_speedup']:.2f}x "
                        f"solver={row['solver_speedup']:.2f}x "
                        f"prep={row['solver_speedup_with_prep']:.2f}x "
                        f"K={row['K']} "
                        f"dist={row['distance']:.2f} "
                        f"assert=PASS"
                    )

            except Exception as error:
                print(
                    f"      ERROR: "
                    f"{type(error).__name__}: {error}"
                )

    if not rows:
        print("\nNo successful cells.")
        return

    gate_speed = np.asarray(
        [
            row["gate_speedup"]
            for row in rows
        ]
    )
    solver_speed = np.asarray(
        [
            row["solver_speedup"]
            for row in rows
        ]
    )
    coverage = np.asarray(
        [
            row["coverage"]
            for row in rows
        ]
    )

    print("\n" + "=" * 104)
    print("SUMMARY")
    print(
        f"cells={len(rows)} "
        f"median gate={np.median(gate_speed):.2f}x "
        f"min={gate_speed.min():.2f}x "
        f"median solver={np.median(solver_speed):.2f}x "
        f"min={solver_speed.min():.2f}x "
        f"median coverage={np.median(coverage):.3f}"
    )
    if np.std(coverage) > 1e-12:
        print(
            f"corr(coverage,solver-speed)="
            f"{np.corrcoef(coverage, solver_speed)[0, 1]:.3f}"
        )
    print(f"wrote {config['out']}")


if __name__ == "__main__":
    main()
