import sys, math, bisect, time
import numpy as np

# ============================ upper envelope of lines y = a + b x ============================
def upper_envelope(lines):
    lines = sorted(lines, key=lambda t: (t[1], t[0]))
    uniq = []
    for a, b in lines:                       # merge equal slopes, keep max intercept
        if uniq and uniq[-1][1] == b:
            if a > uniq[-1][0]:
                uniq[-1] = (a, b)
            continue
        uniq.append((a, b))
    def xcross(p, q):                        # p.b < q.b
        return (p[0] - q[0]) / (q[1] - p[1])
    st, xs = [], []
    for ln in uniq:
        while st:
            x = xcross(st[-1], ln)
            if xs and x <= xs[-1]:
                st.pop(); xs.pop()
            else:
                break
        if not st:
            st.append(ln); xs.append(-math.inf)
        else:
            xs.append(xcross(st[-1], ln)); st.append(ln)
    A = [p[0] for p in st]
    B = [p[1] for p in st]
    xbr = xs + [math.inf]
    return A, B, xbr


# ============================ tail parameters (Rockafellar-Uryasev) ============================
def _tail_parameters(alpha, N):
    if not (0.0 <= alpha < 1.0):
        raise ValueError("alpha must lie in [0,1)")
    if N <= 0:
        raise ValueError("the empirical sample must be nonempty")

    tail_mass = (1.0 - alpha) * N

    nearest = float(round(tail_mass))
    tol = 16.0 * max(math.ulp(tail_mass), math.ulp(nearest))
    if abs(tail_mass - nearest) <= tol:
        tail_mass = nearest

    m = min(N, max(1, int(math.ceil(tail_mass))))
    boundary_weight = min(1.0, max(0.0, tail_mass - (m - 1)))

    return tail_mass, m, boundary_weight


# ============================ CVaR: vectorised sampled baseline, O(N) ============================
def cvar_saa(env, Fsorted, xbr_np, A_np, B_np, alpha):
    N = len(Fsorted)
    tail_mass, m, w = _tail_parameters(alpha, N)
    k = np.searchsorted(xbr_np, Fsorted, side='right') - 1
    np.clip(k, 0, len(A_np) - 1, out=k)
    vals = A_np[k] + B_np[k] * Fsorted
    part = np.partition(vals, N - m)[N - m:]
    boundary = float(part.min())
    return (float(part.sum()) - (1.0 - w) * boundary) / tail_mass


# ============================ CVaR: order statistics, O(p log N) ============================
def cvar_os(env, Fsorted, pre, alpha):
    """Exact empirical CVaR of a convex piecewise-linear loss in O(p log N) time."""
    A, B, xbr = env
    N = len(Fsorted)
    p = len(A)
    tail_mass, m, w = _tail_parameters(alpha, N)

    def phi(i):
        x = Fsorted[i]
        k = bisect.bisect_right(xbr, x) - 1
        if k < 0: k = 0
        if k >= p: k = p - 1
        return A[k] + B[k] * x

    # --- Step 1: valley from the envelope slope sign-change, O(log p + log N).
    # Slopes B increase along the upper envelope, so one bisect over B locates the
    # minimising piece and one bisect over Fsorted places it in the sample. phi is
    # unimodal in i, so the discrete minimiser is a neighbour of that position.
    kstar = bisect.bisect_left(B, 0.0)
    if kstar == 0:
        istar = 0                            # envelope non-decreasing throughout
    elif kstar >= p:
        istar = N - 1                        # envelope non-increasing throughout
    else:
        t = bisect.bisect_left(Fsorted, xbr[kstar])
        istar = min((i for i in (t - 1, t, t + 1) if 0 <= i < N), key=phi)

    # --- Step 2: partition the two non-increasing arms, O(log p log N).
    q, r = istar + 1, N - 1 - istar

    def Aarm(k):
        if k <= 0: return math.inf
        if k > q:  return -math.inf
        return phi(k - 1)

    def Barm(k):
        if k <= 0: return math.inf
        if k > r:  return -math.inf
        return phi(N - k)

    low, high = max(0, m - r), min(m, q)
    a, b, best = low, high, low - 1
    while a <= b:
        mid = (a + b) // 2
        if Aarm(mid) >= Barm(m - mid + 1):
            best = mid; a = mid + 1
        else:
            b = mid - 1
    pL = max(best, low)
    pR = m - pL

    # --- Step 3: piecewise tail sums via factor prefix sums, O(p log N).
    def rangesum(iLo, iHi):
        if iLo > iHi: return 0.0
        tot = 0.0; i = iLo
        while i <= iHi:
            x = Fsorted[i]
            k = bisect.bisect_right(xbr, x) - 1
            if k < 0: k = 0
            if k >= p: k = p - 1
            j = bisect.bisect_left(Fsorted, xbr[k + 1]) - 1
            if j > iHi: j = iHi
            if j < i:   j = i
            tot += A[k] * (j - i + 1) + B[k] * (pre[j + 1] - pre[i])
            i = j + 1
        return tot

    selected = rangesum(0, pL - 1) + rangesum(N - pR, N - 1)
    boundary = min(Aarm(pL), Barm(pR))
    return (selected - (1.0 - w) * boundary) / tail_mass


# ============================ sample helpers ============================
def gen_factor(N, seed):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(N)
    np.clip(z, -2.5, 2.5, out=z)
    z.sort()
    return z

def prep(Fsorted):
    return np.concatenate(([0.0], np.cumsum(Fsorted)))


# ============================ instances ============================
def validate_instance(I):
    """Delivery and pickup must be nonnegative for F in [-factor_bound, factor_bound].

    This is not cosmetic. The exhaustive pricing pass prunes a partial route on a
    capacity violation and never revisits it; that is valid only because the peak
    load is non-decreasing along any extension, which in turn requires every
    delivery to be nonnegative over the factor support.
    """
    zmax = I.get('factor_bound', 2.5)
    da, db, na, nb = I['da'], I['db'], I['na'], I['nb']
    delivery_min = da[1:] - zmax * np.abs(db[1:])
    pickup_min = (da[1:] + na[1:]) - zmax * np.abs(db[1:] + nb[1:])
    if delivery_min.min() < -1e-12 or pickup_min.min() < -1e-12:
        raise ValueError("demands must remain nonnegative over the factor support")


def make_instance(n, seed, dominance=0.85, factor_bound=2.5, Q=65.0):
    """Single generator for every experiment.

    dominance = fraction of customers whose net sensitivity carries the common sign.
    A dominant factor (the paper's premise) makes net sensitivities mostly same-sign,
    so cumulative sensitivities are near-monotone; dominance = 0.5 is the contrast.
    """
    rng = np.random.default_rng(seed)
    X = rng.random(n + 1) * 100
    Y = rng.random(n + 1) * 100
    dist = np.hypot(X[:, None] - X[None, :], Y[:, None] - Y[None, :])
    da = np.zeros(n + 1); db = np.zeros(n + 1)
    na = np.zeros(n + 1); nb = np.zeros(n + 1); dual = np.zeros(n + 1)
    for i in range(1, n + 1):
        dmean = 8.0 + rng.random() * 8.0
        pmean = 8.0 + rng.random() * 8.0
        d_slope = 0.4 + rng.random() * 0.8
        net_mag = 0.25 + rng.random() * 0.65
        da[i] = dmean
        db[i] = d_slope
        na[i] = pmean - dmean
        nb[i] = net_mag * (1.0 if rng.random() < dominance else -1.0)
        dual[i] = 25.0 + rng.random() * 35.0
    I = dict(n=n, dist=dist, da=da, db=db, na=na, nb=nb, dual=dual,
             alpha=0.90, Q=float(Q), factor_bound=factor_bound)
    validate_instance(I)
    return I


def random_route_envelope(pool, rng, lo=2, hi=51):
    ncust = pool['n']
    L = int(rng.integers(lo, hi))
    perm = rng.permutation(np.arange(1, ncust + 1))[:L]
    tda = tdb = 0.0; pn = (0.0, 0.0); pns = [(0.0, 0.0)]
    for j in perm:
        tda += pool['da'][j]; tdb += pool['db'][j]
        pn = (pn[0] + pool['na'][j], pn[1] + pool['nb'][j]); pns.append(pn)
    return upper_envelope([(tda + a, tdb + b) for (a, b) in pns]), L


# ============================ exact elementary pricing ============================
def run_exact_pricing(I, Fsorted, pre, kernel):
    """Exhaustively enumerate every capacity-feasible elementary route.

    No reduced-cost dominance. Dominance on (last node, visited set) would be
    unsound here: two labels with the same visited set and last node differ in the
    order of visits, hence in their peak-load envelope, so the cheaper label is not
    necessarily the more extendable one. Every feasible label is closed at the depot
    and scored, so the reported best reduced cost is the exact pricing optimum.
    """
    validate_instance(I)
    if kernel not in {'os', 'saa'}:
        raise ValueError("kernel must be 'os' or 'saa'")
    n = I['n']; alpha = I['alpha']; Q = I['Q']
    dist = I['dist']; da = I['da']; db = I['db']
    na = I['na']; nb = I['nb']; dual = I['dual']

    checks = 0; feasible = 0; t_env = 0.0; t_cvar = 0.0
    best_rc = math.inf; best_route = None; max_len = 0
    stack = [(0, 0, 0.0, 0.0, [(0.0, 0.0)], 0.0, ())]
    t0 = time.perf_counter()
    while stack:
        node, vis, tda, tdb, pn, rc, route = stack.pop()
        for j in range(1, n + 1):
            bit = 1 << (j - 1)
            if vis & bit:
                continue
            ntda = tda + da[j]; ntdb = tdb + db[j]
            last = pn[-1]; npn = (last[0] + na[j], last[1] + nb[j])
            peak = [(ntda + a, ntdb + b) for (a, b) in pn]
            peak.append((ntda + npn[0], ntdb + npn[1]))

            te = time.perf_counter()
            env = upper_envelope(peak)
            t_env += time.perf_counter() - te

            checks += 1
            tc = time.perf_counter()
            if kernel == 'os':
                cv = cvar_os(env, Fsorted, pre, alpha)
            else:
                cv = cvar_saa(env, Fsorted, np.asarray(env[2]),
                              np.asarray(env[0]), np.asarray(env[1]), alpha)
            t_cvar += time.perf_counter() - tc

            if cv > Q + 1e-10:
                continue
            feasible += 1
            nrc = rc + dist[node, j] - dual[j]
            nroute = route + (j,)
            max_len = max(max_len, len(nroute))
            closed = nrc + dist[j, 0]
            if closed < best_rc:
                best_rc = closed; best_route = nroute
            stack.append((j, vis | bit, ntda, ntdb, pn + [npn], nrc, nroute))
    return dict(total=time.perf_counter() - t0, envelope=t_env, cvar=t_cvar,
                checks=checks, feasible_labels=feasible, max_len=max_len,
                best_rc=best_rc, best_route=best_route)


# ============================ timing helpers ============================
def time_us(fn, reps):
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1e6

def linfit(x, y):
    x = np.asarray(x); y = np.asarray(y)
    mx, my = x.mean(), y.mean()
    sxx = ((x - mx) ** 2).sum(); syy = ((y - my) ** 2).sum()
    slope = ((x - mx) * (y - my)).sum() / sxx
    ssr = ((y - (my - slope * mx + slope * x)) ** 2).sum()
    return slope, (1 - ssr / syy if syy > 0 else 1.0)


# ============================ self-test ============================
def cvar_reference(vals, alpha):
    """Independent Rockafellar-Uryasev CVaR: min over eta of eta + E[(X-eta)+]/(1-alpha).
    Shares no code with _tail_parameters, so a common-mode error cannot pass."""
    x = np.sort(np.asarray(vals, dtype=float))
    if alpha == 0.0:
        return float(x.mean())
    return min(float(e + np.maximum(x - e, 0.0).mean() / (1.0 - alpha)) for e in x)


def selftest():
    rng = np.random.default_rng(7)
    w_pair = w_ref = 0.0
    ALPHAS = [0.0, 0.5, 0.80, 0.85, 0.90, 0.95, 0.99, 0.995]
    for t in range(2000):
        L = 2 + int(rng.integers(0, 12))
        lines = [(float(rng.uniform(-50, 50)), float(rng.uniform(-5, 5))) for _ in range(L)]
        env = upper_envelope(lines)
        A_np = np.array(env[0]); B_np = np.array(env[1]); xbr_np = np.array(env[2])
        N = 200 + int(rng.integers(0, 4000))
        F = gen_factor(N, 1000 + t); pre = prep(F)
        alpha = float(rng.choice(ALPHAS)) if t % 2 else float(rng.uniform(0.0, 0.995))
        a = cvar_os(env, F, pre, alpha)
        b = cvar_saa(env, F, xbr_np, A_np, B_np, alpha)
        w_pair = max(w_pair, abs(a - b))
        if t % 20 == 0:                                     # reference pass is O(N^2)
            k = np.searchsorted(xbr_np, F, side='right') - 1
            np.clip(k, 0, len(A_np) - 1, out=k)
            ref = cvar_reference(A_np[k] + B_np[k] * F, alpha)
            w_ref = max(w_ref, abs(a - ref) / max(1.0, abs(ref)))
    ok = (w_pair < 1e-8) and (w_ref < 1e-8)
    print("[selftest]")
    print(f"  order-stat vs sampled, 2000 configs      : max |delta| = {w_pair:.3e}")
    print(f"  order-stat vs Rockafellar-Uryasev, 100   : max rel     = {w_ref:.3e}")
    print(f"  alpha grid includes 0, 0.95, 0.99, 0.995")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# ============================ experiments ============================
def experiment_pieces(nroutes=10000):
    H = lambda n: sum(1.0 / j for j in range(1, int(n) + 1))
    print("[A] envelope piece count over %d random routes (length 2..50)" % nroutes)
    out = {}
    for tag, dom in [("dominant (0.85)", 0.85), ("mixed (0.50)", 0.50)]:
        pool = make_instance(80, 4242, dominance=dom)
        rng = np.random.default_rng(20240 if dom > 0.5 else 20241)
        pc = []; lens = []
        for _ in range(nroutes):
            env, L = random_route_envelope(pool, rng)
            pc.append(len(env[0])); lens.append(L)
        pc = np.array(pc); lens = np.array(lens)
        q = np.percentile(pc, [50, 90, 99])
        print(f"  {tag:<18} mean={pc.mean():.2f} median={q[0]:.0f} p90={q[1]:.0f} "
              f"p99={q[2]:.0f} max={pc.max()}")
        out[tag] = (pc, lens)
    pc, lens = out["dominant (0.85)"]
    print("  mean p vs Sparre Andersen H_k+1:")
    print("    %-10s %6s %11s %9s %8s" % ("bucket", "kbar", "H_k+1", "mean p", "err"))
    for lo, hi in [(2, 9), (10, 19), (20, 29), (30, 39), (40, 50)]:
        m = (lens >= lo) & (lens <= hi)
        kb = lens[m].mean(); pred = H(round(kb)) + 1.0; obs = pc[m].mean()
        print(f"    {f'{lo}-{hi}':<10} {kb:6.1f} {pred:11.2f} {obs:9.2f} "
              f"{100*(obs-pred)/pred:7.1f}%")
    print()


def experiment_kernel(nroutes=100, Ns=(100, 1000, 10000, 100000, 1000000, 10000000)):
    pool = make_instance(60, 777, dominance=0.85)
    rng = np.random.default_rng(31415)
    routes = [random_route_envelope(pool, rng, 5, 41)[0] for _ in range(nroutes)]
    pmed = int(np.median([len(e[0]) for e in routes]))
    print(f"[B] kernel scaling, median over {nroutes} routes (median p={pmed}), alpha=0.90")
    print(f"{'N':>12} {'order-stat (us)':>17} {'sampled (us)':>14} {'speed-up':>10}")
    lN = []; t_os = []; lt_sa = []; maxdiff = 0.0
    for N in Ns:
        F = gen_factor(N, 2024); pre = prep(F)
        r_os = 3000 if N <= 1e4 else (800 if N <= 1e5 else (150 if N <= 1e6 else 40))
        r_sa = 800 if N <= 1e4 else (150 if N <= 1e5 else (20 if N <= 1e6 else 4))
        tos_l = []; tsa_l = []
        for e in routes:
            A_np = np.array(e[0]); B_np = np.array(e[1]); xbr_np = np.array(e[2])
            maxdiff = max(maxdiff, abs(cvar_os(e, F, pre, 0.9)
                                       - cvar_saa(e, F, xbr_np, A_np, B_np, 0.9)))
            tos_l.append(time_us(lambda e=e: cvar_os(e, F, pre, 0.9), r_os))
            tsa_l.append(time_us(lambda e=e, A_np=A_np, B_np=B_np, xbr_np=xbr_np:
                                 cvar_saa(e, F, xbr_np, A_np, B_np, 0.9), r_sa))
        tos = float(np.median(tos_l)); tsa = float(np.median(tsa_l))
        print(f"{N:>12} {tos:>17.2f} {tsa:>14.1f} {tsa/tos:>9.1f}x")
        lN.append(math.log10(N)); t_os.append(tos); lt_sa.append(math.log10(tsa))
    _, r2 = linfit(lN, t_os); s, _ = linfit(lN, lt_sa)
    print(f"  order-stat : time ~ a + b log10(N), R^2 = {r2:.4f}, "
          f"total rise {t_os[-1]/t_os[0]:.2f}x over {len(Ns)} decades")
    print(f"  sampled    : log10(time) ~ s log10(N), s = {s:.3f} (linear => 1)")
    print(f"  correctness over all routes and N: max |delta| = {maxdiff:.3e}\n")


def experiment_pricing(n=10, seed=12345, Q=65.0, Ns=(1000, 10000, 100000), repeats=3):
    I = make_instance(n, seed, Q=Q)
    print(f"[C] exact elementary pricing; n={n}, alpha={I['alpha']:.2f}, Q={I['Q']:.1f}, "
          f"median of {repeats} runs")
    print(f"{'N':>8} {'C':>8} {'feas':>7} {'len':>4} {'order-stat(s)':>14} {'sampled(s)':>12} "
          f"{'speed-up':>9} {'%CVaR sa':>9} {'%env os':>8}")
    for N in Ns:
        F = gen_factor(N, 2024); pre = prep(F)
        os_t = []; sa_t = []; env_f = []; cv_f = []
        for _ in range(repeats):
            a = run_exact_pricing(I, F, pre, 'os')
            b = run_exact_pricing(I, F, pre, 'saa')
            assert a['checks'] == b['checks'], (a['checks'], b['checks'])
            assert a['feasible_labels'] == b['feasible_labels']
            assert abs(a['best_rc'] - b['best_rc']) <= 1e-8
            assert a['best_route'] == b['best_route']
            os_t.append(a['total']); sa_t.append(b['total'])
            env_f.append(a['envelope'] / a['total']); cv_f.append(b['cvar'] / b['total'])
        to = float(np.median(os_t)); ta = float(np.median(sa_t))
        print(f"{N:>8} {a['checks']:>8} {a['feasible_labels']:>7} {a['max_len']:>4} "
              f"{to:>14.3f} {ta:>12.3f} {ta/to:>8.1f}x "
              f"{100*np.median(cv_f):>8.0f}% {100*np.median(env_f):>7.0f}%")
    print(f"  optimum route {a['best_route']}, reduced cost {a['best_rc']:.4f}")
    print("  no label dominance; matching checks, feasible labels, optimal route and")
    print("  reduced cost are asserted between the two evaluators.")
    print("  envelope construction is a common additive cost in both runs: it dilutes")
    print("  the ratio rather than cancelling, so the speed-up understates the kernel gain.\n")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "--selftest":
        sys.exit(selftest())
    elif arg == "--pricing-only":
        experiment_pricing()
    elif arg == "--quick":
        experiment_pieces(2000)
        experiment_kernel(20, (1000, 10000, 100000, 1000000))
        experiment_pricing(repeats=1)
    else:
        experiment_pieces()
        experiment_kernel()
        experiment_pricing()
