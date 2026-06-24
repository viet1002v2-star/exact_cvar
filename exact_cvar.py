
import sys, math, bisect, time
import numpy as np

# --------------------------- upper envelope of lines y = a + b x ---------------------------
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

def peak_eval(env, x):
    A, B, xbr = env
    k = bisect.bisect_right(xbr, x) - 1
    if k < 0: k = 0
    if k >= len(A): k = len(A) - 1
    return A[k] + B[k] * x

# --------------------------- CVaR: vectorized sample-average baseline (O(N)) ---------------------------
def cvar_saa(env, Fsorted, xbr_np, A_np, B_np, alpha):
    N = len(Fsorted)
    m = max(1, math.ceil((1.0 - alpha) * N))
    k = np.searchsorted(xbr_np, Fsorted, side='right') - 1
    np.clip(k, 0, len(A_np) - 1, out=k)
    vals = A_np[k] + B_np[k] * Fsorted
    part = np.partition(vals, N - m)[N - m:]
    return float(part.mean())

def cvar_os(env, Fsorted, pre, alpha):
    A, B, xbr = env
    N = len(Fsorted)
    m = max(1, math.ceil((1.0 - alpha) * N))
    def phi(i):                              # peak at i-th smallest factor; O(log p)
        x = Fsorted[i]
        k = bisect.bisect_right(xbr, x) - 1
        if k < 0: k = 0
        if k >= len(A): k = len(A) - 1
        return A[k] + B[k] * x
    lo, hi = 0, N - 1
    while hi - lo > 2:
        m1 = lo + (hi - lo) // 3
        m2 = hi - (hi - lo) // 3
        if phi(m1) < phi(m2): hi = m2 - 1
        else:                  lo = m1 + 1
    istar = min(range(lo, hi + 1), key=phi)
    q, r = istar + 1, N - 1 - istar          # |left arm|, |right arm|
    def Aarm(k):                             # A_1=phi(0) .. A_q=phi(istar)  (non-increasing)
        if k <= 0: return math.inf
        if k > q:  return -math.inf
        return phi(k - 1)
    def Barm(k):                             # B_1=phi(N-1) .. B_r=phi(istar+1)  (non-increasing)
        if k <= 0: return math.inf
        if k > r:  return -math.inf
        return phi(N - k)
    # largest pL in [max(0,m-r), min(m,q)] with Aarm(pL) >= Barm(m-pL+1)
    a, b, best = max(0, m - r), min(m, q), max(0, m - r) - 1
    while a <= b:
        mid = (a + b) // 2
        if Aarm(mid) >= Barm(m - mid + 1):
            best = mid; a = mid + 1
        else:
            b = mid - 1
    pL = max(best, max(0, m - r))
    pR = m - pL
    # tail sums over index ranges [0,pL-1] and [N-pR,N-1], piecewise via prefix sums
    def rangesum(iLo, iHi):
        if iLo > iHi: return 0.0
        tot = 0.0; i = iLo
        while i <= iHi:
            x = Fsorted[i]
            k = bisect.bisect_right(xbr, x) - 1
            if k < 0: k = 0
            if k >= len(A): k = len(A) - 1
            xr = xbr[k + 1]
            # last index j with Fsorted[j] < xr
            j = bisect.bisect_left(Fsorted, xr) - 1
            if j > iHi: j = iHi
            if j < i:   j = i
            cnt = j - i + 1
            sumF = pre[j + 1] - pre[i]
            tot += A[k] * cnt + B[k] * sumF
            i = j + 1
        return tot
    s = rangesum(0, pL - 1) + rangesum(N - pR, N - 1)
    return s / m

# --------------------------- sample helpers ---------------------------
def gen_factor(N, seed):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(N)
    np.clip(z, -2.5, 2.5, out=z)
    z.sort()
    return z

def prep(Fsorted):
    pre = np.concatenate(([0.0], np.cumsum(Fsorted)))
    return pre

# --------------------------- instance + pricing labeling (counts CVaR calls) ---------------------------
def make_instance(n, seed, dominance=0.85):
    # dominance = fraction of customers whose net sensitivity shares the common sign;
    # a single dominant factor (the paper's premise) makes net sensitivities mostly same-sign,
    # so cumulative sensitivities are near-monotone. dominance=0.5 is the no-dominant-factor case.
    rng = np.random.default_rng(seed)
    X = rng.random(n + 1) * 100
    Y = rng.random(n + 1) * 100
    dist = np.hypot(X[:, None] - X[None, :], Y[:, None] - Y[None, :])
    da = np.zeros(n + 1); db = np.zeros(n + 1)
    na = np.zeros(n + 1); nb = np.zeros(n + 1); dual = np.zeros(n + 1)
    for i in range(1, n + 1):
        dmean = 8 + rng.random() * 8
        pmean = 8 + rng.random() * 8
        sens  = 1.5 + rng.random() * 2.5
        da[i] = dmean
        db[i] = sens * (0.5 + rng.random())
        na[i] = pmean - dmean
        nb[i] = sens * (1.0 if rng.random() < dominance else -1.0)   # mostly same sign
        dual[i] = 6 + rng.random() * 10
    return dict(n=n, dist=dist, da=da, db=db, na=na, nb=nb, dual=dual,
                alpha=0.90, Q=70.0)

def route_envelope(I, length):
    tda = tdb = 0.0; pn = (0.0, 0.0); pns = [(0.0, 0.0)]
    for j in range(1, length + 1):
        tda += I['da'][j]; tdb += I['db'][j]
        pn = (pn[0] + I['na'][j], pn[1] + I['nb'][j]); pns.append(pn)
    peak = [(tda + p[0], tdb + p[1]) for p in pns]
    return upper_envelope(peak)

def count_pricing_calls(I, Fsorted, pre):
    """Exact forward labeling, elementary, capacity-pruned. Returns #CVaR evaluations."""
    n = I['n']; alpha = I['alpha']; Q = I['Q']
    dist = I['dist']; da = I['da']; db = I['db']; na = I['na']; nb = I['nb']; dual = I['dual']
    calls = 0
    bestrc = {}
    # label: (node, vis, tda, tdb, pn_list, dist, redcost)
    start = (0, 0, 0.0, 0.0, [(0.0, 0.0)], 0.0, 0.0)
    stack = [start]
    while stack:
        node, vis, tda, tdb, pn, d, rc = stack.pop()
        for j in range(1, n + 1):
            bit = 1 << (j - 1)
            if vis & bit:
                continue
            ntda = tda + da[j]; ntdb = tdb + db[j]
            last = pn[-1]
            npn = (last[0] + na[j], last[1] + nb[j])
            peak = [(ntda + p[0], ntdb + p[1]) for p in pn]
            peak.append((ntda + npn[0], ntdb + npn[1]))
            env = upper_envelope(peak)
            calls += 1
            cv = cvar_os(env, Fsorted, pre, alpha)     # counted robust-capacity check
            if cv > Q:
                continue
            nrc = rc + dist[node][j] - dual[j]
            nvis = vis | bit
            key = node_key(j, nvis, n)
            if key in bestrc and bestrc[key] <= nrc + 1e-12:
                continue
            bestrc[key] = nrc
            stack.append((j, nvis, ntda, ntdb, pn + [npn], d + dist[node][j], nrc))
    return calls

def node_key(node, vis, n):
    return node * (1 << n) + vis

# --------------------------- timing ---------------------------
def time_us(fn, reps):
    t0 = time.perf_counter()
    s = 0.0
    for _ in range(reps):
        s += fn()
    t1 = time.perf_counter()
    return (t1 - t0) / reps * 1e6

def linfit(x, y):
    x = np.asarray(x); y = np.asarray(y)
    mx, my = x.mean(), y.mean()
    sxy = ((x - mx) * (y - my)).sum()
    sxx = ((x - mx) ** 2).sum()
    syy = ((y - my) ** 2).sum()
    slope = sxy / sxx
    b0 = my - slope * mx
    ssr = ((y - (b0 + slope * x)) ** 2).sum()
    r2 = 1 - ssr / syy if syy > 0 else 1.0
    return slope, r2

# --------------------------- main ---------------------------
def selftest():
    rng = np.random.default_rng(7)
    worst = 0.0
    for t in range(2000):
        L = 2 + int(rng.integers(0, 12))
        lines = [(float(rng.uniform(-50, 50)), float(rng.uniform(-5, 5))) for _ in range(L)]
        env = upper_envelope(lines)
        A_np = np.array(env[0]); B_np = np.array(env[1]); xbr_np = np.array(env[2])
        N = 200 + int(rng.integers(0, 4000))
        F = gen_factor(N, 1000 + t); pre = prep(F)
        alpha = 0.80 + float(rng.integers(0, 15)) * 0.01
        a = cvar_os(env, F, pre, alpha)
        b = cvar_saa(env, F, xbr_np, A_np, B_np, alpha)
        assert abs(a - b) < 1e-8, f"mismatch {a} vs {b}"
        worst = max(worst, abs(a - b))
    print("[selftest] order-stats vs sample-average CVaR")
    print(f"  random cases : 2000")
    print(f"  max |delta|  : {worst:.3e}")
    print(f"  {'PASS (machine precision)' if worst < 1e-8 else 'FAIL'}")

def main():
    print()

    # ===================== (A) the crux: is p small, and flat in route length? =====================
    def p_sample(pool, nroutes, rng):
        ncust = pool['n']; out = []; lens = []
        for _ in range(nroutes):
            L = int(rng.integers(2, 51))
            perm = rng.permutation(np.arange(1, ncust + 1))[:L]
            tda = tdb = 0.0; pn = (0.0, 0.0); pns = [(0.0, 0.0)]
            for j in perm:
                tda += pool['da'][j]; tdb += pool['db'][j]
                pn = (pn[0] + pool['na'][j], pn[1] + pool['nb'][j]); pns.append(pn)
            peak = [(tda + a, tdb + b) for (a, b) in pns]
            out.append(len(upper_envelope(peak)[0])); lens.append(L)
        return np.array(out), np.array(lens)

    dom = make_instance(80, 4242, dominance=0.85)   # dominant common factor (the paper's premise)
    mix = make_instance(80, 4242, dominance=0.50)   # no dominant factor (contrast)
    pc, lens = p_sample(dom, 10000, np.random.default_rng(20240))
    pcm, _   = p_sample(mix, 10000, np.random.default_rng(20241))
    q  = np.percentile(pc,  [50, 90, 99]); qm = np.percentile(pcm, [50, 90, 99])
    hist = {v: int((pc == v).sum()) for v in range(int(pc.min()), int(pc.max()) + 1)}
    print("[A] piece-count distribution over 10000 random routes (length 2..50)")
    print(f"  dominant (0.85): mean={pc.mean():.2f} median={q[0]:.0f} p90={q[1]:.0f} p99={q[2]:.0f} max={pc.max()}")
    print(f"  mixed   (0.50): mean={pcm.mean():.2f} median={qm[0]:.0f} p90={qm[1]:.0f} p99={qm[2]:.0f} max={pcm.max()}")
    print( "  histogram (dominant) p:count -> " + "  ".join(f"{k}:{v}" for k, v in hist.items()))
    print( "  p vs route length (dominant): ", end="")
    for lo, hi in [(2, 9), (10, 19), (20, 29), (30, 39), (40, 50)]:
        m = (lens >= lo) & (lens <= hi)
        print(f"[{lo}-{hi}]={pc[m].mean():.1f}", end="  ")
    print()
    # Sparre Andersen check: mean p  vs  H_k + 1  (k = mean route length in the bucket)
    H = lambda n: sum(1.0 / j for j in range(1, int(n) + 1))
    print("  mean p vs Sparre-Andersen H_k+1 (k = bucket mean length):")
    print("    %-10s %6s %12s %10s" % ("length", "kbar", "H_k+1 pred", "mean p"))
    for lo, hi in [(2, 9), (10, 19), (20, 29), (30, 39), (40, 50)]:
        m = (lens >= lo) & (lens <= hi)
        kbar = lens[m].mean(); pred = H(round(kbar)) + 1.0
        print(f"    {f'{lo}-{hi}':<10} {kbar:6.1f} {pred:12.2f} {pc[m].mean():10.2f}")
    print("    %% tab:hk rows -> " + " | ".join(
        f"{lo}-{hi} & $\\approx {round(lens[(lens>=lo)&(lens<=hi)].mean())}$ & "
        f"${H(round(lens[(lens>=lo)&(lens<=hi)].mean()))+1:.1f}$ & "
        f"${pc[(lens>=lo)&(lens<=hi)].mean():.1f}$"
        for lo, hi in [(2, 9), (10, 19), (20, 29), (30, 39), (40, 50)]))
    print(f"  %% paper: dominant median {q[0]:.0f}, p99 {q[2]:.0f}, max {pc.max()}; "
          f"mixed p99 {qm[2]:.0f}\n")

    # ===================== (B) Experiment A: kernel scaling, MEDIAN over many routes =====================
    # Times are medians over NROUTES random routes (not one cherry-picked route).
    # NOTE: with NROUTES=100 this loop can take a couple of minutes (the sampled pass at N=1e6-1e7
    # is slow by construction); lower NROUTES if needed -- medians are stable from ~30 routes.
    NROUTES = 100
    poolA = make_instance(60, 777, dominance=0.85)
    ncA = poolA['n']; rngA = np.random.default_rng(31415)
    routes = []
    for _ in range(NROUTES):
        L = int(rngA.integers(5, 41))
        perm = rngA.permutation(np.arange(1, ncA + 1))[:L]
        tda = tdb = 0.0; pn = (0.0, 0.0); pns = [(0.0, 0.0)]
        for j in perm:
            tda += poolA['da'][j]; tdb += poolA['db'][j]
            pn = (pn[0] + poolA['na'][j], pn[1] + poolA['nb'][j]); pns.append(pn)
        peak = [(tda + a, tdb + b) for (a, b) in pns]
        routes.append(upper_envelope(peak))
    pmed = int(np.median([len(e[0]) for e in routes]))
    print(f"[B] kernel scaling, MEDIAN over {NROUTES} random routes (median p={pmed}), alpha=0.90")
    print(f"{'N':>12} {'os med (us)':>13} {'saa med (us)':>14} {'speedup':>10}")
    Ns = [100, 1000, 10000, 100000, 1000000, 10000000]
    rows_scaling = []; lN = []; t_os = []; lt_saa = []; maxdiff = 0.0
    for N in Ns:
        F = gen_factor(N, 2024); pre = prep(F)
        reps_os  = 3000 if N <= 1e4 else (800 if N <= 1e5 else (150 if N <= 1e6 else 40))
        reps_saa = 800  if N <= 1e4 else (150 if N <= 1e5 else (20  if N <= 1e6 else 4))
        tos_l = []; tsaa_l = []
        for e in routes:
            A_np = np.array(e[0]); B_np = np.array(e[1]); xbr_np = np.array(e[2])
            maxdiff = max(maxdiff, abs(cvar_os(e, F, pre, 0.9) - cvar_saa(e, F, xbr_np, A_np, B_np, 0.9)))
            tos_l.append(time_us(lambda e=e: cvar_os(e, F, pre, 0.9), reps_os))
            tsaa_l.append(time_us(lambda e=e, A_np=A_np, B_np=B_np, xbr_np=xbr_np:
                                  cvar_saa(e, F, xbr_np, A_np, B_np, 0.9), reps_saa))
        tos = float(np.median(tos_l)); tsaa = float(np.median(tsaa_l))
        print(f"{N:>12} {tos:>13.3f} {tsaa:>14.1f} {tsaa/tos:>9.1f}x")
        rows_scaling.append((N, tos, tsaa, tsaa / tos))
        lN.append(math.log10(N)); t_os.append(tos); lt_saa.append(math.log10(tsaa))
    os_slope, os_r2 = linfit(lN, t_os); saa_slope, _ = linfit(lN, lt_saa)
    print(f"  os : time ~ a + b*log10(N), R^2 = {os_r2:.4f}")
    print(f"  saa: log10(time) ~ s*log10(N), s = {saa_slope:.3f} (linear => 1)")
    print(f"  correctness over all routes/N: max|delta| = {maxdiff:.3e}\n")

    # ===================== (C) Experiment B: full pricing, both kernels =====================
    # Envelope is rebuilt from scratch each check in BOTH runs, so its cost is common and cancels
    # in the speedup; we also report the share of pricing time spent in CVaR and in envelope build.
    def run_pricing(I, F, pre, kernel):
        n = I['n']; alpha = I['alpha']; Q = I['Q']
        dist = I['dist']; da = I['da']; db = I['db']; na = I['na']; nb = I['nb']; dual = I['dual']
        checks = 0; t_env = 0.0; t_cvar = 0.0; bestrc = {}
        stack = [(0, 0, 0.0, 0.0, [(0.0, 0.0)], 0.0, 0.0)]
        t0 = time.perf_counter()
        while stack:
            node, vis, tda, tdb, pn, d, rc = stack.pop()
            for j in range(1, n + 1):
                bit = 1 << (j - 1)
                if vis & bit: continue
                ntda = tda + da[j]; ntdb = tdb + db[j]
                last = pn[-1]; npn = (last[0] + na[j], last[1] + nb[j])
                peak = [(ntda + a, ntdb + b) for (a, b) in pn]
                peak.append((ntda + npn[0], ntdb + npn[1]))
                te = time.perf_counter(); e = upper_envelope(peak); t_env += time.perf_counter() - te
                checks += 1
                tc = time.perf_counter()
                if kernel == 'os':
                    cv = cvar_os(e, F, pre, alpha)
                else:
                    cv = cvar_saa(e, F, np.array(e[2]), np.array(e[0]), np.array(e[1]), alpha)
                t_cvar += time.perf_counter() - tc
                if cv > Q: continue
                nrc = rc + dist[node][j] - dual[j]; nvis = vis | bit
                key = node_key(j, nvis, n)
                if key in bestrc and bestrc[key] <= nrc + 1e-12: continue
                bestrc[key] = nrc
                stack.append((j, nvis, ntda, ntdb, pn + [npn], d + dist[node][j], nrc))
        return time.perf_counter() - t0, t_env, t_cvar, checks

    Ipr = make_instance(10, 12345)
    print(f"[C] full pricing pass, both kernels; instance n=10, alpha={Ipr['alpha']:.2f}, Q={Ipr['Q']:.1f}")
    print(f"{'N':>10} {'C':>7} {'os tot(s)':>11} {'saa tot(s)':>11} {'speedup':>8} "
          f"{'%CVaR_saa':>10} {'%CVaR_os':>9} {'%env_os':>8}")
    rows_B = []
    for N in [1000, 10000, 100000]:
        F = gen_factor(N, 2024); pre = prep(F)
        tos_t, env_os, cv_os, C  = run_pricing(Ipr, F, pre, 'os')
        tsa_t, env_sa, cv_sa, C2 = run_pricing(Ipr, F, pre, 'saa')
        pe_saa = 100 * cv_sa / tsa_t; pe_os = 100 * cv_os / tos_t; pen_os = 100 * env_os / tos_t
        print(f"{N:>10} {C:>7} {tos_t:>11.3f} {tsa_t:>11.3f} {tsa_t/tos_t:>7.1f}x "
              f"{pe_saa:>9.0f}% {pe_os:>8.0f}% {pen_os:>7.0f}%")
        rows_B.append((N, C, tos_t, tsa_t, tsa_t / tos_t, pe_saa))
    print("  (envelope built from scratch each check in BOTH runs => common cost, cancels in speedup;")
    print("   %CVaR_saa shows CVaR is the pricing bottleneck, removed by the order-statistics kernel.)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        selftest()
    else:
        main()
