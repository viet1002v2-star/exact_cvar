Directed information-cut pilot v2 audit

Changes from v1
---------------
1. The CLI parameter batch now controls incumbent information cuts.
2. Incumbent cuts are added only when genuinely violated at the current integer solution.
3. Only proper contiguous subpaths of an exact-route-violated incumbent are considered.
4. CSV diagnostics include candidate/violated/added counts, median coefficient retention,
   and the final-route information gap (Qbar-QF)/Qbar.

Why this matters
----------------
In v1, batch affected only root separation. With root=0 it had no effect, explaining why
batch=5,10,25 produced identical cut counts and node counts. V1 also added every positive
subpath coefficient encountered after an exact-route violation, without checking whether its
cut was currently violated. Those cuts are valid, but this is proactive cut enrichment rather
than actual separation.

Local rerun on CON3-0, n=8, routes=2, seeds 0..2, p in {0.50,0.80,0.95,0.99}
--------------------------------------------------------------------------------
Proper violated-cut separation (v2):
- positive node reduction: 5/12 cells
- median node reduction: 0.0%
- mean node reduction: 6.15%
- time wins: 5/12 cells
- median speedup: 0.981x
- pooled speedup: 0.937x (HYBRID 6.7% slower)
- seed 2: no violated information subpath cut was found in any p cell

Interpretation
--------------
The large v1 node reductions came mainly from proactively adding valid but not currently
violated subpath cuts. That is not incorrect, but it is a different algorithmic mechanism.
Under actual separation, the signal is weak on this pilot.
