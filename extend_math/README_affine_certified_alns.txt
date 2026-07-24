DETHLOFF AFFINE CERTIFIED ALNS

The main script is self-contained. It copies the parser and ALNS/ILS core from
dethloff_runner.py and does not import or modify that file.

The uncertainty overlay is multivariate affine because the exact certificate
requires max-affine route load. Dethloff distances, customer mean demands, and
capacity are retained exactly.

1. Audit one small run
----------------------
python dethloff_affine_certified_alns.py ^
  dir=Dethloff max=1 profiles=moderate ^
  N=5000 iters=5 reps=1 audit

Audit computes FULL on every certified decision and asserts equality. Do not
use audit timings as the fair solver comparison because the extra checks are
deliberately included in wall time.

2. Fair four-instance gate
--------------------------
del results_affine_certified_alns.csv 2>nul

python dethloff_affine_certified_alns.py ^
  dir=Dethloff ^
  regex="(CON3-0|CON8-0|SCA3-0|SCA8-0)$" ^
  profiles=concentrated,moderate,diffuse ^
  N=50000 d=10 iters=50 reps=3 ^
  policy=both direction=SLOPE

3. Summarize
------------
python summarize_affine_certified_alns.py

4. Full 40-instance run
-----------------------
del results_affine_certified_alns.csv 2>nul

python dethloff_affine_certified_alns.py ^
  dir=Dethloff ^
  profiles=concentrated,moderate,diffuse ^
  N=50000 d=10 iters=50 reps=5 ^
  policy=both direction=SLOPE

Protocol
--------
- same scenarios and same random seed;
- fixed outer iterations, not wall-clock stopping;
- randomized FULL/CERT order per repetition;
- equal route-check counts asserted;
- equal final plan and objective asserted;
- preprocessing reported separately and conservatively included in
  solver_speedup_with_prep.
