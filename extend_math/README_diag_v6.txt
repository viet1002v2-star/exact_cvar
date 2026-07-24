Dethloff exact multivariate CVaR certificate — diagnostic v6
============================================================

This is a self-contained copy of the solver/diagnostic. It does not import the
older v5 file. v6 keeps all v5 spectral/coherence diagnostics and adds the
missing residual-direction factor

    C_e(u) = empirical CVaR_alpha(||(I-u u^T)(X-mu)||).

Why this matters
----------------
The M-coherence theorem controls J_gamma(u), but the actual certificate
half-width is gamma_r(u) C_e(u). At population level:

    Psi(u) = C_e(u)^2 J_gamma(u).

v6 therefore reports:

  Ce_u                         C_e at the trained SLOPE direction
  Ce_best_found_min/max        numerical multistart directional extrema
  Ce_best_found_spread         max/min over the best-found search
  Ce_rho_best_found            Ce_u / best-found minimum (descriptive)
  Ce_global_lower_bound        rigorous direction-uniform lower bound
  Ce_rho_certified_upper       Ce_u / rigorous lower bound (conservative)
  certificate_factor_best_found
                               sqrt(mu_max R_M) * Ce_rho_best_found
  certificate_factor_certified_upper
                               sqrt(mu_max R_M) * Ce_rho_certified_upper

The best-found minimum is numerical, so certificate_factor_best_found is an
observed diagnostic rather than a proof. The global lower bound is rigorous:

  C_e(u) >= mean ||e(u)||
         >= mean ||e(u)||^2 / max_i ||X_i-mu||
         >= [tr(Sigma)-lambda_1(Sigma)] / max_i ||X_i-mu||.

The rigorous bound may be loose; it is included to keep the logical status of
both reported factors explicit.

Recommended final 40-instance run (Windows CMD)
------------------------------------------------

del diagnostics_width40_*.csv 2>nul

python dethloff_affine_certified_alns_diag_v6.py ^
  dir=Dethloff ^
  profiles=moderate ^
  N=10000 ^
  d=10 ^
  iters=5 ^
  reps=1 ^
  policy=SAA ^
  direction=SLOPE ^
  diagnostic=1 ^
  diag_sample_rate=0 ^
  diag_tier2=0 ^
  diag_best_sample=0 ^
  diag_ce=1 ^
  diag_ce_random=2048 ^
  diag_ce_batch=64 ^
  diag_ce_local_starts=6 ^
  diag_ce_local_steps=7 ^
  diag_ce_local_trials=24 ^
  diag_out_prefix=diagnostics_width40

The global C_e directional search is reused across all 40 instances because the
factor sample X is identical under the same N, d, support, alpha and seed. On a
10,000 x 10 sample it took about five seconds in the build smoke test.

Summary command
---------------

python -c "import pandas as pd; d=pd.read_csv('diagnostics_width40_spectral_summary.csv'); x=d[d.label=='search_call_weighted']; cols=['Ce_u','Ce_best_found_min','Ce_best_found_max','Ce_best_found_spread','Ce_rho_best_found','Ce_global_lower_bound','Ce_rho_certified_upper','theorem_factor_exact_observed','certificate_factor_best_found','certificate_factor_certified_upper']; print(x[cols].describe(percentiles=[.5,.9,.95]).to_string())"

Interpretation discipline
-------------------------

- Ce_rho_best_found and certificate_factor_best_found are descriptive because
  the directional minimum is found numerically.
- Ce_rho_certified_upper and certificate_factor_certified_upper are rigorous but
  may be conservative.
- Do not call either quantity a universal constant. They are functionals of the
  realized empirical factor sample and search stream.
