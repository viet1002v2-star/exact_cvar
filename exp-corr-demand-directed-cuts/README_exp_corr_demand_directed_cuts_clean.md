# Correlated-Demand Directed Information Cuts

> **Status: archived / no-go as a standalone research direction**

This repository records an exploratory attempt to extend path-based lower-bounding cuts for the Vehicle Routing Problem with Stochastic Demands (VRPSD) from independent demands to a common-factor correlated-demand setting.

The branch produced a valid mathematical correction and several clean counterexamples. However, the resulting cuts did not provide sufficiently consistent computational gains in the pilot experiments, so the direction was not continued as a standalone paper.

## 1. Research idea

Under independent customer demands, the optimal-restocking recourse cost satisfies a superadditivity property. This property is useful because it supports strong path-based cuts.

With correlated demands, observations made on the first part of a route update the conditional distribution of the remaining demands. The later part of the route can therefore use information revealed earlier.

The research question was:

> Can the loss of superadditivity under correlated demands be quantified and repaired in a way that still gives useful exact cuts?

## 2. Demand model

The toy model uses one binary latent factor shared by all customers.

Each demand is either low or high. A parameter controls how strongly the demands move together through the latent factor, while keeping every individual customer’s marginal demand distribution unchanged.

This allows the experiments to isolate the effect of dependence rather than changing the average demand or variance of each customer.

The recourse model includes:

- failure restocking;
- preventive restocking;
- Bayesian updating after observing demands;
- metric travel costs;
- nonnegative demands;
- preventive-restocking penalties no larger than failure penalties.

## 3. Main mathematical finding

Under independence, the tested superadditivity property holds.

Under common-factor correlation, it can fail even after choosing the better of the two route orientations.

The failure mechanism is informational:

- a route prefix may have zero standalone recourse cost;
- the same prefix can still reveal useful information about the latent factor;
- this information lets the suffix make better restocking decisions;
- as a result, the concatenated path can have lower recourse cost than the sum of the two standalone paths.

The exact toy enumeration found:

- no violations under independence;
- persistent violations under correlation;
- a maximum best-orientation relative violation of about **11.1%** in the tested regime;
- a minimal-total-length counterexample with four customers in the toy setting.

## 4. Directed repair

The difficulty comes partly from using an undirected route coefficient that takes the better of the two orientations.

The proposed repair keeps the route orientation explicit.

For a directed path, the lower-bound coefficient is defined as the expected recourse cost when the latent factor is revealed before departure.

This coefficient has three useful properties:

- it depends only on the directed path;
- it is valid regardless of which route prefix comes before it;
- it is inexpensive to compute when the latent factor has only a few states.

In the toy experiments, this directed coefficient retained most of the original recourse value:

- median retention was approximately **98%–100%**;
- no tested coefficient became zero;
- the worst observed retention was roughly **51%**.

## 5. Important correction

The directed information cuts are valid lower-bounding cuts, but they do **not** by themselves reproduce the exact recourse function.

Exact full-route cuts are still required for correctness.

The tested hybrid formulation therefore used:

- exact full-route cuts to guarantee convergence and correctness;
- directed information subpath cuts to strengthen the relaxation.

## 6. Computational pilot

The pilot used small Dethloff-derived instances:

- Dethloff customer geometry;
- delivery-demand magnitudes rescaled into a simplified VRPSD setting;
- 7- and 8-customer subsets;
- two routes;
- exact full-route cuts;
- directed information subpath cuts.

This is not a full simultaneous pickup-and-delivery model. Dethloff was used only as a source of realistic geometry and demand magnitudes.

Two cut strategies were tested.

### Version 1: proactive cut enrichment

After encountering a route, the solver added several valid subpath cuts even when those cuts were not currently violated.

This version often reduced the branch-and-bound node count. Some cells showed reductions above 40% or 50%.

However:

- runtime improvements were inconsistent;
- several cells became slower;
- the result depended strongly on the selected subset and seed;
- the benefit looked more like generic proactive cut-pool enrichment than a clean consequence of the information correction.

### Version 2: violated-cut separation

The callback was then corrected so that it:

- considered only proper contiguous subpaths;
- added only currently violated cuts;
- reported candidate, violated, and added cuts separately;
- used the batch parameter correctly.

Under this cleaner protocol:

- only about half of the tested cells reduced the node count;
- the median node reduction was approximately zero;
- the median speedup was below 1;
- total runtime was worse than the baseline;
- some instances produced no violated information cuts at all.

## 7. Why the branch was stopped

The mathematics survived:

- correlation genuinely breaks undirected superadditivity;
- the best route orientation does not remove all violations;
- a clean directed lower bound can be derived;
- the coefficient is path-specific and computationally tractable.

The algorithmic story did not survive strongly enough:

- exact route cuts are still required;
- root separation produced no useful cuts in the pilot;
- violated-only separation did not give stable node or runtime improvements;
- proactive enrichment occasionally helped, but the effect was inconsistent;
- the evidence was not strong enough for an EJOR/COR-level algorithmic contribution.

## 8. Final conclusion

The strongest defensible conclusion is:

> Correlated demand can invalidate undirected superadditivity because a route prefix reveals information about the remaining demands. A directed information-based coefficient restores a valid lower bound, but the resulting cuts were not computationally strong enough in the pilot to justify a standalone algorithmic project.

This repository is kept as:

- a reproducible negative result;
- a record of the counterexamples and corrected theory;
- a warning that strong coefficient retention does not automatically imply useful cuts;
- a possible starting point for a future formulation with cheaper or stronger directed separation.

## 9. Repository status

No further tuning, larger Dethloff experiments, or solver engineering is planned for this branch.

Suggested repository or branch name:

```text
exp-corr-demand-directed-cuts
```
