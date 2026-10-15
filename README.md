# WC-Graph: Weighted, Length-Stratified Conformal Certification of Variable-Length Agent Trajectories

Code, data pipeline, results and figures for the paper
*"Weighted Conformal Certification of Tool-Using Agents over Variable-Length Action Graphs:
Coverage Bounds, Algorithms, and Complexity"* (prepared for the Computer Science Journal of Moldova).

> **Scope of the experiments.** All results were produced on CPU without access to large language models.
> The five real benchmarks are **hierarchical sequential agents**: an agent walks a class taxonomy, and at
> every internal node a probabilistic "tool" picks the next step. This is the variable-length
> action-graph setting of the paper, but it is **not** an LLM agent. A sixth, synthetic tool-use environment
> has a known density ratio and is used to test the theory. `wcgraph/traces.py` runs the identical pipeline
> on exported LLM-agent logs (JSONL), so an LLM study can be added without changing the method.

---

## 1. Problem

An agent answers an input *x* by following a root-to-terminal path τ in a rooted action graph *G = (V, E)*.
The length |τ| differs from one trajectory to another. Each step *e* carries a score
*s<sub>e</sub>(x) = 1 − p<sub>e</sub>(x)*, where *p<sub>e</sub>* is the agent's probability for that step.

We want a set *C(x)* of trajectories that

1. contains the correct trajectory τ\* with probability ≥ 1 − α **for every trajectory length** (not only on
   average over lengths), and
2. remains valid when deployment inputs follow a different distribution from calibration inputs
   (covariate shift: *Q<sub>X</sub> ≠ P<sub>X</sub>*, with *P(τ | x)* unchanged).

**Why this matters.** In our experiments, every method that calibrates marginally under-covers the longest
trajectories, by 3–11 points, while its marginal coverage looks correct (Fig. 2). Long, multi-step
trajectories are the ones an agent is most likely to get wrong, so this is the failure that marginal
guarantees hide.

## 2. Method (WC-Graph)

1. **Trajectory score.** *S(x, τ) = 1 − ∏<sub>e∈τ</sub> (1 − s<sub>e</sub>(x))*. This is a monotone
   transformation of the additive score Σ −log p<sub>e</sub>(x).
2. **Length strata.** Lengths are merged into contiguous strata so that every stratum has an *expected*
   calibration size ≥ *n<sub>min</sub>* = 5/α. The expected sizes come from **training**-label frequencies,
   so the partition is fixed before any calibration label is seen (`length_strata`).
3. **Density-ratio weights.** *ŵ(x) = (n<sub>s</sub>/n<sub>t</sub>) · h(x)/(1 − h(x))*, where *h* is a domain
   classifier fitted on training inputs (source) against unlabeled target inputs. The classifier is an
   L2-regularised logistic regression with C chosen by 5-fold cross-validated log-loss. Weights are clipped at
   the 99th percentile of the source weights. All weighted methods, including baselines, use these same
   weights.
4. **Per-stratum weighted quantile** (Tibshirani et al., 2019). A candidate τ in stratum *k* is kept iff
   *S(x, τ) ≤ q<sub>k</sub>(x)*, where *q<sub>k</sub>(x)* is the (1 − α)-quantile of
   Σ<sub>i∈I<sub>k</sub></sub> p<sub>i</sub>(x) δ<sub>S<sub>i</sub></sub> + p<sub>n+1</sub>(x) δ<sub>+∞</sub>,
   *I<sub>k</sub>* is the set of calibration examples whose true trajectory lies in stratum *k*, and
   *p<sub>i</sub>(x) = ŵ(X<sub>i</sub>) / (Σ<sub>j∈I<sub>k</sub></sub> ŵ(X<sub>j</sub>) + ŵ(x))*.

Implementation: `wcgraph/methods.py::wc_graph`.

## 3. Theory

**Theorem 1 (coverage).** Let the stratum partition be fixed in advance.

- **(a) Known weights.** If *w = dQ<sub>X</sub>/dP<sub>X</sub>* is known, then for every stratum *k*:
  *Q(τ\* ∈ C(X) | κ(τ\*) = k) ≥ 1 − α*. Hence marginal coverage is also ≥ 1 − α.

  *Proof sketch.* Conditional on stratum *k*, the source law of *(X, τ\*)* is *P(·|k)*, and the target law is
  *Q(x, τ | k) ∝ w(x) P(x, τ | k)*. This is again a covariate shift, with a ratio proportional to *w*; the
  constant cancels in the normalised weights. Conditional on |I<sub>k</sub>|, the calibration points in stratum
  *k* are i.i.d. from *P(·|k)*. Weighted exchangeability (Tibshirani et al., 2019, Thm. 2) then gives the
  claim. Averaging over *k* gives marginal coverage.

- **(b) Estimated weights.** If *ŵ* is estimated independently of the calibration data, coverage in stratum
  *k* is ≥ 1 − α − Δ<sub>k</sub>, where
  Δ<sub>k</sub> = ½ E<sub>P(·|k)</sub> | ŵ/E ŵ − w/E w | (Lei & Candès, 2021, Thm. 3, applied per stratum).
  Clipping is part of *ŵ*, so its bias is included in Δ<sub>k</sub>.

**Theorem 2 (complexity of trajectory sets on a DAG).** For a fixed input, all with exact algorithms in
`wcgraph/graph.py`:

| Problem | Complexity |
|---|---|
| (a) Max-step score: membership, counting \|C(x)\|, answer set | O(\|V\| + \|E\|) |
| (a) Length-Mondrian thresholds (one per length): counting | O(D(\|V\| + \|E\|)) |
| (b) Additive score Σ s<sub>e</sub> ≤ q: answer set | O(\|V\| + \|E\|) (DAG shortest path) |
| (b) Additive score, per-length thresholds: answer set | O(D(\|V\| + \|E\|)) (min-sum DP over node × length) |
| (c) Additive score: counting \|C(x)\| | **#P-complete** |
| (c) Additive score, integer weights ≤ B: counting | O(\|E\| · B) (pseudo-polynomial) |

*Proof of (c).* `knapsack_reduction` gives a parsimonious reduction from #Knapsack. Each item becomes a
two-route gadget (skip: weight 0; take: weight *w<sub>i</sub>*). Paths are then in bijection with subsets, and
path weight equals subset weight, so counting paths with weight ≤ *B* counts knapsack solutions.

**Empirical check.** All seven algorithms agree with brute-force enumeration on 300/300 random DAGs each
(`results/complexity_correctness.csv`). Counting 1.47 M trajectories takes 0.08 ms with the DP versus 6.0 s
by enumeration (Fig. 8).

## 4. Experimental protocol

### Benchmarks

| Benchmark | Action graph | Leaves | Path lengths | Shift |
|---|---|---|---|---|
| **20NG** (text, TF-IDF) | newsgroup-name hierarchy | 20 | 1–3 | **natural**: senders with `.edu` addresses (source) → all other senders (target) |
| **Letter** | Ward clustering of class centroids | 26 | 3–7 | exponential tilting on PC1, β = 1 |
| **PenDigits** | Ward | 10 | 3–4 | tilting, β = 1 |
| **MNIST** (20k subsample, PCA-50) | Ward | 10 | 2–6 | tilting, β = 1 |
| **SatImage** | Ward | 6 | 1–4 | tilting, β = 1 |
| **ToolGraph-Syn** (3 random graphs) | random tool tree, branching 2–4 | 29–44 | 1–5 | Gaussian mean shift δ, **known w** |

Agent accuracy on the target domain is 0.73–0.90 (`results/table1_datasets.csv`).

### Splits, held fixed for every method

- The agent is trained once per dataset (seed 0).
- Each repetition draws, from the held-out pool:
  - *n<sub>cal</sub>* = 1000 labelled source examples;
  - 1000 target test examples (772 for SatImage);
  - 1000 unlabeled target examples, used only for the weights.
- Methods that need a tuning split (TRAQ-Alloc, and the LA-score ablation) take 30% of the same 1000
  calibration examples.
- Number of repetitions:
  - main experiment: 100;
  - sweeps and agent-seed robustness: 50;
  - synthetic environment: 3 graphs × 100.

### Baselines

| # | Method | What it is |
|---|---|---|
| B1 | Indep-CP | per-step split CP at level α, with no joint guarantee |
| B2 | Bonferroni-CP | per-step split CP at α/D (union bound) |
| B3 | PASC | joint max-step score, single split-CP quantile (Kotte, 2026) |
| B4 | Answer-SCP | final-answer LAC score 1 − ∏ p, single split-CP quantile |
| B5 | TRAQ-Alloc | union bound with per-step budgets tuned by random search on a tuning split (TRAQ-style; Li et al., 2024) |
| B6 | MiCP-Alloc | union bound with geometrically decaying per-turn budgets (MiCP-style; Zhou et al., 2026) |
| B7 | WCP-Answer | weighted CP under covariate shift (Tibshirani et al., 2019) on the answer score |

B5 and B6 are *re-implementations of the allocation ideas* of TRAQ and MiCP adapted to this setting, not the
authors' original code.

**Ablations:**

| Ablation | What changes |
|---|---|
| −weights | no density-ratio weights |
| −Mondrian | no length stratification; this is exactly B7 |
| max-score | max-step score in place of the trajectory-likelihood score |
| LA-score | the first design we tried (see §7) |
| raw weights | domain classifier without cross-validation or clipping |
| oracle w | true density ratio (synthetic environment only) |

### Metrics and statistics (`experiments/analyze.py`)

**Metrics:**

- marginal coverage;
- mean set size (number of trajectories);
- **worst-length coverage (WLC)**, the minimum over lengths of the rep-averaged coverage given |τ\*|;
- the gap between the best and worst length;
- empty-set rate.

**Statistical tests:**

- *Validity:* one-sided t-test of H0: E[coverage] ≥ 1 − α, Holm-corrected within each dataset.
- *Comparisons:* paired Wilcoxon signed-rank tests (WC-Graph vs each baseline) on per-repetition set size
  and per-repetition WLC, Holm-corrected, with Cliff's δ as the effect size.
- *Ranking:* Friedman test with Nemenyi critical difference over all (dataset × α × β) blocks.

**Decision rules, fixed before the final runs:**

- A method is *valid* if mean coverage ≥ 1 − α − 0.01.
- A method is *length-valid* if WLC ≥ 1 − α − 0.02.
- The validity-aware ranking score is set size ÷ largest set size in the block for valid methods, and
  1 + coverage shortfall otherwise.

## 5. Results

### Main experiment (α = 0.1, target 0.90, 100 repetitions)

Each cell gives marginal coverage / WLC / mean set size. In the WC-Graph row, WLC is in bold, and set size is
in bold where it is the smallest set among methods that are both valid and length-valid. No method is
length-valid on 20NG.

| Method | 20NG | Letter | PenDigits | MNIST | SatImage |
|---|---|---|---|---|---|
| Indep-CP | 0.745 / 0.702 / 1.05 | 0.529 / 0.409 / 0.73 | 0.668 / 0.566 / 0.70 | 0.686 / 0.656 / 0.72 | 0.726 / 0.701 / 0.82 |
| Bonferroni-CP | 0.901 / 0.847 / 3.01 | 0.893 / 0.875 / 4.38 | 0.906 / 0.861 / 1.13 | 0.936 / 0.931 / 1.37 | 0.929 / 0.860 / 1.35 |
| PASC | 0.870 / 0.749 / 2.24 | 0.841 / 0.802 / 2.92 | 0.882 / 0.811 / 1.02 | 0.892 / 0.861 / 1.05 | 0.898 / 0.877 / 1.12 |
| Answer-SCP | 0.865 / 0.829 / 1.73 | 0.838 / 0.788 / 2.42 | 0.881 / 0.809 / 1.01 | 0.892 / 0.852 / 1.04 | 0.897 / 0.869 / 1.11 |
| TRAQ-Alloc | 0.895 / 0.800 / 2.81 | 0.884 / 0.866 / 4.44 | 0.900 / 0.846 / 1.12 | 0.937 / 0.935 / 1.56 | 0.931 / 0.926 / 1.33 |
| MiCP-Alloc | 0.896 / 0.780 / 2.61 | 0.867 / 0.831 / 5.00 | 0.892 / 0.829 / 1.09 | 0.916 / 0.886 / 2.03 | 0.923 / 0.823 / 1.47 |
| WCP-Answer | 0.866 / 0.830 / 1.74 | 0.901 / 0.868 / 3.86 | 0.902 / 0.842 / 1.06 | 0.898 / 0.863 / 1.06 | 0.895 / 0.865 / 1.10 |
| **WC-Graph (ours)** | 0.870 / **0.855** / 1.73 | 0.904 / **0.888** / **4.06** | 0.901 / **0.897** / **1.10** | 0.901 / **0.899** / **1.09** | 0.896 / **0.894** / **1.12** |

Full table with confidence intervals and p-values: `results/RESULTS_TABLES.md` (Table 2).

### Findings

1. **Length-conditional validity.** On Letter, PenDigits, MNIST and SatImage, WC-Graph is valid both marginally
   and at every trajectory length. Its worst-to-best length gap is 0.007–0.078, compared with 0.054–0.143 for
   PASC, Answer-SCP and WCP-Answer on the same four datasets.

   It is also the most efficient length-valid method:
   - **MNIST:** sets 30% smaller than TRAQ-Alloc and 20% smaller than Bonferroni (1.09 vs 1.56 and 1.37; Holm
     p = 2.7e-17, |δ| ≥ 0.998).
   - **SatImage:** sets 16% smaller than TRAQ-Alloc (1.12 vs 1.33).
   - **Letter and PenDigits:** WC-Graph is the only length-valid method.

2. **The price of the per-length guarantee is small.** Against the best marginal method under shift
   (WCP-Answer), WC-Graph's sets range from 0.3% smaller (20NG) to 5.2% larger (Letter). In return, its WLC is significantly higher on PenDigits,
   MNIST, SatImage and 20NG (Holm p ≤ 0.012). On Letter the per-repetition WLC is tied (p = 0.72), but the
   rep-averaged WLC is 0.888 vs 0.868.

3. **Weights are necessary under covariate shift.** On Letter, the unweighted single-quantile methods (PASC,
   Answer-SCP and the −weights ablation) fall to 0.84 coverage. Only the conservative union-bound methods stay
   near 0.89 without weights, and they pay for it with larger sets (Table 2, Table 5).

4. **Weight stabilisation is necessary** (Table 5b, `run_weight_stability.py`). With raw weights (logistic
   regression C = 1, no cross-validation, no clipping) on the same MNIST splits:
   - the effective calibration size fell below 10 in 11 of 100 repetitions (minimum 1.01);
   - WCP-Answer's mean coverage dropped to 0.858, with 0.000 in the worst repetition;
   - WC-Graph's worst repetition fell to 0.756 and its mean set size grew to 1.30.

   With stabilised weights, the minimum effective sample size is 679 and the worst repetition is 0.860. On
   the other four datasets the two estimators behave alike.

5. **Negative result: 20NG.** The `.edu` → other-sender shift is *not* a pure covariate shift. Agent accuracy
   drops from 0.82 to 0.75, and every non-conservative method, weighted or not, under-covers by 3–4 points.
   Only the union-bound methods stay within the tolerance (Bonferroni 0.901, at 1.7× the set size;
   TRAQ-Alloc 0.895; MiCP-Alloc 0.896), and **no method is length-valid**. Among the eight main methods,
   WC-Graph still has the highest WLC (0.855) and the smallest gap between lengths (0.035), at essentially the
   smallest set size (1.73; Answer-SCP 1.73). Theorem 1 does not apply to this shift, and the paper should say
   so.

6. **Synthetic graph with known w (Theorem 1).**
   - Coverage within every length stratum stays at 0.898–0.905 for WC-Graph with both oracle and estimated
     weights, at every shift level δ ∈ [0, 1.5].
   - PASC reaches only 0.78–0.82 and WCP-Answer 0.73–0.79.
   - Marginal coverage is never significantly below the Theorem-1 bound (one-sided p ≥ 0.86).
   - Clipped estimated weights are far more efficient than oracle weights at strong shift (set size 2.87 vs
     5.71 at δ = 1.5), at no loss of per-stratum coverage.
   - Caveat: the rarest length (5) is merged with length 4. Coverage of length 5 alone
     therefore drops to 0.73 at δ = 1.5. The guarantee is per *stratum*, and *n<sub>min</sub>* controls this
     trade-off (Fig. 10).

7. **Robustness.** The pattern holds for two further agent-training seeds (Table 10, 50 repetitions each).
   - WC-Graph's WLC is 0.853–0.900 across all 10 seed × dataset runs.
   - That is higher than PASC, Answer-SCP, WCP-Answer and MiCP-Alloc in every run.
   - The union-bound methods Bonferroni and TRAQ exceed it on MNIST, SatImage and 20NG (seed 2), with larger
     sets.

### Sweeps (50 repetitions per cell; Tables 6–9, Figs. 4, 5, 7, 10)

- **Shift strength** (β = 0 → 2, four tabular datasets):
  - WC-Graph's marginal coverage stays within 0.887–0.910 and its WLC within 0.882–0.902 in all 20
    (dataset, β) cells.
  - WCP-Answer's WLC is 0.837–0.879.
  - The unweighted single-quantile methods degrade steadily with β: on Letter, coverage falls from 0.90 to
    0.81 and WLC from 0.85 to 0.77.
  - With no shift (β = 0), WC-Graph's sets are 2–8% larger than the smallest (Answer-SCP). That is the cost of
    the per-length guarantee.
  - The weight estimator is not perfect: at β = 2 on SatImage every non-union-bound method, WC-Graph
    included, drops to 0.884–0.890.
- **Miscoverage level** (α ∈ {0.05, 0.1, 0.2}; *n<sub>min</sub>* = 5/α):
  - On Letter, PenDigits, MNIST and SatImage, WC-Graph's WLC stays within 0.02 of 1 − α at every level:
    0.945–0.952 at α = 0.05 and 0.780–0.804 at α = 0.2.
  - 20NG under-covers at every level, as expected for a shift that is not a covariate shift.
- **Overall ranking** (Friedman test over 31 blocks; Nemenyi CD = 1.89):
  - *Length-validity-aware score:* WC-Graph ranks first (average rank 1.35; Friedman p = 4e-29). It is
    significantly better than WCP-Answer (4.13), MiCP-Alloc (5.19), PASC (5.26), Answer-SCP (5.84) and
    Indep-CP (8.00). Its lead over Bonferroni-CP (2.94) and TRAQ-Alloc (3.29) is **not** significant at the
    Nemenyi level.
  - *Marginal-validity-aware score:* WCP-Answer ranks first (2.26) and WC-Graph second (3.39). The difference
    is not significant. This is expected, because WC-Graph trades a few percent of set size for the
    per-length guarantee.
- **Minimum stratum size**:
  - On the real datasets, results are identical for *n<sub>min</sub>* ∈ {10, 25, 50, 100} on four of five
    datasets.
  - Letter is the exception: WLC is 0.899 at *n<sub>min</sub>* ≤ 25 and 0.888 at 50–100.
  - At *n<sub>min</sub>* = 200, strata merge and WLC drops (Letter 0.875).
  - In synthetic graph 0, the rarest length has an expected 32 calibration points, so
    *n<sub>min</sub>* = 50 merges it with its neighbour. Keeping it separate would need
    *n<sub>min</sub>* ≤ 25, which leaves a small, heavily weighted stratum whose quantile is often infinite
    under strong shift.

## 6. Figures (`figures/`, 300 dpi PNG)

| File | Content |
|---|---|
| `fig1_main_coverage_size.png` | coverage and set size, all methods, all datasets (mean ± 95% CI) |
| `fig2_length_conditional_coverage.png` | coverage as a function of trajectory length |
| `fig3_size_vs_worst_length_coverage.png` | efficiency vs length-conditional validity |
| `fig4_shift_sweep.png` | marginal coverage and WLC vs tilt strength β |
| `fig5_alpha_calibration.png` | coverage vs nominal level 1 − α |
| `fig6_synthetic_theorem1.png` | synthetic graph: coverage vs shift, the Theorem-1 region, set size |
| `fig7_critical_difference.png` | Friedman/Nemenyi critical-difference diagrams |
| `fig8_complexity_runtime.png` | exact DP vs enumeration runtime (Theorem 2) |
| `fig9_ablations.png` | ablations |
| `fig10_nmin_sensitivity.png` | sensitivity to *n<sub>min</sub>* |
| `fig11_20ng_action_graph.png` | the 20NG action graph |

## 7. Development history (reported for transparency)

- **First design (now the LA-score ablation).** A single weighted quantile on a length-adjusted,
  depth-normalised max score, *(max<sub>e</sub> F<sub>d(e)</sub>(s<sub>e</sub>))<sup>|τ|</sup>*. A 20-repetition
  pilot showed that it inflated set sizes without removing the under-coverage of long trajectories. It was
  replaced by length stratification *before* the final runs, and it is reported as an ablation.
- **Minimum stratum size.** *n<sub>min</sub>* = 5/α was fixed a priori. Fig. 10 / Table 8 report the
  sensitivity to it; it was not tuned on test results.
- **Weight stabilisation.** Cross-validated regularisation and 99th-percentile clipping were added after the
  first full run exposed the MNIST weight collapse. They apply to all weighted methods, and every result above
  was re-run with them. The collapse itself is reproduced in `run_weight_stability.py` / Table 5b.

## 8. Reproduce

```bash
pip install -r requirements.txt
./run_all.sh                 # about 1.5–2 h on 2 CPU cores; downloads data automatically
python -m pytest -q          # 8 unit tests: DP vs brute force, #Knapsack reduction, CP validity, trace loader
```

Or open `notebooks/WC_Graph_experiments.ipynb`. It is stored executed, and includes a 20-repetition live
re-run.

## 9. Repository layout

```
wcgraph/            library
  graph.py          action graphs, exact DP algorithms, #Knapsack reduction (Theorem 2)
  conformal.py      split / weighted conformal quantiles, trajectory scores
  methods.py        WC-Graph, 7 baselines, ablations
  weights.py        stabilised density-ratio estimator, ESS, Delta bound
  data.py           datasets, taxonomies, hierarchical agents, shift protocols
  synthetic.py      tool-graph environment with known density ratio
  traces.py         adapter for real LLM-agent JSONL traces
  runner.py         repetition loops
  stats.py          tests (t, binomial, Wilcoxon, Holm, Cliff's delta, Friedman/Nemenyi)
experiments/        run_main.py, run_sweeps.py, run_synthetic.py, run_complexity.py,
                    run_weight_stability.py,
                    analyze.py, make_figures.py
results/            raw per-repetition CSVs, summary tables, RESULTS_TABLES.md
figures/            PNG figures
notebooks/          WC_Graph_experiments.ipynb (+ build_notebook.py)
tests/              pytest suite
```

## 10. Using real LLM-agent traces

Export one JSON line per episode. Each line holds:

- `steps`: the candidate tool calls at each decision point, with their probabilities (normalised log-probs or
  rollout frequencies);
- `gold_path`: the verified correct trajectory;
- `features`: a task embedding;
- `split`: `cal`, `test` or `unlabeled`.

Then:

```python
from wcgraph.traces import load_traces
from wcgraph.weights import DomainClassifierRatio
from wcgraph.methods import CalContext, wc_graph
tax, d = load_traces("my_agent_traces.jsonl")
w = DomainClassifierRatio().fit(d["cal"]["F"], d["unlabeled"]["F"])
ctx = CalContext(tax, 0.1, d["cal"]["S"], d["cal"]["y"], w(d["cal"]["F"]),
                 d["test"]["S"], w(d["test"]["F"]))
sets = wc_graph(ctx)          # boolean matrix: test episodes x candidate trajectories
```

The trace adapter builds a **tree** from observed prefixes. For DAG-shaped agents in which different paths
reach the same answer, use the answer-set algorithms in `graph.py`.

## 11. Limitations

- **No LLM agents** were evaluated (see the scope note at the top).
- **The guarantees assume covariate shift.** 20NG shows that coverage degrades under natural shifts that change
  *P(τ | x)*.
- **Rare lengths are merged.** The guarantee is per stratum, so a rare length inside a merged stratum can be
  under-covered (synthetic graph, length 5).
- **Δ cannot be computed on real data**, where the true density ratio is unknown. It is computed only in the
  synthetic environment.
- **Weights need a separate unlabeled target sample** (1000 inputs here).

## References

- Tibshirani, Barber, Candès, Ramdas (2019). Conformal prediction under covariate shift. *NeurIPS*.
- Barber, Candès, Ramdas, Tibshirani (2023). Conformal prediction beyond exchangeability. *Ann. Statist.*
- Lei, Candès (2021). Conformal inference of counterfactuals and individual treatment effects. *JRSS-B*.
- Vovk, Gammerman, Shafer (2005). *Algorithmic Learning in a Random World* (Mondrian CP).
- Sadinle, Lei, Wasserman (2019). Least ambiguous set-valued classifiers with bounded error levels. *JASA*.
- Li, Park, Lee, Bastani (2024). TRAQ: Trustworthy retrieval augmented question answering via conformal
  prediction. *NAACL*.
- Zhou et al. (2026). Adaptive stopping for multi-turn LLM reasoning (MiCP). arXiv:2604.01413.
- Kotte (2026). PASC: Pipeline-aware conformal prediction with joint coverage guarantees. arXiv:2605.18812.
- Demšar (2006). Statistical comparisons of classifiers over multiple data sets. *JMLR*.
- Romano, Le, Olson, Moore (PMLB). Penn Machine Learning Benchmarks.



License: MIT.
