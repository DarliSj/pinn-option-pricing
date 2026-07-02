# Pre-Stage 2: Walk-Forward Benchmark Plan

> **STATUS: HISTORICAL (Apr 2026) — completed and superseded.** The benchmark
> was executed and extended well beyond this plan (final grid: B0–B12 plus the
> Stage 2 2×2; results in `reports/master_table.csv`). Kept for provenance.
> Current plan of record: `docs/HYBRID_PARETO_PLAN.md`.

**Purpose:** Establish proper walk-forward benchmarks for ALL models before
implementing learnable volatility. The single Nov–Dec split was used to develop
the architecture; these 9-fold results are the real evaluation.

**Context:** We used a simpler single split (Jan–Oct / Nov–Dec) to develop our
architecture iteratively — testing standard MLP, then Modified MLP + RWF, then
tuning μ. Now we benchmark the final architecture (and variants) properly with
walk-forward validation, alongside all non-PINN baselines.

---

## Walk-Forward Configuration

- **Window:** Expanding (all data up to fold start)
- **Burn-in:** 3 months minimum (Jan–Mar)
- **Test folds:** Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec 2020
- **9 folds** covering: post-COVID recovery, summer, pre-election,
  post-election, TSLA S&P 500 inclusion
- **Reporting:** Final-epoch RMSE (no early stopping, no epoch selection)

---

## Reporting Protocol (READ THIS BEFORE FILLING IN ANY CELL)

Every model (PINN or baseline) is evaluated on the same 9 folds and reports
the same set of statistics. Read this section once and use the exact same
metric definitions for every row of the master table — comparing different
statistics across cells is silently misleading.

### The headline cell value: **Pooled RMSE**

For each model, the master table displays a single dollar number per fold
(the per-fold RMSE) and a **pooled** RMSE in the footer:

    pooled_rmse = sqrt( Σ_folds Σ_options residual² / Σ_folds n_test )
                = sqrt( Σ_folds sum_sq_err_fold      / total_n_test  )

Equivalently: concatenate the residuals from all 9 fold test sets and take
one RMSE over the combined ~44k options.

**Why pooled and not mean-of-folds?** Pooled treats every option equally
and is directly comparable to single-split RMSEs reported in the literature.
Mean-of-folds (= arithmetic mean of 9 per-fold RMSEs) silently up-weights
small folds and is optimistic when folds are heteroskedastic — for the BS
baseline on this dataset, mean-of-folds = $6.95 vs pooled = $8.79, a non-
trivial gap purely from the metric definition. Pooled is the cell value.

### Secondary diagnostics (always report, separate from the headline)

Alongside the pooled cell, every model row also reports:

| Statistic | Definition | What it tells you |
|---|---|---|
| **Pooled RMSE** | as above | the headline number, model accuracy |
| **Mean(folds) ± Std(folds)** | mean and std of per-fold RMSEs | regime stability (low std = consistent across months) |
| **Worst-fold RMSE** | max over per-fold RMSEs | robustness to hard regimes |
| **Per-fold RMSEs** | all 9 values | regime analysis — which months are hard for which models |
| **drift_gap** (PINN only) | final-epoch − best-epoch RMSE per fold | late-training stability — large positive gap = pathological drift |

The full per-fold breakdown is **required**, not optional. The whole point
of walk-forward is to see which months are hardest and how each model
degrades across regimes — a single aggregate number throws that away.

### Final-epoch only (no epoch selection)

All RMSEs in the master table are computed on the **final-epoch** model
state. PINN runners track best-epoch alongside as a stability diagnostic
(`drift_gap = final − best`), but the best-epoch number is **never**
restored or reported as the cell value. Reporting best-epoch on a
test set would be test-set epoch selection bias — explicitly forbidden by
CLAUDE.md "Methodology Rules" and `docs/CLAUDE_CODE_BRIEFING.md` §11.4.

---

## Part A: Non-PINN Baselines (fast, run first)

### A1. Black-Scholes (constant-σ)

No training. Per fold: compute scalar σ_fixed (median ATM IV from training
data) and scalar r_fixed (median train interest rate), evaluate the BS
analytical formula on every option in the test set, return per-fold
RMSE/MAE plus the per-fold sum-of-squared-errors needed for pooling.

Reference implementation: `src/baselines.py::bs_baseline_fold`. Runner:
`run_bs_baseline.py` (writes `results/bs_baseline/results.json`).

```python
def bs_baseline_fold(train_df, test_df):
    config = compute_constants(train_df)              # scalar σ_fixed, r_fixed
    bs_prices = bs_call_price(
        S=test_df["forward_price"].values,
        K=test_df["strike_price"].values,
        T=test_df["time_to_exp"].values,
        r=config["r_fixed"], sigma=config["sigma_fixed"],
    )
    residuals = test_df["mid_price"].values - bs_prices
    sum_sq  = float(np.sum(residuals ** 2))           # for pooling across folds
    sum_abs = float(np.sum(np.abs(residuals)))
    n_test  = len(test_df)
    return {
        "rmse_mkt":  float(np.sqrt(sum_sq / n_test)),
        "mae_mkt":   float(sum_abs / n_test),
        "sum_sq_err": sum_sq, "sum_abs_err": sum_abs,
        "sigma_fixed": config["sigma_fixed"], "r_fixed": config["r_fixed"],
        "n_train": len(train_df), "n_test": n_test,
    }
```

Runtime: seconds total.

#### Measured baseline (recorded 2026-04-09)

| Statistic | Value |
|---|---|
| **Pooled RMSE (headline)** | **$8.79** |
| Mean(folds) ± Std(folds) | $6.95 ± $4.15 |
| Worst-fold RMSE | $15.15 (Sep 2020) |
| Best-fold RMSE | $2.58 (May 2020) |
| Total options pooled | 43,984 |

Per-fold breakdown:

| Fold | σ_fixed | RMSE | MAE | n_test |
|---|---|---|---|---|
| Apr2020 | 0.729 | $3.05 | $2.38 | 3,094 |
| May2020 | 0.776 | $2.58 | $1.53 | 3,188 |
| Jun2020 | 0.738 | $2.72 | $1.63 | 3,265 |
| Jul2020 | 0.712 | $8.87 | $6.40 | 5,083 |
| Aug2020 | 0.733 | $5.32 | $3.36 | 5,167 |
| **Sep2020** | 0.737 | **$15.15** | $12.96 | 6,361 |
| Oct2020 | 0.776 | $5.43 | $3.97 | 5,968 |
| Nov2020 | 0.772 | $7.35 | $5.42 | 6,420 |
| **Dec2020** | 0.755 | **$12.12** | $9.06 | 5,438 |

The hard regimes are **Sep 2020** (TSLA S&P 500 inclusion announcement,
extreme idiosyncratic vol) and **Dec 2020** (post-election plus year-end
inclusion-effect spillover). Apr–Jun 2020 are easy because realized vol
in those months was actually below σ_fixed → BS happens to mis-price by
small amounts. Any PINN cell in the master table should be compared to
$8.79 pooled, not the per-fold mean of $6.95 and *certainly* not the
legacy single-split figure described below.

#### ⚠ Legacy "$17.23" figure — DO NOT use as the BS baseline

Earlier project documents (`docs/CLAUDE_CODE_BRIEFING.md`, `docs/TODO.md`,
`notebooks/stage0_pinn.ipynb`) report a BS RMSE of **$17.23** on the single
Jan–Oct/Nov–Dec split. This is **not** the same baseline that §A1 specifies.

The notebook computes `bs_theoretical` per row using
`r=df["interest_rate"]` and `sigma=df["daily_vol_proxy"]` — i.e. a
**1-day-lagged daily mean IV across all contracts**, not a constant
ATM-fit σ. That daily mean is contaminated by the volatility smile (deep
OTM/ITM IVs inflate the mean), which biases the BS prices upward and
produces large residuals.

Reproducing the exact same Jan–Oct/Nov–Dec split with the **constant-σ**
baseline (this section's `bs_baseline_fold`) gives **$10.64**, not $17.23.
σ_fixed matches the notebook's reported value (0.771868) to six decimal
places, so the discrepancy is entirely in which BS pricing model is being
called the "baseline".

**Implication:** every claim in the legacy docs of the form "PINN beats
BS by ~50%" was beating the smile-contaminated daily-IV-proxy pricer, not
a properly fit constant-σ BS. The corrected benchmark bar to clear is
**$8.79**, which is meaningfully harder.

The legacy GAM ($12.57) and laGP ($9.63) numbers from the same documents
are also suspect — they were either evaluated on the same biased baseline
or used `daily_vol_proxy` as a feature themselves. When the R code is
adapted for walk-forward (A2/A3), GAM and laGP must be re-evaluated on the
same preprocessed dataset and against the constant-σ BS reference, and
their cells in the master table must be derived fresh.

### A2. GAM

Adapt R code from `STA-325-option.qmd`. Same features as original report:
moneyness, time_to_exp, daily_vol_proxy, log_volume.

Runtime: minutes total.

### A3. laGP

Adapt R code. k=50 nearest neighbors. Also record 95% coverage and
interval width (needed later for Stage 3 UQ comparison).

Runtime: minutes total.

---

## Part B: PINN Benchmarks

The 6 PINN configs form a deliberate 2×2 ablation grid (with the right
column extended over μ ∈ {0.5, 0.75, 1.0} for μ selection):

|              | Physics-only (no data loss) | Hybrid (data + PDE)           |
|---           |---                          |---                            |
| **Std MLP**  | **B0** (Std physics)        | **B1** (Std hybrid)           |
| **Mod MLP**  | **B2** (Mod physics μ=1)    | **B3 / B4 / B5** (Mod hybrid μ=0.5/0.75/1.0) |

The grid lets you read off three clean ablations:

1. **Initial introduction of data on the naive arch:** B0 → B1.
   Same architecture (Std MLP + Fourier + grad-norm), only the data loss is
   added. This is the "what happens when you naively introduce market data"
   experiment — the motivation for everything that follows.
2. **Initial introduction of data on the improved arch:** B2 → B5 (μ=1).
   Same architecture (Modified MLP + RWF μ=1 + Fourier + grad-norm), only
   the data loss is added. This is the "does the architectural fix actually
   make data introduction work?" experiment.
3. **Architecture contribution at each mode:** B0 vs B2 (physics column)
   and B1 vs B5 (hybrid column at μ=1). Isolates whether Modified MLP + RWF
   matters in physics-only training (probably less) vs in hybrid training
   (probably much more).

The headline narrative from this grid is supposed to be:
**B2 establishes physics-only convergence to BS** (sanity check) →
**B0 confirms physics-only also works for the naive arch** (the architecture
fix isn't doing magic in physics mode) →
**B1 shows naive arch + data drifts/struggles** (the motivation for the
fix is now isolated, with B0 as the matched control) →
**B3/B4/B5 show improved arch + data is stable and selects μ** (the
resolution).

### B0. Standard MLP, Physics Mode (matched control for B1)

**Question answered:** Does the naive architecture converge to BS in
physics-only mode? Establishes the baseline RMSE that the *same* network
achieves when no market data is in the loss — the matched control for B1.
Without B0, the "B1 drifts when you add data" claim has no clean control.

```bash
python run_walk_forward.py --mode physics --epochs 12000 --arch standard
```

Architecture: Fourier features + standard `nn.Linear` (no RWF) + standard
MLP (no Modified MLP encoders) + grad-norm balancing. Same network as B1,
just trained without the data loss term.

**Expected behavior:** Should converge to a reasonable RMSE_bs across folds
(no data competition, so the late-training drift pathology shouldn't appear).
If it *does* drift in physics mode too, that's a finding — architecture
matters even without data competition. If it converges cleanly, that
isolates the data introduction as the cause of B1's pathology.

Config: LR=1e-3, 12k epochs, standard MLP with Fourier features. No data
loss term, so per-epoch cost is slightly lower than B1.
Runtime: ~2.5–3 hours.

### B1. Standard MLP, Hybrid Mode (the "naive PINN" with data)

**Question answered:** What does the naive architecture do when market
data is introduced into the loss? Paired with **B0** as the matched
"+ data" arm of the naive-arch ablation.

```bash
python run_walk_forward.py --mode hybrid --epochs 12000 --arch standard
```

Architecture: Identical to B0 (Fourier + plain `nn.Linear` + standard MLP +
grad-norm). The **only** difference vs B0 is the addition of the market
data loss term. This is essentially the original notebook architecture
before the drift fix.

**Important:** This model drifts on the single split. With walk-forward,
some folds may drift too. Report final-epoch RMSE regardless — the drift
itself is a finding, and the per-fold `drift_gap` column will quantify it.
Compare per-fold RMSE and drift_gap to B0 to isolate the effect of
introducing data on the same architecture.

Config: LR=1e-3, 12k epochs, standard MLP with Fourier features.
Runtime: ~3 hours.

### B2. Modified MLP + RWF, Physics Mode (matched control for B5)

**Question answered:** What does physics-only training achieve with the
improved architecture? Establishes that the Modified MLP + RWF reaches BS
without seeing any market data — the matched control for B5.

```bash
python run_walk_forward.py --mode physics --epochs 15000 --rwf_mu 1.0
```

Architecture: Modified MLP encoders (U/V gating) + RWF (μ=1.0) + Fourier +
grad-norm. Same network as B5, just trained without the data loss term.

Config: μ=1.0 (proven stable for physics), LR=1e-3, 15k epochs.
Runtime: ~3 hours.

### B3. Modified MLP + RWF, Hybrid Mode, μ=0.5

**Question answered:** Our best single-split config — does it generalize?

```bash
python run_walk_forward.py --mode hybrid --epochs 12000 --rwf_mu 0.5
```

Config: μ=0.5, LR=1e-3, 12k epochs.
Runtime: ~3 hours.

**Save checkpoints per fold** — these are the warm-start weights for Stage 2.

### B4. Modified MLP + RWF, Hybrid Mode, μ=0.75

**Question answered:** Does the middle-ground μ perform differently across
market regimes?

```bash
python run_walk_forward.py --mode hybrid --epochs 12000 --rwf_mu 0.75
```

Config: μ=0.75, LR=1e-3, 12k epochs.
Runtime: ~3 hours.

### B5. Modified MLP + RWF, Hybrid Mode, μ=1.0

**Question answered:** Does the slow-start μ catch up when evaluated
per-fold? Does it win in volatile regimes (Apr/May post-COVID) where
stronger PDE scaffolding might matter more?

```bash
python run_walk_forward.py --mode hybrid --epochs 15000 --rwf_mu 1.0
```

Config: μ=1.0, LR=1e-3, 15k epochs (needs more epochs due to slow start).
Runtime: ~4 hours.

---

## Part C: Analysis

### C1. Master Benchmark Table

Cells are **per-fold final-epoch RMSE in dollars**. The footer carries the
three aggregate statistics defined in the Reporting Protocol section above:
**Pooled (headline), Mean(folds) ± Std(folds), Worst-fold**. All three are
required — the pooled row is the headline metric, mean±std and worst-fold
are the stability/robustness diagnostics used in §C3 for μ selection.

Columns are grouped by the 2×2 design: non-PINN baselines, then the
Std-MLP physics→hybrid pair (B0/B1), then the Mod-MLP physics→hybrid pair
(B2/B3/B4/B5). Reading the table left-to-right within each row tells the
"add data, then improve architecture" story.

```
Walk-Forward Results: TSLA 2020 Call Options — RMSE($), final-epoch
=================================================================================================
            | non-PINN baselines     | Std MLP         | Mod MLP — physics |  Mod MLP — hybrid
            |                        | (naive arch)    | & matched hybrid  |  μ ablation
Fold        |  BS   |  GAM  | laGP   |  B0     |  B1   |  B2     |  B5     |  B3     |  B4
            |       |       |        |  Phys   | Hybrid|  Phys   | Hybrid  | Hybrid  | Hybrid
            |       |       |        |  Std    |  Std  |  μ=1.0  |  μ=1.0  |  μ=0.5  |  μ=0.75
------------|-------|-------|--------|---------|-------|---------|---------|---------|--------
Apr 2020    |  3.05 |       |        |         |       |         |         |         |
May 2020    |  2.58 |       |        |         |       |         |         |         |
Jun 2020    |  2.72 |       |        |         |       |         |         |         |
Jul 2020    |  8.87 |       |        |         |       |         |         |         |
Aug 2020    |  5.32 |       |        |         |       |         |         |         |
Sep 2020    | 15.15 |       |        |         |       |         |         |         |
Oct 2020    |  5.43 |       |        |         |       |         |         |         |
Nov 2020    |  7.35 |       |        |         |       |         |         |         |
Dec 2020    | 12.12 |       |        |         |       |         |         |         |
------------|-------|-------|--------|---------|-------|---------|---------|---------|--------
Pooled      |  8.79 |       |        |         |       |         |         |         |   ← headline
Mean(folds) |  6.95 |       |        |         |       |         |         |         |
Std(folds)  |  4.15 |       |        |         |       |         |         |         |
Worst-fold  | 15.15 |       |        |         |       |         |         |         |
```

The two highlighted ablations (read down within a column-pair):

- **B0 → B1**  (Std MLP physics → Std MLP hybrid): introduces the data
  loss while holding the naive architecture fixed. Δ in pooled RMSE and
  per-fold drift_gap quantifies "what does introducing data do to a naive
  PINN?"
- **B2 → B5**  (Mod MLP physics → Mod MLP hybrid μ=1): introduces the data
  loss while holding the improved architecture fixed. Δ in pooled RMSE
  quantifies "does the architectural fix make data introduction work?"

The two architecture comparisons (read across between physics pair / hybrid pair):

- **B0 vs B2** (physics column): architecture contribution **without** data.
- **B1 vs B5** (hybrid column at μ=1): architecture contribution **with** data.

#### Companion stability table (PINN cells only)

`drift_gap = final − best` per fold, recorded by `run_walk_forward.py` in
`run_info`. Large positive values indicate the model peaked early and
degraded — the original drift pathology that the Modified MLP + RWF
architecture was meant to fix. Tracking this per-fold (not just on average)
shows whether any fold destabilizes a given configuration.

```
PINN Stability — drift_gap ($) per fold (final − best, smaller is better)
==================================================================================
Fold        |  B0     |  B1     |  B2     |  B5     |  B3     |  B4
            | Std     | Std     | Mod     | Mod     | Mod     | Mod
            | Phys    | Hybrid  | Phys    | Hybrid  | Hybrid  | Hybrid
            |         |         | μ=1.0   | μ=1.0   | μ=0.5   | μ=0.75
------------|---------|---------|---------|---------|---------|--------
Apr 2020    |         |         |         |         |         |
...         |         |         |         |         |         |
Dec 2020    |         |         |         |         |         |
------------|---------|---------|---------|---------|---------|--------
Mean        |         |         |         |         |         |
Worst       |         |         |         |         |         |
```

Reading the drift_gap table is the cleanest way to see the data-introduction
pathology and its resolution: B0 should have ~0 drift_gap everywhere
(physics-only is stable), B1 should show large positive drift_gap on at
least some folds (the original pathology), B2 should be ~0, and B3/B4/B5
should be ~0 (the architectural fix resolved it). If B5 has nontrivial
drift_gap on any fold, μ=1.0 is not actually stable on that regime even
with the improved architecture, and μ selection should weight that heavily.

### C2. Key Questions the Table Answers

1. **Does architecture matter?** Two clean comparisons read directly off
   the 2×2 grid:
   - **In physics-only mode:** B0 (Std physics) vs B2 (Mod physics μ=1).
     Same training objective, different architecture. If RMSE_bs is similar,
     the Modified MLP + RWF machinery is doing little in physics mode —
     consistent with the hypothesis that the architectural fix matters
     specifically because of *data competition*, not because of the PDE
     residual itself.
   - **In hybrid mode:** B1 (Std hybrid) vs B5 (Mod hybrid μ=1). Same
     training objective, different architecture. This is where the
     architecture is expected to matter most. A large gap here (especially
     with B1 showing positive `drift_gap` and B5 showing ~0) is the
     headline architecture-validation result.

2. **What does introducing market data do to each architecture?** This is
   the "initial data introduction" experiment, also read off the grid:
   - **Naive arch:** B0 → B1 (Std physics → Std hybrid). Δ pooled RMSE
     and per-fold `drift_gap` together quantify whether adding data
     destabilizes the naive PINN. Expectation: B1 drifts on at least
     some folds (the original pathology), confirming the motivation for
     the architectural fix.
   - **Improved arch:** B2 → B5 (Mod physics μ=1 → Mod hybrid μ=1). Same
     transition for the improved architecture. Expectation: B5 improves
     on B2 (data helps because the model can now use it without drifting)
     and `drift_gap` stays small across folds.

   These two paired transitions are the core finding of the benchmark
   table. B0 is included specifically to make B0 → B1 a *matched control*
   ablation (same architecture, with vs without data) rather than a
   between-architectures confound.

3. **Does μ selection generalize?** Compare μ=0.5 vs μ=0.75 vs μ=1.0 across folds.
   - If μ=0.5 wins most folds: single-split selection confirmed
   - If μ varies by regime: report this as a finding about PDE scaffolding depth
   - If μ=1.0 wins volatile folds (Apr/May): interesting — stronger PDE
     scaffolding helps when market data is noisier

4. **Which folds are hardest?** All models will struggle on certain months
   (probably Apr 2020 post-crash, Dec 2020 S&P inclusion). This is regime
   analysis that strengthens the paper.

5. **Does PINN beat GAM/laGP?** The core baseline comparison. **Note:** the
   legacy laGP figure of $9.63 and GAM figure of $12.57 from the briefing
   are not directly comparable — they were derived against the daily-IV-
   proxy BS baseline (see §A1 ⚠ box) and may also use `daily_vol_proxy` as
   a feature. They MUST be re-derived against the same constant-σ BS
   reference and the same 9 walk-forward folds before they can enter the
   master table. Until that recompute lands, leave the GAM/laGP columns
   blank rather than dropping in the legacy numbers.

### C3. μ Selection for Stage 2

After the table is complete, select the μ for Stage 2 based on:

1. **Pooled RMSE** across all 9 folds (primary criterion — the headline cell).
2. **Stability:** Std(folds) of RMSE (lower = more uniform across regimes).
3. **Robustness:** Worst-fold RMSE (which μ degrades least on Sep/Dec).
4. **Late-training drift:** distribution of `drift_gap` across folds (large
   positive gaps anywhere = the architecture is unstable on that regime).

If one μ dominates all four criteria, use it for Stage 2. If there is a
trade-off (e.g. μ=0.5 best pooled but μ=0.75 best worst-fold and lowest
drift on hard months), report all four columns and pick based on which
criterion matters for the downstream application — Stage 2's learnable
volatility module is more sensitive to drift on hard regimes than to a
small pooled-RMSE difference, so I'd weight stability/drift heavily.

**μ is not pre-committed.** Earlier docs claimed μ=0.5 wins; that came from
single-split best-epoch metrics and is not load-bearing. The walk-forward
sweep over {0.5, 0.75, 1.0} is the actual selection step.

The selected μ's per-fold checkpoints become the Stage 2 warm-start weights.

---

## Implementation Notes

### Standard MLP (used by B0 and B1)

Need to add to `src/model.py`:

```python
class StandardPricingNet(nn.Module):
    """Original architecture: Fourier + standard MLP, no Modified MLP, no RWF."""
    def __init__(self, hidden_dims=None, fourier_features=64, fourier_scale=1.0):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64, 64, 64]
        self.embedding = FourierFeatureEmbedding(in_dim=2, n_features=fourier_features, scale=fourier_scale)
        layers = []
        in_dim = self.embedding.out_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, m, tau):
        if m.dim() == 1: m = m.unsqueeze(1)
        if tau.dim() == 1: tau = tau.unsqueeze(1)
        x = torch.cat([m, tau], dim=1)
        x = self.embedding(x)
        return self.net(x)
```

Add `--arch` flag to `run_walk_forward.py`:
```python
parser.add_argument("--arch", choices=["standard", "modified"], default="modified")
```

### Parallel Execution

B0–B5 are independent. If you have access to multiple machines or can run
overnight across multiple nights:

- Night 1: B0 (std physics) + B1 (std hybrid) — completes the naive-arch ablation pair
- Night 2: B2 (mod physics) + B5 (mod hybrid μ=1) — completes the improved-arch ablation pair
- Night 3: B3 (mod hybrid μ=0.5) + B4 (mod hybrid μ=0.75) — completes the μ sweep

This ordering also means each night yields a *complete* ablation result on
its own, so you can review intermediate findings before queuing the next
night's runs (e.g. if B1 doesn't drift on any fold, the architecture story
needs to be re-thought before launching B2–B5).

Or if single machine: queue them sequentially (~18–19 hours total).

---

## Output Structure

Directory names mirror the runner's `run_tag` (`{arch}_{mode}` for standard,
`{arch}_{mode}_mu{rwf_mu}` for modified). The `results.json` in each
directory has the `{folds, summary}` structure produced by
`run_walk_forward.py`, including the pooled-RMSE summary block consumed by
the master-table compiler.

```
results/
├── walk_forward_benchmarks.csv
├── walk_forward_benchmarks.md
├── bs_baseline/
│   └── results.json
├── gam_baseline/                       ← deferred (R-side recompute)
│   └── results.json
├── lagp_baseline/                      ← deferred (R-side recompute)
│   └── results.json
├── standard_physics/                   ← B0 (matched control for B1)
│   ├── results.json
│   └── fold_*.pt
├── standard_hybrid/                    ← B1
│   ├── results.json
│   └── fold_*.pt
├── modified_physics_mu1.0/             ← B2 (matched control for B5)
│   ├── results.json
│   └── fold_*.pt
├── modified_hybrid_mu0.5/              ← B3
│   ├── results.json
│   └── fold_*.pt          ← Stage 2 warm-start candidate
├── modified_hybrid_mu0.75/             ← B4
│   ├── results.json
│   └── fold_*.pt          ← Stage 2 warm-start candidate
└── modified_hybrid_mu1.0/              ← B5
    ├── results.json
    └── fold_*.pt          ← Stage 2 warm-start candidate
```

---

## Completion Criteria

Before starting Stage 2:
- [x] Part A1 complete: BS baseline on all 9 folds (pooled $8.79)
- [ ] Part A2/A3 deferred: GAM and laGP need R-side walk-forward adaptation
- [ ] Part B complete: all **6** PINN configurations (B0–B5) on all 9 folds
- [ ] Both physics→hybrid ablations verified: B0→B1 and B2→B5 deltas
      computed and documented
- [ ] Master table compiled (C1) with the 2×2 grid layout
- [ ] drift_gap stability table compiled and inspected per-fold
- [ ] μ selected for Stage 2 based on walk-forward results (C3)
- [ ] Selected μ's per-fold checkpoints saved for warm-start
- [ ] Training stability confirmed for all PINN folds (no large positive
      drift_gap on any fold for the selected μ)

---

## Estimated Total Compute

| Run | Model | Config | Runtime | Pair |
|-----|-------|--------|---------|------|
| A1 | BS | — | seconds | — |
| A2 | GAM | R script (deferred) | minutes | — |
| A3 | laGP | R script (deferred) | minutes | — |
| **B0** | PINN Std MLP Physics | 12k epochs | ~2.5–3 hours | matched ctrl for B1 |
| B1 | PINN Std MLP Hybrid | 12k epochs | ~3 hours | + data on naive arch |
| B2 | PINN Mod MLP Physics | μ=1.0, 15k ep | ~3 hours | matched ctrl for B5 |
| B3 | PINN Mod MLP Hybrid | μ=0.5, 12k ep | ~3 hours | μ sweep |
| B4 | PINN Mod MLP Hybrid | μ=0.75, 12k ep | ~3 hours | μ sweep |
| B5 | PINN Mod MLP Hybrid | μ=1.0, 15k ep | ~4 hours | + data on improved arch |
| **Total** | | | **~18.5–19 hours + R** | |
