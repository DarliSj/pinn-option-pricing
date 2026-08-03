# PINN Option Pricing — Project TODO

*Last updated: 2026-08-03. Authoritative next-steps plan:
`docs/HYBRID_PARETO_PLAN.md` (§4 resequenced: balancer first). Advisor
discussion doc: `docs/TRAINING_VALIDATION_DISCUSSION.md`. This file tracks
status + actionable items.*

## Where we are

Staged PINN framework for TSLA 2020 call-option pricing under a strict
walk-forward protocol (9 monthly folds, Apr–Dec 2020).

- **Stage 0 (constant-σ dev split):** DONE — architecture settled.
- **Stage 1 (walk-forward ablation B0–B12 + baselines):** DONE — all configs run,
  master table built (`reports/master_table.csv`).
- **Stage 2 (learnable σ, 2×2: B10/B12 × C-Vol/A-Vol):** DONE — all four cells run.
- **Active workstream: R1 (regime-conditioned pricing net)** — W0/W1 are done and
  redirected the plan. The balancer defect was real but not the root cause; the
  root cause is a missing input (the net is never told the vol regime). R1 is
  implemented and queued; R2 extends it to Stage 2. W2–W7 follow.
- **Stage 3 (UQ):** deferred — only after the point model is sound.

### Headline results (current)

| | mean-fold RMSE | pooled RMSE | butterfly% | calendar% |
|---|---|---|---|---|
| BS (per-fold σ) | 6.96 | 8.79 | — | — |
| B10 mod-hyb μ=0.75 +warmup (best RMSE) | 6.75 | 8.49 | 32 | 19 |
| B1 standard hybrid | 6.95 | 8.70 | 6.6 | 0.6 |
| S2_B12 / A-Vol (cleanest arb) | 7.52 | 9.41 | 8.2 | 0.27 |

**The gap:** no single model beats BS on RMSE _and_ is arbitrage-clean. Low RMSE
comes with severe arbitrage violations (B10 calendar max-neg hits −301 in some
folds); clean arbitrage comes with worse-than-BS RMSE.

### Key findings from the 2026-06-02 diagnostic dive

- Adding the data loss to the modified architecture _worsens_ test RMSE
  (physics ~8.76 → hybrid ~10.0) AND explodes arbitrage — constant-σ physics and
  smile-bearing quotes genuinely conflict; a high-capacity net resolves it by
  breaking convexity.
- The Modified-MLP+RWF stack is itself an arbitrage source: physics-mode butterfly
  is 3.7% (standard) vs 16/25/38% (modified μ=0.5/0.75/1.0). **μ is an arbitrage
  knob and μ=1.0 (chosen for "stability") is the worst.**
- No no-arbitrage term exists in the loss today — violations are measured, never
  trained against.
- B1's apparent edge is protocol-linked (val-best rescues its late drift), not a
  better model.

### RESULTS from W0 + W1 (ran 2026-08) — three findings, two corrections

**W1 (balancer): the defect is real and fixable, but it was NOT the root cause.**
λ_data runaway confirmed at **9.4×10⁴** (B6, gradnorm); ReLoBRaLo bounds it to
**~2** — five orders of magnitude. Warmup only *delays* it (B10's λ_data sits at
1 through warmup then climbs 1→309→780→946 the moment data switches on).
Test RMSE on Aug/Oct/Nov (μ=0.75):

| config | RMSE | bfly% | cal% |
|---|---|---|---|
| **B3 physics (matched control)** | **5.90** | 25.0 | 0.4 |
| B6 hybrid gradnorm (v1) | 7.56 | 36.4 | 39.7 |
| B6 hybrid **relobralo** | 6.76 | 24.8 | 18.8 |
| B6 hybrid renorm | 8.38 | 32.8 | 30.1 |
| B6 hybrid fixed | 6.79 | 22.1 | 17.9 |
| B10 (warmup) gradnorm (v1) | 6.25 | 36.6 | 16.8 |
| B10 (warmup) relobralo | 7.57 | 21.6 | 17.8 |

- **ReLoBRaLo is the pick**; **renorm is NOT viable** — Σλ renormalisation bounds
  the total but lets λ_pde *collapse* to ~2×10⁻⁴ (physics effectively switched
  off → the Aug blow-up at 11.65). It converts a runaway into a collapse.
- **Warmup and bounded balancing are substitutes, not complements** — bounding
  helps without warmup (7.56→6.76) and hurts with it (6.25→7.57).
- **Hybrid still loses to physics (5.90).** → root cause is elsewhere: see R1.

**W0 audit 1 — ⚠ CORRECTION: the butterfly metric is largely an artifact.**
Running the same finite-difference check on the *analytic BS* surface gives a
noise floor of 4.015 (the payoff kink). PINN excess over that floor: standard
+0.1%, μ=0.5 +1.4%, μ=0.75 +3.1%, μ=1.0 +15%. **The earlier claim that the
Modified-MLP is a primary arbitrage source is retracted for butterfly** — the
dramatic *rates* (3.7% vs 38%) are sign-flips in near-zero-curvature regions.
What IS real: μ=1.0 has 10× worse PDE fidelity (err 0.006 vs 0.0004) and genuine
calendar violations (floor is exactly 0.0, μ=1.0 = 0.031). μ=1.0 is a bad *PDE
solve* — that is the defensible claim.

**W0 audit 2 — ⚠ CORRECTION: Stage 2's σ_θ does not track market IV.**
corr(σ_θ, IV) ∈ [−0.26, +0.29]; beats a flat σ in only 3/8 cases (twice by
~0.0004); **B10/A-Vol learned a literally constant σ = 0.594**; B12/C-Vol spans
0.005. Stage 2 is doing **global σ recalibration, not smile learning**. The
paper's §5 "pressure-valve / absorbed substantial smile" narrative rests on grid
plots — at actual quote locations the variation nearly vanishes. **Paper revision
required.**

### Key findings from the training/validation dive (2026-06-03)

- **Selection regret is material and regime-dependent.** Each month is the test
  month in fold f and the validation month in fold f+1, so the val-selected vs
  directly-best gap is measurable from existing results: mean $0.57–1.18/month,
  max $3.14 (Oct, whose epoch was selected on Sep — the split outlier), min ≈ 0.
- **The config ranking flips with the selection rule** (reported: B12 best;
  selection-stripped: B1/B10 ahead, B12 near-worst). Accuracy and
  selection-robustness are being conflated — configs are not compared fairly.
- **Selection is seed-fragile:** two runs of identical B10 picked epochs 5,000
  vs 8,500 on the same fold.
- **Literature check (see discussion doc, Appendix B):** our balancer is the
  *unbounded* member of one of the weakest-performing families in published
  benchmarks; bounded loss-stats methods (ReLoBRaLo) and augmented-Lagrangian
  methods (PECANN/CAPU — the only family built for noisy-data + physics
  fusion) are the two live candidates.

---

## Active workstream — fix the foundation, then close the Pareto gap

Sequenced per `HYBRID_PARETO_PLAN.md` §4 (resequenced 2026-06-03: balancer
first). Advisor analysis: `TRAINING_VALIDATION_DISCUSSION.md`.

### W0 — Instrument + analysis-only audits (no training) — GATE
- [x] **Output versioning in place** (`EXPERIMENTS.md`): v1 frozen
      (`reports/v1/`); v2 routed via `RUN_ROOT=runs/v2`; `compare_methodologies.py`
      diffs them. All W1+ runs use the v2 tree.
- [x] Arbitrage **severity** metrics: `butterfly_int_neg` / `calendar_int_neg`
      (mean integrated negative part) added to `compute_arbitrage_ratios`,
      per-fold results, summaries, and master-table columns
      (`arb_bfly_sev`/`arb_cal_sev`; NaN for v1 — backfill via
      `scripts/audit_physics_fidelity.py --folds all`). Dollar-scale variant
      deferred to the fidelity audit (needs strike context).
- [x] `track_test_curve` exposed as `--track_test_curve` in
      `run_walk_forward.py` + `run_stage2.py` (diagnostic only — never selection).
- [x] **Gradient-alignment score** cos(∇L_data, ∇L_pde) logged to
      `history["grad_align_data_pde"]` at balancer cadence (all balancers).
- [ ] σ_θ-vs-market-IV overlay — **script ready**
      (`scripts/overlay_vol_surface.py`, all 4 Stage 2 cells × Nov/Dec);
      RUN ON DCC in pinn_env.
- [ ] Physics-vs-analytic-BS fidelity audit — **script ready**
      (`scripts/audit_physics_fidelity.py`, incl. analytic-surface
      finite-diff noise floor); RUN ON DCC in pinn_env.
- [x] Reporting switch: master table now leads with **mean-fold RMSE + MAE**,
      adds `rmse_mae_gap` + `medape%` (medAPE recorded by `validate()` for
      new runs; NaN for v1); pooled demoted to secondary columns.

### W1 — Balancer fix (the source; blocks everything downstream)
- [ ] Cheap falsification: **ReLoBRaLo** drop-in + Σλ-renormalization control on
      B6/B10, 3 folds. Acceptance: weights plateau, no runaway, hybrid stops
      degrading vs physics.
      **Code ready:** `--balancer {gradnorm,gradnorm_renorm,relobralo,fixed}`
      in `run_walk_forward.py`/`run_stage2.py` (default `gradnorm` = v1,
      byte-identical). LAUNCH: `sbatch --array=0-5 slurm/run_w1_balancer.sh`
      ({B6,B10} × {relobralo,renorm,fixed}, folds Aug/Oct/Nov, → runs/v2/,
      track_test_curve on; v1 gradnorm runs are the control).
- [ ] Principled target: **augmented Lagrangian** (PECANN/CAPU-style) — data =
      objective; PDE/TC/BC = equality constraints; arbitrage = inequality
      constraints with self-limiting multipliers. Implement after the cheap
      test confirms bounding is the lever.
- [x] Fixed-schedule robustness baseline: `--balancer fixed` (no adaptation;
      included in the W1 test array).
- [ ] Decision gate: ReLoBRaLo vs ALM vs fixed-schedule on weight convergence +
      RMSE + implementation risk.

### R1 — Regime-conditioned pricing net (NEW; the actual root-cause fix) ← RUNNING

**Why.** W1 proved the balancer was not the root cause. The real defect is a
**missing input**: BS needs (S,K,τ,r,σ); the net gets S,K via `m=F/K` and τ, but
σ is *frozen* at σ_fixed per fold while the market repriced σ daily. So the data
term is fitting a **one-to-many** map — the same (m,τ) had different correct
prices in March vs September (measured across-date spread within a fine bucket:
0.0142 ≈ **$4.71**). The data loss can only learn a regime-average, which is wrong
for the test month and corrupts the shape physics had right. This one cause
explains hybrid<physics, the −0.5 gradient conflict, the flat σ_θ, AND why
GAM/laGP also lost to BS.

**Evidence (bucket models, 9-fold, mean-fold RMSE — no training):**
BS 6.95 | BS+resid(m,τ) 6.72 | **BS+resid(m,τ,ν) 6.23** | BS with ν used *as σ*
8.61 ✗. The regime-conditioned correction beats every model in the benchmark
(B10 6.75, B3 physics 7.05, GAM 7.26), winning 8/9 folds.

**What changed.** The **loss functions are unchanged** — `L_data` is still plain
MSE. Only the input space changed: `v̂_θ(m, τ, ν)` with ν = 1-day-lagged
ATM-median IV (per quote's own date). Physics stays ν-independent (PDE at
σ_fixed, TC, BC) so it anchors every ν-slice to one reference solution; only the
data term differentiates slices. Collocation/TC/BC ν is drawn from the training
quotes' empirical ν distribution. **ν is an INPUT, never σ** (as σ it scores
8.61 — noisy daily estimate × vega). Full table of what-is-fed-where in
`TRAINING_VALIDATION_DISCUSSION.md` §P6.

- [x] `atm_iv_lag` in `load_and_preprocess` (no-look-ahead, ffill/bfill);
      exposed via `df_to_arrays` as `"nu"`.
- [x] `PricingNet` / `StandardPricingNet` take `n_inputs=3`; forward accepts `nu`.
- [x] `compute_pde_residual` / `compute_individual_losses` thread ν (derivatives
      in (m,τ) only — ν is a parameter, not a coordinate).
- [x] `make_batch(regime=True)`; `validate(use_nu=)`; arbitrage check evaluated
      at the fold's median test ν.
- [x] `--regime_input {none,atm_iv_lag}` (default `none` = v1 byte-identical);
      `_nuatm` run-tag suffix; `regime_input` recorded in results.json + run_info.
- [x] Verified locally on CPU: both paths train; ν changes the output; 3-input
      checkpoint **fails loudly** against a 2-input net.
- [ ] **RUN:** `bash slurm/submit_r1.sh` → 6-task array, 9 folds, → `runs/v2/r1/`
      (own subtree so it cannot overwrite W1's arms):
      {hybrid+warmup} × {gradnorm, relobralo} × {no-ν, +ν}, plus physics and
      physics+ν as a **null control** (physics is ν-independent → task 5 should
      ≈ task 4, isolating "data became identifiable" from "extra capacity").
- [ ] **ACCEPTANCE:** the +ν task beats BOTH its no-ν twin AND the physics
      control (5.90 / 7.05 mean-fold) — the first time market data would help.
- [ ] Aggregate: `build_master_table.py --runs_dir runs/v2/r1 --output_dir
      reports/v2_r1 --label r1`, then `compare_methodologies.py`.

### R2 — Stage 2 under regime conditioning (after R1 lands)
- [ ] `σ_θ(m, τ, ν)` — the flat-σ_θ finding is a *specification* limit, so the
      same fix applies. Ablate: **(a) minimal** (vol net gains ν, anchor stays
      σ_fixed) vs **(b) regime-anchored C-Vol** `σ² = μ(m,τ,ν)·ν²`, where the
      multiplier learns smile shape *relative to the current ATM level*
      (sticky-moneyness). (b) is more expressive but re-exposes ν's level noise
      via the anchor → pair with a weakened (μ−1)² reg (already planned).
- [ ] Hard constraint: 3-input Stage 2 must warm-start from a **3-input** Stage 1
      checkpoint — regime runs chain to regime runs (shape check enforces this).
- [ ] Re-run the σ_θ-vs-IV overlay; success = corr(σ_θ, IV) clearly > 0 and
      beating flat σ on most folds (v1 baseline: corr ∈ [−0.26,+0.29], 3/8).

### W2 — Architecture gate (~2 runs; parallel to W1)
- [ ] modified-**physics** at μ ∈ {0.0, 0.25}: if arb-clean at low μ → keep
      Modified-MLP confidently; if still >15% butterfly → promote standard MLP
      to a deliverable candidate.

### W3 — Protocol lock (after W1; gates the re-sweep)
- [ ] Re-measure selection regret + reported-vs-stripped ranking under the
      stable balancer.
- [ ] If convergence verified (final ≈ best): **drop the validation set**, fold
      the month back into training (expanding window), report fixed-budget
      final-epoch; optional last-K *prediction* averaging (never weight-SWA).
- [ ] One-time transparency: publish both-protocol rankings.

### W4 — Arbitrage objective
- [ ] Under ALM: inequality constraints (self-modulating). Under a grad-norm-
      family balancer: soft penalty `relu(−v̂_mm)²`, `relu(−v̂_τ)²`, post-warmup
      ramp, outside the balancer. Butterfly-in-K ≡ convexity in m:
      `∂²C/∂K² = (m²/K)·v̂_mm` (derivatives already computed for the PDE).
- [ ] Align training penalty grid with the diagnostic grid; smoke-test on B10.
- [ ] Hard/architectural constraint (ICNN-style) = FALLBACK ONLY (fights tanh,
      over-constrains τ, breaks comparability).

### W5 — Stage 1 fractional re-sweep (new balancer + protocol)
- [ ] arch × μ × arbitrage{off,on} first (highest-signal axes); Fourier-scale
      {0.5, 1.0} second (isolate ripple source).
- [ ] μ selected on **arbitrage + RMSE jointly**, not PDE-dominance.
- [ ] Pick the lowest-arb backbone that holds BS-level RMSE.

### W6 — Stage 2 re-sweep + the 2×2 closure
- [ ] {Stage 1, Stage 2} × {arb off, on} attribution table ("enable vs enforce").
- [ ] Break warm-start lock-in: pricing_lr {1e-4, 3e-4}; C-Vol reg weight
      {full, half, 0} (it pulls σ_θ → inert); vol_type {cvol, avol};
      warm-start vs from-scratch; grow vol capacity / Fourier scale if σ_θ
      underfits the smile.
- [ ] Validate σ_θ against market IV (the decisive scientific check).

### W7 — Re-baseline + reporting
- [ ] Rebuild master table (mean-fold + MAE primary, severity columns).
- [ ] Level σ-field across baselines (BS_A1/PINN use single median σ; GAM/laGP
      use time-varying `daily_vol_proxy`) or document the asymmetry.
- [ ] Add smile-aware-σ reference baseline (σ as a function of moneyness).
- [ ] Regenerate figures; update `paper/paper.qmd` narrative.

### Success criteria (from plan §0)
mean-fold RMSE ≤ 6.96 (≤ BS) AND mean MAE ≤ 5.19 AND combined butterfly+calendar
< 12% AND bounded max-neg (no −100s) AND pooled not worse than BS 8.79.

---

## Stage 3: Uncertainty Quantification — DEFERRED (future work)

Only after the point model meets the Pareto target — UQ bands around an
arbitrage-violating surface inherit the pathology. Design retained from the
framework proposal:

- **A-UQ (heteroscedastic):** variance head ŝ²(m,τ)=exp(NN_var), data loss MSE→NLL.
- **C-UQ (proportional):** ŝ²=ν²+η²(K·v̂)², two learnable scalars.
- **Four combos:** AA / AC / CA / CC (vol parametrization × UQ head).
- **Calibration:** empirical coverage @ {85,90,95}%, interval width, calibration
  plots; compare to GAM (86.85% @95%) and laGP/GP (96.26% @95%).
- Anchored ensembles (Kazemian et al.) as the implementation route; laGP is the
  non-PINN credible-interval comparator on the same folds.

---

## Reference — current hyperparameters (⚠ several under revision)

Items flagged ⚠ are being re-decided in the active workstream.

### Pricing network
| Parameter | Value | Notes |
|---|---|---|
| Architecture | Modified MLP + RWF + Fourier | ⚠ standard kept as scored control |
| Hidden layers | 4 × 64 | |
| Activation | tanh | required for PDE 2nd derivs |
| Fourier features / scale | 64 (128-dim) / 1.0 | ⚠ scale sweep {0.5,1.0} planned |
| RWF μ | hybrid 0.75 (B10) | ⚠ re-select on arbitrage+RMSE, not dominance |
| RWF σ | 0.1 | Wang et al. default |
| Learning rate | 1e-3 (Stage 0/1) / 1e-4 (Stage 2 warm) | ⚠ Stage 2 pricing_lr ablation |
| LR schedule | cosine → LR×0.01 | |
| Loss balancing | grad-norm, freq 1000 / α 0.9 | ⚠ **ReLoBRaLo is the W1 pick** (`--balancer relobralo`); renorm rejected (λ_pde collapse) |
| Regime input ν | none (v1) / `atm_iv_lag` (R1) | `--regime_input`; 3-input net, `_nuatm` tag. ν is an INPUT, never σ |
| Weight floor | 10 (TC, BC) | |
| Collocation | 5000/iter, 70/30 uniform/kink | ⚠ add wing-weighted sampling |
| Market batch | 8000 | hybrid |
| Epochs / warmup | 15k / 5000 PDE-warmup | warmup is load-bearing |
| Reporting | val-best snapshot (current runs) | ⚠ retiring the argmin — fixed-budget final-epoch once W1 verifies convergence (W3) |

### Volatility network (Stage 2)
| Parameter | Value | Notes |
|---|---|---|
| Architecture | Simple MLP + RWF | ⚠ grow if σ_θ underfits smile |
| Hidden layers | 2 × 32 | |
| Fourier scale | 0.5 | ⚠ raise if too smooth |
| Learning rate (vol) | 1e-3 | |
| σ₀ (C-Vol) | σ_fixed | |
| C-Vol reg | (μ_C − 1)² | ⚠ ablate weight {full,half,0} — pulls σ_θ inert |

### Non-dimensionalization
m = S/K (forward/strike), τ = T−t, v̂ = V/K. PDE:
R = ∂v̂/∂τ − ½σ̂²m²∂²v̂/∂m² − r·m·∂v̂/∂m + r·v̂.
