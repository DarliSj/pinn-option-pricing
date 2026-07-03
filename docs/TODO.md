# PINN Option Pricing — Project TODO

*Last updated: 2026-06-03. Authoritative next-steps plan:
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
- **Active workstream:** fix the training/validation foundation first
  (W0–W3: balancer, protocol), then close the RMSE-vs-arbitrage Pareto gap
  (W4–W7).
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
| Loss balancing | grad-norm, freq 1000 / α 0.9 | ⚠ under replacement — ReLoBRaLo vs augmented Lagrangian (W1) |
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
