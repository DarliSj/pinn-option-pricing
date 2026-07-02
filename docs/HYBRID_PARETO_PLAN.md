# Closing the Pareto Gap — Hybrid Investigation & Re-Sweep Plan

**Status:** living plan (drafted 2026-06-02). Supersedes the "next steps" notes in
`results_summary.md`. Built from the per-fold diagnostic dive of 2026-06-02.

**North star:** produce a single model that **beats BS on RMSE (mean-fold
primary, pooled secondary) _and_ is arbitrage-consistent** — i.e. close the hole
in the RMSE-vs-arbitrage frontier.
**Budget:** full re-sweep approved (grid + Stage 2 cells with new loss terms).

**Decisions locked (2026-06-02):**
- **Architecture:** keep Modified-MLP + RWF as the _developed_ model; keep the
  standard MLP as a _scored control_. The evidence convicts **μ**, not the gating
  (arb tracks μ 16→25→38% at fixed arch; B8 μ=0.0 already 9.95%; the
  standard-vs-modified gap was measured at μ=0.5, not the floor). **Gate:** run
  modified-_physics_ at μ ∈ {0.0, 0.25} first — if arb-clean there, keep modified
  confidently; if still >15% butterfly, promote standard to deliverable candidate.
- **Arbitrage penalty:** soft penalty (L1), **off during warmup → ramped
  post-warmup → kept _outside_ grad-norm** (never let the balancer amplify it the
  way it amplified λ_data). Hard/architectural constraint (ICNN-style) is a
  _fallback only_: it fights the tanh/2nd-derivative requirement, over-constrains
  τ (BS prices are not convex in τ), and breaks comparability. Revisit only if
  soft penalties plateau with un-killable residual violations, and then as a
  partial convexity-preserving transform in m only.
- **μ re-selection:** select on arbitrage + RMSE jointly, _not_ PDE-dominance
  alone (the old criterion picked μ=1.0, the worst arb setting).

**Decisions/updates (2026-06-03) — from the training & validation dive
(companion doc: `TRAINING_VALIDATION_DISCUSSION.md`):**
- **Balancer first.** The grad-norm drift (unbounded Σg/gᵢ feedback, λ_data →
  ~10⁵, weights never converge) is the *source* defect behind the selection and
  reporting problems — fix it before any re-sweep. Candidates: **ReLoBRaLo**
  (bounded, cheap drop-in; fast falsification) vs **augmented Lagrangian**
  (PECANN/CAPU-style; data-fusion-native; absorbs arbitrage + Stage 2 terms as
  constraints). Fixed-schedule weights stay as the robustness baseline.
- **Arbitrage handling follows the balancer choice:** under the augmented
  Lagrangian it becomes an *inequality constraint* with a self-limiting
  multiplier (supersedes the "outside grad-norm" phrasing above, which applies
  only if a grad-norm-family scheme is kept).
- **Retire the per-fold val-best argmin.** Evidence: selection regret mean
  \$0.57–1.18/month, max \$3.14 (Oct, selected on the Sep outlier); the config
  ranking *flips* between reported and selection-stripped metrics; E* is
  seed-dependent (5,000 vs 8,500 for identical B10). If the fixed balancer
  verifiably converges (weights plateau; final ≈ best), **drop the validation
  set entirely**: fold the month back into training and report fixed-budget
  final-epoch (+ optional last-K *prediction* averaging; never weight-SWA).

---

## 0. Objective & success criteria

We currently have no model on the desirable frontier corner. The two extremes:

| | pooled RMSE | butterfly% | calendar% | cal max-neg (worst fold) |
|---|---|---|---|---|
| BS (bar to beat) | 8.79 | — | — | — |
| B10 (best RMSE) | **8.49** | 37 | 19 | **−301** |
| S2_B12 / A-Vol (cleanest) | 9.41 | 8.2 | 0.27 | small |

**Primary metrics (decision 2026-06-02): lead with mean-fold RMSE + MAE; pooled
is secondary/appendix.** Mean-fold weights each month equally (robust to the
severe Apr-vs-Dec fold-size heterogeneity); MAE is linear (not hostage to a few
huge wing errors). Pooled stays for comparability with the historical single-split
\$17.23 and the "every option equal" view. **Report the RMSE−MAE gap as a tail
diagnostic** (it widens exactly where arbitrage-violating wing blow-ups live), and
add one relative metric (median absolute percentage error) so dollar numbers are
interpretable across TSLA's price range. BS reference: mean-fold RMSE 6.96, mean
MAE 5.19, pooled RMSE 8.79.

**Quantitative target for "success":**
- mean-fold RMSE **≤ 6.96** (≤ BS), ideally ≤ B10's 6.75; mean MAE **≤ 5.19**, AND
- combined butterfly + calendar violation rate **< 12%** (≤ B1's level), AND
- calendar / butterfly **max-neg bounded** (no −100s blow-ups in any fold), AND
- pooled RMSE reported as secondary (and not _worse_ than BS's 8.79).

A model meeting all four is the deliverable. Failing that, a clean documented
frontier with the mechanism explained is the fallback.

---

## 1. Consolidated diagnosis — the five independent levers

The 2026-06-02 dive established these facts (all from per-fold `results.json`):

1. **The hybrid degradation is real and architecture-specific.** Turning on the
   data loss takes modified-physics → modified-hybrid from ~8.76 to ~10.0–10.7
   pooled RMSE, with arbitrage rate _and severity_ exploding (cal max-neg
   −0.6 → −2 → −5 as μ rises). Cause is twofold: (i) numerical — grad-norm
   amplifies λ_data before PDE/TC/BC settle; (ii) model — constant-σ physics and
   smile-bearing quotes are genuinely inconsistent, and a high-capacity net
   "resolves" the conflict by bending the price surface into non-convex shapes.

2. **B1's (standard hybrid) ranking is protocol-dependent — not a better model.**
   Val-best is a _valid_ protocol (no test leakage; test month touched once). But
   it _rescues drifters_: B1 snapshots at epoch 500–2000, before the standard
   MLP's known late drift, whereas modified-physics snapshots at ~15,000. Under
   the briefing's original "report final-epoch" rule, B1 would look terrible. So
   the B1-vs-modified ranking depends on the reporting protocol. B1 also can't
   manufacture violations (low capacity → 2–13% butterfly, cal max-neg ≈ 0).
   Takeaway: this is about comparability/variance, not bias — and it shows that
   **constraining capacity / penalizing arbitrage is the lever.**

3. **The Modified-MLP + RWF stack is itself an arbitrage source.** In _pure
   physics mode_ (no data): B0 standard = 3.7% butterfly / 0.3% calendar;
   B2/B3/B4 modified = 16/25/38% butterfly, with μ=1.0 also hitting 21% calendar.
   Both use Fourier features, so the differentiator is the gating + RWF
   specialization. **μ is a direct arbitrage knob and it points the wrong way:
   the value chosen for "stability" (μ=1.0) is the worst for arbitrage.**

4. **B10's low RMSE hides violent local distortions.** Warmup fixes the λ_data
   runaway (RMSE 10.0 → 8.49) but not the structural freedom to violate
   no-arbitrage: cal max-neg reaches −301 (Aug), −198 (Sep), −184 (Nov). Pooled
   RMSE and violation-_rate_ both hide this; we are under-measuring severity.

5. **BS \$8.79 is a weak, outlier-driven bar.** Single ATM σ ignores the smile;
   three folds (Sep 15.15, Dec 12.12, Jul 8.87) drive the pooled number; the
   other six average ≈ \$3.7; mean-fold BS = \$6.96. Fair as a _controlled
   comparator_ (PINN physics asymptotes to it: B4 = 8.76), weak as a pricer.

**Five levers fall out, and they are largely independent:**

| Lever | Moves RMSE? | Moves arbitrage? | Cost |
|---|---|---|---|
| L1 — direct arbitrage penalty `relu(−v̂_mm)`, `relu(−v̂_τ)` | ~neutral | ↓↓ (targeted) | ~free per step |
| L2 — architecture / RWF-μ (capacity ↔ arb trade-off) | small | ↓↓ at low μ | re-sweep |
| L3 — Stage 2 learnable σ (removes model-error conflict) | ↓ (if freed) | ↓ | re-sweep |
| L4 — wing-weighted collocation (violations live in wings) | ~neutral | ↓ | ~free |
| L5 — convergent loss balancing (grad-norm → ReLoBRaLo/ALM replacement, W1) | ↓↓ | ↓ | re-sweep prerequisite |

The headline bet (§3) combines L1 + L3 + L2 + L4 on top of L5.

---

## 2. Problem families

### A. The constant-σ ⟂ smile conflict (model error)

- **Evidence:** physics-modified hugs BS (~8.8); adding data → ~10 + arb blow-up.
  The conflict is structural: a smile cannot live in a single-σ price surface
  without breaking convexity.
- **Hypotheses:** (a) giving σ freedom (Stage 2) removes the conflict so data and
  physics stop fighting; (b) the conflict is _why_ warmup is needed at all.
- **Fixes / explorations & trade-offs:**
  - Stage 2 learnable σ is the principled resolution (see family D).
  - Interim: a **moneyness-dependent σ in the constant baseline** (a cheap smile,
    σ = f(m) fit on train) would test how much of the gap is pure
    σ-misspecification vs modeling difficulty. Trade-off: muddies the clean
    "constant-σ" story; keep it as a diagnostic baseline, not a headline model.
- **Experiments:** physics-only fidelity audit (does the PINN even reproduce
  arb-free BS? — B4 says _no_, 38%/21%); smile-aware-σ reference baseline.
- **Success check:** Stage 2 hybrid no longer degrades vs its physics counterpart.

### B. Architecture & RWF-μ as an arbitrage source (newly identified)

- **Evidence:** physics-mode butterfly 3.7% (standard) vs 16/25/38% (modified
  μ=0.5/0.75/1.0). This contradicts the code convention "always Modified-MLP+RWF,
  μ=1.0."
- **Hypotheses:** (a) gating + RWF per-neuron specialization injects local
  curvature ripples; (b) higher μ = larger init scales = more ripple; (c) Fourier
  scale (1.0) contributes high-frequency wing oscillation.
- **Fixes / explorations & trade-offs:**
  - **RWF-μ sweep re-framed around arbitrage**, not just PDE-dominance: include
    μ ∈ {0.0, 0.25, 0.5, 0.75} (B8/B12/B11/B10 already span this — extend
    analysis to severity, not just rate). Trade-off: lower μ may slow PDE
    convergence; combine with warmup.
  - **Fourier-scale sweep** {0.5, 1.0} and a **no-Fourier modified** variant to
    isolate the ripple source. Trade-off: lower scale may blunt the payoff kink.
  - **Depth/width reduction** (capacity control) as an alternative to penalties.
  - Re-examine whether the "always Modified-MLP" convention should be relaxed for
    the arb-clean objective. Trade-off: it's our Wang-et-al. novelty hook.
- **Experiments:** physics-mode arch × μ × Fourier-scale grid, scored on
  arbitrage rate AND severity AND RMSE.
- **Success check:** identify the lowest-arb architecture that still solves the
  PDE to BS-level RMSE; use it as the Stage 2 backbone.

### C. Arbitrage is an untrained objective and is under-measured

- **Evidence:** `src/losses.py` has only {pde, tc, bc, data, reg}; nothing
  penalizes `relu(−v̂_mm)` or `relu(−v̂_τ)`. We _measure_ violations but never
  _train_ against them. Severity (max-neg) is logged but not headlined; it blows
  up where the rate looks merely "high."
- **Design (clean — gradients already computed for the PDE residual):**
  Butterfly no-arb in strike space reduces exactly to convexity in m:
  `∂²C/∂K² = (m²/K)·v̂_mm ≥ 0  ⟺  v̂_mm ≥ 0`. Calendar: `∂C/∂τ = K·v̂_τ ≥ 0 ⟺ v̂_τ ≥ 0`.
  So add
  `L_bfly = mean(relu(−v̂_mm)²)`,  `L_cal = mean(relu(−v̂_τ)²)`
  on the collocation grid. **These are compatible with the smile** (an arb-free
  smile surface _is_ convex in K and monotone in τ), so unlike lowering μ they
  should cut violations _without_ an RMSE penalty — this is the central bet.
- **Weighting trade-offs:** (a) fixed/floored λ_arb (safer given grad-norm
  runaway history); (b) include in grad-norm balancing (risk: amplified like
  λ_data); (c) ramp λ_arb up after warmup. Recommend (a)/(c); ablate magnitude.
  _Update 2026-06-03: under the augmented Lagrangian (W1 target) this whole
  question dissolves — arbitrage becomes an inequality constraint with a
  self-limiting multiplier; (a)/(c) apply only if a grad-norm-family balancer
  is kept._
- **Measurement upgrades (do first, they gate every "arb-clean" claim):**
  - report **integrated negative part** `∫ relu(−v̂_mm)` and `∫ relu(−v̂_τ)`
    (severity), not only the violating-grid fraction;
  - report a **dollar butterfly** metric (value of the most-violated butterfly
    spread) for economic interpretability;
  - align the training penalty grid with the diagnostic grid.
- **Experiments:** L1 penalty on/off × {B10, B12} × {fixed, ramped} λ_arb.
- **Success check:** B10-level RMSE with violation rate < 12% and bounded max-neg.

### D. Stage 2 doesn't beat constant-σ on RMSE; is σ_θ even real?

- **Evidence:** S2_B10/C-Vol 8.85 > B10 8.49. Improves arbitrage but not
  accuracy. C-Vol `reg` penalizes (μ_C − 1)², actively pulling σ toward σ_fixed
  (inert, esp. B12/C-Vol ≈ 1.0). pricing_lr=1e-4 nearly freezes the price net.
- **Hypotheses:** (a) warm-start lock-in — price net frozen, vol net can't
  compensate; (b) reg over-smooths σ_θ to inertness; (c) vol net too small
  (2×32, Fourier scale 0.5); (d) σ_θ is fitting freedom, not real vol.
- **Critical analysis (no training):** **overlay learned σ_θ(m,τ) on the
  empirical IV smile/term-structure** for Nov/Dec. If it tracks market IV →
  scientifically meaningful even at flat RMSE; if not → it's just slack.
- **Fixes / explorations & trade-offs:**
  - unfreeze price net (pricing_lr 1e-4 → 3e-4/1e-3); trade-off: may re-introduce
    instability — pair with warmup + L1.
  - reduce/remove C-Vol deviation reg; trade-off: less smooth σ_θ — pair with L1
    so arbitrage is still controlled.
  - grow vol net capacity / raise vol Fourier scale; trade-off: rougher σ_θ.
  - from-scratch vs warm-start ablation.
- **Experiments:** full Stage 2 cell re-sweep (below) with L1 enabled.
- **Success check:** a Stage 2 cell reaches the §0 target; σ_θ resembles market IV.

### E. Protocol & measurement integrity

- **Val-best is valid — the concern is variance + comparability, not bias.**
  Val-best snapshotting has _no test leakage_ (test month touched once); reported
  test RMSE is an honest held-out estimate. Two legitimate caveats: (1)
  **selection variance** — the val month is a single, sometimes-unrepresentative
  month, so E* is high-variance (range 500→15,000, std up to ~3,900) and inflates
  the variance of the reported number; (2) **protocol-dependence of cross-config
  ranking** — val-best rescues late-drifting configs (standard MLP) but is neutral
  to monotone ones, so B1-vs-modified depends on the protocol (the briefing's
  original rule was final-epoch). Fixes: keep val-best, but cut variance with
  **last-K-snapshot averaging (SWA-style)** near the val optimum, and report a
  couple of configs under _both_ val-best and final-epoch so the ranking's
  protocol-sensitivity is visible. Not a bias/leakage problem. _Update
  2026-06-03: superseded by the balancer-first decision (§0) — retire the
  argmin; if the fixed balancer verifiably converges, drop the val set and
  report fixed-budget final-epoch._
- **Val-month non-representativeness:** Sep-as-val for Oct gives val RMSE 16.9 vs
  test 8.1 (B6) — the selection signal is noisier than the target. Consider a
  less regime-fragile val scheme; trade-off: temporal val is methodologically
  cleanest — at minimum document the mismatch.
- **σ-field consistency across baselines:** BS_A1 + PINN use a single median σ;
  GAM/laGP fit residuals on a _time-varying_ `daily_vol_proxy` BS base. Level the
  σ field or document the asymmetry.
- **Reporting (decided):** lead with **mean-fold RMSE + MAE**; pooled to the
  appendix; show the **RMSE−MAE gap** as a tail diagnostic; add **median absolute
  percentage error** for cross-price interpretability; report std across folds;
  flag Sep as a regime outlier; add the smile-aware-BS reference (family A). All
  already in the master table — emphasis change, not new computation.

---

## 3. The headline path & experiment matrix

**Bet:** _a convergent balancer (ReLoBRaLo or augmented Lagrangian — W1) +
learnable σ (removes the model conflict) + the arbitrage objective (inequality
constraint under ALM; post-warmup-ramped penalty otherwise — removes residual
violations, smile-compatible) + a low-arb backbone (low-μ or capacity-controlled)
+ wing collocation._ This is the only combination where RMSE and arbitrage can
both reach target, because each lever attacks a different cause.

*(The matrices below are drafted pre-W1; the arbitrage and warmup axes get
finalized after the balancer decision — under ALM, "penalty schedule" collapses
to constraint on/off and warmup may be subsumed by the multiplier schedule.)*

**Stage 1 re-sweep (physics + hybrid), scored on RMSE + arb rate + arb severity:**

| Axis | Values |
|---|---|
| arch | standard, modified |
| RWF μ | 0.0, 0.25, 0.5, 0.75 |
| Fourier scale | 0.5, 1.0 |
| arbitrage penalty L1 | off, fixed, ramped |
| warmup | 0, 5000 |

(Don't run the full cross-product — see §4 for the fractional design.)

**Stage 2 re-sweep (with L1 on, low-arb backbone from Stage 1):**

| Axis | Values |
|---|---|
| vol_type | cvol, avol |
| backbone | best low-arb Stage-1 config |
| C-Vol reg weight | {full, half, 0} |
| pricing_lr | 1e-4, 3e-4 |
| init | warm-start, from-scratch |

---

## 4. Sequencing (resequenced 2026-06-03 — balancer first)

The training/validation dive showed the loss-balancer drift is the *source*
defect: it produces the noisy trajectories that make epoch selection unfair
(regret up to \$3.14/month; ranking flips with the selection rule) and reporting
unstable. So the balancer is fixed and the protocol locked *before* the
re-sweep — otherwise the grid is generated under a rule known to reorder configs.

1. **Instrument + analysis-only audits** (no training): arbitrage severity
   metrics, σ_θ-vs-IV overlay, physics-vs-analytic-BS fidelity audit on existing
   checkpoints, B1-under-final-epoch, `track_test_curve` wiring (diagnostic
   only), and a **gradient-alignment score** between ∇L_data and ∇L_pde
   (directional-conflict check). _Gate: nothing downstream launches until these
   exist._
2. **Balancer fix (the source):** ReLoBRaLo drop-in + Σλ-renormalization control
   on B6/B10 over 2–3 folds (acceptance: weights plateau, no runaway, hybrid
   stops degrading vs physics). Principled target: augmented Lagrangian (data =
   objective; PDE/TC/BC = equality constraints; arbitrage = inequality
   constraints). Decision gate: ReLoBRaLo vs ALM vs fixed-schedule.
3. **Architecture gate (~2 runs, parallel to 2):** modified-_physics_ at
   μ ∈ {0.0, 0.25}, scored on arbitrage rate + severity → keep modified
   confidently or promote standard to deliverable candidate.
4. **Protocol lock:** re-measure selection regret under the stable balancer; if
   convergence is verified, drop the val set (expanding window + fixed-budget
   final-epoch; optional last-K *prediction* averaging). One-time both-protocol
   ranking for transparency.
5. **Arbitrage objective:** under ALM → inequality constraint (self-modulating);
   under a grad-norm-family balancer → soft penalty, post-warmup ramp, outside
   the balancer. Smoke-test on B10; align the training penalty grid with the
   diagnostic grid.
6. **Stage 1 fractional re-sweep** (new balancer + protocol): arch × μ ×
   arbitrage{off,on} first, Fourier scale {0.5, 1.0} second; μ selected on
   arbitrage + RMSE jointly; pick the lowest-arb backbone that holds BS-level
   RMSE.
7. **Stage 2 re-sweep + the 2×2 closure:** {Stage 1, Stage 2} × {arbitrage
   off, on} attribution ("enable vs enforce"); pricing_lr {1e-4, 3e-4}; C-Vol
   reg {full, half, 0}; validate σ_θ against market IV.
8. **Re-baseline:** rebuild master table (mean-fold RMSE + MAE primary, severity
   columns), regenerate figures, update the paper narrative.

---

## 5. Risks & falsification

- **L1 may fight RMSE more than predicted** if the data genuinely wants a locally
  non-convex (i.e. arbitrageable, noisy) fit — that would itself be a finding
  (the quotes contain static arbitrage). Test: does L1 raise RMSE on clean folds?
- **σ_θ may not match market IV** → Stage 2's contribution is mechanical, not
  economic; pivot the claim to "regularization device" honestly.
- **Low-μ backbone may underfit the kink** → RMSE floor above BS; mitigate with
  Fourier scale / TC weighting, or accept modified-μ + heavier L1.
- **No config hits all four §0 criteria** → fall back to a documented frontier
  with the mechanism (families A–C) as the contribution.
