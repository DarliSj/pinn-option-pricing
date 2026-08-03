# Training & Validation Methodology: Issues, Rationale, Candidate Fixes

*Discussion document for advisor review (drafted 2026-06-02). Scope: the
**training** (loss balancing) and **validation/model-selection** layer — the
foundation under every downstream result. Not a ready-to-implement spec; a menu
of options with trade-offs. Stage 2 (learnable σ) and the arbitrage objective are
out of scope here except where they motivate a choice.*

---

## The one linking insight

We found three problems, but they are **causally chained**, not independent:

> **Unstable loss balancing (P1)** produces noisy, non-converging training
> trajectories → which makes **validation-based epoch selection fragile and
> unfair (P2)** → which forces an awkward **reporting-protocol choice (P3)**.

So the spine of the plan is: **fix the source (P1) first**; much of P2 and P3
dissolves once the trajectory converges. Each problem below follows the same
four-part structure: (1) the weird result, (2) what we did and why, (3) why it
breaks, (4) candidate fixes with pros/cons.

> **⚠ UPDATE (2026-08, after W0/W1 ran).** The chain above is confirmed but is
> **not the whole story**, and two claims in this document were revised by
> measurement. W1 showed the balancer defect is real and fixable, yet fixing it
> does **not** make hybrid beat physics — so the balancer was never the root
> cause of the headline defect. The actual root cause is a **specification**
> error: the network is never told the volatility regime a quote came from, so
> the data term is fitting a one-to-many map. See **§P6** (new) for the
> diagnosis, evidence, and the R1 fix now implemented. Read P1–P5 as the
> (still-valid) analysis of the *training* layer, and P6 as what actually
> governs whether market data helps at all.

---

## P1 — The loss balancing drifts (grad-norm runaway)

**1. The weird result.** Adding the market-data loss to the expressive
architecture makes test RMSE *worse*, not better: modified-physics ≈ \$8.76 →
modified-hybrid ≈ \$10.0–10.7. Arbitrage severity grows in lockstep (calendar
worst-case slope −0.6 → −2 → −5 as μ rises). The adaptive weight λ_data climbs to
~10⁵ and the weights never settle — they are still moving at the final epoch.

**2. What we did & why.** We use the Wang et al. (2023) gradient-norm scheme,
`λ_i = (Σ_j ‖∇L_j‖)/‖∇L_i‖` with EMA smoothing. Motivation: it auto-balances
heterogeneous PINN loss terms (PDE / terminal / boundary) without hand-tuning,
and it worked well in **physics-only** mode. When hybrid mode misbehaved we added
a 5,000-epoch **PDE-warmup** (data loss off early) as a patch.

**3. Why it breaks (structural faults).**
- **Unbounded with positive feedback.** `Σg/g_i` upweights the term with the
  *smallest* gradient. The data term's gradient shrinks as it fits — so once it
  starts fitting (including fitting noise), λ_data grows, which pushes harder on
  data, which shrinks its gradient further → runaway. There is a floor on TC/BC
  but **no ceiling anywhere**.
- **Wrong prior.** Equalizing gradient norms treats a noisy, soft data-fit as
  *co-equal* with hard physics constraints. Market-quote noise gets the same pull
  as the PDE.
- **Moving target.** Gradient norms shift every step; the rule reacts with a lag
  and never reaches a fixed point. Warmup defers the onset but doesn't make it
  converge.

**4. Candidate fixes.**

| Option | What it is | Pros | Cons | Effort/Risk |
|---|---|---|---|---|
| **F1a Normalize** | Renormalize so Σλ = const (orig. GradNorm) | tiny change; provably bounded; keeps intent | a term can still grab most mass; verify it stops noise-fitting | low / low |
| **F1b Softmax-relative** | ReLoBRaLo / SoftAdapt: weights = softmax of relative loss *improvement* | bounded by construction; uses progress not raw gradient (avoids the trap); cheap (no per-term backward pass); strong benchmark record incl. inverse problems (Appendix B) | new rule + temperature/lookback to set | med / low |
| **F2 Constrained opt** | Augmented Lagrangian: minimize data loss *s.t.* PDE/TC/BC ≈ 0; multipliers via dual ascent | principled; multipliers self-limit (grow only while violated — opposite of runaway); matches the constraint-vs-objective asymmetry; precedent that it cures the pathology (PECANN, Basir–Senocak 2022; CAPU 2025 adds per-constraint adaptive penalties — Appendix B); arbitrage drops in as another constraint | more to implement; dual step ρ to tune; can oscillate | high / med |
| **F3 Fixed/scheduled** | Hand-set or deterministically ramped weights, no adaptation | zero drift by construction; reproducible; makes final-epoch reliable; we already found warmup (a schedule) does the heavy lifting | lose auto-balancing; must tune; less novel | low / low |

*Lean (updated after the literature survey, Appendix B):* run **F1b (ReLoBRaLo)**
as the cheap bounded drop-in, with **F1a** as a one-line control; target **F2**
as the principled headline — it is the only family with a data-fusion track
record, and it unifies the no-arbitrage objective; keep **F3** as a robustness
baseline.

---

## P2 — Validation epoch-selection is high-variance and unfair

**1. The weird result.** The val-best epoch is unstable: across folds it ranges
500–15,000, and *two runs of the identical config* chose epochs 5,000 vs 8,500.
The "cost" of val-selection vs the directly-best epoch on a month (computable
because each month is a test month in fold *f* and the validation month in fold
*f+1*) is large and **regime-dependent**: mean ≈ \$0.6–1.2, max **\$3.14**, min
≈ 0 — worst exactly when the validation month is an outlier (Sept-as-validation
for the October fold). Most damning: the **config ranking flips** depending on the
selection rule —
`reported: B12 < B1 < B10 < B11 < B6` vs `selection-stripped: B1 < B10 < B11 < B12 < B6`.
B12 looks best under our protocol and near-worst once selection is removed.

**2. What we did & why.** Per-fold **val-best snapshot selection**: train on the
window, snapshot whenever RMSE on the held-out month-before-test improves,
restore the best snapshot, evaluate the test month once. Motivation: a standard,
**leakage-free** early-stopping rule (test touched exactly once) that avoids
reporting a drifted/overfit terminal model. It replaced the briefing's original
"report final-epoch" rule.

**3. Why it breaks.** The signal is **one calendar month** — small, and sometimes
a different regime than the test month. An argmin over a noisy/flat val curve is
near-random among many near-equal epochs. The regret it injects is
**config-dependent**: it rewards selection-robust configs and penalizes
high-ceiling ones, so the cross-config comparison conflates *accuracy* with
*selection robustness* — they are not compared fairly. (Note: this is a
variance/fairness problem, **not** leakage — the numbers are honest.) It is
downstream of P1: the noisy, non-converging trajectory is what produces the flat,
unreliable val basins in the first place.

**4. Candidate fixes.**

| Option | What it is | Pros | Cons | Effort/Risk |
|---|---|---|---|---|
| **(prereq) Fix P1** | converge the trajectory | sharpens val basins; shrinks regret; may make selection a non-issue | doesn't fully remove residual noise | — |
| **F4 Drop argmin → fixed budget** | report a fixed epoch for all configs | uniform; zero selection variance; single model | hostage to terminal-iterate noise *unless* P1 converges | low / low |
| **F5 Last-K averaging** | average final K snapshots (prediction-average, not weights) | variance reduction; for the linear BS operator preserves PDE residual & convexity; uniform | only coherent if weights are frozen/converged; reports an ensemble; +1 hyperparameter (K) | med / low |
| **F6 Wider/robust val** | multi-month validation for selection | less regime-fragile signal | costs training data on early folds; still selection | low / med |
| **F7 Report both rankings** | publish reported *and* selection-stripped metric | exposes whether ranking is an artifact; uses existing data | not a fix, a transparency measure | low / low |

*Lean:* once P1 converges, **F4** (fixed budget) is likely sufficient; add **F5**
for variance reduction (and arbitrage smoothing) and **F7** as a transparency
check. Avoid weight-averaging (nonlinear net → no PDE/convexity guarantee).

---

## P3 — No stable reporting protocol

**1. The weird result.** "Report final-epoch" is unreliable because the weights
(and thus the solution) are still moving at the end (P1); "report val-best" is
unfair (P2). We have no rule that is simultaneously *uniform, low-variance, and
honest about drift*.

**2. What we did & why.** Defaulted to val-best (see P2). The original intent
(briefing) was fixed-budget final-epoch to avoid selection bias — abandoned when
trajectories drifted.

**3. Why it breaks.** It's a symptom of P1+P2, not an independent disease. Any
reporting rule is only as good as the stability of what it reports. With a
non-converging balancer, *every* rule is bad: final-epoch catches the runaway,
val-best injects regret, averaging mixes models trained under different (still-
moving) weightings.

**4. Candidate fixes.** Mostly determined by P1:
- If the balancer **converges** (F1/F2/F3): final-epoch becomes reliable → F4 is
  enough; F5 optional polish.
- If we keep an adaptive scheme that doesn't fully converge: add an explicit
  **freeze phase** (stop adapting λ before the reporting window) so reported
  models solve one fixed objective — necessary for averaging to even be coherent.

---

## P4 — Arbitrage is measured but never *trained against* (and Stage 2 only half-fixes it)

**1. The weird result.** Even *physics-only* PINNs violate no-arbitrage badly
(B4 physics: 38% butterfly, 21% calendar) — a pure Black–Scholes solve *should*
be convex in strike and calendar-monotone by construction. Hybrid is worse, and
severity hides behind the rate: B10's "37% butterfly" conceals a calendar
worst-case slope of **−301** on some folds. Stage 2 (learnable σ) helps but does
not close it — the cleanest cell still sits at **8.2%** butterfly.

**2. What we did & why.** We *measure* butterfly (∂²v̂/∂m² < 0) and calendar
(∂v̂/∂τ < 0) violations on a dense grid as diagnostics, and assumed the physics
constraint plus (later) learnable σ would keep the surface arbitrage-free.
Motivation: "physics-informed" *sounds* like it should imply "arbitrage-
consistent," and Stage 2 was designed to absorb the smile.

**3. The catch.** "Physics-informed" does **not** imply "arbitrage-free." A small
*mean-squared* PDE residual does not enforce *pointwise* convexity — convexity is
a property of the solution, not something the residual penalizes. So the network
can satisfy the PDE on average while leaving non-convex ripples in the wings
(amplified by the modified architecture + high μ). And we never *optimize* against
arbitrage — we only score it afterward, with a *rate* metric that understates
severity.

**4. Candidate fixes & the key verdict.** (a) Train against it: soft penalties
`relu(−v̂_mm)`, `relu(−v̂_τ)` on the collocation grid (derivatives already
computed for the PDE residual; smile-compatible, so cheap on RMSE). (b) Report
*severity* (integrated negative part / worst-case slope), not just rate.
(c) Architecture/μ as a knob (lower μ is far cleaner). (d) Under the Augmented
Lagrangian (Appendix) arbitrage is just an inequality *constraint* whose
multiplier activates only where violated.

> **Necessary-but-not-sufficient:** Stage 2 *enables* arbitrage-freeness (removes
> the constant-σ↔smile conflict that *forces* non-convexity); the penalty
> *enforces* it (closes the residual gap from finite/noisy fitting + ripples).
> Test both via a **2×2: {Stage 1, Stage 2} × {penalty off, on}.** Expectation:
> the penalty *costs* RMSE under constant σ (it fights the conflict) but is *cheap*
> under learnable σ — and that cost gap quantifies, in dollars, the value of
> Stage 2.

---

## P5 — How Stage 2 interacts with all of the above (sequencing)

Stage 2 (jointly learn σ_θ(m,τ)) is downstream, but it changes the calculus for
P1–P4:

- **It amplifies the balancing instability → P1 must be fixed first.** Stage 2
  adds a loss term (σ-regularization), the arbitrage penalty, *and* a second
  network with its own learning rate. More terms + two nets = more surface area
  for the runaway. Launching Stage 2 on the current balancer inherits the chaos.
  The balancer fix is a **prerequisite**, not a parallel track.
- **It enables but does not enforce arbitrage-freeness** (see P4) — complementary
  with the penalty, not a substitute.
- **It partially weakens one clean guarantee.** Last-K *prediction*-averaging (F5)
  preserved the PDE residual *because the constant-σ operator is linear in v̂*; in
  Stage 2, σ_θ multiplies v̂_mm, so residual-averaging no longer holds. (Butterfly
  convexity is still preserved — ∂²v̂/∂m² depends on v̂ alone, not σ.)
- **Dropping the validation set is harder to justify in Stage 2.** Two networks
  with differential learning rates make convergence subtler to verify, so the
  "drop val + final-epoch" move (clean in Stage 1 once the balancer converges)
  needs its own convergence check in Stage 2.
- **The Augmented Lagrangian scales to all of it.** Its biggest practical draw:
  σ-regularization and arbitrage just become additional constraints with their own
  self-limiting multipliers. One framework covers Stage 1 balancing, Stage 2's
  extra terms, and the no-arbitrage objective.

---

## P6 — The data term is ILL-POSED: the net is never told the vol regime (R1)

*Added 2026-08 after W0/W1 ran. This is the finding that reframes P1–P5.*

**1. The weird result.** Fixing the balancer worked — and hybrid *still* lost to
physics. On folds Aug/Oct/Nov (μ=0.75 throughout):

| config | test RMSE | butterfly% | calendar% |
|---|---|---|---|
| **B3 physics (matched control)** | **5.90** | 25.0 | 0.4 |
| B6 hybrid, gradnorm (v1) | 7.56 | 36.4 | 39.7 |
| B6 hybrid, **relobralo** | 6.76 | 24.8 | 18.8 |
| B6 hybrid, renorm | 8.38 | 32.8 | 30.1 |
| B6 hybrid, fixed | 6.79 | 22.1 | 17.9 |
| B10 hybrid (warmup), gradnorm (v1) | 6.25 | 36.6 | 16.8 |
| B10 hybrid (warmup), relobralo | 7.57 | 21.6 | 17.8 |

Bounding the balancer recovered a lot without warmup (7.56 → 6.76) and roughly
halved the arbitrage rates — but **physics-only still wins at 5.90**. Adding
market data makes the model worse no matter how the terms are weighted.

**2. Why we expected otherwise.** The whole premise of the hybrid mode is that
market quotes carry information the constant-σ PDE lacks (the smile). We assumed
the reason it wasn't landing was the λ_data runaway — i.e. an *optimization*
problem. W1 falsified that.

**3. The actual cause — a missing input, not a bad weight.** Black–Scholes needs
(S, K, τ, r, σ). Our network sees:

| BS argument | in the network? |
|---|---|
| S, K | ✅ via `m = F/K`, refreshed per observation |
| τ | ✅ |
| **σ** | ❌ **frozen** at `σ_fixed`, one constant per fold |
| r | ❌ frozen at `r_fixed` |

So the net is asked to price with a **stale volatility** while the market
repriced σ daily through a violently non-stationary year. The data loss then
tries to teach it the market's σ — but with no σ input the only learnable object
is a **time-average over the training window**, which is wrong for the test month
and corrupts the shape the PDE had right.

The target is therefore genuinely **one-to-many**: the same (m, τ) had different
correct prices in March vs September. Measured: within a fine (m,τ) bucket, the
across-date spread of v̂ is **0.0142 ≈ \$4.71** at the median strike — comparable
to the ~\$6 RMSE we are trying to beat, and irreducible for *any* model whose
only inputs are (m, τ).

This single cause explains four separate puzzles:

1. **hybrid < physics** — data fits a regime-mixed average.
2. **cos(∇L_data, ∇L_pde) → −0.5** (the new W0 diagnostic; starts ≈ +0.1, ends
   −0.19 to −0.48) — data pulls toward the average σ while the PDE holds
   σ_fixed. Opposed gradients cannot be reconciled by *any* weighting scheme,
   which is exactly why W1 could not fix this.
3. **Stage 2's σ_θ came out flat** (W0 audit: corr(σ_θ, IV) ∈ [−0.26, +0.29];
   B10/A-Vol learned a literally constant σ = 0.594) — because `σ_θ(m, τ)` has
   **no regime input either**, so it too can only learn a time-average, and the
   time-average of a wildly moving σ is ≈ a constant.
4. **GAM and laGP also lost to BS** — same featurization, same blind spot.

**4. Evidence that the fix works** (bucket models, 9-fold walk-forward,
mean-fold RMSE — `scripts/` run locally, no training):

| model | mean-fold | pooled |
|---|---|---|
| BS(σ_fixed) | 6.95 | 8.79 |
| BS + residual(m, τ) | 6.72 | — |
| **BS + residual(m, τ, ν)** | **6.23** | **7.78** |
| BS(σ = ν) — ν used *as* sigma | 8.61 ✗ | 11.47 ✗ |

with ν = 1-day-lagged ATM-median IV. The regime-conditioned correction beats
**every model in the benchmark** (B10 hybrid 6.75, B3 physics 7.05, GAM 7.26,
laGP 8.81), wins 8/9 folds against BS, and gains most exactly on the
regime-break folds (Nov −1.69, Dec −1.76, Sep −1.65).

**The critical negative result:** feeding ν in *as the pricing σ* is much worse
(6.95 → 8.61). ν is a noisy daily estimate and the price is exposed to it through
vega. **ν is a conditioning INPUT, never σ.** σ_fixed survives precisely because
it is a heavily smoothed estimator.

### What changed in the code (R1) — and what did NOT

**The loss functions are unchanged.** No new term, no reweighting. `L_data` is
still plain MSE:

    L_data = mean_i ( v̂_θ(m_i, τ_i, ν_i) − v̂_market,i )²

The only difference is the third argument. Each quote now carries the vol state
of its own date, so the regression stops being one-to-many. Same loss,
well-posed.

**What is fed where** (`--regime_input atm_iv_lag`, default `none` = v1):

| term | inputs | ν source | target |
|---|---|---|---|
| `L_pde` | (m, τ, ν) at collocation pts | sampled i.i.d. from the **training quotes'** empirical ν distribution | residual → 0, **σ = σ_fixed** |
| `L_tc` | (m, τ≈0, ν) | same empirical draw | payoff `max(m−1,0)` (ν-independent) |
| `L_bc` | (m_min/m_max, τ, ν) | same empirical draw | analytic BS at σ_fixed (ν-independent) |
| `L_data` | (m, τ, ν) at quotes | **the quote's own date** | `v̂_market` |

Two deliberate properties follow:

- **Physics is ν-independent**: derivatives are taken in (m, τ) only — ν is a
  *parameter* of the surface, not a PDE coordinate — so every ν-slice must
  satisfy the *same* σ_fixed BS equation. Physics anchors all slices to one
  reference solution; only the data term differentiates them. Where there are no
  quotes, every slice collapses back to the BS surface, reproducing the
  "fall back to BS on unseen buckets" behaviour that made the bucket model
  conservative.
- **Collocation ν is drawn from the empirical training distribution**, so the PDE
  is enforced across the regimes that actually occurred, density-weighted.

Data plumbing: `atm_iv_lag` is built in `load_and_preprocess` (per-date median IV
of quotes with moneyness ∈ [0.95, 1.05], shifted one day, ffill/bfill) — the same
no-look-ahead construction as the existing `daily_vol_proxy`, but unbiased (the
smile-wide mean sits ≈ +0.11 high, which is why it is the *worse* conditioner).
`df_to_arrays` exposes it as `"nu"`; `make_batch(regime=True)` attaches it to
every point type; `PricingNet/StandardPricingNet` take `n_inputs=3`.

Reproducibility guards: default is off and byte-identical; all extra RNG draws
sit inside `if regime:` so the v1 sampling stream is untouched; run dirs get a
`_nuatm` suffix; and the 3-input Fourier matrix means a regime checkpoint
**fails loudly** rather than silently loading into a 2-input net.

**Operational caveat to state in the paper:** ν commits us to a daily vol feed at
inference. It is strictly lagged (no look-ahead), and it is the same standing as
the forward price we already consume per-observation — the model already sees
today's *spot*; R1 simply stops hiding today's *vol level*.

### Stage 2 under R1 — the same fix, and a design choice

The W0 audit's flat-σ_θ finding is a *specification* limitation, not a training
failure, so Stage 2 inherits the same remedy: **`σ_θ(m, τ, ν)`**. Two variants to
ablate once Stage 1 confirms the transfer:

- **(a) minimal** — vol net simply gains the ν input; the C-Vol anchor stays
  σ_fixed. Lowest risk.
- **(b) regime-anchored C-Vol** — `σ² = μ(m, τ, ν) · ν²`, so the multiplier
  learns the **smile shape relative to the current ATM level** (sticky-moneyness,
  which is how practitioner surfaces are actually quoted). More expressive, and
  it directly targets the inertness we measured. Risk: it re-exposes the model to
  ν's level noise through the anchor — mitigated because μ can learn a systematic
  correction, which argues for **weakening the (μ−1)² regulariser** (already on
  the plan for independent reasons).

Hard constraint either way: **a 3-input Stage 2 must warm-start from a 3-input
Stage 1 checkpoint** — regime runs chain to regime runs, never across (enforced
by the shape check above).

---

## The plan (discussion-level, sequenced by causality)

Not a build spec — a sequence of decisions, each with a cheap test that informs
the next. The ordering follows the causal chain: **source → selection →
reporting → arbitrage/Stage 2.**

**Step 0 — Instrument (free, do regardless).** Log the things that diagnose
everything: weight trajectories, PDE-dominance ratio (does it plateau?), and the
post-hoc test-vs-val-vs-oracle curves (`track_test_curve`, diagnostic only —
never feeds selection). Decide every later step on evidence.

**Step 1 — Attack the source (P1).**
- *Cheap proof-of-principle:* add normalization (F1a) or a λ_data cap to the
  existing scheme; re-run the worst drifter on a few folds; check the runaway
  flattens and weights converge. *Question: is bounding enough, or do we want the
  principled reformulation?*
- *Principled target:* Augmented Lagrangian (F2, see Appendix). It fixes
  balancing, encodes the physics-as-constraint structure honestly, and later
  absorbs σ-regularization + the no-arbitrage objective in one framework.
  *Decision:* F2 vs F1b vs F3, judged on weight-convergence + RMSE +
  does-hybrid-stop-degrading.

**Step 2 — Re-examine selection (P2), and — if converged — drop validation.**
- Recompute the selection regret and the reported-vs-oracle ranking under the
  stable balancer. *Hypothesis:* a converged trajectory shrinks the regret and
  removes the ranking flip.
- **If convergence is verified** (weights plateau, final-epoch ≈ best-epoch),
  **drop the validation set entirely**: fold that month back into training
  (expanding window grows by one) and report final-epoch — the briefing's
  *original* protocol, now *justified* rather than assumed. This removes selection
  variance **and** the val-month mismatch in a single move, and reclaims the most
  recent month for training. Keep last-K averaging (F5) optional for variance
  reduction (it works without a val set); keep a non-selecting monitoring peek if
  desired.

**Step 3 — Lock the reporting protocol (P3).** Likely just fixed-budget
final-epoch (+ optional last-K). Add a freeze phase only if we keep a
non-converging adaptive scheme.

**Step 4 — Arbitrage 2×2 (P4).** {Stage 1, Stage 2} × {penalty off, on}, scored on
RMSE + arbitrage *rate* + *severity*. Attribute the gain to *enable* (Stage 2) vs
*enforce* (penalty). Under F2 the penalty is a constraint whose multiplier
self-modulates — active where violated, ~0 where Stage 2 already keeps it convex.

**Step 5 — Re-baseline.** Only now do headline numbers mean something stable
*and* fairly comparable across configs.

---

## Open questions to pitch

1. **How principled do we go on balancing?** Cheap bounding (F1) gets us a stable
   trajectory fast; the Augmented Lagrangian (F2) is a defensible methodological
   contribution that also unifies the no-arbitrage objective — but it's more work
   and risk. Is the reformulation worth it for the paper?
2. **Is "fixed-budget, no selection" (F4) acceptable** once training converges, or
   do reviewers expect some held-out model selection? (We can always report both.)
3. **Do we report the selection-stripped ranking (F7)** as a robustness result, or
   does surfacing the ranking flip raise more questions than it answers?
4. **Scope check:** is it acceptable that none of this fixes the constant-σ ↔ smile
   *model* conflict (that's Stage 2's job) — i.e. a stable balancer may be
   "stably mediocre" until σ is learnable? We should be explicit that balancer
   stability ≠ accuracy ceiling.
5. **Commit to one unifying framework?** The Augmented Lagrangian would cover loss
   balancing, the no-arbitrage objective, and Stage 2's extra terms with a single
   mechanism (self-limiting multipliers). Worth the implementation lift to unify,
   or keep them as separate, simpler mechanisms (bounded grad-norm + a hand-set
   arbitrage penalty)?

---

## Appendix — The Augmented Lagrangian, explained from scratch

*For readers unfamiliar with constrained optimization. The goal is to convey what
it is, why it fixes our drift, and what it enables.*

### A1. What we do now, in optimization language

Our training loss is a **weighted sum**:
`L = λ_pde·L_pde + λ_tc·L_tc + λ_bc·L_bc + λ_data·L_data`, with the λ's auto-tuned
by grad-norm. In constrained-optimization terms this is the **penalty method**:
treat the physics terms as soft penalties and scale them up until they're
"satisfied." The penalty method has a well-known flaw — **to truly enforce a
constraint you must send its weight to infinity**, and large weights make the
optimization stiff and unstable. Our `λ_data → 10⁵` runaway is the textbook
symptom of this ill-conditioning.

### A2. What we *should* be saying

Not "minimize a blend of physics and data as if they're the same kind of thing,"
but: **minimize the market-data misfit *subject to* obeying the Black–Scholes PDE,
the boundary/terminal conditions, and no-arbitrage.** Formally:

```
minimize_θ    L_data(θ)                      ← the objective (fit the market)
subject to    L_pde(θ) = 0                    ← equality constraints
              L_tc(θ)  = 0
              L_bc(θ)  = 0
              butterfly(θ) ≤ 0                ← inequality constraints
              calendar(θ)  ≤ 0
```

This separation — data is the **objective**, physics + no-arbitrage are
**constraints** — is the honest structure of the problem. The weighted sum throws
it away by treating everything as one blended loss.

### A3. The three classical ways to solve a constrained problem

1. **Penalty method (what we do):** add `(ρ/2)·Σ c_k²` and crank ρ. Needs ρ→∞ for
   exact satisfaction → ill-conditioned. ← our drift.
2. **Plain Lagrangian / dual ascent:** introduce a *multiplier* `μ_k` per
   constraint, `L = L_data + Σ μ_k c_k`; ascend on μ, descend on θ. Elegant, but
   unstable for non-convex deep networks.
3. **Augmented Lagrangian (the standard fix; Hestenes & Powell, 1969):** combine
   the two —
   ```
   L_A(θ, μ; ρ) = L_data(θ) + Σ_k μ_k c_k(θ) + (ρ/2) Σ_k c_k(θ)²
   ```
   — and alternate: (a) a few gradient-descent steps on θ; (b) a *multiplier
   update* `μ_k ← μ_k + ρ·c_k`  (for inequalities, `μ_k ← max(0, μ_k + ρ·c_k)`).

### A4. Why the augmented version works (the key intuition)

The multiplier `μ_k` **accumulates the constraint violation** across iterations
and does the job that ρ→∞ would do — but with a **fixed, moderate ρ**, so no
ill-conditioning. And it is **self-limiting**: once a constraint is satisfied
(`c_k ≈ 0`), the update `μ_k += ρ·c_k` adds ≈ 0, so `μ_k` stops growing and settles
at exactly the value needed to hold the constraint. Compare our penalty weights,
which grow without bound precisely *because* satisfaction requires it.

> **Analogy.** The penalty method is like fining someone ever-larger amounts until
> they comply — the fine must escalate toward infinity. The augmented Lagrangian
> is a **thermostat / integral controller**: the multiplier integrates the error
> and converges to exactly the "pressure" needed to hold the constraint, then
> holds steady. No runaway.

### A5. How it maps onto our PINN

- **Objective:** `L_data` (fit market quotes).
- **Constraints, each with its own multiplier:** PDE residual, terminal, boundary
  (equalities → driven to ~0); butterfly, calendar (inequalities → multiplier
  activates *only when violated*); plus, in Stage 2, σ-regularization.
- The multipliers `μ` replace the fragile adaptive `λ`'s. They self-limit → no
  runaway → the trajectory **converges** → final-epoch becomes reliable → we can
  **drop the validation set** (this is the thread that ties P1 → P2 → P3 together).

### A6. What it enables (why it may be worth the lift)

1. **Fixes the drift at the source** — self-limiting multipliers, fixed ρ.
2. **Encodes the right structure** — physics & no-arbitrage are constraints,
   market data is the objective; not co-equal losses.
3. **Unifies the no-arbitrage objective** — arbitrage is just an inequality
   constraint, self-modulating (active where violated, ~0 where Stage 2 already
   keeps the surface convex). "Train against arbitrage in both stages" becomes
   automatic rather than a hand-tuned extra term.
4. **Scales to Stage 2** — σ-regularization and the second network's terms are
   simply more constraints.

### A7. Costs & honest caveats

- One real hyperparameter, ρ (penalty strength), plus the multiplier-update
  cadence. Standard schedules exist (raise ρ slowly if a constraint isn't
  decreasing). It's *one robust knob* vs. several fragile adaptive weights.
- Outer/inner-loop structure (descend θ, periodically update μ) — a bit more code
  than a plain weighted sum.
- Convergence theory is for convex problems; for deep nets it's heuristic but
  empirically strong in PINNs (e.g. **PECANN**, Basir & Senocak 2022, which argues
  the weighted-sum is ill-conditioned and replaces it with exactly this;
  **hPINN**, Lu et al. 2021).
- Equality constraints that can't reach exactly 0 (the PDE residual won't): the
  multiplier converges to a *finite* value reflecting the residual floor, and the
  method finds the best feasible trade-off automatically — which is the behavior
  we want.

---

## Appendix B — Literature landscape: loss balancing for PINNs *with data* (surveyed 2026-06)

*Context for the F1/F2 decision. Orienting fact: most published balancing
benchmarks are **forward** problems (PDE + IC/BC only). Ours is a
**data-fusion / inverse-flavoured** problem — the supervised market term is the
destabilizer — so forward-problem track records transfer only partially. The camp
with the data-native track record is the constrained-optimization family.*

| Family | Members | Signal used | Bounded? | Data/inverse-tested? | Cost | Relevance to us |
|---|---|---|---|---|---|---|
| Gradient statistics | **ours** (Wang grad-norm variant), GradNorm, NTK weighting | per-term gradient norms / NTK trace | ours: **no**; GradNorm: Σλ renorm; NTK: partial | mostly forward | high (backward pass per term) | our unbounded variant is the failure mode; renorm is the 1-line control |
| Loss statistics | SoftAdapt, **ReLoBRaLo**, uncertainty weighting (lbPINN) | softmax of relative loss improvement / learned 1/2σ² | **yes** by construction | ReLoBRaLo: forward **+ inverse** | **low** (no per-term backward) | cheap bounded drop-in; benchmark across 19 problems: LR-annealing best on 9, ReLoBRaLo on 7, SoftAdapt & GradNorm weakest |
| Constrained optimization | **PECANN**, AL-PINN, **CAPU** (2025) | constraint values → multipliers (dual ascent) | **yes** (self-limiting) | **yes — incl. inverse + multi-fidelity data fusion** | low | best structural match; CAPU adds per-constraint penalties, monotone max-updates, conditional dual steps (updates only when loss plateaus) |
| Gradient surgery | PCGrad, dual-cone descent (2025), DB-PINN (IJCAI 2025) | gradient *directions* (conflict projection) | partial | mostly forward | high | targets *directional* conflict (our constant-σ↔smile); complementary, not a runaway fix |
| Per-term optimizers | AutoBalance (2025) | separate adaptive optimizer per loss term, aggregate preconditioned updates | n/a (no shared weights to blow up) | forward | high (memory) | novel, orthogonal, least validated |

**Implications for our decision:**
1. Our scheme is the *unbounded* member of one of the *weakest-performing*
   families — the mechanism critique (P1.3) and the published benchmarks point
   the same way.
2. Two live candidates: **ReLoBRaLo** (F1b) as the cheap bounded drop-in — fast
   falsification of "bounding cures the drift" — and the **augmented Lagrangian**
   (F2) as the principled target: the only family designed for "noisy data
   objective + physics constraints," which is literally our problem statement.
   Arbitrage (P4) and Stage 2 terms (P5) fold in as constraints for free.
3. Worth one extra diagnostic: a **gradient-alignment score** between ∇L_data and
   ∇L_pde. If the conflict is *directional* (not just magnitude), gradient-surgery
   ideas (dual-cone, DB-PINN) become relevant as a complement.

**References:** ReLoBRaLo — Bischof & Kraus, arXiv:2110.09813; PECANN — Basir &
Senocak, J. Comput. Phys. 2022; adaptive-ALM PECANN — arXiv:2306.04904; CAPU —
Hu, Basir & Senocak, arXiv:2508.15695; AL-PINN — arXiv:2205.01059; lbPINN —
Xiang et al., Neurocomputing 2022; DB-PINN — arXiv:2505.11117 (IJCAI 2025);
AutoBalance — arXiv:2510.06684; gradient alignment / dual-cone — arXiv:2502.00604;
SA-PINN — McClenny & Braga-Neto; NVIDIA PhysicsNeMo "advanced schemes" docs
(practitioner reference).
