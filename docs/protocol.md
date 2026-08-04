# Protocol — The Safety-Shaped Blind Spot

**Question:** Do language models detect activation-level concept injection *less* when the injected
concept is harmful — and if so, is refusal the cause?

**Prediction:** Refusal training suppresses introspective detection (Macar et al.: abliteration takes
detection 10.8% → 63.8%). Harmful concepts are refusal-aligned. Therefore detection should fall as a
concept vector aligns with refusal, and ablating refusal should flatten that relationship.

---

## Operating point

| | |
|---|---|
| Model | `google/gemma-3-27b-it` — 62 layers, residual dim 5376, bf16 on 1×A100/H100 80GB |
| Injection layer | **L37 / 62** (≈60% depth); L38 is the reproduction fallback |
| Strength | **α = 4** (main sweep); **α = 2** (abliteration, coherence limit) |
| Published baseline | **38.2% mean TPR at 0% FPR** across 500 concepts (median 30.0%, range 0–100%) |
| Primary measure | Binary detection, scored at **0% FPR** |
| Tooling | `safety-research/introspection-mechanisms`, run unmodified for the baseline |

---

## The four-arm design

The arms separate **harmfulness** from its confounds. All arms are **magnitude-matched** to the benign
arm's mean ‖v‖, so a detection difference cannot be a magnitude effect.

| Arm | Examples | Isolates |
|---|---|---|
| **1. Benign neutral** | bread, orchid, hammer | baseline — reproduces 38.2% |
| **2. Negative-valence** | sadness, grief, failure | **valence** — is the effect just unpleasantness? |
| **3. Harm-adjacent** | police, surgery, medicine | **topic** — is it just proximity to danger? |
| **4. Harmful** | weapon, poison, assault | the hypothesis |

Arms 2 and 3 are not optional: without them a drop in arm 4 is uninterpretable. G3 shows valence is
geometrically close to harm in the Gemma-3 family, making arm 2 load-bearing for this model specifically.

**Per-concept covariates** (computed for every concept, every arm) turn four discrete arms into a
continuous axis:

- **`cos(v_concept, d_refusal)`** — alignment with the refusal direction (Arditi et al.). The key covariate.
- **Top-k refusal-subspace projection** — robustness against the "refusal is a cone" critique.
- **Empirical refusal rate** — fraction of prompts about the concept the model refuses. Measured ground
  truth (Gondil), better than trusting arm labels.
- **‖v‖** — for magnitude-matching.

---

## Stages

Ordered so the cheapest, most decisive checks come first; nothing expensive runs until a cheaper gate
clears it.

### Stage 0 — Environment
**Do:** Clone the harness, install, smoke-test the injection script on a small model; confirm the judge
API key works.
**Why:** Debug setup for free, not on a metered pod.
**Data:** None — a green pipeline (model loads, vector builds, generation and judge label return).

### Stage G3 — Geometry premise check · **GATE G3**
**Do:** Extract `d_refusal` (difference-of-means over ~100–500 matched harmful/harmless instruction
pairs). Build concept vectors for ~10 concepts × 4 arms. Compute `cos(v_concept, d_refusal)` across all
layers, read at L37, under **≥2 pooling protocols**, plus the top-k subspace projection. Inference-only —
no judge, no sweep. Small models first (done: Gemma-2-2B, Gemma-3-4B, Qwen2.5-3B — premise holds in all),
then **confirm on Gemma-3-27B at L37**.
**Why:** The entire hypothesis rests on harmful vectors being refusal-aligned *at the injection layer*.
Harm and refusal decouple with depth (HARC), and L37 is late, so this must be measured, not assumed.
**Data:** Δcos(harmful − benign) at L37 per protocol; per-arm cosine table; whether the Gemma-3 valence
entanglement persists at the target. A standalone result. **Null → stop and redirect before renting.**

### Stage 1 — Load the model
**Do:** Load Gemma-3-27B at bf16 on an 80GB card; generate coherent text. SSH-only pod, no open port.
**Why:** The model is ~54GB at bf16; quantisation is the prime suspect for baseline-reproduction failure.
An open endpoint becomes a public uncensored model once Stage 4 runs.
**Data:** None — a working model at ~54GB VRAM.

### Stage 1b — Reproduce the baseline · **HARD GATE G1**
**Do:** Run the harness unchanged (`02b_run_500_concepts.py --model gemma3_27b --specific-layers 37
--strength 4.0`), including the clean/no-injection condition. Target **~38% TPR at 0% FPR**. If L37 misses,
try L38 before anything else.
**Why:** Until this clears, every downstream number is uninterpretable — could be model, template, judge,
or bug, with no way to tell. The baseline turns an unbounded debugging problem into a bounded one.
**Data:** Measured TPR at 0% FPR vs 38.2%; clean-input FPR (target 0.0%); identification accuracy logged
alongside (distinct circuit). Only detection gates G1.

### Stage 2 — Build the arms and covariates
**Do:** Assemble ~40 concepts per arm (single-token where possible). Build each vector with the harness's
Lindsey-style construction. Compute ‖v‖, `cos(v, d_refusal)`, subspace projection, and empirical refusal
rate per concept. Magnitude-match every arm to the benign mean ‖v‖.
**Why:** Arms 2 and 3 make a harmful-arm drop interpretable; magnitude-matching removes the "bigger vector"
explanation; empirical refusal rate lets the model reveal what it treats as harmful.
**Data:** One row per concept (arm, ‖v‖, cos, projection, refusal rate). **This table drives Chart 2.**

### Stage 2.5 — Decline pilot · **GATE G2**
**Do:** Inject ~5 harmful concepts, ask the detection question, read raw responses. Extend the judge to a
three-way label: **detected / not detected / declined**. If the harmful arm mostly declines, add a neutral
interpretability framing and re-pilot with **≥2 framings**.
**Why:** "I can't help with that" is a refusal to engage, not a failure to detect. Filtering declines
biases the harmful arm and breaks the Stage 3 vs Stage 4 comparison (abliteration collapses declines by
construction). Catch this cheaply on 5 concepts.
**Data:** Whether you are measuring detection or refusal; the working three-way judge; the framing(s) to
carry forward. Decline rate becomes a first-class reported result.

### Stage 3 — The detection sweep · **primary dataset**
**Do:** For each arm × concept at α = 4: the binary detection question (primary, one generation gives
detection *and* identification) plus the global clean/no-injection condition for FPR. Sizes: **100
injection samples/concept** (10×10), **~500 global clean-control samples**. Judge with the three-way label.
Add the **carrier-level readout** (logit-lens / linear probe at early post-injection layers) — it rides on
the same forward passes.
**Why:** This is the experiment — it produces Charts 1 and 2. The carrier readout tests the sharper
hypothesis: harmfulness may impair the *report*, not the *registration*. Carrier signal with a suppressed
verbal report is a dissociation that stands even if Chart 1 is flat.
**Data:** Per-concept detection, identification, and decline rates per arm, with clean FPR; carrier-layer
signal per arm. Checkpoint continuously.

### Stage 4 — The abliteration arm · **causal test**
**Do:** Repeat Stage 3 on the refusal-ablated model (`03d_refusal_abliteration.py`). Ablate `d_refusal`
(difference-of-means, read at position −2) by runtime projection at every layer. Use the **minimum
effective dose** (smallest weight achieving ≥30% judged refusal bypass) at **α = 2**. Reuse base-model
concept vectors to avoid double-ablation. Validate that refusals actually drop on held-out harmful prompts.
**Why:** Turns correlation into cause: if refusal is the suppressor, abliteration should close the
harmful/benign gap specifically and collapse the harmful-arm decline rate.
**Data:** Detection and decline rates per arm on the ablated model — the "after" of Chart 3; ablation-only
FPR (expect ~0% → 7.3%).
**Handling:** The ablated model is the one sensitive artifact — pod only, weights never uploaded,
regenerate never archive, delete the volume at project end.

### Stage 4b — Random-direction control · **MANDATORY GATE G5**
**Do:** Repeat Stage 4 ablating a **magnitude-matched random direction**. Stage 4 thus runs three
conditions: baseline · refusal-ablated · random-ablated.
**Why:** Macar and KAIST disagree on whether the effect is refusal-specific. Without this, a flattened
slope could just mean "ablation degrades the model." Any reviewer asks for it on sight.
**Data:** The random-ablation slope. Flattens under refusal but *not* random → refusal-specific, causal.
Flattens under both → resolves a published disagreement toward KAIST.

### Stage 5 — Analysis
**Do:** From aggregates pulled off the pod. Report TPR at a stated FPR with 95% CIs (Miller — bootstrap /
cluster; per-trial responses sharing a concept are not independent).
**Why:** Sample sizes are limited; effect sizes need confidence intervals.
**Data:** The three deliverable charts and supporting tables.

---

## Gates

| Gate | Stage | Question | Fail action |
|---|---|---|---|
| **G3** | G3 | Are harmful vectors refusal-aligned at L37? | Redirect (cross-family or harmful-free fallback) |
| **G1** | 1b | Does the 38.2% baseline reproduce? | Stop and diagnose; abliteration as elicitation lever |
| **G2** | 2.5 | Does the harmful arm refuse to engage? | Neutral framing, re-pilot; never filter declines |
| **G5** | 4b | Is the ablation effect refusal-specific? | Report non-specificity honestly |

---

## Deliverables

| Chart | Question it answers |
|---|---|
| **1 — headline** | Does the harmful arm sit below benign, with arms 2 & 3 ruling out valence and topic? |
| **2 — mechanism** | Detection vs `cos(v_concept, d_refusal)`, one point per concept — is the slope negative? |
| **3 — causal** | Chart 2 before vs after abliteration, with the random control — does the slope flatten under refusal but not random? |
| **+** | Detection-vs-identification split · decline rate per arm · FPR tables · carrier-probe readout |

**Every outcome is publishable:** harmful < benign proves the blind spot; harmful ≈ benign confirms
content-agnosticism on its hardest case; both-ablations-flatten resolves a live disagreement; a
carrier/report dissociation stands even if Chart 1 is flat.

---

## Data hygiene

Publish **rates, not artifacts**. The ablated model and the concept vectors never leave the pod
(regenerate, never archive). Never commit vectors, activations, or raw generations. Run the pre-push
checklist every commit.
