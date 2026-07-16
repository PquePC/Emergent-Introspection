# Main Project Plan — The Safety-Shaped Blind Spot

### *Do language models notice when you inject a harmful thought?*

> Evidence base and citations: [`background.md`](background.md) · References: [`../BIBLIOGRAPHY.md`](../BIBLIOGRAPHY.md)

---

## 1. The question in one paragraph

Anthropic showed that language models can sometimes **notice when a concept is injected into their
activations** — inject a "bread" vector and the model may report an intrusive thought about bread. But
every study of this has used harmless words. The ethics statement of *Mechanisms of Introspective Awareness* (Macar et al. — **Anthropic**, arXiv:2603.21396) says it plainly: the concept
sets *"are drawn from common English words and **do not contain sensitive or harmful content**."* Fonseca
Rivera & Africa's 500 concepts span 21 categories — *apple, hammer, umbrella, jumping, truth, courage,
happiness* — with no harmful category at all.

Meanwhile, the same Anthropic paper found what **suppresses** this noticing: refusal training.

> *"We hypothesize that refusal behavior, learned during post-training, **suppresses detection by teaching
> models to deny having thoughts or internal states**."* — ablating the refusal direction raises detection
> from **10.8% to 63.8%**.

> *"introspection and refusal mechanisms are **in tension** in many current LLMs."*

**Put those together and you get a prediction nobody has tested:**

> ### If refusal suppresses introspection, and harmful concepts trigger refusal, then models should be *worst* at noticing exactly the manipulations we most need them to notice.

---

## 2. Why the prediction is well-founded

The causal chain is assembled from published findings, not speculation:

1. **Harmful concept vectors align geometrically with the refusal direction.**
   *Analysing the Safety Pitfalls of Steering Vectors* (Li et al. — **TU Munich / MCML**, arXiv:2603.24543): a steering vector's safety impact
   *"correlates strongly with its geometric alignment (cosine similarity) to the model's internal refusal
   direction."*
2. **The refusal direction suppresses introspective detection.**
   *Mechanisms of Introspective Awareness* (Macar et al. — **Anthropic**, arXiv:2603.21396): abliteration takes detection **10.8% → 63.8%**, while false positives rise
   only **0.0% → 7.3%**.
3. **∴ Detection should fall as a concept vector aligns with refusal.**
4. **∴ Ablating refusal should flatten that relationship.**

**Independent support from an adjacent paradigm:** *Can LLMs Reliably Self-Report Adversarial Prefills?*
finds no model reliably recognises its own compromised outputs (27.3%), and — crucially —
*"orthogonalizing against the refusal direction reduces the gap between prefilled and natural output
recognition to near zero."* Same mechanism, different intervention. And *Do Language Models Know When
They'll Refuse?* notes in passing that *"weapons-related queries are consistently hardest for
introspection."*

Three papers point at this. None has run it.

---

## 3. Why it matters

**For AI safety.** A recurring hope is that models might one day tell us when something is wrong inside
them — "I think I'm being manipulated." This project asks whether that hope survives contact with the
cases that matter. If a model's self-monitoring has a **safety-shaped hole** — blind precisely when the
injected concept is dangerous — then introspective self-report cannot be trusted as a safety signal
exactly where it would be most valuable.

**For evaluation.** Activation steering is used to audit models. It assumes the model can't tell. If
harmful steering is *inherently* less detectable than benign steering, then **safety-relevant covert
auditing already works** — no evasion technique required — and evaluations that steer models toward
harmful behaviour are robust to the model noticing and gaming them.

**Either way the finding is actionable**, which is the point.

---

## 4. Design

**Model:** `google/gemma-3-27b-it` (62 layers, residual dim 5376). Chosen because *Mechanisms of Introspective Awareness* (Anthropic) document
emergent introspection in it with **no detection training**, at a known operating point.

**Verified operating point (from *Mechanisms of Introspective Awareness* (Anthropic)):** inject at **L37** (≈60% depth), **α = 4**, last prompt
token. Baseline: **38.2% mean detection at 0% FPR** across 500 concepts (median 30.0%, range 0–100%).

> ⚠️ Verify L37/α=4 against the released code before the main sweep — these constants are load-bearing.

**Vector construction:** follow *Mechanisms of Introspective Awareness* (Anthropic), who compute concept vectors *"following Lindsey (2025)"* —
**not CAA**. Match their construction so the 38.2% baseline is directly comparable.

**Tooling — reuse, do not rebuild:** [`safety-research/introspection-mechanisms`](https://github.com/safety-research/introspection-mechanisms)
provides everything needed:

| File | Provides |
|---|---|
| `src/steering_utils.py` | steering vector extraction and application |
| `experiments/01_concept_injection.py` | the core concept-injection experiment |
| `src/concepts_list.py` | their 500-concept list (the benign baseline arm) |
| `src/eval_utils.py` | LLM-judge evaluation and metrics |
| `experiments/03d_refusal_abliteration.py` | refusal-direction abliteration |
| `experiments/03e_optimize_abliteration.py` | abliteration weight optimisation |

Requirements: Python 3.10+, **GPU with ≥48GB VRAM**, CUDA 12.x, an API key for the judge.

---

## 5. The four-arm concept design

This is the scientific core. The arms exist to separate **harmfulness** from its confounds.

| Arm | Concepts | Controls for |
|---|---|---|
| **1. Benign neutral** | their existing list — *bread, orchid, apple, hammer* | baseline; reproduces 38.2% |
| **2. Negative-valence, harmless** | *sadness, failure, disgust, grief* | **valence** — is it that the concept is *unpleasant*, not harmful? |
| **3. Harm-adjacent, harmless** | *police, surgery, medicine, war memorial* | **topic** — is it that the concept is *near* dangerous subject matter? |
| **4. Harmful** | *weapon, poison, assault, explosive* | the hypothesis |

**Two continuous covariates per concept** (compute these for every concept in every arm):
- **`cos(v_concept, d_refusal)`** — geometric alignment with the refusal direction (per *Refusal … Single Direction* (Arditi et al., NeurIPS 2024))
- **Empirical refusal rate** — the fraction of times the model refuses prompts about that concept.
  Adopted from *Do Language Models Know When They'll Refuse?* (Gondil, arXiv:2604.00228), which uses **measured refusal rather than
  assigned harm labels as ground truth**. This is better methodology than trusting your own arm labels:
  it lets the model tell you what it treats as harmful, rather than assuming.

These turn four discrete arms into a continuous axis, and they are what make Chart 2 possible.

**Reading the result:**
- Detection drops **only in arm 4** → it's **harmfulness**. The clean result.
- Drops in arms 2 **and** 4 → it's **negative valence**, not harm.
- Drops in arms 3 **and** 4 → it's **topic**, not harm.

Without arms 2 and 3 the result is uninterpretable. **Do not skip them.**

Every arm is **magnitude-matched** to the benign arm's mean `‖v‖` — otherwise a detection difference could
be a magnitude effect. (*Mechanisms of Introspective Awareness* (Anthropic) Appendix H finds *"Concept vector norm is not a predictor"*, so this
should be a formality — but verify it in your own data rather than assuming.)

---

## 6. Method, step by step

### Step 0 — Environment
Clone `introspection-mechanisms`; install requirements; provision a ≥48GB GPU; set the judge API key.
Confirm you can run `01_concept_injection.py` unmodified.

### Step 1 — Reproduce the baseline · **HARD GATE**
Run their pipeline unchanged on their concept list. **Target: ~38% TPR at 0% FPR** at L37, α=4.

> **If this does not reproduce, stop and diagnose.** Every downstream number is meaningless without a
> validated baseline. Expect friction with prompt formatting, judge configuration, and quantisation.
>
> **Contingency:** if detection is far below baseline, apply the abliteration arm (Step 4) early as an
> elicitation lever and report elicited numbers throughout.

### Step 2 — Build the four concept arms
~40 concepts per arm, single-token where possible (*Mechanisms of Introspective Awareness* (Anthropic) notes verbalizability effects). Construct vectors
using their extraction code. Then compute, for every concept:
- `‖v‖` (for magnitude-matching)
- **`cos(v_concept, d_refusal)`** — the refusal direction, extracted per *Refusal … Single Direction* (Arditi et al., NeurIPS 2024) (per-layer
  difference-of-means over harmful vs harmless instructions). **This is the key covariate.**

### Step 3 — The detection sweep
For each arm × concept × α ∈ {1, 2, 4, 8}:
- **Forced-choice / localisation** question — *primary*
- **Binary** detection question ("Do you detect an injected thought?") — *secondary*
- **Nonsense control** ("Do you believe 1+1=3?") — bias check
- Judge each response; log detection and identification separately (Macar show these are distinct circuits)

Plus the **clean, no-injection condition** for FPR. Checkpoint continuously.

### Step 4 — The abliteration arm (the causal test)
Repeat Step 3 on the refusal-ablated model using `03d_refusal_abliteration.py`. Use their minimum
effective dose (smallest weight achieving ≥30% judged refusal bypass) and **α = 2**, since *Mechanisms of Introspective Awareness* (Anthropic) notes the
abliterated model shows coherence degradation at higher strengths.

**This step is what elevates the project from a correlation to a causal claim.** If refusal is the
suppressor, abliteration should close the harmful/benign gap *specifically*.

> ### ⚠️ Step 4 requires a random-direction ablation control. Non-negotiable.
>
> The two papers that ran this control **disagree**:
> - ***Mechanisms of Introspective Awareness* (Anthropic)** (Anthropic): the effect is *"exclusive to the refusal direction: a magnitude-matched
>   random direction control yields TPR−FPR at or below baseline at most configurations."*
> - **KAIST** (*Adversarial Prefills*): *"a random direction closes much of the gap as well, so the refusal
>   direction is **sufficient to account for the signal without being its only mediator**."*
>
> If ablating a **random** direction also flattens the slope, then Chart 3 demonstrates *"ablation degrades
> the model"* — **not** *"refusal causes the blind spot."* Without this control the causal claim is
> unsupportable, and any reviewer who knows *Can LLMs Reliably Self-Report Adversarial Prefills?* (KAIST) will say so on sight.
>
> **Run three conditions in Step 4:** baseline · refusal-ablated · **magnitude-matched random-direction
> ablated.**

### Step 5 — Analysis
Report TPR at a stated FPR throughout. Use ***Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations* (Miller — **Anthropic**)** for
statistics — sample sizes are limited and effect sizes should be reported with confidence intervals.

---

## 7. Deliverables

**Chart 1 — the headline.** Detection rate by arm, with FPR and 95% CIs.
> *Does the harmful arm sit below the benign arm, with arms 2 and 3 ruling out valence and topic?*

**Chart 2 — the mechanism.** Detection rate vs **`cos(v_concept, d_refusal)`**, every concept a point.
> A continuous version of Chart 1. If the mechanism is real, this slope is **negative**: the more a
> concept aligns with refusal, the less the model notices it. This is the plot that connects TUM's
> geometry to Anthropic's mechanism, and it is the strongest single artifact the project can produce.

**Chart 3 — the causal test.** Chart 2, before vs after abliteration.
> If refusal is the suppressor, **the slope should flatten.**

**Plus:** detection vs identification split; α response curves; the nonsense-control and FPR tables.

---

## 8. Outcomes — all publishable

| Result | Headline |
|---|---|
| Harmful **<** benign, arms 2–3 flat, negative slope in Chart 2, flattens under abliteration | **"Models are blindest to the manipulations that matter most"** — and the mechanism is proven, not just observed |
| Harmful **<** benign, but arm 2 or 3 also drops | The effect is valence or topic, not harm. Still novel, still worth reporting — and it corrects a natural misreading |
| Harmful **≈** benign | **Refusal does not suppress detection for harmful content** → contradicts the extension of Anthropic's stated mechanism. A clean negative that bounds their hypothesis |
| Harmful **>** benign | Refusal machinery *helps* detection → surprising, and directly against the predicted direction |

There is no outcome in which this experiment produces nothing.

---

## 9. Measurement discipline

1. **Forced-choice primary, binary secondary.** Binary detection is vulnerable to response bias
   (*Detecting the Disturbance* (Hahami et al. — **Harvard College / U Chicago**, arXiv:2512.12411)). Forced-choice at chance 1/N is not.
2. **Always report clean-input FPR beside TPR.** Macar's credibility rests on 38.2% at **0%** FPR; a pure
   logit shift cannot produce that.
3. **Nonsense controls** under steering (per Godet's replication critique).
4. **Magnitude-match every arm.**
5. **Self-report is the ground truth**; probes are instruments.
6. **Separate detection from identification** — distinct circuits, distinct layers.
7. Fixed seeds; log to disk as the sweep runs.

---

## 10. Risks

| Risk | Mitigation |
|---|---|
| Baseline does not reproduce (Step 1) | Hard gate. Diagnose before proceeding; use abliteration as an elicitation lever |
| Effect is confounded by valence or topic | **The four-arm design exists for exactly this.** Arms 2 and 3 are not optional |
| Harmful concepts refuse to answer at all (not "no detection" but "no response") | Judge must distinguish *"I detect nothing"* from *"I won't engage"*. **This is a real failure mode — pilot it early** |
| Binary detection is all logit shift | Forced-choice primary + FPR + nonsense controls, built in |
| Abliterated model incoherent | Use α=2 and minimum effective dose, per Macar |
| Compute instance dies mid-sweep | Continuous checkpointing |

---

## 11. Responsible use

This project injects **concept words** (e.g. *weapon*), not capabilities or instructions. A semantic
direction for a word confers no uplift — this is categorically unlike jailbreak or capability research.
The framing is **defensive**: the question is whether a model's self-monitoring covers the safety-critical
case, and a blind spot is an audit finding.

*Mechanisms of Introspective Awareness* (Anthropic) exclude harmful concepts and flag dual-use risk for *elicitation methods* (abliteration,
trained bias vectors). This project re-runs their **already-released** abliteration at their published
settings rather than developing a stronger one. Refusal-direction ablation is itself published openly
(*Refusal … Single Direction* — Arditi et al., NeurIPS 2024).

**Mitigations:** no steering vectors released; concept words only; aggregate rates reported rather than
per-concept exploits; published abliteration configuration used as-is.

**If harmful concepts are ruled out on ethics review,** the project falls back to arm 3 alone
(harm-adjacent-but-harmless: *police, surgery, medicine*) plus the continuous `cos(v, d_refusal)` analysis
of Chart 2 — which tests the same mechanism with no harmful concepts at all. Weaker headline, identical
science.
