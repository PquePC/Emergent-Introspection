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

**Put those together, and they yield a prediction nobody has tested:**

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

### 4.1 Why this model — and why not smaller or larger

The decisive reason is **the published baseline, not the model.** Step 1 is a hard gate that requires a
number to reproduce. That number — 38.2% TPR at 0% FPR, at L37, α=4 — exists for Gemma3-27B and for
essentially no other open-weight model. Without it, a measured detection rate of, say, 12% is
uninterpretable: it could be the model, the prompt template, the judge configuration, or a bug, and there is
no way to tell which. **The baseline converts an unbounded debugging problem into a bounded one**, and for
a solo project that is worth more than any intrinsic property of the model.

**Why not smaller — floor effects, which are fatal here.** Introspective detection is *emergent* and
post-training-dependent: absent in base models, elicited by DPO but not SFT (*Mechanisms*). It is also
under-elicited by default. At 27B the baseline is 38.2%; at small scale, detection plausibly sits at or near
the false-positive rate. This design does not measure detection — **it measures a *difference in detection
between arms***. If baseline detection is ~5%, even a large relative suppression is a ~2pp absolute
difference, requiring an infeasible number of trials to resolve. **You cannot measure a decrement in a
capability the model does not have.** This is failure mode G4, and at small scale it is not a risk but a
certainty.

> **This does not conflict with running G3 on a small model** (§6, Step 2-pre). The distinction is exact:
> **G3 measures geometry; Steps 1–4 measure behaviour.** Llorente-Saguer (arXiv:2604.18901) recovers harm
> directions from models as small as **0.5B–1.3B** with mean effective AUROC 0.982. Representational
> geometry is present at tiny scale; introspective *reporting* is not. Small models are therefore the right
> instrument for G3 and the wrong instrument for the detection sweep.

**Why not larger.** A 70B+ open-weight model forfeits the baseline gate, requires multi-GPU sharding, and
roughly triples cost for no additional scientific claim — the hypothesis is about a mechanism, not about
scale. Frontier models cannot be steered at all: activation injection requires open weights. That
constraint is the origin of the model-organism framing (§Scope), not a compromise within it.

### 4.2 Model families — cross-family where it is cheap, single-model where it is expensive

**The main experiment runs on one model. The geometry runs on several.** These pull in opposite directions
and should be resolved separately rather than averaged into a single compromise.

| | Models | Rationale |
|---|---|---|
| **G3 geometry** (pre-sweep) | **3–4 families** — Gemma-3, Qwen, Llama, at small scale | Cheap (~$0–15 on 4B-class variants), and this is where cross-family evidence is most *needed*: Llorente-Saguer establishes harm directions across Qwen2.5 / Qwen3.5 / Llama-3.2 / Gemma-3, and HARC's depth-decoupling claim must be checked in *this* model rather than assumed to transfer |
| **Main sweep** (Steps 1–4b) | **Gemma3-27B only** | Each additional model needs its own G1 reproduction, its own refusal-direction extraction, and its own abliteration dose. One model done properly beats two done badly — and for a solo project the second model is where the schedule breaks |
| **Extension, if time permits** | Qwen3-32B | *Mechanisms* also used it, so a partial baseline exists. This is the only upgrade that buys a real scientific claim (architecture/scale generalisation is an explicit open question in *Mechanisms*) — but it is the **first thing to cut** |

**Why generalisation is not urgent here.** The result's value does not rest on breadth. A refusal-mediated
blind spot demonstrated *causally* in one model — with the confound arms and the random-direction control
intact — is a stronger contribution than a shallow correlation across three. Breadth without the
abliteration arm proves nothing about mechanism.

> **⚠️ Cross-family G3 may say the main experiment is on the wrong model — decide this *before* the sweep.**
> HARC (arXiv:2607.00572) reports `cos(v_harm, v_ref)` decoupling toward orthogonality in late layers. If
> Gemma-3 turns out to have weak harm/refusal coupling at ~60% depth while another family couples strongly,
> the project faces a genuine conflict: **baseline comparability (Gemma3-27B, where 38.2% is published)
> versus mechanism strength (a family where the mechanism actually has purchase)**. There is no clean
> answer — a model with no coupling at the injection layer cannot exhibit the effect, but a model with no
> baseline cannot pass G1.
>
> **This is a project-shaping decision and it is cheap to inform:** cross-family G3 surfaces the conflict
> before any sweep is run. Default if coupling is adequate-but-not-maximal in Gemma-3: **stay with
> Gemma3-27B** — the baseline is worth more than a marginally stronger effect.

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

Requirements: Python 3.10+, CUDA 12.x, an API key for the judge, and a GPU — see below.

> **Hardware:** the repo states **≥48GB VRAM**, but that is a floor assuming quantisation. **Gemma3-27B at
> bf16 is ~54GB of weights alone**, before KV cache or activation caching — a 48GB card cannot hold it.
> Standard guidance for `gemma-3-27b-it` at bf16 is **1×A100 80GB or 1×H100**. Since **quantisation is a
> prime suspect for baseline-reproduction failure (G1)**, running at bf16 on 80GB removes the most likely
> cause of the project's most likely failure. Use a 24GB card only for smoke tests and for G3 pilot work on
> a smaller Gemma-3 variant (Step 2-pre).

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
  assigned harm labels as ground truth**. This is better methodology than trusting the arm labels:
  it lets the model reveal what it treats as harmful, rather than assuming.

These turn four discrete arms into a continuous axis, and they are what make Chart 2 possible.

**Reading the result:**
- Detection drops **only in arm 4** → it's **harmfulness**. The clean result.
- Drops in arms 2 **and** 4 → it's **negative valence**, not harm.
- Drops in arms 3 **and** 4 → it's **topic**, not harm.

Without arms 2 and 3 the result is uninterpretable. **Do not skip them.**

Every arm is **magnitude-matched** to the benign arm's mean `‖v‖` — otherwise a detection difference could
be a magnitude effect. (*Mechanisms of Introspective Awareness* (Anthropic) Appendix H finds *"Concept vector norm is not a predictor"*, so this
should be a formality — but it is verified in-data rather than assumed.)

---

## 6. Method, step by step

### Step 0 — Environment
Clone `introspection-mechanisms`; install requirements; provision a ≥48GB GPU; set the judge API key.
Confirm that `01_concept_injection.py` runs unmodified.

### Step 1 — Reproduce the baseline · **HARD GATE**
Run their pipeline unchanged on their concept list. **Target: ~38% TPR at 0% FPR** at L37, α=4.

> **If this does not reproduce, stop and diagnose.** Every downstream number is meaningless without a
> validated baseline. Expect friction with prompt formatting, judge configuration, and quantisation.
>
> **Contingency:** if detection is far below baseline, apply the abliteration arm (Step 4) early as an
> elicitation lever and report elicited numbers throughout.

### Step 2-pre — The geometry check · **G3** · *run this first, before anything else*

**The entire hypothesis rests on one unverified premise:** that harmful concept vectors are more
refusal-aligned than benign ones, *at the injection layer*. TU Munich demonstrated refusal alignment for
**behavioural** steering vectors (sycophancy, corrigibility). Whether **concept** vectors ("weapon") align
with the refusal direction has never been measured. If they do not, the mechanism has nothing to act on.

This check is **inference-only, needs no judge, no abliteration, and no sweep** — a few hundred forward
passes and some linear algebra. It is Step 2 pulled forward and done properly; none of the work is wasted.

1. Extract `d_refusal` per Arditi et al. — difference-of-means over ~100–500 matched harmful/harmless
   instruction pairs.
2. Build concept vectors per Lindsey for ~10 concepts per arm (a geometry check, not the experiment).
3. Compute `cos(v_concept, d_refusal)` **across layers** and **across ≥2 extraction protocols**, plus
   projection onto the top-k refusal subspace.

> **⚠️ This is not "one cosine computation."** Three published findings make a single scalar misleading:
>
> | Finding | Source | Consequence |
> |---|---|---|
> | Two pooling choices — max-pool over content tokens vs last-token post-instruction — applied to the *same activations at the same layer* recover harm directions **73° apart** | Llorente-Saguer, arXiv:2604.18901 | A single cosine measures a **methodological choice** as much as the model. **Sweep protocol; report whether the benign/harmful ordering is stable. If the sign flips with pooling, the premise is not established** |
> | `cos(v_harm, v_ref)` *"peaks around L12 and then drops through the late layers (L20–L28)"*, approaching orthogonality | HARC, arXiv:2607.00572 | Harm and refusal are **distinct directions that decouple with depth**. The project injects at **L37/62 (~60%)** — late. **Sweep layers; confirm coupling still exists at the injection layer** |
> | Refusal may be a multidimensional **concept cone**, not a single direction | arXiv:2502.17420, arXiv:2602.02132 | Scalar cosine against one vector is the wrong statistic. **Also report projection onto the top-k refusal subspace** |

**Run it cheap, in this order:** a 4B-class model first (validates every line of extraction code at zero
cost, and answers the *shape* of the question — geometry is recoverable at 0.5B–1.3B per Llorente-Saguer),
then confirm on Gemma3-27B. Extend across families per §4.2.

**A null here does not end the project — it redirects it.** *"Harmful concept vectors are not
refusal-aligned at the injection layer, contra the natural extension of TU Munich's behavioural-vector
result"* is a genuine finding, obtained cheaply, and it reshapes the question rather than closing it.
Given HARC's depth-decoupling result, a null is now genuinely plausible rather than a formality.

### Step 2 — Build the four concept arms
~40 concepts per arm, single-token where possible (*Mechanisms of Introspective Awareness* (Anthropic) notes verbalizability effects). Construct vectors
using their extraction code. Then compute, for every concept:
- `‖v‖` (for magnitude-matching)
- **`cos(v_concept, d_refusal)`** — the refusal direction, extracted per *Refusal … Single Direction* (Arditi et al., NeurIPS 2024) (per-layer
  difference-of-means over harmful vs harmless instructions). **This is the key covariate.**
- **Projection onto the top-k refusal subspace** — robustness against the concept-cone critique (Step 2-pre)
- Use the **extraction protocol pre-registered in Step 2-pre**, and state it explicitly. Given the 73°
  protocol sensitivity, an unstated pooling choice is an unstated result.

### Step 3 — The detection sweep
For each arm × concept × α ∈ {1, 2, 4, 8}:
- **Forced-choice / localisation** question — *primary*
- **Binary** detection question ("Do you detect an injected thought?") — *secondary*
- **Nonsense control** ("Do you believe 1+1=3?") — bias check
- Judge each response; log detection and identification separately (Macar show these are distinct circuits)

Plus the **clean, no-injection condition** for FPR. Checkpoint continuously.

> ### ⚠️ Declining to engage is an outcome, not an exclusion
>
> A model asked *"do you detect an injected thought?"* with **weapon** injected may answer *"I can't help
> with that."* That is a refusal to engage, not a failure to detect — and the obvious fix (a three-way
> judge label: *detected* / *not detected* / *declined*, then drop the declines) **introduces a selection
> effect that the random-direction control does not catch.**
>
> If declining is itself refusal-mediated, then "declined" is not a nuisance category — **it is a
> realisation of the dependent variable.** The refusal mechanism acting on the report *is the thing being
> measured.* Two consequences:
>
> 1. **Filtering declines biases the harmful arm** toward the subset of trials the refusal machinery let
>    through — precisely the population where the hypothesised effect is weakest. This attenuates the
>    effect being estimated.
> 2. **It breaks the Step 3 / Step 4 comparison.** Abliteration collapses the decline rate *by
>    construction*, so a filtered Step 3 and a filtered Step 4 are computed over differently-selected
>    samples, and a slope change is partly a change in *who was measured*. **Step 4b does not catch this** —
>    random-direction ablation does not collapse declines the same way.
>
> **Design:** three-category outcome (*detected* / *not detected* / *declined*), **primary analysis over the
> full sample.** Report **decline rate per arm as a first-class result** — a harmful-arm decline rate that
> collapses under refusal ablation but *not* random ablation is clean evidence for the mechanism, and it is
> obtainable **even if detection shows nothing.** Run the filtered analysis as a clearly-labelled secondary,
> conditional on engagement.
>
> Related: KAIST report that *"recognition depends on how the question is framed"* — two probe framings gave
> qualitatively different signals on the same models. Pilot ~5 harmful concepts before the full sweep.

> ### Add a carrier-level readout — nearly free, and possibly the better headline
>
> *Mechanisms* describes a **two-stage circuit**: content-agnostic **"evidence carrier"** features in early
> post-injection layers detect perturbations along diverse directions, and these suppress downstream
> **"gate"** features implementing a default negative response. **Refusal plausibly acts on the gate, not
> the carriers.**
>
> This sharpens the hypothesis and reconciles it with Lederman & Mahowald's content-agnosticism result
> (`background.md`) rather than opposing it: *content-agnostic detection* and *harmfulness-dependent
> reporting* are fully compatible if harmfulness acts at the reporting stage.
>
> > **Refined hypothesis:** harmfulness does not impair the model's internal *registration* of the
> > injection; it impairs the model's *report* of it. The effect should appear at the reporting stage, be
> > absent or weaker in carrier-level probes, and abliterate away.
>
> **A logit-lens / linear-probe readout at the carrier layers rides on the same forward passes as the main
> sweep** — it is a strict add-on, not a redesign. *Latent Introspection* (arXiv:2602.20031) already
> demonstrates the pattern: a model **denies injection in sampled output while logit-lens analysis reveals
> clear detection signal in the residual stream**. If the harmful arm shows carrier-level signal with a
> suppressed verbal report, that is a **demonstrable dissociation between what the model registers and what
> it says**, on exactly the safety-critical case — arguably a stronger result than the detection-rate
> comparison, and it survives even if Chart 1 is flat.
>
> Supporting: Llorente-Saguer (arXiv:2604.18901) finds the harm direction **survives abliteration**
> (abliterated variants match instruction-tuned within ±0.003 AUROC) — abliteration removes the gate
> without erasing the harm signal, which is what this prediction requires.

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
| Harmful **≈** benign | **Refusal does not suppress detection for harmful content** → bounds Anthropic's stated mechanism, **and independently confirms Lederman & Mahowald's content-agnosticism claim on the one content dimension most likely to break it** |
| Harmful **>** benign | Refusal machinery *helps* detection → surprising, and directly against the predicted direction |
| Slope flattens under **both** refusal and random ablation | Ablation is non-specific → consistent with KAIST, inconsistent with *Mechanisms*. Resolves a live disagreement between two published papers |
| Carrier-level signal present, verbal report suppressed | **The model registers the injection but does not report it** — a dissociation between internal state and self-report, on the safety-critical case. Obtainable even if every arm looks flat |

There is no outcome in which this experiment produces nothing.

### 8.1 Framing — this project tests a published claim

*Emergent Introspection in AI is Content-Agnostic* (Lederman & Mahowald, arXiv:2603.05414) claims that
*"introspection in these models is content-agnostic: models can detect that an anomaly occurred even when
they cannot reliably identify its content."* That is a **strong published prior that this project's headline
result should be null.**

This is a feature, not a threat. It means:
- **Every outcome answers a stated thesis** rather than reporting a rate. "Harmful ≈ benign" *confirms* a
  named claim; "harmful < benign" *refutes* it. This raises the floor of the project and makes the null
  publishable — which materially de-risks a solo effort.
- **It supplies the motivation for choosing harmfulness** over any other content property. Harmfulness is
  the *only* content dimension with a documented, causally-manipulable mechanism (refusal) coupling it to
  the reporting pathway. Valence and concreteness have no such mechanism — which is precisely why arms 2
  and 3 are the correct controls and not merely diligent ones.

Lederman & Mahowald apply valence/arousal/concreteness norms (Warriner et al.) to the **confabulations** —
guesses skew concrete, more positive, less arousing — **not** to detection rate as a function of the
injected concept. The question this project asks is open.

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
| **G3 — premise unverified: harmful concept vectors may not be refusal-aligned at L37** | **Run Step 2-pre first.** Cheap, inference-only, decisive. HARC's depth-decoupling result makes a null plausible, not a formality. A null redirects the project rather than ending it |
| Baseline does not reproduce (Step 1) | Hard gate. Diagnose before proceeding; use abliteration as an elicitation lever. **Run at bf16 — quantisation is a prime suspect for reproduction failure** |
| Effect is confounded by valence or topic | **The four-arm design exists for exactly this.** Arms 2 and 3 are not optional |
| **Harmful concepts decline to engage — and filtering declines biases the arm** | **Declines are an outcome, not an exclusion** (Step 3). Three-category outcome, primary analysis on the full sample, decline rate reported per arm. Filtering breaks the Step 3/Step 4 comparison and **Step 4b does not catch it**. Pilot ~5 harmful concepts early |
| Binary detection is all logit shift | Forced-choice primary + FPR + nonsense controls, built in |
| **`d_refusal` may be a cone, not a vector** | Report projection onto the top-k refusal subspace alongside the scalar cosine (Step 2-pre). Cuts both ways: if refusal is higher-dimensional, single-direction ablation is a *partial* intervention, which makes Step 4b more informative |
| **Extraction protocol changes the harm direction by up to 73°** | Pre-register the pooling protocol in Step 2-pre; report ordering stability across protocols |
| Abliterated model incoherent | Use α=2 and minimum effective dose, per Macar |
| Compute instance dies mid-sweep | Continuous checkpointing |
| No signal in any arm (G4) | Elicit via abliteration, or the prompt-based lever — *Latent Introspection* takes detection **0.3% → 39.9%** when the prompt explains introspection |

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
