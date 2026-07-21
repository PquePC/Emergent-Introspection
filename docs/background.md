# Background

What this project rests on. Only the parts relevant to the safety-shaped blind spot — quoted verbatim
where the exact wording matters.

> Main plan: [`project-plan.md`](project-plan.md) · References: [`../BIBLIOGRAPHY.md`](../BIBLIOGRAPHY.md)

---

## 1. Key concepts

**Residual stream.** Every token position in a transformer carries a vector (Gemma3-27B: 5376
dimensions). Layers read from it and **add** their output back — a shared, accumulating "highway."
Because layers *add* to it, a vector can be injected by simple addition. That is what makes this
whole research area possible.

**Activation steering.** Adding a vector to the residual stream at a chosen layer and token position
during the forward pass. Changes behaviour **without touching any weights**.

**Concept injection.** the paradigm of *Emergent Introspective Awareness in Large Language Models* (Lindsey — **Anthropic**): inject a known concept's vector in an unrelated context and ask
whether the model notices — requiring it to report the concept *before* verbalising it, which separates
genuine introspection from confabulation.

**Refusal direction.** *Refusal in Language Models Is Mediated by a Single Direction* (Arditi, Obeso, Syed, Paleka, Panickssery & Nanda — **NeurIPS 2024**) showed refusal is mediated by **a single direction** in the
residual stream. Ablating it removes refusal across many prompts with minimal capability loss
("abliteration").

> ⚠️ **"Single direction" is contested — see §6.4.** Two papers argue refusal is a multidimensional **concept
> cone**. This matters because `cos(v_concept, d_refusal)` is the project's headline covariate.

**Harmfulness direction ≠ refusal direction.** A distinction the project originally elided, and which the
2026 geometry literature makes explicit: *harmfulness* (what the input **is**) and *refusal* (what the model
**does**) are **separate directions**. HARC (arXiv:2607.00572) reports their cosine similarity *"peaks around
L12 and then drops through the late layers (L20–L28)"* — they **decouple with depth** — and its central
premise is that jailbreaks exploit exactly that dissociation. See §3.2 and §6.5.

**TPR / FPR.** True-positive rate = detection when steering *was* applied. False-positive rate = claimed
detection when it *wasn't*. A model that always says "yes" scores 100% TPR and is worthless — **TPR is
only meaningful at a stated FPR.**

---

## 2. Established (multi-paper agreement)

1. **Models can sometimes detect injected concepts.** ~20% for Opus 4/4.1 (*Emergent Introspective Awareness* — Lindsey, **Anthropic**); **38.2% mean TPR at
   0% FPR** across 500 concepts for Gemma3-27B (*Mechanisms of Introspective Awareness* (Anthropic)).
2. **Inject at ~60–67% depth.** *Emergent Introspective Awareness* (Lindsey, Anthropic) (~2/3), *Mechanisms of Introspective Awareness* (Anthropic) (**L37/62**), *Steering Awareness* (Fonseca Rivera & Africa, UT Austin) (67%). Robust.
3. **It's a post-training phenomenon.** Absent in base models; elicited by **DPO, not SFT** (Anthropic).
4. **A strength sweet spot exists.** Too weak → nothing; too strong → garbled. *Emergent Introspective Awareness* (Lindsey): strengths 2 and 4;
   *Mechanisms of Introspective Awareness* (Anthropic) optimum: **α = 4**.
5. **Self-report + LLM judge is the standard measurement.** Probes are instruments, not ground truth.
6. **Detection and identification are distinct** — different circuits, different layers (Anthropic).
7. **Introspection is heavily under-elicited.** Refusal ablation: **10.8% → 63.8%** (Anthropic).

---

## 3. The two facts this project stands on

### 3.1 Refusal suppresses introspection — *Mechanisms of Introspective Awareness* (Macar et al. — **Anthropic**, arXiv:2603.21396)

> *"**Refusal ablation ("abliteration") increases true detection.** We hypothesize that refusal behavior,
> learned during post-training, **suppresses detection by teaching models to deny having thoughts or
> internal states**."*

> *"abliteration increases TPR from **10.8% to 63.8%** and introspection rate from 4.6% to 24.1% (at α=2),
> while increasing FPR only modestly from **0.0% to 7.3%**."*

> *"introspection and refusal mechanisms are **in tension** in many current LLMs."*

Corroborated internally: LoRA finetuning on preference pairs that *affirm* rather than deny having
internal states also raises detection substantially.

### 3.2 Steering vectors align with the refusal direction — *Analysing the Safety Pitfalls of Steering Vectors* (TU Munich)

> *"this phenomenon is correlated with the **directional overlap between steering vectors and the model's
> refusal behavior direction**."*

> *"steering vectors **partially align with the one-dimensional refusal direction**, and use this geometric
> overlap to examine how activation shifts can push the model off its safety manifold."*

> *"By **ablating the refusal-aligned component** from steering vectors, we consistently mitigate the
> vector's impact on ASR, providing **causal validation for the geometric interference hypothesis**."*

⚠️ **Important caveat:** *Analysing the Safety Pitfalls of Steering Vectors* (TU Munich) demonstrates this for **behavioural** steering vectors (sycophancy,
corrigibility). Whether a **concept** vector for "weapon" aligns with refusal is **untested** — which is
why the plan makes it an early gate rather than an assumption.

⚠️⚠️ **And the 2026 geometry literature makes a null plausible.** Three findings, none available when the
original chain was drafted, bear directly on this premise:

| Finding | Source | Consequence for §3.3 |
|---|---|---|
| `v_harm` and `v_ref` are **distinct** directions whose alignment *"peaks around L12 and then drops through the late layers (L20–L28)"*, approaching orthogonality at the most decoupled layer | **HARC**, arXiv:2607.00572 | The project injects at **L37/62 (~60% depth)** — late. **If harm and refusal are decoupled there, the mechanism has no purchase at the injection layer.** Must be measured, not assumed. (HARC's absolute indices are for a smaller model; convert to relative depth before drawing conclusions) |
| Two pooling choices — max-pool over content tokens vs last-token post-instruction — applied to the **same activations at the same layer** recover harm directions **73° apart** | **Llorente-Saguer**, arXiv:2604.18901 | A single cosine measures a **methodological choice** as much as the model. The premise check must sweep extraction protocol and report ordering stability |
| The harm direction **survives abliteration** — abliterated variants match instruction-tuned within **±0.003 AUROC** | **Llorente-Saguer**, arXiv:2604.18901 | **Good news.** Harm *representation* is independent of refusal *behaviour*: abliteration removes the reporting gate without erasing the harm signal. This is what the gate-vs-carrier prediction (§7.1) requires |

### 3.3 The prediction
Refusal suppresses detection (3.1). Harmful concepts should be refusal-aligned (3.2, extended). Therefore
detection should **fall** with refusal alignment, and abliteration should **flatten** that slope.

**Status of each link.** 3.1 is published and replicated. 3.2-as-extended-to-concept-vectors is **a premise,
not a result**, and the table above shows it is *contestable* rather than merely unverified. The plan
therefore opens with the premise check (`project-plan.md` Step 2-pre) before any sweep.

---

## 4. The gap

**Nobody has injected a harmful concept and measured detection.**

| Paper | Concepts used | Harmful? |
|---|---|---|
| ***Mechanisms of Introspective Awareness* (Anthropic)** | 500 concepts, 20 categories | **Excluded by policy:** *"drawn from common English words and **do not contain sensitive or harmful content**"* |
| ***Steering Awareness* (Fonseca Rivera & Africa, UT Austin)** | 500 concepts, **21 categories**: Concrete Nouns, Verbs, Adjectives, Abstract, Emotions, Animals, Nature, Food, Spatial, Temporal, Colors, Languages ×3, Technical, Professions, Events, Body Parts, Places, Materials, Quantities. Examples: *apple, hammer, umbrella, jumping, truth, courage, happiness* | **None** |
| ***Emergent Introspective Awareness*** (Lindsey, Anthropic) | 50-word random list | **None** |
| ***Emergent Introspection in AI is Content-Agnostic*** (Lederman & Mahowald, UT Austin) | Lindsey replication | **None** |

### 4.1 …but one paper predicts the answer — *Emergent Introspection in AI is Content-Agnostic* (Lederman & Mahowald — **UT Austin**, arXiv:2603.05414)

The row above is true of its *concept set* and undersells the *paper*. Its central claim is a direct prior
on this project's headline:

> *"We first extensively replicate Lindsey (2025)'s thought injection detection paradigm in large
> open-source models. We show that **introspection in these models is content-agnostic: models can detect
> that an anomaly occurred even when they cannot reliably identify its content.** The models confabulate
> injected concepts that are high-frequency and concrete (e.g. "apple"). They also **require fewer tokens to
> detect an injection than to guess the correct concept** (with wrong guesses coming earlier). We argue that
> a content-agnostic introspective mechanism is consistent with leading theories in philosophy and
> psychology."*

**If introspection is content-agnostic, detection should not vary with harmfulness** — i.e. this project's
Chart 1 should be flat.

**Why this does not preempt the project, and in fact strengthens it:**
- They apply valence/arousal/concreteness norms (Warriner et al. 2013) to the **confabulations** —
  guesses skew *more concrete, more positive, less arousing* — **not** to detection rate as a function of
  the injected concept. The question here is untouched.
- Content-agnosticism is a **claim to be tested**, and this project tests it on **the one content dimension
  with a documented, causally-manipulable mechanism coupling content to the reporting pathway.** Valence and
  concreteness have no such mechanism. Harmfulness has refusal. That asymmetry is the project's entire
  motivation, and it is also why arms 2 and 3 are the right controls rather than merely diligent ones.
- **It makes every outcome answer a stated thesis.** Flat *confirms* a named published claim on its hardest
  case; a drop *refutes* it. The null becomes publishable — which is exactly what a solo project needs.

---

## 5. Adjacent work — three papers point here, none runs it

***Can LLMs Reliably Self-Report Adversarial Prefills, and How?* (Nguyen, Ahmed & Kim — **KAIST**, arXiv:2606.23671)** — *the closest work; cite prominently.*
Ten open-weight models (3B–70B). *"No model reliably recognizes its compromised outputs"* — **27.3%** claim
rate. And the mechanism:
> *"ablating the refusal direction **collapses the recognition gap to near zero** on every ablated model."*

**But:** text **prefills**, not activation injection — a different intervention, and a different question
(*"did I author this?"* vs *"was a thought injected?"*). This project asks whether the mechanism holds in
the injection paradigm.

⚠️ **Their caveat drives a required control:**
> *"In most cases, **a random direction closes much of the gap as well**, so the refusal direction is
> **sufficient to account for the signal without being its only mediator**."*

This **directly conflicts** with *Mechanisms of Introspective Awareness* (Anthropic), which reports the abliteration effect is *"exclusive to the refusal
direction: a magnitude-matched random direction control yields TPR−FPR at or below baseline."* The conflict
is unresolved — hence the mandatory random-ablation control in the plan.

*Do Language Models Know When They'll Refuse?* (Gondil, arXiv:2604.00228) — 300 requests, **10 sensitive topics** (weapons, drugs,
hacking, self-harm, hate speech, fraud, privacy, illegal activities, manipulation, violence) × **5 harm
levels** (L1 clearly safe educational → L5 clearly harmful/AdvBench). Finds d′ = 2.4–3.5, and:
> *"**weapons-related queries are consistently hardest for introspection**."*

**But:** behavioural **self-prediction** of refusal, no injection, no steering. Two methods worth borrowing:
their **graded harm taxonomy**, and their use of **empirical refusal rates as ground truth rather than
assigned harm labels**.

**Analysing the Safety Pitfalls of Steering Vectors — *Analysing the Safety Pitfalls of Steering Vectors* (TU Munich)** — see §3.2. No introspection at all, but it
supplies the geometric link.

---

## 6. Contested — handle with care

**Is binary detection valid?** *Detecting the Disturbance* (Hahami et al. — **Harvard College / U Chicago**, arXiv:2512.12411) (Llama-3.1-8B
only) argues apparent detection is *"entirely explained by global logit shifts that bias models toward
affirmative responses."* **Partial rebuttal:** *Mechanisms of Introspective Awareness* (Anthropic) reports 38.2% TPR at **0% FPR** across 500 concepts —
a pure logit shift cannot produce 0% FPR.

**Resolution adopted here:** the 0% FPR calibration *is* the rebuttal — report clean-input FPR beside every
TPR, and calibrate the detection threshold at a genuine 0% FPR, where a pure logit shift cannot survive.
Combined with magnitude-matching across arms, this is the standing bias defense. Their fuller prescription
(forced-choice localisation at chance 1/N; nonsense controls) is kept as a **deferred confirmatory probe** —
neither is in the released harness, and both are pulled back only if a harmful-vs-benign gap appears, over
the affected concepts only. Good practice, but not on the critical path.

**Does framing matter?** Yes. KAIST: *"recognition depends on how the question is framed"* — their two
probes gave qualitatively different signals on the same models. Report multiple framings.

### 6.4 Is refusal a single direction at all?

The design takes `d_refusal` from Arditi et al. as **one vector**, and `cos(v_concept, d_refusal)` is the
headline covariate. Two papers contest the premise:

- *The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence*
  (arXiv:2502.17420) — refusal is a multidimensional **cone**, not a single direction, with semi-independent
  components.
- *There Is More to Refusal in Large Language Models than a Single Direction* (arXiv:2602.02132).

**Why it matters:** if refusal is a cone, a cosine against one extracted vector understates alignment by an
unknown amount, and a flat slope in Chart 2 is ambiguous between *"no relationship"* and *"wrong basis."*

**Mitigation (cheap):** report alignment against the **top-k refusal subspace** — the projection norm onto
the cone — alongside the scalar cosine. One extra column; converts a possible confound into a robustness
check.

**Note this cuts *for* the project too:** if refusal is higher-dimensional, then single-direction ablation
in Step 4 is a **partial** intervention. That weakens *Mechanisms*' *"exclusive to the refusal direction"*
claim and makes the Step 4b random-direction control **more** informative, not less.

### 6.5 Harmfulness vs refusal — the project must not conflate them

The original chain (§3.3) slides between *"harmful concepts"* and *"refusal-aligned concepts"* as though
they were the same axis. **They are not.** HARC (arXiv:2607.00572) is built on the distinction: `v_harm`
(what the input *is*) and `v_ref` (what the model *does*) are separate directions, their alignment is
**depth-dependent**, and *"same-concept cross-position pairs remain aligned while cross-concept pairs become
near-orthogonal at the most decoupled layer."* HARC's method — coupling them via margin hinge losses — exists
precisely because they naturally dissociate, and jailbreaks exploit the gap.

**Consequence for this project:** `cos(v_concept, d_refusal)` is the *right* covariate (the hypothesis is
about the **refusal** mechanism suppressing reports, not about harm representation per se). But the arm
labels are a *harm* construct while the covariate is a *refusal* construct, and the two need not track each
other at L37. This is a further reason to prefer **empirical refusal rate** as ground truth over assigned
harm labels (§5, Gondil) — and a further reason to run the premise check first.

---

## 7. What this project adds

| Question | Status |
|---|---|
| Do models detect **harmful** concept injection? | **Open** — excluded by policy in the Tier-1 paper |
| Does detection fall with **refusal alignment** of the concept vector? | **Open** — nobody has plotted it |
| Do **harmful concept** vectors even align with refusal? | **Open, and newly contestable** — TU Munich showed it for behavioural vectors only; **HARC** now shows harm/refusal decouple with depth, and the injection layer is late |
| Is introspection **content-agnostic**, on the one content dimension with a mechanism? | **Open** — Lederman & Mahowald claim yes (§4.1); this project tests it where refusal predicts no |
| Does abliteration **close the gap**, and is the effect **refusal-specific**? | **Open, and contested** — *Mechanisms of Introspective Awareness* (Anthropic) and *Can LLMs Reliably Self-Report Adversarial Prefills?* (KAIST) disagree on the random-direction control |
| Does the model **register** an injection it will not **report**? | **Open** — see §7.1 |

### 7.1 The sharpest form of the hypothesis — gate, not carrier

*Mechanisms* describes a **two-stage circuit**: content-agnostic **"evidence carrier"** features in early
post-injection layers detect perturbations along diverse directions, and these **suppress downstream "gate"
features** implementing a default negative response.

**Refusal plausibly acts on the gate, not the carriers.** This reconciles the project with §4.1 rather than
opposing it — *content-agnostic detection* and *harmfulness-dependent reporting* are fully compatible if
harmfulness acts at the reporting stage.

> **Refined hypothesis:** harmfulness does not impair the model's internal **registration** of the
> injection; it impairs the model's **report** of it. The effect should appear at the reporting stage, be
> absent or weaker in carrier-level probes, and abliterate away.

Two independent results support the shape of this:
- *Latent Introspection* (arXiv:2602.20031) — a Qwen-32B model **denies injection in sampled output while
  logit-lens analysis reveals clear detection signal in the residual stream**. Exactly the
  registers-but-denies dissociation, already demonstrated. (Also the source of the prompt-based elicitation
  lever: detection **0.3% → 39.9%** when the prompt explains introspection.)
- *Detecting the Disturbance* / *Feeling the Strength but Not the Source* (arXiv:2512.12411) — models can
  detect *that* and *how strongly* without the *what*.

**Measurement consequence:** a logit-lens or linear-probe readout at the carrier layers **rides on the same
forward passes as the main sweep** — a strict add-on, not a redesign. If the harmful arm shows carrier-level
signal with a suppressed verbal report, that is a demonstrable dissociation between what the model registers
and what it says, on the safety-critical case — **and it survives even if Chart 1 is flat.**
