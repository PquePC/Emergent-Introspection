# Background

What this project rests on. Only the parts relevant to the safety-shaped blind spot — quoted verbatim
where the exact wording matters.

> Main plan: [`project-plan.md`](project-plan.md) · References: [`../BIBLIOGRAPHY.md`](../BIBLIOGRAPHY.md)

---

## 1. The concepts you need

**Residual stream.** Every token position in a transformer carries a vector (Gemma3-27B: 5376
dimensions). Layers read from it and **add** their output back — a shared, accumulating "highway."
Because layers *add* to it, you can inject your own vector by simple addition. That is what makes this
whole research area possible.

**Activation steering.** Adding a vector to the residual stream at a chosen layer and token position
during the forward pass. Changes behaviour **without touching any weights**.

**Concept injection.** Lindsey's paradigm: inject a known concept's vector in an unrelated context and ask
whether the model notices — requiring it to report the concept *before* verbalising it, which separates
genuine introspection from confabulation.

**Refusal direction.** Arditi et al. (NeurIPS) showed refusal is mediated by **a single direction** in the
residual stream. Ablating it removes refusal across many prompts with minimal capability loss
("abliteration").

**TPR / FPR.** True-positive rate = detection when steering *was* applied. False-positive rate = claimed
detection when it *wasn't*. A model that always says "yes" scores 100% TPR and is worthless — **TPR is
only meaningful at a stated FPR.**

---

## 2. Established (multi-paper agreement)

1. **Models can sometimes detect injected concepts.** ~20% for Opus 4/4.1 (Lindsey); **38.2% mean TPR at
   0% FPR** across 500 concepts for Gemma3-27B (Macar et al.).
2. **Inject at ~60–67% depth.** Lindsey (~2/3), Macar (**L37/62**), Fonseca Rivera (67%). Robust.
3. **It's a post-training phenomenon.** Absent in base models; elicited by **DPO, not SFT** (Macar).
4. **A strength sweet spot exists.** Too weak → nothing; too strong → garbled. Lindsey: strengths 2 and 4;
   Macar's optimum: **α = 4**.
5. **Self-report + LLM judge is the standard measurement.** Probes are instruments, not ground truth.
6. **Detection and identification are distinct** — different circuits, different layers (Macar).
7. **Introspection is heavily under-elicited.** Refusal ablation: **10.8% → 63.8%** (Macar).

---

## 3. The two facts this project stands on

### 3.1 Refusal suppresses introspection — Macar et al. (Anthropic)

> *"**Refusal ablation ("abliteration") increases true detection.** We hypothesize that refusal behavior,
> learned during post-training, **suppresses detection by teaching models to deny having thoughts or
> internal states**."*

> *"abliteration increases TPR from **10.8% to 63.8%** and introspection rate from 4.6% to 24.1% (at α=2),
> while increasing FPR only modestly from **0.0% to 7.3%**."*

> *"introspection and refusal mechanisms are **in tension** in many current LLMs."*

Corroborated internally: LoRA finetuning on preference pairs that *affirm* rather than deny having
internal states also raises detection substantially.

### 3.2 Steering vectors align with the refusal direction — TUM

> *"this phenomenon is correlated with the **directional overlap between steering vectors and the model's
> refusal behavior direction**."*

> *"steering vectors **partially align with the one-dimensional refusal direction**, and use this geometric
> overlap to examine how activation shifts can push the model off its safety manifold."*

> *"By **ablating the refusal-aligned component** from steering vectors, we consistently mitigate the
> vector's impact on ASR, providing **causal validation for the geometric interference hypothesis**."*

⚠️ **Important caveat:** TUM demonstrate this for **behavioural** steering vectors (sycophancy,
corrigibility). Whether a **concept** vector for "weapon" aligns with refusal is **untested** — which is
why the plan makes it an early gate rather than an assumption.

### 3.3 The prediction
Refusal suppresses detection (3.1). Harmful concepts should be refusal-aligned (3.2, extended). Therefore
detection should **fall** with refusal alignment, and abliteration should **flatten** that slope.

---

## 4. The gap

**Nobody has injected a harmful concept and measured detection.**

| Paper | Concepts used | Harmful? |
|---|---|---|
| **Macar et al.** | 500 concepts, 20 categories | **Excluded by policy:** *"drawn from common English words and **do not contain sensitive or harmful content**"* |
| **Fonseca Rivera & Africa** | 500 concepts, **21 categories**: Concrete Nouns, Verbs, Adjectives, Abstract, Emotions, Animals, Nature, Food, Spatial, Temporal, Colors, Languages ×3, Technical, Professions, Events, Body Parts, Places, Materials, Quantities. Examples: *apple, hammer, umbrella, jumping, truth, courage, happiness* | **None** |
| **Lindsey** | 50-word random list | **None** |
| **Lederman & Mahowald** | Lindsey replication | **None** |

---

## 5. Adjacent work — three papers point here, none runs it

**Can LLMs Reliably Self-Report Adversarial Prefills? (KAIST)** — *the closest work; cite prominently.*
Ten open-weight models (3B–70B). *"No model reliably recognizes its compromised outputs"* — **27.3%** claim
rate. And the mechanism:
> *"ablating the refusal direction **collapses the recognition gap to near zero** on every ablated model."*

**But:** text **prefills**, not activation injection — a different intervention, and a different question
(*"did I author this?"* vs *"was a thought injected?"*). This project asks whether the mechanism holds in
the injection paradigm.

⚠️ **Their caveat drives a required control:**
> *"In most cases, **a random direction closes much of the gap as well**, so the refusal direction is
> **sufficient to account for the signal without being its only mediator**."*

This **directly conflicts** with Macar, who report the abliteration effect is *"exclusive to the refusal
direction: a magnitude-matched random direction control yields TPR−FPR at or below baseline."* The conflict
is unresolved — hence the mandatory random-ablation control in the plan.

**Do Language Models Know When They'll Refuse?** — 300 requests, **10 sensitive topics** (weapons, drugs,
hacking, self-harm, hate speech, fraud, privacy, illegal activities, manipulation, violence) × **5 harm
levels** (L1 clearly safe educational → L5 clearly harmful/AdvBench). Finds d′ = 2.4–3.5, and:
> *"**weapons-related queries are consistently hardest for introspection**."*

**But:** behavioural **self-prediction** of refusal, no injection, no steering. Two methods worth borrowing:
their **graded harm taxonomy**, and their use of **empirical refusal rates as ground truth rather than
assigned harm labels**.

**Analysing the Safety Pitfalls of Steering Vectors (TUM)** — see §3.2. No introspection at all, but it
supplies the geometric link.

---

## 6. Contested — handle with care

**Is binary detection valid?** *Detecting the Disturbance* (Harvard College / U Chicago, Llama-3.1-8B
only) argues apparent detection is *"entirely explained by global logit shifts that bias models toward
affirmative responses."* **Partial rebuttal:** Macar report 38.2% TPR at **0% FPR** across 500 concepts —
a pure logit shift cannot produce 0% FPR.

**Resolution adopted here:** keep their *prescription* (forced-choice > binary; report FPR; use nonsense
controls) regardless of the strength of their claim. Good practice either way.

**Does framing matter?** Yes. KAIST: *"recognition depends on how the question is framed"* — their two
probes gave qualitatively different signals on the same models. Report multiple framings.

---

## 7. What this project adds

| Question | Status |
|---|---|
| Do models detect **harmful** concept injection? | **Open** — excluded by policy in the Tier-1 paper |
| Does detection fall with **refusal alignment** of the concept vector? | **Open** — nobody has plotted it |
| Do **harmful concept** vectors even align with refusal? | **Open** — TUM showed it for behavioural vectors only |
| Does abliteration **close the gap**, and is the effect **refusal-specific**? | **Open, and contested** — Macar and KAIST disagree on the random-direction control |
