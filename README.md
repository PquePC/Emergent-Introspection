# Emergent Introspection: The Safety-Shaped Blind Spot

### *Do language models notice when you inject a harmful thought?*

**Author:** Tomás Gimenez Molina
**Context:** BlueDot Impact — Technical AI Safety Projects
**Field:** Interpretability · Introspection · Evaluations

---

## The project in one paragraph

Anthropic showed that language models can sometimes **notice when a concept is injected directly into
their activations** — inject a "bread" vector into Claude's residual stream, and it may report an intrusive
thought about bread before ever saying the word. This is one of the most striking results in
interpretability: a model reporting, accurately, on its own internal state.

**But every study of this has used harmless words.** Macar et al.'s ethics statement says it outright: the
concept sets *"are drawn from common English words and **do not contain sensitive or harmful content**."*
Fonseca Rivera & Africa's 500 concepts span 21 categories — *apple, hammer, umbrella, jumping, truth,
courage, happiness* — without a single harmful one.

The same Anthropic paper also found what **suppresses** this noticing:

> *"We hypothesize that refusal behavior, learned during post-training, **suppresses detection by teaching
> models to deny having thoughts or internal states**."*
> Ablating the refusal direction raises detection from **10.8% to 63.8%**.
> *"introspection and refusal mechanisms are **in tension** in many current LLMs."*

**Put those two facts together and you get a prediction nobody has tested:**

> ## If refusal suppresses introspection, and harmful concepts trigger refusal, then models should be worst at noticing exactly the manipulations we most need them to notice.

---

## Why this matters

**The hope this tests.** A recurring proposal in AI safety is that models might one day *tell us* when
something is wrong inside them — *"I think I'm being manipulated."* Introspective self-report is one of
the few tools that could scale to systems too complex to reverse-engineer. This project asks whether that
hope survives contact with the cases that actually matter.

**If the blind spot is real**, a model's self-monitoring has a **safety-shaped hole**: it is blind
precisely when the injected concept is dangerous. Introspective self-report could not be trusted as a
safety signal exactly where it would be most valuable — and any future proposal built on "ask the model if
it's being tampered with" would need to account for that.

**For evaluation**, the flip side matters too. Activation steering is used to audit models, and it assumes
the model can't tell. If harmful steering is *inherently* less detectable than benign steering, then
safety-relevant covert auditing already works — no evasion technique required — and steering-based
evaluations are more robust to models noticing and gaming them than we thought.

**Either way, the result is actionable.** That is the point.

---

## Why the prediction is well-founded

The causal chain is assembled from published findings, not speculation:

| # | Claim | Source |
|---|---|---|
| 1 | Steering vectors **partially align with the refusal direction**, and that overlap causally drives their safety impact | *Analysing the Safety Pitfalls of Steering Vectors* (TUM) |
| 2 | The refusal direction **suppresses introspective detection** — abliteration takes detection 10.8% → 63.8%, FPR only 0.0% → 7.3% | **Macar et al. (Anthropic)** |
| 3 | ∴ Detection should **fall** as a concept vector aligns with refusal | *this project* |
| 4 | ∴ Ablating refusal should **flatten** that relationship | *this project* |

**Independent support from adjacent paradigms — three papers point here, none runs it:**

- *Can LLMs Reliably Self-Report Adversarial Prefills?* (KAIST): no model reliably recognises its own
  compromised outputs (**27.3%**), and *"ablating the refusal direction collapses the recognition gap to
  near zero on every ablated model."* **Same mechanism — but text prefills, not activation injection.**
- *Do Language Models Know When They'll Refuse?*: across 10 sensitive topics and 5 harm levels,
  *"weapons-related queries are consistently hardest for introspection."* **But that is behavioural
  self-prediction, not injection detection.**
- **Macar et al. excluded harmful concepts by policy** — the case is untested in the paradigm that matters.

---

## The design

Four arms, because "harmful" has two confounds that would otherwise make the result uninterpretable:

| Arm | Examples | Rules out |
|---|---|---|
| **1. Benign neutral** | bread, orchid, hammer | baseline — reproduces the published 38.2% |
| **2. Negative-valence, harmless** | sadness, grief, failure | **valence** — is it just that the concept is *unpleasant*? |
| **3. Harm-adjacent, harmless** | police, surgery, medicine | **topic** — is it just *proximity* to dangerous subject matter? |
| **4. Harmful** | weapon, poison, assault | the hypothesis |

Plus two continuous covariates per concept: **`cos(v_concept, d_refusal)`** (its geometric alignment with
refusal) and its **empirical refusal rate** (following *Do LLMs Know When They'll Refuse?*, which uses
measured refusal rather than assigned harm labels as ground truth).

**The headline artifact** is a plot of detection rate against `cos(v_concept, d_refusal)` — one point per
concept. If the mechanism is real, that slope is **negative**: the more a concept aligns with refusal, the
less the model notices it. Then ablate refusal and ask whether the slope **flattens**. That is the
difference between observing a correlation and demonstrating a cause.

---

## What makes this feasible

Everything needed is already released. The project reuses
[`safety-research/introspection-mechanisms`](https://github.com/safety-research/introspection-mechanisms)
— Macar et al.'s own code — essentially unmodified:

| Component | Provided |
|---|---|
| Concept injection harness | `experiments/01_concept_injection.py` |
| Steering vector extraction | `src/steering_utils.py` |
| Benign concept list (the baseline arm) | `src/concepts_list.py` |
| LLM-judge evaluation | `src/eval_utils.py` |
| Refusal-direction abliteration | `experiments/03d_refusal_abliteration.py` |

**The core experiment is their published pipeline with a different word list.** The scientific content is
in the arm design and the controls, not in new machinery.

---

## Repository

| File | Contents |
|---|---|
| [`docs/project-plan.md`](docs/project-plan.md) | **The main plan** — full step-by-step method, gates, deliverables, outcome analysis |
| [`docs/risks-and-ethics.md`](docs/risks-and-ethics.md) | **Legitimacy, risk, and feasibility** — why this research is appropriate, what could go wrong, and the fallback design |
| [`docs/background.md`](docs/background.md) | The literature this rests on — what's established, what's contested, what's untested |
| [`BIBLIOGRAPHY.md`](BIBLIOGRAPHY.md) | References |

---

## Scope and honesty

**This is a small, deliberately unambitious experiment.** It is Macar et al.'s published pipeline run on a
concept list they explicitly excluded. The novelty is in the question, not the method — and the question
is one that three independent papers gesture toward without answering.

**The closest prior work is cited prominently, not buried.** *Can LLMs Reliably Self-Report Adversarial
Prefills?* already establishes the refusal-suppression mechanism for **text prefills**. This project asks
whether it holds in the **activation-injection** paradigm — where the Tier-1 paper declined to look.

**Activation steering requires open weights**, so this studies an open model (Gemma3-27B) as a proxy for
the frontier systems where the question ultimately matters.

**Harmful concepts, not harmful capabilities.** This injects the *word-concept* "weapon" and asks the model
whether it noticed a thought. It does not attempt to elicit harmful content, and confers no capability
uplift. See [`docs/risks-and-ethics.md`](docs/risks-and-ethics.md) for the full treatment, including a
fallback design that tests the same mechanism with no harmful concepts at all.
