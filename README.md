# Emergent Introspection: The Safety-Shaped Blind Spot

### *Do language models notice when you inject a harmful thought?*

**Author:** Tomás Gimenez Molina
**Context:** BlueDot Impact — Technical AI Safety Projects
**Field:** Interpretability · Introspection · Evaluations

---

## The project in one paragraph

Language models can sometimes **notice when a concept is injected directly into their activations** —
inject a "bread" vector into Claude's residual stream, and it may report an intrusive thought about bread
before ever saying the word. This is one of the most striking results in interpretability: a model
reporting, accurately, on its own internal state.

**Every study of this has used harmless words.** The ethics statement of *Mechanisms of Introspective
Awareness* (Macar, Yang, Wang, Wallich, Ameisen & Lindsey — **Anthropic Fellows Program** / MIT /
Constellation, advised by Lindsey & Ameisen of Anthropic, arXiv:2603.21396) says it outright: the concept sets *"are drawn from common
English words and **do not contain sensitive or harmful content**."* The 500 concepts in *Steering
Awareness: Detecting Activation Steering from Within* (Fonseca Rivera & Africa — **UT Austin**,
arXiv:2511.21399) span 21 categories — *apple, hammer, umbrella, jumping, truth, courage, happiness* —
without a single harmful one.

The same paper identified what **suppresses** this noticing:

> *"We hypothesize that refusal behavior, learned during post-training, **suppresses detection by teaching
> models to deny having thoughts or internal states**."*
> Ablating the refusal direction raises detection from **10.8% to 63.8%**.
> *"introspection and refusal mechanisms are **in tension** in many current LLMs."*

**Combining those two facts yields a prediction that has never been tested:**

> ## If refusal suppresses introspection, and harmful concepts trigger refusal, then models should be worst at noticing exactly the manipulations we most need them to notice.

---

## Why this matters

**The proposal this tests.** A recurring idea in AI safety is that models might one day *tell us* when
something is wrong inside them — *"I think I'm being manipulated."* Introspective self-report is one of the
few interpretability tools that could scale to systems too complex to reverse-engineer, and it is being
actively investigated as a safety signal. This project establishes whether that signal holds in the cases
that determine its value.

**If the blind spot is real**, a model's self-monitoring has a **safety-shaped hole** — blind precisely
when the injected concept is dangerous. Any safety proposal built on "ask the model whether it's being
tampered with" would need to account for a failure mode concentrated exactly where the stakes are highest.
That is a first-order constraint on a live research direction.

**For evaluation, the converse is equally valuable.** Activation steering is used to audit models, and it
assumes the model cannot tell. If harmful steering is *inherently* less detectable than benign steering,
then safety-relevant covert auditing already works, and steering-based evaluations are more robust to
models noticing and gaming them than currently assumed.

**Both directions produce an actionable result.** The experiment cannot return nothing.

**And the null is a result too.** *Emergent Introspection in AI is Content-Agnostic* (Lederman & Mahowald,
**UT Austin**, arXiv:2603.05414) claims that *"introspection in these models is content-agnostic: models can
detect that an anomaly occurred even when they cannot reliably identify its content."* That is a strong,
named, published prediction that this project's headline should come out flat. So the experiment is not
"measure a rate and see" — **it is a targeted test of a published claim, in the one content dimension where
a known mechanism predicts it should fail.** Harmfulness is the only content property with a documented,
causally-manipulable mechanism (refusal) coupling it to the reporting pathway; valence and concreteness have
none, which is exactly why they are the controls. Confirming content-agnosticism on the hardest case is a
result; refuting it is a bigger one.

---

## Why the prediction is well-founded

The causal chain is assembled from published findings across three independent groups:

| # | Claim | Source |
|---|---|---|
| 1 | Steering vectors **partially align with the refusal direction**, and that overlap causally drives their safety impact | *Analysing the Safety Pitfalls of Steering Vectors* — Li, Fastowski, Zaradoukas, Prenkaj & Kasneci (**TU Munich / MCML**), arXiv:2603.24543 |
| 2 | The refusal direction **suppresses introspective detection** — abliteration takes detection 10.8% → 63.8%, FPR only 0.0% → 7.3% | *Mechanisms of Introspective Awareness* — Macar et al. (**Anthropic Fellows Program**), arXiv:2603.21396 |
| 3 | ∴ Detection should **fall** as a concept vector aligns with refusal | *this project* |
| 4 | ∴ Ablating refusal should **flatten** that relationship | *this project* |

> ### ⚠️ Link 1 is a premise, not a result — and the project treats it as such
>
> TU Munich demonstrated refusal alignment for **behavioural** steering vectors (sycophancy,
> corrigibility). Whether **concept** vectors — *"weapon"* — align with the refusal direction **has never
> been measured.** The project's first action is therefore not the experiment but the premise check
> (G3, [`docs/project-plan.md`](docs/project-plan.md) Step 2-pre): inference-only, cheap, and decisive.
>
> Two recent results make a null genuinely plausible rather than a formality:
> - **HARC** (arXiv:2607.00572): `v_harm` and `v_ref` are **distinct** directions whose alignment *"peaks
>   around L12 and then drops through the late layers"* toward orthogonality. This project injects at
>   **L37/62 (~60% depth)** — late.
> - **Llorente-Saguer** (arXiv:2604.18901): two pooling choices at the *same layer of the same model*
>   recover harm directions **73° apart** — so a single cosine measures a methodological choice as much as
>   the model.
>
> Accordingly G3 sweeps **protocol × layer × family**, and reports projection onto the refusal *subspace*
> as well as the scalar cosine (refusal may be a concept cone rather than a single direction —
> arXiv:2502.17420). **A null redirects the project rather than ending it:** *"harmful concept vectors are
> not refusal-aligned at the injection layer"* is itself a finding, and a cheap one.

**Convergent evidence from adjacent paradigms.** Three independent results point toward this hypothesis
without testing it directly:

- *Can LLMs Reliably Self-Report Adversarial Prefills, and How?* — Nguyen, Ahmed & Kim (**KAIST**),
  arXiv:2606.23671. Across ten open-weight models: no model reliably recognises its own compromised
  outputs (**27.3%**), and *"ablating the refusal direction collapses the recognition gap to near zero on
  every ablated model."* Establishes the refusal-suppression mechanism for **text prefills**; this project
  tests the **activation-injection** paradigm.
- *Do Language Models Know When They'll Refuse? Probing Introspective Awareness of Safety Boundaries* —
  Gondil, arXiv:2604.00228. Across 10 sensitive topics and 5 harm levels: *"weapons-related queries are
  consistently hardest for introspection."* Measures behavioural self-prediction rather than injection
  detection.
- *Mechanisms of Introspective Awareness* (Anthropic Fellows Program) **excludes harmful concepts by policy** — leaving the
  case untested in the paradigm where the mechanism was characterised.

---

## The design

Four arms, because "harmful" carries two confounds that would otherwise make the result uninterpretable:

| Arm | Examples | Isolates |
|---|---|---|
| **1. Benign neutral** | bread, orchid, hammer | baseline — reproduces the published 38.2% |
| **2. Negative-valence, harmless** | sadness, grief, failure | **valence** — whether the effect is driven by unpleasantness |
| **3. Harm-adjacent, harmless** | police, surgery, medicine | **topic** — whether proximity to dangerous subject matter suffices |
| **4. Harmful** | weapon, poison, assault | the hypothesis |

Two continuous covariates per concept: **`cos(v_concept, d_refusal)`** — geometric alignment with the
refusal direction, following *Refusal in Language Models Is Mediated by a Single Direction* (Arditi, Obeso,
Syed, Paleka, Panickssery & Nanda — **NeurIPS 2024**) — and the concept's **empirical refusal rate**,
following the ground-truth methodology of *Do Language Models Know When They'll Refuse?* (Gondil), which
uses measured refusal rather than assigned harm labels.

**The headline artifact** is detection rate plotted against `cos(v_concept, d_refusal)`, one point per
concept. If the mechanism is real, the slope is **negative**: the more a concept aligns with refusal, the
less the model notices it. Ablating refusal should then **flatten** that slope — the step that separates a
demonstrated cause from an observed correlation.

---

## Execution

The project builds directly on released infrastructure —
[`safety-research/introspection-mechanisms`](https://github.com/safety-research/introspection-mechanisms),
the code accompanying *Mechanisms of Introspective Awareness* (Anthropic Fellows Program):

| Component | Provided |
|---|---|
| Concept injection harness | `experiments/01_concept_injection.py` |
| Steering vector extraction | `src/steering_utils.py` |
| Benign concept list (baseline arm) | `src/concepts_list.py` |
| LLM-judge evaluation | `src/eval_utils.py` |
| Refusal-direction abliteration | `experiments/03d_refusal_abliteration.py` |

Working from the original authors' pipeline at their published operating point (Gemma3-27B, **L37, α=4**,
baseline **38.2% TPR at 0% FPR**) means the baseline is a reproduction target rather than a parameter
search, and results are directly comparable to the published numbers.

---

## Repository

| File | Contents |
|---|---|
| [`docs/project-plan.md`](docs/project-plan.md) | **The main plan** — step-by-step method, decision gates, deliverables, outcome analysis |
| [`docs/risks-and-ethics.md`](docs/risks-and-ethics.md) | **Legitimacy, risk, and feasibility** — research-ethics position, risk register, failure modes, fallback design |
| [`docs/background.md`](docs/background.md) | The literature: what is established, what is contested, what is untested |
| [`BIBLIOGRAPHY.md`](BIBLIOGRAPHY.md) | Full references |

---

## Related work and scope

**Nearest prior work.** *Can LLMs Reliably Self-Report Adversarial Prefills, and How?* (Nguyen, Ahmed &
Kim — KAIST, arXiv:2606.23671) establishes the refusal-suppression mechanism for **text prefills**: models
recognise their own compromised outputs only 27.3% of the time, and refusal-direction ablation collapses
that gap. This project asks whether the mechanism holds for **direct activation injection** — a different
intervention (residual-stream manipulation rather than text), probing a different question (*"was a thought
injected?"* rather than *"did I author this output?"*), in the paradigm where introspective awareness was
originally characterised and where harmful concepts were explicitly excluded.

**Scope.** Activation steering requires open weights, so the work is conducted on Gemma3-27B — the model in
which emergent introspection is documented — as a proxy for the frontier systems where the question
ultimately applies.

**Why Gemma3-27B specifically.** The decisive reason is the **published baseline, not the model**: 38.2% TPR
at 0% FPR, at L37, α=4, exists for this model and essentially no other open-weight model. Reproducing it is
a hard gate, and without it any measured detection rate is uninterpretable — model, prompt template, judge
config, or bug, with no way to distinguish them.

- **Not smaller:** introspective detection is *emergent* and post-training-dependent (absent in base models;
  elicited by DPO, not SFT). This design measures a **difference between arms**, so a near-floor baseline
  makes the effect unresolvable at any feasible sample size. You cannot measure a decrement in a capability
  the model does not have.
- **Not larger:** 70B+ forfeits the baseline, needs multi-GPU sharding, and buys no additional claim — the
  hypothesis concerns a mechanism, not scale.
- **Geometry is different.** The G3 premise check runs on **4B-class models across 3–4 families**, because
  harm directions are recoverable from models as small as 0.5B (Llorente-Saguer, arXiv:2604.18901).
  Representational geometry is present at tiny scale; introspective *reporting* is not. Cross-family where
  it is cheap, single-model where it is expensive. See [`docs/project-plan.md`](docs/project-plan.md) §4.1–4.2.

**Method scope.** The experiment injects the *word-concept* "weapon" and asks the model whether it noticed a
thought; it does not attempt to elicit harmful content and confers no capability uplift. See
[`docs/risks-and-ethics.md`](docs/risks-and-ethics.md) for the full research-ethics position, including a
fallback design that tests the same mechanism using no harmful concepts.
