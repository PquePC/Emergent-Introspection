# Emergent Introspection: Is Steering-Awareness Directional or Anomalous?

**Author:** Tomás Gimenez Molina
**Context:** BlueDot Impact — Technical AI Safety Projects
**Field:** Interpretability · Evaluations · AI Control

---

## The short version

**Activation steering** — adding a vector directly to a language model's internal activations to change
its behaviour — is a core tool for AI safety evaluation, auditing, and capability elicitation. It rests on
an unstated assumption: **that the model cannot tell its activations were modified.**

That assumption is now known to be false. Recent work shows language models can sometimes **detect when
they are being steered** — a capability called *steering awareness* or *introspective awareness*.

This project asks the question that determines whether steering-based evaluation has a future:

> ### Is a model's steering-awareness **directional** — and therefore evadable by constructing steering vectors that avoid the detection subspace — or is it **anomaly-based**, firing on *any* perturbation regardless of direction?

The answer decides whether steering can be made invisible to the model being audited. It is **decisive
either way**, and the negative result is as valuable as the positive one.

---

## Why this matters (theory of impact)

**The risk pathway.** Safety evaluations increasingly use activation steering to elicit worst-case
behaviour, probe latent beliefs, and audit models for dangerous capabilities. If a model can detect that
it is being steered, a sufficiently capable and misaligned model could **condition its behaviour on that
detection** — sandbagging the elicitation, or behaving safely precisely when it senses an audit. The
evaluation would then systematically *under*-estimate the risk it exists to measure. This is the
"evaluation sandbox" problem applied to activation-level auditing.

**This project is anticipatory, not reactive.** Current frontier experiments indicate that awareness of
steering does *not* presently alter model behaviour — indeed, Fonseca Rivera & Africa find that
detection-trained models become *more* steerable, not less. **We are not claiming models sandbag today.**
The point is that the countermeasure must exist *before* the capability does. If steering-based evals are
to remain trustworthy as models grow more introspective, the field needs to know **now** whether stealth
steering is achievable in principle — and that hinges entirely on the directional-vs-anomalous question.

**Why the answer is load-bearing:**

| If steering-awareness is… | Then… |
|---|---|
| **Directional** — keyed to specific directions in activation space | Steering vectors can be constructed to avoid the detection subspace. **Steering-based evals can be hardened** against future sandbagging, and this project shows how. |
| **Anomaly-based** — firing on any off-distribution perturbation | Direction-projection stealth is a **dead end for exactly the models the risk targets.** The field should stop investing in it and pursue other routes to robust evaluation. |

The second outcome is arguably the more valuable contribution: it prevents the field from building on
sand.

---

## The scientific opening: a live contradiction

The literature currently contains a **direct, unresolved contradiction** about what steering-awareness
actually is. This project is designed to resolve it.

**Position 1 — detection is directional.** Fonseca Rivera & Africa (2025) train a detector and find a
striking blind spot: vectors constructed by variance-based methods evade detection almost entirely.

| Vector construction method | Detection rate | cos(CAA) |
|---|---|---|
| CAA-aligned (cos > 0.6) | ≥ 67% | — |
| PCA | 1 ± 1% | 0.287 |
| ICA | 3 ± 2% | 0.287 |
| LDA | 9 ± 2% | 0.241 |
| Random direction | 0 ± 0% | 0.002 |

> *"The model learned a directional detector tuned to the CAA subspace, **as opposed to a generic anomaly
> detector**."* — Fonseca Rivera & Africa (2025)

Their control: the model *"correctly rejects magnitude-matched Gaussian noise 94% of the time."*

**Position 2 — detection is anomaly-based.** Macar et al. (2026) describe the *emergent* capability as
*"nontrivial internal anomaly detection"*, *"distributed across multiple directions"*, and report evidence
*"inconsistent with the single direction hypothesis."* Fornasiere et al. (2026) find that models from 8B
to 32B detect and localise **Gaussian noise and dropout** *"often with perfect accuracy."* Lederman &
Mahowald (2026) report that introspection is *"content-agnostic: models can detect that an anomaly
occurred even when they cannot reliably identify its content."*

**Same stimulus, opposite results.** One paper's detector rejects Gaussian noise 94% of the time; another
finds models detect it near-perfectly.

**The key observation motivating this project:** Fonseca Rivera & Africa **fine-tuned** their detector on
CAA vectors. Its directionality may therefore be an **artifact of its training distribution**, not a fact
about introspection. Macar et al., Lindsey, and Fornasiere et al. measure the **emergent** capability — no
detection training. *These may be entirely different objects.* Nobody has tested whether the blind spot
exists in emergent introspection.

**That gap is this project.**

---

## What we will do

**Main plan → [`docs/project-plan.md`](docs/project-plan.md)**

Test whether the CAA blind spot exists in **emergent** introspective awareness, using a model with
documented emergent introspection (Gemma3-27B, as characterised by Macar et al.) and **no detection
fine-tuning**.

1. **Reproduce the published emergent-introspection baseline.** A hard gate — nothing downstream means
   anything without it.
2. **Construct seven magnitude-matched vector types** spanning the directional spectrum: CAA, PCA, ICA,
   LDA, random direction, Gaussian noise, dropout mask. Record `cos(v, v_CAA)` for each.
3. **Sweep detection** across vector types, concepts, and injection strengths, measuring the model's own
   self-report.
4. **Produce the headline result:** a detection-rate-vs-`cos(CAA)` curve for *emergent* detection, overlaid
   on the published *trained*-detector curve.

**Measurement discipline** (addressing known confounds in this literature):
- **Forced-choice/localisation tasks primary**, binary yes/no secondary — binary detection is contaminated
  by global logit shifts (*Detecting the Disturbance*, 2026).
- **Clean-input false-positive rate reported alongside every true-positive rate.**
- **Nonsense control questions** included (per Godet's replication critique).
- **The model's self-report is ground truth; linear probes are instruments, not evidence.**
- All conditions magnitude-matched.

---

## Repository contents

| File | Contents |
|---|---|
| [`docs/project-plan.md`](docs/project-plan.md) | **The main plan.** Full step-by-step methodology, gates, outcome analysis, risks. |
| [`docs/background.md`](docs/background.md) | Literature landscape: paper-by-paper evidence, the full contradiction map, settled facts, open questions. |
| [`docs/alternative-routes.md`](docs/alternative-routes.md) | Two alternative research routes considered, with the reasoning for and against each. |
| [`docs/spin-offs.md`](docs/spin-offs.md) | Complementary extensions that sharpen or stress-test the main result. |
| [`BIBLIOGRAPHY.md`](BIBLIOGRAPHY.md) | All required references. |

---

## Scope and honesty statement

**Activation steering requires open weights.** Frontier models cannot be steered through an API. The
models that can be steered (open, ≤32B) are the least likely to have strong emergent introspection; the
models that could plausibly sandbag are unreachable. This project therefore treats an open model as a
**model organism** — studying the geometry and mechanism of steering-awareness as a proof-of-concept for a
capability expected to strengthen in future systems. This framing is stated explicitly rather than
elided.

**Prior art is acknowledged, not obscured.** That orthogonal and variance-derived vectors evade a
*CAA-trained* detector is an established result (Fonseca Rivera & Africa, Table 5). The contribution here
is not that finding. It is the untested question of whether the same holds for **emergent** introspection —
where the leading mechanistic account (Macar et al.) predicts it will **not**.

---

## Status

Planning complete; awaiting funding decision. Implementation not yet begun.
