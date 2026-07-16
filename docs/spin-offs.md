# Spin-Offs and Complementary Extensions

Optional extensions that sharpen, stress-test, or generalise the main result. Each is self-contained and
sequenced strictly **after** the main plan — none is a prerequisite, and none should be started before the
headline result is secured.

> Main plan: [`project-plan.md`](project-plan.md) · Evidence: [`background.md`](background.md)

Ordered by expected value.

---

## ⭐ S1 — Refusal ablation × the blind spot

### *Is the blind spot real, or just under-elicitation?*

**The finding it builds on.** Macar et al. (2026) report that introspection is heavily **suppressed by the
model's own refusal machinery**: ablating refusal directions raises detection from **10.8% to 63.8%**. They
conclude *"models possess greater introspective capability than is elicited by default."*

**The question.** **Does refusal-ablation also close the blind spot?** That is: does a refusal-ablated
model suddenly detect random, PCA, and noise vectors that it otherwise ignores?

**Why this is the highest-value extension.** It can **flip the main plan's headline.** If the blind spot
exists only because introspection is suppressed by refusal circuitry, then "stealth steering works" is an
artifact of under-elicitation — and a future model with better-elicited introspection closes the blind spot
for free, with no adversarial effort at all. That is a direct hit on the project's theory of impact, and it
is far better to discover it here than to have a reviewer point it out.

Conversely, if the blind spot **survives** refusal ablation, the main result is substantially strengthened:
the blind spot is not merely a suppression effect, but a genuine property of what the model can sense.

**Method.** Reuse Macar et al.'s released refusal-ablation code. Re-run the main plan's detection sweep on
the ablated model. Compare blind-spot depth (detection on PCA/ICA/random/noise) before and after ablation.

**Deliverable.** A before/after blind-spot comparison — the single cheapest experiment that could change the
project's conclusion.

---

## S2 — Layer sweep: reconciling the depth contradiction

**The contradiction.** *Detecting the Disturbance* (2026) reports that introspective capabilities are
*"confined to early-layer injections and collapse to chance thereafter"* — directly contradicting the
60–67% depth consensus of Fonseca Rivera (67%), Macar et al. (60%), and Lindsey (*"about two thirds of the
way through"*).

**The likely resolution.** They may be measuring different things: *Detecting the Disturbance* evaluates
**localisation and strength-discrimination**, while the consensus papers evaluate **binary detection and
identification**. If introspection is early-layer for one task family and mid-layer for another, that is a
structural fact about the capability that nobody has stated.

**Method.** Inject across a layer range spanning early to late depth, measuring detection on both task
families. A cheap bolt-on to the main sweep — same harness, one additional loop.

**Deliverable.** A detection-vs-depth curve per task type, resolving contradiction C3.

---

## S3 — Norm vs angle ablation

**The question.** Does detection depend on the **magnitude** of the injected vector, or on its **direction**?

**Why it matters.** This tests Aparin & Gaintseva (2026) directly — *"concepts are represented primarily in
angular structure, [but] norm remains important for the stability and downstream effects of steering."* The
answer determines whether direction-projection is even the right lever:
- If detection is **angle-driven**, projection is the correct approach, and Route B's H2 problem (that
  renormalisation cannot restore projected-away angular content) is real and unavoidable.
- If detection is **norm-driven**, the entire stealth-steering framework needs rethinking — you would evade
  detection by shrinking the vector, not by rotating it.

**Method.** Two orthogonal sweeps: vary `‖v‖` at fixed direction; vary direction at fixed `‖v‖`. Measure
detection across both.

**Deliverable.** A decomposition of detection into angular and radial dependence. Feeds directly into
Route B's viability.

---

## S4 — Multi-model generalisation

**The question.** Does the main plan's result hold across model families and scales, or is it a fact about
one model?

**Why it matters.** Scale and architecture generalisation is an **explicit open question** in Macar et al.
A single-model result is a curiosity; a result replicated across families is a claim about *introspection
as a phenomenon*. This is the extension that most increases the work's reach.

**Method.** Repeat the main plan's headline measurement on additional open-weight instruction-tuned models
spanning families and scales — candidates include Qwen3-32B, Llama-3.3-70B, and OLMo-32B (the latter used
by Macar et al. for training-stage analysis).

**Deliverable.** The detection-vs-`cos(CAA)` curve replicated across families. This is the extension with
the strongest case for additional compute resources.

---

## S5 — Dropout as a "training awareness" cue

**The hypothesis it tests.** Fornasiere et al. (2026) raise the possibility of a *"data-agnostic **training
awareness** signal"*: dropout is a **training-time** regularisation technique, while Gaussian noise is
sometimes added at **inference** time. A model may therefore have learned a prior associating dropout with
"I am being trained."

**The question.** Does dropout injection shift the model's *evaluation-aware* behaviour, in a way Gaussian
noise does not?

**Why it is interesting.** It would be the first bridge between the **steering-awareness** and
**evaluation-awareness** literatures — connecting "can the model tell it's being perturbed?" to "can the
model tell it's being tested?", which is the actual risk pathway motivating this project. Speculative, but
genuinely novel.

**Method.** Inject dropout masks vs magnitude-matched Gaussian noise; measure not just detection but
downstream behaviour on evaluation-sensitive prompts.

---

## S6 — Format effects as a standalone methodological note

**The question.** Does binary vs forced-choice task format explain the field's Gaussian-noise contradiction?

Fonseca Rivera's detector *"correctly rejects magnitude-matched Gaussian noise 94% of the time"* using a
**binary** detection question. Fornasiere et al. find models detect Gaussian noise *"often with perfect
accuracy"* using **forced-choice** localisation. *Detecting the Disturbance* argues binary detection is
*"entirely explained by global logit shifts."*

If task format alone accounts for the contradiction, that is a correction the entire subfield needs —
every binary-detection result in this literature would require re-examination.

**Method.** Already built into the main plan as Chart 2. This spin-off expands it into a standalone
head-to-head across all conditions, with the logit-shift controls made explicit.

**Deliverable.** A methodological note on measurement validity in introspection research.
