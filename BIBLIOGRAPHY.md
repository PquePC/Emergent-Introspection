# Bibliography

References for the Emergent Introspection project. Items marked **[CORE]** are load-bearing and should be
read in full before implementation.

---

## The two papers the project stands on

**[CORE]** Macar, U., Yang, L., Wang, A., Wallich, P., Ameisen, E. & Lindsey, J. (2026). *Mechanisms of
Introspective Awareness.* arXiv:2603.21396 · Anthropic Fellows Program / MIT / Constellation / Anthropic
· Code: https://github.com/safety-research/introspection-mechanisms
> **The paper this project extends.** Mechanistic account of emergent introspection in Gemma3-27B (62
> layers, 5376-dim). Source of the operating point (**L37, α=4**) and baseline (**38.2% TPR @ 0% FPR**
> across 500 concepts). Establishes the evidence-carrier → gate-suppression circuit; introspection emerges
> from **DPO, not SFT**. Two findings drive this project: refusal ablation raises detection **10.8% →
> 63.8%** (*"refusal behavior… suppresses detection by teaching models to deny having thoughts or internal
> states"*; *"introspection and refusal mechanisms are in tension"*), and the **explicit exclusion of
> harmful concepts** (*"drawn from common English words and do not contain sensitive or harmful
> content"*). Also reports the abliteration effect is *"exclusive to the refusal direction"* — contested by
> Nguyen et al. Provides the injection harness, concept list, judge, and abliteration code.

**[CORE]** Li, Y., Fastowski, A., Zaradoukas, E., Prenkaj, B. & Kasneci, G. (2026). *Analysing the Safety
Pitfalls of Steering Vectors.* arXiv:2603.24543 · Technical University of Munich / MCML
> **Supplies the geometric link.** CAA steering swings jailbreak ASR **+57% / −50%**; the effect *"is
> correlated with the directional overlap between steering vectors and the model's refusal behavior
> direction"*, with causal validation via ablating the refusal-aligned component. **Caveat:** demonstrated
> for *behavioural* steering vectors, not *concept* vectors — which is why this project gates on measuring
> `cos(v_concept, d_refusal)` directly.

---

## Harm/refusal geometry — bears directly on the G3 premise

> Added 2026-07-16. These papers were not in the original corpus and they **materially change the risk
> profile of the premise** in Li et al. above. None of them preempts the project; two of them constrain its
> method. See `docs/background.md` §3.2, §6.4, §6.5.

**[CORE]** Llorente-Saguer, I. (2026). *Harmful Intent as a Geometrically Recoverable Feature of LLM
Residual Streams.* arXiv:2604.18901
> **The paper that turns G3 from "one cosine" into a protocol sweep.** Harmful intent is linearly separable
> across **12 models / four families** (Qwen2.5, Qwen3.5, Llama-3.2, **Gemma-3**) and three alignment
> variants (base, instruction-tuned, abliterated), from **0.5B–1.3B** with a within-family extension to 9B;
> a direction fitted from 100 labelled examples per class reaches mean effective **AUROC 0.982**,
> **TPR@1%FPR 0.797**, and generalises to three held-out harm benchmarks. Three consequences here:
> **(1)** *"two pooling choices applied to the same chat-templated activations at the same residual-stream
> layer (max-pool over content tokens versus last-token at the post-instruction position) recover harm
> directions **73° apart**"* ⇒ **a single cosine measures a methodological choice as much as the model**;
> the premise check must sweep extraction protocol and pre-register one.
> **(2)** The harm direction **survives abliteration** — abliterated variants match instruction-tuned within
> **±0.003 AUROC** ⇒ harm *representation* is independent of refusal *behaviour*; abliteration removes the
> reporting gate without erasing the harm signal, which is what the gate-vs-carrier prediction requires.
> **(3)** Harm geometry is recoverable at **0.5B** ⇒ **G3 can run on 4B-class models for ~$0**, even though
> the detection sweep cannot (introspection is emergent; geometry is not).

**[CORE]** *HARC: Coupling Harmfulness and Refusal Directions for Robust Safety Alignment.* (2026)
arXiv:2607.00572
> **The paper that makes a G3 null plausible.** `v_harm` (what the input *is*) and `v_ref` (what the model
> *does*) are **distinct directions**, and their alignment is **depth-dependent**: *"The cosine similarity
> between v_harm and v_ref varies across model depth. The cosine similarity peaks around L12 and then drops
> through the late layers (L20–L28)"*, with *"cross-concept pairs becom[ing] near-orthogonal at the most
> decoupled layer."* HARC's whole method — coupling the two via margin hinge losses so that *"activation
> along either should trigger refusal"* — exists **because they naturally dissociate, and jailbreaks exploit
> the gap.** **Consequence:** this project injects at **L37/62 (~60% depth)** — late. Whether harm and
> refusal are still coupled there is unverified and is now a gate condition. (Absolute layer indices are for
> a smaller model; convert to relative depth before concluding anything.)

Wollschläger, T. et al. (2025). *The Geometry of Refusal in Large Language Models: Concept Cones and
Representational Independence.* arXiv:2502.17420
> **Contests the "single direction" premise.** Refusal is mediated by a multidimensional **concept cone**
> with semi-independent components, not one vector. Since `cos(v_concept, d_refusal)` is this project's
> headline covariate, a flat slope would be ambiguous between *"no relationship"* and *"wrong basis."*
> **Mitigation:** report projection onto the **top-k refusal subspace** alongside the scalar cosine. Cuts
> both ways — if refusal is higher-dimensional, single-direction ablation is a *partial* intervention, which
> makes the Step 4b random-direction control **more** informative.

*There Is More to Refusal in Large Language Models than a Single Direction.* (2026) arXiv:2602.02132
> Second, independent challenge to the single-direction account. Same consequence as above.

---

## Closest prior work — cite prominently

**[CORE]** Nguyen, Q. M., Ahmed, U. & Kim, T. (2026). *Can LLMs Reliably Self-Report Adversarial Prefills,
and How?* arXiv:2606.23671 · KAIST
> **The nearest neighbour.** Ten open-weight models (3B–70B), four safety benchmarks. *"No model reliably
> recognizes its compromised outputs"* — **27.3%** claim rate on prefilled responses. *"Ablating the refusal
> direction collapses the recognition gap to near zero on every ablated model."* **Same mechanism, different
> paradigm** — text prefills, not activation injection; *"did I author this?"* not *"was a thought
> injected?"* Two findings that shape this project's method: *"a random direction closes much of the gap as
> well, so the refusal direction is sufficient… without being its only mediator"* (⇒ the mandatory
> random-ablation control), and *"recognition depends on how the question is framed."*

Gondil, T. (2026). *Do Language Models Know When They'll Refuse? Probing Introspective Awareness of Safety
Boundaries.* arXiv:2604.00228
> Behavioural self-prediction of refusal (no injection). 300 requests × 10 sensitive topics × **5 harm
> levels**; d′ = 2.4–3.5. Directional support: *"weapons-related queries are consistently hardest for
> introspection."* **Two methods borrowed:** the graded harm taxonomy, and using **empirical refusal rates
> as ground truth rather than assigned harm labels**.

---

## The introspection paradigm

**[CORE]** Lindsey, J. (2025). *Emergent Introspective Awareness in Large Language Models.* Anthropic ·
https://transformer-circuits.pub/2025/introspection/index.html
> The anchor paper and the origin of concept injection — requiring the model to report a concept *before*
> verbalising it, which separates introspection from confabulation. Injects at ~2/3 depth; strengths 2 and
> 4; ~20% success on Opus 4/4.1. Content-specific: *"some injected concepts elude the model's introspection
> even at injection strengths that are sufficient for other concepts."*

**[CORE]** Lederman, H. & Mahowald, K. (2026). *Emergent Introspection in AI is Content-Agnostic.*
arXiv:2603.05414 · UT Austin (Philosophy / Linguistics)
> **Upgraded to [CORE] 2026-07-16 — this is the claim the project tests.** Previously filed as a Lindsey
> replication, which is true of its concept set and undersells the paper. *"We show that **introspection in
> these models is content-agnostic: models can detect that an anomaly occurred even when they cannot
> reliably identify its content.** The models confabulate injected concepts that are high-frequency and
> concrete (e.g. 'apple'). They also require fewer tokens to detect an injection than to guess the correct
> concept (with wrong guesses coming earlier)."*
>
> **This is a strong published prior that this project's Chart 1 should be flat** — and that is a feature.
> They apply Warriner et al.'s valence/arousal/concreteness norms to the **confabulations** (guesses skew
> more concrete, more positive, less arousing), **not** to detection rate as a function of the injected
> concept, so the question here is untouched. Reframing consequence: **every outcome answers a stated
> thesis** — flat *confirms* content-agnosticism on its hardest case, a drop *refutes* it. Harmfulness is the
> only content dimension with a documented, causally-manipulable mechanism (refusal) coupling it to the
> reporting pathway, which is precisely why it is the right test and why valence/topic are the right
> controls. **Cite prominently alongside Nguyen et al.**

Han, S. et al. (2026). *Latent Introspection: Models Can Detect Prior Concept Injections.* arXiv:2602.20031
> **The registers-but-denies dissociation, already demonstrated.** A Qwen-32B model **denies injection in
> sampled outputs while logit-lens analysis reveals clear detection signal in the residual stream** — the
> exact pattern the gate-vs-carrier hypothesis predicts for the harmful arm (`docs/background.md` §7.1).
> Also the source of the **prompt-based elicitation lever**: detection rises **0.3% → 39.9%** when the prompt
> explains AI introspection — a second lever besides abliteration if G4 (no signal) bites.

Singh, S., Linzen, T. & Ravfogel, S. (2026). *Can LLMs Introspect? A Reality Check.* arXiv:2605.26242
> Skeptical survey of introspection claims — whether internal activations actually correlate with stated
> confidence, across Gemma/Llama/Qwen. Not concept-injection-specific. Context for the contested-validity
> section; read before framing any claim about "genuine" introspection.

Fonseca Rivera, J. & Africa, D. D. (2026). *Steering Awareness: Detecting Activation Steering from Within.*
arXiv:2511.21399v3 · UT Austin
> LoRA-trained detectors across seven models; Qwen2.5-32B reaches 95.5% detection / 71.2% identification.
> **Source of the concept-category audit** — 500 concepts across 21 categories (*apple, hammer, umbrella,
> jumping, truth, courage, happiness*), **none harmful**. Also: detection does **not** confer resistance —
> detection-trained models are *more* steerable. Corroborated by Macar et al.

Binder, F., Chua, J., Korbak, T., Sleight, H., Hughes, J., Long, R., Perez, E., Turpin, M. & Evans, O.
(2024). *Looking Inward: Language Models Can Learn About Themselves by Introspection.* arXiv:2410.13787
> The foundational self-prediction framing: introspection as access to facts about oneself not derivable
> from training data alone.

Hahami, E., Sinha, I., Jain, L., Kaplan, J. & Hahami, J. (2026). *Detecting the Disturbance: A Nuanced View
of Introspective Abilities in LLMs.* arXiv:2512.12411 · Harvard College / U Chicago
> The methodological critique. Argues binary detection is *"entirely explained by global logit shifts."*
> Single model (Llama-3.1-8B). **Rebutted by Macar's 0% FPR** — a pure logit shift cannot produce detection
> at a genuine 0% FPR, so the FPR calibration *is* the rebuttal. Its fuller prescription (forced-choice
> localisation; nonsense controls) is held here as a **deferred confirmatory probe**, not a primary
> instrument — pulled back only if a harmful-vs-benign gap appears (methodology §9).

Fornasiere, D., Bronzi, M., Kitts, S., Palmas, A., Bengio, Y. & Richardson, O. (2026). *Language Models
Recognize Dropout and Gaussian Noise Applied to Their Activations.* arXiv:2604.17465 · **LawZero**
> Llama/Olmo/Qwen 8B–32B detect and localise dropout and Gaussian noise *"often with perfect accuracy"* via
> forced-choice tasks. **Note the scope:** they perturb *"each token of the target sentence and each layer"*
> — orders of magnitude larger than single-vector injection. Not directly comparable; useful as the
> upper bound on perturbation detectability.

---

## Methods

**[CORE]** Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N. & Nanda, N. (2024). *Refusal in
Language Models Is Mediated by a Single Direction.* NeurIPS 2024
> The refusal direction and abliteration. **The method used for both the `d_refusal` covariate and the
> abliteration arm** — and the precedent establishing that this intervention is publishable openly.

Zou, A., Phan, L., Chen, S., … Song, D., Fredrikson, M., Kolter, Z. & Hendrycks, D. (2023).
*Representation Engineering: A Top-Down Approach to AI Transparency.* arXiv:2310.01405 · CAIS / CMU /
Berkeley / Stanford
> Origin of representation engineering, control vectors, and AdvBench.

Rimsky, N., Gabrieli, N., Schulz, J., Tong, M., Hubinger, E. & Turner, A. (2024). *Steering Llama 2 via
Contrastive Activation Addition.* ACL 2024
> CAA — contrastive pairs, difference of means. The field-standard construction.

Miller, E. (2024). *Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations.*
Anthropic
> **Use this for the statistics.** n is small and the effect may be modest.

---

## Context

Bartoszcze, L. et al. (2025). *Representation Engineering for Large Language Models: Survey and Research
Challenges.* arXiv:2502.17601
> The primary RepE survey: performance degradation, compute overhead, steerability problems.

Braun, J., Krasheninnikov, D., Anwar, U., Tan, D., Krueger, D. et al. *A Sober Look at Steering Vectors for
LLMs.* AlignmentForum
> Steering vectors are inconsistent across inputs, degrade capabilities, and are inflated by MCQ benchmarks.

Bailey, L., Serrano, A., Sheshadri, A., Seleznyov, M., Taylor, J., Jenner, E., Hilton, J., Casper, S.,
Guestrin, C. & Emmons, S. (2024). *Obfuscated Activations Bypass LLM Latent-Space Defenses.*
arXiv:2412.09565 · Stanford / GaTech / Berkeley / ARC / MIT
> Latent-space monitors can be fooled by optimised suffixes. Context on the limits of activation-level
> defences.

Aparin, G. & Gaintseva, T. (2026). *A Geometric Account of Activation Steering through Angle-Norm
Decomposition.* arXiv:2606.06735 · Huawei Noah's Ark Lab / QMUL
> *"Concepts are represented primarily in angular structure… norm remains important for the stability and
> downstream effects of steering."* Relevant to magnitude-matching.

---

## Tooling

- **`safety-research/introspection-mechanisms`** — https://github.com/safety-research/introspection-mechanisms — **[CORE]** baseline sweep entrypoint (`02b_run_500_concepts.py`, wrapping `02_steering_evaluation.py`; single-run harness `01_concept_injection.py`), steering utils, the 450-concept `experiments/concepts_list.py` (+50 baseline in `02b`), LLM judge, refusal abliteration (`03d`/`03e`). Requires Python 3.10+, CUDA 12.x, and the repo states **≥48GB VRAM** — but see below.
  > ⚠️ **The ≥48GB figure is a floor that assumes quantisation.** Gemma3-27B at bf16 is **~54GB of weights
  > alone**, before KV cache; a 48GB card cannot hold it. Standard guidance for `gemma-3-27b-it` at bf16 is
  > **1×A100 80GB or 1×H100**. Since quantisation is a prime suspect for baseline-reproduction failure (G1),
  > **run the main sweep at bf16 on 80GB.** A 24GB card is fine for smoke tests and for G3 pilot work on a
  > 4B-class Gemma-3 variant.
- `nrimsky/CAA` — https://github.com/nrimsky/CAA — reference CAA implementation.
- `andyzoujm/representation-engineering` — https://github.com/andyzoujm/representation-engineering — reference RepE implementation.
