# Background: The Steering-Awareness Literature

Evidence base for the main plan. Paper-by-paper claims (quoted verbatim where marked), the contradiction
map, settled facts, and open questions.

> Full references: [`../BIBLIOGRAPHY.md`](../BIBLIOGRAPHY.md)
> ⚠️ = available in abstract form only at time of writing; verify against full text before relying on.

---

## 1. Key concepts

**Residual stream.** Each token position in a transformer carries a vector of dimension `d` (Gemma3-27B:
5376). Layers *read* from it and *add* their outputs back — a shared accumulating "highway." Because
layers add to it, an external vector can be injected by simple addition. This is what makes activation
steering possible.

**Activation steering.** Adding a vector to the residual stream at a chosen layer and token position
during the forward pass, changing behaviour without modifying any weights.

**CAA — Contrastive Activation Addition.** The field-standard recipe for constructing a steering vector:
run contrastive prompt pairs that differ only in the target property, cache activations at a layer, and
take the difference of means. Also called *mean-difference*. **Critically: this is the training
distribution of the published steering-detector.**

**Concept injection.** Lindsey's paradigm — adding a concept steering vector specifically to test whether
the model notices, and requiring the model to report it *before* verbalising the concept, which separates
introspection from confabulation.

**Steering awareness / introspective awareness.** A model's ability to detect that its own activations
have been modified.

**TPR / FPR.** True-positive rate (detection when steering *was* applied) and false-positive rate (claimed
detection when it was *not*). A model that always answers "yes" has 100% TPR and is worthless — **TPR is
only meaningful at a stated FPR.**

---

## 2. Paper-by-paper

### 2.1 Fonseca Rivera & Africa (2025) — *Steering Awareness: Detecting Activation Steering from Within*

- **Models:** seven instruction-tuned — Gemma2-9B, Qwen2.5-7B/32B, QwQ-32B, Llama3-8B/70B, DeepSeek-7B.
- **Best result:** *"Qwen 2.5 32B achieves 95.5% detection and 71.2% concept identification"*, 0% FPR on
  clean inputs.
- **The detector is LoRA-fine-tuned** to answer detection questions; responses classified by a
  *"dual-judge system (keyword regex + GPT-4o-mini)."* **← Trained, not emergent. The critical caveat.**
- **Injection:** ~67% depth, final prompt token. Gemma2-9B L28/42 · Llama3-8B L21/32 · Qwen2.5-32B L43/64.
- **Readout:** final layer (L41 for Gemma2-9B).
- **Mechanism — directional, explicitly not anomaly-based:**
  > *"The model learned a directional detector tuned to the CAA subspace, **as opposed to a generic anomaly
  > detector**."*
  > *"Detection correlates strongly with cosine similarity... detection depends on **direction** v through
  > alignment with the learned template."*
- **Cross-layer rotation:**
  > *"cos(Δc(ℓ), d̂) rises monotonically from **∼0.2 at injection layer to ∼0.75 at layer 41**"*
  > *"A single affine map... fit on 140 training concepts, achieves cosine similarity 0.85 on held-out
  > concepts."*

  **Interpretation:** the detector never reads the injected vector directly — it reads the **rotated image**
  of it, produced by a learned affine map whose support is the CAA subspace. This explains the blind spot:
  vectors outside that subspace are not that they are "invisible," but that the learned map has no purchase
  on them.
- **The blind spot (Table 5):**

  | Method | Detection | cos(CAA) |
  |---|---|---|
  | CAA-aligned (cos > 0.6) | ≥67% | — |
  | PCA | 1 ± 1% | 0.287 |
  | ICA | 3 ± 2% | 0.287 |
  | LDA | 9 ± 2% | 0.241 |
  | Random direction | 0 ± 0% | 0.002 |

  > *"methods producing vectors aligned with CAA (cos(CAA)>0.6) achieve ≥67% detection, while methods below
  > this threshold achieve ∼0%"*
- **Noise control:** *"model correctly rejects magnitude-matched Gaussian noise 94% of the time."*
- **Headline safety finding:** **detection does not confer resistance** — detection-trained models are
  *more* susceptible to steering. *"Activation steering should not be considered an invisible intervention
  in safety evaluations."*
- **Open questions:** does the geometric detector generalise beyond the CAA training distribution; can the
  blind spot be closed; why does detection increase susceptibility?

### 2.2 Macar, Yang, Lindsey, Ameisen et al. (2026) — *Mechanisms of Introspective Awareness*

**Emergent capability — no detection training.** Code: `safety-research/introspection-mechanisms`.

- **Models:** Gemma3-27B (primary; 62 layers, 5376-dim residual), Qwen3-235B (robustness), OLMo-3.1-32B
  (training-stage analysis).
- **Injection:** L37/62 (≈60% depth); also tested L29, swept L25–55.
- **Circuit:** *"the injected concept vector activates evidence carriers in early post-injection layers,
  which in turn suppress late-layer gates via directions that are both steering-aligned and
  gate-connected."* Evidence carriers peak **L38**; gate features **L45–61**.
- **Mechanism — anomaly-ish and distributed:**
  > detection reflects *"nontrivial internal anomaly detection"* rather than directional bias
  > *"the underlying computation is distributed across multiple directions... the capability is not
  > attributable to the uninteresting explanation that some concept vectors happen to be correlated with a
  > direction that promotes affirmative responses."*
  > *"23.3% of S-S pairs, both opposite directions trigger detection, compared to only 3.2% for F-F pairs.
  > **This is inconsistent with the single direction hypothesis**."*
- **But content matters somewhat:** *"Detection depends on the injected concept producing a sufficiently
  distinctive perturbation."*
- **Measurement:** model self-report — *"An LLM judge classifies each response for detection (whether the
  model reports sensing something unusual) and identification."*
- **Numbers:** **38.2% average TPR at 0% FPR** (emergent baseline). Ablating refusal directions raises
  detection **10.8% → 63.8%**. Trained steering vectors add ~75 points.
- **Origin:** emerges in post-training via **DPO, not SFT**.
- **Under-elicitation:** *"models possess greater introspective capability than is elicited by default."*
- **Random/noise vectors: not tested.** ← a direct, citable gap.
- **Open questions:** scale/architecture generalisation; role of attention; simulated vs genuine
  introspection; whether self-report predicts reliable internal-state reporting generally.

### 2.3 Lindsey (2025) — *Emergent Introspective Awareness in Large Language Models* — Anthropic

- **Method:** concept injection; the model must report the concept *before* verbalising it.
- **Injection:** peaks *"at a specific layer about two thirds of the way through the model"* (Opus 4.1).
- **Strength:** swept over 50 random words × layers; *"strengths 2 and 4 produced successful results most
  often."* A sweet spot is required — too weak, nothing; too strong, garbled output.
- **Measurement:** **verbal self-report.** Success requires affirmative answer, correct concept
  identification, detection *before* saying the word, and coherent output.
- **Rate:** ~20% for Opus 4/4.1. Highly unreliable and context-dependent.
- **Content-specific:** *"some injected concepts elude the model's introspection even at injection
  strengths that are sufficient for other concepts."*
- Describes the model as noticing *"the presence of an unexpected pattern in its processing"*, but
  explicitly declines to pin a mechanism, warning *"the mechanisms underlying our results could still be
  rather shallow and narrowly specialized."*
- **Origin:** emerges from post-training; absent in base models.
- **Dual-use warning:** introspection could raise situational awareness and help models evade oversight.

### 2.4 Lederman & Mahowald (2026) — *Emergent Introspection in AI is Content-Agnostic* ⚠️

- **Central claim:**
  > *"introspection in these models is content-agnostic: models can detect that an anomaly occurred even
  > when they cannot reliably identify its content."*
- **Evidence:** *"models... require fewer tokens to detect an injection than to guess the correct concept
  (with wrong guesses coming earlier)"*; *"The models confabulate injected concepts that are high-frequency
  and concrete (e.g., 'apple')."*
- Replicates Lindsey's paradigm on *"large open-source models."*

### 2.5 Fornasiere et al. (2026) — *Language Models Recognize Dropout and Gaussian Noise Applied to Their Activations*

- **Abstract (verbatim):**
  > *"We provide evidence that language models can detect, localize and, to a certain degree, verbalize the
  > difference between perturbations applied to their activations. More precisely, we either (a) mask
  > activations, simulating dropout, or (b) add Gaussian noise to them, at a target sentence. We then ask a
  > multiple-choice question such as "Which of the previous sentences was perturbed?" or "Which of the two
  > perturbations was applied?". We test models from the Llama, Olmo, and Qwen families, with sizes between
  > 8B and 32B, **all of which can easily detect and localize the perturbations, often with perfect
  > accuracy**. These models can also learn, when taught in context, to distinguish between dropout and
  > Gaussian noise. Notably, Qwen3-32B's zero-shot accuracy in identifying which perturbation was applied
  > improves as a function of the perturbation strength and, moreover, decreases if the in-context labels
  > are flipped, suggesting a prior for the correct ones -- even modulo controls. Because dropout has been
  > used as a training-regularization technique, while Gaussian noise is sometimes added during inference,
  > we discuss the possibility of a data-agnostic **"training awareness"** signal and the implications for
  > AI safety."*
- **Task format is forced-choice / localisation**, not binary detection. ← methodologically crucial.

### 2.6 *Detecting the Disturbance: A Nuanced View of Introspective Abilities in LLMs* (2026)

The methodological critique. **Model:** Meta-Llama-3.1-8B-Instruct.

- **The critique:**
  > *"apparent detection accuracy is **entirely explained by global logit shifts** that bias models toward
  > affirmative responses regardless of question content."*
- **Prescription:** move beyond binary detection; use *"tasks requiring differential sensitivity."*
- **Positive findings — introspection is real but narrow:**
  - localise injected sentences: **88%** (vs 10% chance)
  - discriminate injection strengths: **83%** (vs 50% chance)
  - > *"these capabilities are **confined to early-layer injections and collapse to chance thereafter**."*
- **Verdict:** introspection is *"a real but layer-dependent phenomenon."*

### 2.7 *Obfuscated Activations Bypass LLM Latent-Space Defenses*

The closest prior art to the optimisation fallback in [`alternative-routes.md`](alternative-routes.md).

- **Method:** joint optimisation `min ℒ_beh(M(x)) + λ_obf · ℒ_obf(ℛ(x))` — behavioural loss pushes the
  desired output, obfuscation loss fools the monitor. **Model frozen; only the adversarial suffix
  optimised.**
- **What is optimised:** *"(1) hard prompt suffixes (real tokens appended to the end of an input) and (2)
  embedding suffixes (a matrix of values appended to the end of a token sequence embedding)."*
  **They optimise a suffix, not a steering vector.**
- **Monitors evaded:** linear/MLP harmfulness probes, Gaussian and VAE OOD detectors, SAE+linear probes,
  Circuit Breakers, Latent Adversarial Training.
- **Results:** embedding attacks achieve *"0% recall against obfuscation attacks"* while maintaining *"90%
  jailbreaking rate."*
- **The critical limitation:**
  > *"**We are unable to achieve near-orthogonality while preserving behavior**"*
- **The tradeoff is real and published:**
  > *"When obfuscating against probes that detect SQL code, the model still generates SQL, but its code is
  > incorrect more often than without obfuscation."*
- **Adversarial training does not close it:** *"Even after over 70 epochs, the attacker can still find a new
  suffix that fools the latest probe"* — obfuscated representations are *"abundant."*

### 2.8 Aparin & Gaintseva (2026) — *A Geometric Account of Activation Steering through Angle-Norm Decomposition* ⚠️

- Steering methods differ in *"how they couple two geometric effects: changing a token's angular alignment
  with a concept direction and changing its hidden-state norm."*
- **Key claim:** *"concepts are represented primarily in **angular structure**, supporting the motivation
  for spherical methods, but that **norm remains important for the stability and downstream effects of
  steering**."*
- Revisits the assumption that *"hidden-state norm does not carry concept-relevant information."*

---

## 3. Settled facts (multi-paper agreement)

1. **Inject at ~60–67% depth.** Fonseca Rivera (67%), Macar (60%), Lindsey (~67%). Robust consensus.
2. **Introspection is a post-training phenomenon.** Absent in base models (Lindsey); elicited by **DPO, not
   SFT** (Macar).
3. **Baseline emergent detection is weak and unreliable.** ~20% (Lindsey, Opus); 38.2% TPR @ 0% FPR (Macar,
   Gemma3-27B). Highly context-dependent.
4. **An injection-strength sweet spot exists.** Too weak → nothing; too strong → garbled (Lindsey: 2, 4).
5. **Self-report plus an LLM judge is the standard measurement.** Probes are instruments, not ground truth.
6. **Introspection is under-elicited.** Refusal machinery suppresses it (Macar: 10.8% → 63.8% under
   refusal-direction ablation).
7. **Concepts live primarily in angular structure**; norm matters for stability (Aparin & Gaintseva).
8. **Steering vectors are unreliable in general** — hyperparameter-sensitive, OOD-fragile,
   capability-degrading (*A Sober Look at Steering Vectors*).

---

## 4. The contradiction map

### C1 — Directional detector vs generic anomaly detector **[the central question]**

| Position | Source | Evidence |
|---|---|---|
| **Directional**, not anomaly | Fonseca Rivera | *"directional detector tuned to the CAA subspace, as opposed to a generic anomaly detector"*; rejects Gaussian noise 94%; PCA 1%, ICA 3%, random 0% |
| **Anomaly**, distributed | Macar et al. | *"nontrivial internal anomaly detection"*; *"distributed across multiple directions"*; opposite directions both trigger (23.3% vs 3.2%) → *"inconsistent with the single direction hypothesis"* |
| **Anomaly**, near-perfect | Fornasiere et al. | 8B–32B detect and localise **dropout and Gaussian noise** *"often with perfect accuracy"* |
| **Content-agnostic** | Lederman & Mahowald | *"models can detect that an anomaly occurred even when they cannot reliably identify its content"* |

**A direct stimulus-level contradiction:** Fonseca Rivera's detector *rejects* Gaussian noise 94% of the
time; Fornasiere's models detect Gaussian noise *near-perfectly*. Same stimulus, opposite result.

**Leading reconciliation hypotheses — all untested:**

- **(a) Trained vs emergent.** Fonseca Rivera **LoRA-trained** their detector on CAA vectors, so it is
  directional *by construction of its training distribution*. Macar/Fornasiere/Lindsey measure the
  **emergent** capability. These may be different objects, and a CAA-trained detector's blind spot may say
  nothing about emergent introspection. **← the main plan tests this.**
- **(b) Task format.** Fonseca Rivera asks a **binary** question; Fornasiere asks **forced-choice**. Binary
  detection is contaminated by global logit shift (*Detecting the Disturbance*); forced-choice is not. The
  noise discrepancy may be a task-format artifact. **← tested as a secondary axis of the main plan.**
- **(c) Perturbation site and magnitude.** Fornasiere perturbs across a target sentence (many token
  positions); Fonseca Rivera injects one vector at one layer and token. Not magnitude-matched across
  papers.

### C2 — Content-agnostic vs content-specific
- **Agnostic:** Lederman & Mahowald — detect *that*, not *what*.
- **Specific:** Lindsey — *"some injected concepts elude the model's introspection even at injection
  strengths that are sufficient for other concepts."*
- **Middle:** Macar — *"Detection depends on the injected concept producing a sufficiently distinctive
  perturbation."*

### C3 — Layer depth
- **60–67% works:** Fonseca Rivera, Macar, Lindsey.
- **Contradicted:** *Detecting the Disturbance* — *"confined to early-layer injections and collapse to
  chance thereafter."*
- **Possible resolution:** different task families (localisation vs binary detection). Unresolved.

### C4 — Is binary detection valid at all?
- **No:** *Detecting the Disturbance* — *"entirely explained by global logit shifts."* Godet concurs:
  steered models say "yes" to nonsense controls.
- **Partial defence:** Macar reports **0% FPR** alongside 38.2% TPR — a pure logit shift cannot produce
  that. Fonseca Rivera also reports 0% FPR.
- **Emerging consensus:** prefer **forced-choice / localisation / differential-sensitivity** tasks.

### C5 — Detection ≠ resistance vs endogenous resistance
- Fonseca Rivera: detection-trained models are **more** susceptible to steering.
- *Endogenous Resistance to Activation Steering*: models internally **counteract** injected steering.
- Both single-paper findings; neither replicated.

---

## 5. Open questions, ranked

1. **Is emergent steering-awareness directional (evadable) or anomaly-based (not)?** Resolves C1;
   determines whether detection-evading steering is possible at all for future models. **← the main plan.**
2. **Do random / Gaussian / dropout perturbations trigger *emergent* detection?** Macar explicitly did not
   test this; Fonseca Rivera says no (trained); Fornasiere says yes (emergent).
3. **Does task format explain the noise contradiction?** Methodological, and would clean up the literature.
4. **Can a detector trained on diverse vector types close the CAA blind spot?** Determines
   detection-evasion's shelf life. → [`alternative-routes.md`](alternative-routes.md), Route C.
5. **Does the blind spot survive refusal ablation, or is it an under-elicitation artifact?**
   → [`spin-offs.md`](spin-offs.md), S1.
6. **Does norm-restoration recover post-projection semantics?** Theory (Aparin & Gaintseva) says no.
   → Route B.
7. **Is introspection early-layer only?** Reconcile C3.
8. **Why does detection increase steerability, and how does that square with endogenous resistance?** (C5)

---

## 6. Novelty landscape

| Claim | Status |
|---|---|
| Orthogonal/PCA/random vectors evade a **CAA-trained** detector | **Established** — Fonseca Rivera, Table 5 |
| Projecting onto the orthogonal complement to defeat a linear probe | **Established** — LEACE / INLP / concept erasure |
| Optimising an input to fool latent-space monitors while preserving behaviour | **Established** — Obfuscated Activations (suffix-based) |
| Coherence-preserving activation attacks | **Established** — *Steering in the Shadows* |
| Steering vectors as a red-teaming attack | **Established** — *Trojan Activation Attack* |
| **Does orthogonal projection evade *emergent* introspection?** | **OPEN ← the main plan** |
| **Are random/noise vectors detected by *emergent* introspection?** | **OPEN** — Macar explicitly did not test |
| **Does binary vs forced-choice format explain the noise contradiction?** | **OPEN** |
| **Can the CAA blind spot be closed by diverse-vector training?** | **OPEN** — Route C |
| **Does norm-restoration recover semantics after projection?** | **OPEN** — theory says no — Route B |
