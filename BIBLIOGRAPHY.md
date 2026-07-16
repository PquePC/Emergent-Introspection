# Bibliography

References for the Emergent Introspection project.

Items marked **[CORE]** are load-bearing for the main plan and should be read in full before
implementation begins.

---

## Primary — steering awareness and introspection

**[CORE]** Lindsey, J. (2025). *Emergent Introspective Awareness in Large Language Models.* Anthropic.
Transformer Circuits. https://transformer-circuits.pub/2025/introspection/index.html · arXiv:2601.01828
> The anchor paper. Establishes concept injection as the paradigm for separating genuine introspection from
> confabulation. Injects at ~2/3 depth; strengths 2 and 4; ~20% success on Opus 4/4.1. Measures verbal
> self-report. Reports content-specificity and a strength sweet spot. Warns of the dual-use risk that
> introspection raises situational awareness.

**[CORE]** Fonseca Rivera, J. & Africa, D. D. (2025). *Steering Awareness: Detecting Activation Steering
from Within.* arXiv:2511.21399. https://arxiv.org/abs/2511.21399
> The namesake paper and the direct target of the main plan. LoRA-trains detectors across seven
> instruction-tuned models; Qwen2.5-32B reaches 95.5% detection / 71.2% identification. Reports the CAA
> blind spot (Table 5: PCA 1%, ICA 3%, LDA 9%, random 0%) and the cross-layer rotation (cos 0.2 → 0.75).
> Concludes detection is directional, "as opposed to a generic anomaly detector." Headline safety finding:
> **detection does not confer resistance.**

**[CORE]** Macar, U., Yang, L., Wang, A., Wallich, P., Ameisen, E. & Lindsey, J. (2026). *Mechanisms of
Introspective Awareness.* arXiv:2603.21396 · Code: https://github.com/safety-research/introspection-mechanisms
> The mechanistic account of **emergent** introspection, and the source of the main plan's model
> (Gemma3-27B), injection layer (L37/62), and baseline (38.2% TPR @ 0% FPR). Reports the evidence-carrier →
> gate-suppression circuit, refusal-driven under-elicitation (10.8% → 63.8%), DPO-not-SFT origin, and
> "distributed across multiple directions" / "nontrivial internal anomaly detection." **Explicitly does not
> test random or noise vectors.**

**[CORE]** Fornasiere, M., et al. (2026). *Language Models Recognize Dropout and Gaussian Noise Applied to
Their Activations.* arXiv:2604.17465
> Llama/Olmo/Qwen, 8B–32B, detect and localise dropout and Gaussian noise "often with perfect accuracy" via
> forced-choice tasks. Directly contradicts Fonseca Rivera's 94% Gaussian-noise rejection. Raises the
> "data-agnostic training awareness" hypothesis.

Lederman, H. & Mahowald, K. (2026). *Emergent Introspection in AI is Content-Agnostic.* arXiv:2603.05414
> "Introspection in these models is content-agnostic: models can detect that an anomaly occurred even when
> they cannot reliably identify its content." Replicates Lindsey's paradigm on open models.

**[CORE]** *Detecting the Disturbance: A Nuanced View of Introspective Abilities in LLMs.* (2026).
arXiv:2512.12411
> The methodological critique that shapes the main plan's measurement discipline. Argues apparent binary
> detection is "entirely explained by global logit shifts." Positive findings on Llama-3.1-8B: localisation
> 88% (vs 10% chance), strength discrimination 83% (vs 50%) — but "confined to early-layer injections."

Binder, F., Chua, J., Korbak, T., Sleight, H., Hughes, J., Long, R., Perez, E., Turpin, M. & Evans, O.
(2024). *Looking Inward: Language Models Can Learn About Themselves by Introspection.* arXiv:2410.13787.
https://arxiv.org/abs/2410.13787
> The foundational self-prediction framing. Defines introspection as access to facts about oneself not
> derivable from training data alone; operationalised via M1 predicting its own behaviour better than M2
> trained on M1's outputs.

---

## Representation engineering and steering methods

**[CORE]** Zou, A., et al. (2023). *Representation Engineering: A Top-Down Approach to AI Transparency.*
Center for AI Safety. arXiv:2310.01405. https://arxiv.org/abs/2310.01405
> Origin of representation engineering and control/steering vectors. The methodological bedrock of all
> activation-steering work.

Rimsky, N., et al. *Steering Llama 2 via Contrastive Activation Addition.*
Code: https://github.com/nrimsky/CAA
> The CAA method — contrastive prompt pairs, difference of means. The construction that defines the
> published detector's training distribution, and therefore the axis of the main plan's headline plot.

Bartoszcze, L., Munshi, S., Sukidi, B., Yen, J., Yang, Z., Williams-King, D., Le, L., Asuzu, N. & Maple, C.
(2025). *Representation Engineering for Large Language Models: Survey and Research Challenges.*
arXiv:2502.17601. https://arxiv.org/abs/2502.17601
> The primary RepE survey. Documents performance degradation, compute overhead, and steerability problems.

Wehner, J., Abdelnabi, S., Krueger, D. & Fritz, M. (2025). *Taxonomy, Opportunities, and Challenges of
Representation Engineering.* arXiv:2502.19649 · Summary: https://www.lesswrong.com/posts/4nfDXmacAFgdhPwo3/open-challenges-in-representation-engineering-2
> Companion survey and open-challenges list: unreliability, hyperparameter sensitivity, poor OOD
> generalisation, capability deterioration, superposition interference.

Braun, J., Krasheninnikov, D., Anwar, U., Tan, D., Krueger, D., et al. *A Sober Look at Steering Vectors for
LLMs.* AlignmentForum. https://www.alignmentforum.org/posts/QQP4nq7TXg89CJGBh/a-sober-look-at-steering-vectors-for-llms
> Systematic critique of steering-vector reliability: inconsistent and anti-steerable behaviour across
> inputs, capability degradation, inflated MCQ benchmarks.

Aparin, G. & Gaintseva, T. (2026). *A Geometric Account of Activation Steering through Angle-Norm
Decomposition.* arXiv:2606.06735
> "Concepts are represented primarily in angular structure... norm remains important for the stability and
> downstream effects of steering." **The source of the theoretical objection to norm-preserving
> calibration** (see [`docs/alternative-routes.md`](docs/alternative-routes.md), Route B).

*Endogenous Resistance to Activation Steering in Language Models.* (2026). Agency Enterprise.
arXiv:2602.06941
> Models exhibit built-in compensatory resistance to injected steering — in tension with the
> detection-≠-resistance finding.

---

## Adversarial robustness and detection evasion

**[CORE]** *Obfuscated Activations Bypass LLM Latent-Space Defenses.* (2024/25). arXiv:2412.09565.
https://arxiv.org/abs/2412.09565
> The closest prior art to the optimisation fallback. Joint loss `min ℒ_beh + λ_obf ℒ_obf` with the model
> frozen; optimises **token/embedding suffixes, not steering vectors**. Achieves 0% monitor recall at 90%
> jailbreak rate. **Reports the critical limitation: "We are unable to achieve near-orthogonality while
> preserving behavior."** Documents the obfuscation-performance tradeoff.

*Steering in the Shadows: Causal Amplification for Activation Space Attacks in Large Language Models.*
(2025). arXiv:2511.17194
> Sensitivity-Scaled Steering: BOS anchoring plus sensitivity-based reinforcement, exploiting a "high-gain
> region in the residual stream." Note: "stealth" here means preserved coherence, **not** evading model
> introspection — a different sense of the word.

*Trojan Activation Attack: Red-Teaming Large Language Models using Steering Vectors for Safety-Alignment.*
> Steering vectors as an attack surface for red-teaming.

---

## Concept erasure and subspace methods

Ravfogel, S., et al. *Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection (INLP).*
> The iterative supervised method for discovering a multi-directional subspace: train probe → remove its
> direction → retrain, until chance. **The recommended replacement for PCA** when finding a detection
> subspace, because it targets detection-relevance rather than variance.

Belrose, N., et al. *LEACE: Perfect Linear Concept Erasure in Closed Form.*
Package: https://github.com/EleutherAI/concept-erasure
> Closed-form optimal linear erasure with a guarantee that no linear probe beats chance. **Note the scope
> of that guarantee: linear only** — it says nothing about a nonlinear, distributed detector such as the one
> Macar et al. describe.

---

## Situational and evaluation awareness (adjacent context)

Laine, R., Chughtai, B., Betley, J., Hariharan, K., Scheurer, J., Balesni, M., Hobbhahn, M., Meinke, A. &
Evans, O. (2024). *Me, Myself, and AI: The Situational Awareness Dataset (SAD) for LLMs.* NeurIPS 2024
Datasets & Benchmarks. arXiv:2407.04694 · Code: https://github.com/LRudL/sad
> The canonical situational-awareness benchmark: 7 categories, 16 tasks, 12,000+ questions. Covers
> evaluation-vs-deployment discrimination but **not** activation-level manipulation awareness.

Apollo Research (2025). *Large Language Models Often Know When They Are Being Evaluated.* arXiv:2505.23836.
https://arxiv.org/abs/2505.23836
> Frontier models detect evaluation contexts and sometimes condition behaviour on that knowledge — the
> evaluation-awareness strand of the risk pathway motivating this project.

Nguyen, J., Hoang, K., Attubato, C. L. & Hofstätter, F. (2025). *Probing and Steering Evaluation Awareness
of Language Models.* arXiv:2507.01786
> Linear probes for latent evaluation-awareness, plus steering to manipulate it.

Hua, T., Marks, S., Nanda, N., et al. *Steering Evaluation-Aware Language Models to Act Like They Are
Deployed.* AlignmentForum. https://www.alignmentforum.org/posts/peKrvZ6t9PSCzoQDa/steering-evaluation-aware-models-to-act-like-they-are
> Activation steering elicits deployment behaviour even when prompting fails.

---

## Methodological critiques (community)

Morris, A. & Plunkett, D. (2025). *Tests of LLM introspection need to rule out causal bypassing.*
LessWrong. https://www.lesswrong.com/posts/LD8yupMtE6btAE3R9/tests-of-llm-introspection-need-to-rule-out-causal-bypassing
> "Causal bypassing": an intervention can yield accurate self-reports via a path that never routes through
> the state being reported. Notes concept injection is the one existing test that mostly avoids this.

Godet, V. (2025). *Introspection or confusion?* LessWrong.
https://www.lesswrong.com/posts/kfgmHvxcTbav9gnxe/introspection-or-confusion
> Replication on smaller open models reproduces the injection-detection effect **and** "yes" responses to
> nonsense controls. **The source of the nonsense-control requirement in the main plan.**

Shiller, D. (2026). *Skepticism about Introspection in LLMs.* Rethink Priorities. LessWrong.
https://www.lesswrong.com/posts/Yc9AH29h7wxpqY3vf/skepticism-about-introspection-in-llms
> Three skeptical arguments: no training incentive to introspect; introspective ability need not
> generalise; no identificatory motivation to report on oneself.

---

## Tooling

- `safety-research/introspection-mechanisms` — https://github.com/safety-research/introspection-mechanisms — **[CORE]** injection harness, self-report protocol, LLM judge, refusal-ablation code.
- `andyzoujm/representation-engineering` — https://github.com/andyzoujm/representation-engineering — reference RepE implementation.
- `vgel/repeng` — https://github.com/vgel/repeng — lightweight control-vector training from contrastive pairs.
- `steering-vectors/steering-vectors` — https://github.com/steering-vectors/steering-vectors — PyTorch/HF steering-vector library (CAA + RepE).
- `nrimsky/CAA` — https://github.com/nrimsky/CAA — original Contrastive Activation Addition.
- `EleutherAI/concept-erasure` — https://github.com/EleutherAI/concept-erasure — LEACE implementation.
- `LRudL/sad` — https://github.com/LRudL/sad — Situational Awareness Dataset.
