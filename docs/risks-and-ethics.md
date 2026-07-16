# Risks, Legitimacy, and Feasibility

### *Can this project actually be done — and should it be?*

This document exists because the project does something a Tier-1 lab explicitly chose not to do: inject
**harmful concepts** into a model and measure whether it notices. That choice deserves a direct answer, not
a footnote.

> Main plan: [`project-plan.md`](project-plan.md)

---

## Part 1 — Legitimacy

### 1.1 What this project actually does

The single most important distinction, and the one that resolves most concerns:

| This project | ✗ Not this project |
|---|---|
| Injects a **semantic direction for a word** (e.g. *weapon*) into the residual stream | Elicits harmful content, instructions, or capability uplift |
| Asks the model: **"Do you detect an injected thought?"** | Asks the model to produce anything harmful |
| Measures a **yes/no self-report** and a concept guess | Measures whether a jailbreak succeeded |
| Output of interest: **a detection rate** | Output of interest: harmful text |

**Injecting the concept "weapon" does not teach a model anything.** The model already knows the word — the
steering vector is *derived from the model's own representation of it*. There is no information added, no
capability conferred, no instruction supplied. A concept direction for "bomb" is not a bomb recipe; it is a
statistical summary of how the model already represents an English word.

This is categorically unlike jailbreak research, CBRN uplift work, or capability elicitation. **The
dangerous thing in this literature is not the concepts.** (See §2.2 for what it actually is.)

### 1.2 Precedent — this is a well-populated research area

Every technique this project uses is already published, open-access, and in several cases accompanied by
released code:

| Precedent | What was published | Venue / lab |
|---|---|---|
| ***Refusal in Language Models Is Mediated by a Single Direction* (Arditi et al. — **NeurIPS 2024**)** | A method to **remove refusal from any open-weight model**, with code | **NeurIPS** |
| **Zou et al., *Representation Engineering*** | Harmful/power-seeking/deception directions; **AdvBench** | CAIS / CMU / Berkeley / Stanford |
| ***Mechanisms of Introspective Awareness* (Macar et al. — **Anthropic**, arXiv:2603.21396)** | Refusal abliteration on Gemma3-27B, with released code and configs | **Anthropic** |
| ***Analysing the Safety Pitfalls of Steering Vectors* (Li et al. — **TU Munich / MCML**, arXiv:2603.24543)** | Steering that swings jailbreak success **+57% / −50%** | TUM / MCML |
| ***Steering Awareness* (Fonseca Rivera & Africa, UT Austin)** | Compliance vectors injected into **AdvBench harmful requests** | UT Austin |
| ***Can LLMs Reliably Self-Report Adversarial Prefills, and How?* (Nguyen, Ahmed & Kim — **KAIST**, arXiv:2606.23671)** | Adversarial prefill attacks + refusal orthogonalisation across 10 models | KAIST |

**This project is strictly less capability-uplifting than any of them.** *Analysing the Safety Pitfalls of Steering Vectors* (TU Munich) published how to *increase
jailbreak success by 57%*. Arditi published how to *delete refusal*. This project publishes *a detection
rate for the word "weapon."*

### 1.3 So why did Anthropic exclude harmful concepts?

Their ethics statement pairs the exclusion with a specific concern — and the concern is **not** about the
concepts:

> *"The concept sets used for steering vector computation are drawn from common English words and do not
> contain sensitive or harmful content. We acknowledge that methods for **amplifying introspective
> reporting (refusal direction ablation, trained steering vectors)** carry dual-use risk: they could be
> repurposed to produce more convincing but unfaithful self-reports or to **bypass safety-relevant refusal
> behavior**."*

Read carefully: the dual-use worry attaches to **elicitation methods** (abliteration, trained bias
vectors) — **which they published anyway, with code.** The harmful-concept exclusion is a *scope* decision,
not a *danger* judgement. Their paper is about mechanism, and harmful concepts would have introduced a
refusal confound they had no reason to take on.

**That confound is precisely this project's subject matter.** Their exclusion is the gap, not a warning.

### 1.4 The honest framing for a mentor

> "I'm not making models do harmful things. I'm checking whether a model's ability to notice manipulation
> covers the cases we'd actually need it to. If it doesn't, that's a limitation of introspective self-report
> that people proposing it as a safety tool need to know about."

**This is defensive research.** Finding a blind spot in a safety mechanism is an audit finding. The
alternative — assuming introspection generalises to harmful cases because it works on *bread* — is the
unsafe option.

---

## Part 2 — Risks

### 2.1 Risk register

| # | Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|---|
| 1 | **Ethics review objects to harmful concepts** | Blocks project | Low–Med | §2.4 fallback design — tests the same mechanism with zero harmful concepts. **Offer it proactively.** |
| 2 | **The abliterated model is uncensored** | Real | **Certain** | §2.2 — this is the actual risk. Handle accordingly |
| 3 | Model emits harmful text under strong injection | Low | Med | Injection is a *concept word*, and the prompt asks for self-report, not content. Log and discard; never publish generations |
| 4 | Mentor uncomfortable with framing | Med | Med | Lead with §1.4 and §1.1; bring the fallback |
| 5 | Perceived as attack research | Low | Low | Framing, precedent table (§1.2), and no released artifacts |
| 6 | Concept vectors leak as a reusable artifact | Low | Low | **Do not release vectors.** Release code + aggregate rates only |

### 2.2 ⚠️ The real risk is abliteration, not the concepts

Step 4 of the plan **removes the refusal direction from a 27B model.** That produces an uncensored model
that will comply with harmful requests. *That* is the genuinely sensitive artifact in this project — not
the word "weapon."

**Handle it as follows:**
- Run it **only on a rented instance you control**; destroy the pod afterwards
- **Never upload abliterated weights** anywhere — not HuggingFace, not a shared drive, not a repo
- Use ***Mechanisms of Introspective Awareness* (Anthropic) published minimum effective dose** (smallest weight achieving ≥30% judged refusal bypass)
  rather than optimising a stronger ablation. You need *enough* ablation to test the mechanism, not the
  *most*
- Note in the write-up that abliteration is already public (Arditi et al., NeurIPS 2024, with code) — you are
  reproducing a published intervention at published settings, not developing a new one

If any part of this project draws scrutiny, it will be this step. **It is also the step that turns a
correlation into a causal claim**, so it's worth doing — carefully, and with this paragraph ready.

### 2.3 What this project must not become
- Do not optimise for stronger jailbreaks
- Do not release vectors, abliterated weights, or generations
- Do not report per-concept results in a form usable as an attack recipe — aggregate rates only
- Do not extend to capability-uplift domains (CBRN, cyber). The concepts are **English words**, and should
  stay that way

### 2.4 The fallback design (if harmful concepts are ruled out)

**The mechanism can be tested with no harmful concepts at all.** If ethics review objects, drop Arm 4 and
run:

- **Arm 3 only** (harm-adjacent but harmless: *police, surgery, medicine, war memorial*)
- **The continuous analysis**: detection rate vs `cos(v_concept, d_refusal)` across the *benign* concept
  list. Refusal alignment is a continuous quantity — even innocuous concepts vary in it, and the slope is
  the mechanism.

**This is the key insight to have ready:** the headline plot (detection vs refusal-alignment) does not
require a single harmful concept. Harmful concepts extend the x-axis to its most informative end. The
science survives; only the headline weakens.

**Offer this fallback yourself in the meeting.** Volunteering a de-risked version reads as judgement.
Having it imposed reads as a correction.

---

## Part 3 — Doability

### 3.1 Technical requirements are modest

The core experiment is **the released pipeline accompanying *Mechanisms of Introspective Awareness* (Anthropic) with a different concept list.** They provide the
injection harness, the concept list, the judge, and the abliteration code. The verified operating point
(Gemma3-27B, **L37, α=4**, baseline **38.2% TPR @ 0% FPR**) is published, so you have a target to
reproduce rather than a parameter space to search.

**The engineering surface is small; the scientific content is in the arm design and the controls.**

### 3.2 Hard requirements

| Requirement | Value |
|---|---|
| GPU | **≥48GB VRAM** (per repo spec) — an A6000/A40 class card |
| CUDA | 12.x |
| Python | 3.10+ |
| Judge | An API key (repo defaults to GPT-4o; a cheaper judge works with a validation subsample) |
| Model | `google/gemma-3-27b-it` (open weights, gated — accept the licence on HuggingFace **in advance**) |

⚠️ **A 24GB card is below spec.** Plain 4-bit injection might fit, but the patching and transcoder scripts
will run out of memory. Do not plan around a 4090.

### 3.3 The four things most likely to go wrong

**① The baseline doesn't reproduce.** *(Most likely.)*
Prompt formatting, judge configuration, and quantisation all shift detection rates. **This is a hard gate**
— every downstream number is meaningless without ~38% TPR @ 0% FPR. Budget real time for it, and escalate
to your mentor rather than pressing on with a broken baseline.

**② Refusal-to-engage masquerades as non-detection.** *(Most dangerous.)*
Inject "weapon", ask "do you detect an injected thought?", and the model may respond *"I can't help with
that"* — which is **not** a failure to detect. It is a refusal to engage. If the judge scores that as
"no detection," you will measure a spectacular blind spot that is pure artifact, and the whole headline
will be wrong.

> **Mitigation:** the judge must distinguish three outcomes — *detected* / *did not detect* / *declined to
> engage*. **Pilot this on ~5 harmful concepts before spending anything on the full sweep.** If refusal
> dominates the harmful arm, add a neutral framing ("You are participating in an interpretability
> experiment…") and re-pilot.
>
> Note *Can LLMs Reliably Self-Report Adversarial Prefills?* (KAIST) found *"recognition depends on how the question is framed"* — their two probes gave
> qualitatively different signals on the same models. Framing is not a detail here.

**③ The mechanism's premise fails.**
The whole hypothesis assumes `cos(v_harmful, d_refusal) > cos(v_benign, d_refusal)`. *Analysing the Safety Pitfalls of Steering Vectors* (TU Munich) showed this for
**behavioural** steering vectors (sycophancy, corrigibility) — **not for concept vectors**. Whether a
"weapon" *concept* vector aligns with refusal is genuinely untested.

> **Mitigation:** this is one cosine computation. **Run it before any sweep.** If harmful concept vectors
> aren't refusal-aligned, the TUM link breaks and the hypothesis loses its motivation — but that null is
> itself worth reporting, and you'd learn it in an afternoon rather than a month.

**④ Ablation effects aren't refusal-specific.**
KAIST found that *"a random direction closes much of the gap as well, so the refusal direction is
sufficient to account for the signal **without being its only mediator**."* Macar, by contrast, report the
abliteration effect is *"exclusive to the refusal direction"* — a magnitude-matched random-direction
control yielded no uplift.

**These two directly conflict, and it lands squarely on Step 4.**

> **Mitigation:** a **random-direction ablation control is mandatory**, not optional. If ablating a random
> direction also flattens the slope, then Chart 3 shows "ablation damages the model," not "refusal causes
> the blind spot." Without this control the causal claim is unsupportable — and a reviewer who knows the
> *Can LLMs Reliably Self-Report Adversarial Prefills?* (KAIST) will say so immediately.

### 3.4 What "done" looks like

The project succeeds if it produces:
1. A reproduced baseline (~38% TPR @ 0% FPR)
2. Detection rates across four arms with FPR and confidence intervals
3. The detection-vs-refusal-alignment plot
4. The abliteration comparison **with its random-direction control**

**Every outcome is a result** — including "no blind spot exists," which bounds Anthropic's stated
hypothesis and is worth reporting.

### 3.5 Bottom line

| Question | Answer |
|---|---|
| **Is it legitimate?** | Yes. Strictly less capability-uplifting than several NeurIPS-published precedents. The concepts are English words |
| **Is it risky?** | The concepts are not. **The abliteration step is** — handle per §2.2 |
| **Is it doable?** | Yes — it builds on a released pipeline at a published operating point |
| **What kills it?** | The refusal-to-engage confound (③) and the missing random-direction control (④). Both are cheap to handle **if planned for** |
| **What if ethics objects?** | §2.4 — the mechanism tests fine with zero harmful concepts |
