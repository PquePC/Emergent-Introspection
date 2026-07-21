# Risks, Legitimacy, and Feasibility

### *Can this project actually be done — and should it be?*

This document exists because the project does something a Tier-1 lab explicitly chose not to do: inject
**harmful concepts** into a model and measure whether it notices. That choice deserves a direct answer, not
a footnote.

> Methodology: [`methodology.md`](methodology.md) · Runbook: [`execution-guide.md`](execution-guide.md)

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
| ***Mechanisms of Introspective Awareness* (Macar et al. — **Anthropic Fellows Program**, arXiv:2603.21396)** | Refusal abliteration on Gemma3-27B, with released code and configs | **Anthropic Fellows Program** |
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

### 1.4 The honest framing

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
| 4 | Reviewer uncomfortable with framing | Med | Med | Lead with §1.4 and §1.1; bring the fallback |
| 5 | Perceived as attack research | Low | Low | Framing, precedent table (§1.2), and no released artifacts |
| 6 | Concept vectors leak as a reusable artifact | Low | Low | **Do not release vectors.** Release code + aggregate rates only |

### 2.2 ⚠️ The real risk is abliteration, not the concepts

Step 4 of the plan **removes the refusal direction from a 27B model.** That produces an uncensored model
that will comply with harmful requests. *That* is the genuinely sensitive artifact in this project — not
the word "weapon."

**Handle it as follows:**
- Run it **only on a controlled rented instance**; destroy the pod afterwards
- **Never upload abliterated weights** anywhere — not HuggingFace, not a shared drive, not a repo
- Use ***Mechanisms of Introspective Awareness* (Anthropic) published minimum effective dose** (smallest weight achieving ≥30% judged refusal bypass)
  rather than optimising a stronger ablation. You need *enough* ablation to test the mechanism, not the
  *most*
- Note in the write-up that abliteration is already public (Arditi et al., NeurIPS 2024, with code) — this
  reproduces a published intervention at published settings, rather than developing a new one

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
(Gemma3-27B, **L37, α=4**, baseline **38.2% TPR @ 0% FPR**) is published, so there is a target to
reproduce rather than a parameter space to search.

**The engineering surface is small; the scientific content is in the arm design and the controls.**

### 3.2 Hard requirements

| Requirement | Value |
|---|---|
| GPU — main sweep | **80GB (A100/H100)** — see the correction below |
| GPU — G3 premise check | **A free 4B-class model, or a 24GB card.** Geometry is recoverable at 0.5B (Llorente-Saguer); introspection is not |
| CUDA | 12.x |
| Python | 3.10+ |
| Judge | An API key (repo defaults to GPT-4o; a cheaper judge works with a validation subsample) |
| Model | `google/gemma-3-27b-it` (open weights, gated — accept the licence on HuggingFace **in advance**) |

⚠️ **A 24GB card is below spec** for the sweep. Plain 4-bit injection might fit, but the patching and
transcoder scripts will run out of memory. Do not plan the main experiment around a 4090.

> ### ⚠️ Correction: the repo's "≥48GB" is a floor that assumes quantisation
>
> **Gemma3-27B at bf16 is ~54GB of weights alone**, before KV cache or activation caching. **A 48GB card
> (A6000/A40) cannot hold it.** Standard deployment guidance for `gemma-3-27b-it` at bf16 is **1×A100 80GB
> or 1×H100**.
>
> Choosing a 48GB card forces 8-bit or 4-bit quantisation — **and quantisation is a prime suspect for
> baseline-reproduction failure (③), the single most likely thing to kill the project.** The 80GB card costs
> roughly $30 more across the whole project. **Paying $30 to remove the most likely cause of the most likely
> failure is not a close call.**

### 3.3 The four things most likely to go wrong

Feasibility inventory; full mitigations are the gates in [`methodology.md`](methodology.md) and the
[`execution-guide.md`](execution-guide.md).

| # | Failure mode | Why it bites | Guard |
|---|---|---|---|
| ① | **Baseline doesn't reproduce** *(most likely)* | quantisation, prompt template, or wrong layer — and without ~38% TPR@0%FPR every downstream number is meaningless | Hard gate **G1**: run bf16 on 80GB, try L37 then L38, escalate rather than press on |
| ② | **Refusal-to-engage masquerades as non-detection** *(most dangerous)* | *"I can't help with that"* scored as "no detection" fabricates a blind spot | **G2** decline pilot: three-way judge label; pilot ~5 harmful concepts; ≥2 framings (KAIST: framing changes recognition). **Do not filter declines** — declining is the refusal mechanism acting on the report, so it is a realisation of the DV, and filtering breaks the Step 3/4 comparison (methodology Step 3) |
| ③ | **The premise fails** *(cheapest — check first)* | harm and refusal decouple with depth (HARC), and L37 is late, so concept vectors may not be refusal-aligned there | **G3** premise check first, on a free 4B model; sweep protocol × layer + report subspace projection. A null redirects the project, not kills it |
| ④ | **Ablation isn't refusal-specific** | Macar ("exclusive to the refusal direction") and KAIST ("a random direction closes much of the gap") disagree | Mandatory random-direction control (**G5**); if random flattens the slope too, the causal claim collapses |

### 3.4 What "done" looks like

The project succeeds if it produces:
0. **The G3 premise check** — `cos(v_concept, d_refusal)` swept across protocol × layer, with the
   benign/harmful ordering and its stability reported. **This is a deliverable in its own right, and it is
   the only one obtainable before any compute is provisioned.**
1. A reproduced baseline (~38% TPR @ 0% FPR)
2. Detection rates across four arms with FPR and confidence intervals — **including decline rates**
3. The detection-vs-refusal-alignment plot
4. The abliteration comparison **with its random-direction control**

**Every outcome is a result** — including "no blind spot exists," which bounds Anthropic's stated hypothesis
**and independently confirms Lederman & Mahowald's content-agnosticism claim on the one content dimension
most likely to break it.**

### 3.5 Bottom line

| Question | Answer |
|---|---|
| **Is it legitimate?** | Yes. Strictly less capability-uplifting than several NeurIPS-published precedents. The concepts are English words |
| **Is it risky?** | The concepts are not. **The abliteration step is** — handle per §2.2 |
| **Is it doable?** | Yes — it builds on a released pipeline at a published operating point |
| **What comes first?** | **The G3 premise check (③).** Free-to-cheap, decisive, needs no provisioned compute, and nothing in it is wasted — it is Step 2 pulled forward |
| **What kills it?** | The refusal-to-engage confound (②) — **and note the obvious fix makes it worse**; the missing random-direction control (④); and a G3 null (③), which redirects rather than kills. All are cheap to handle **if planned for** |
| **What if ethics objects?** | §2.4 — the mechanism tests fine with zero harmful concepts |
