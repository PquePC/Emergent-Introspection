# The Safety-Shaped Blind Spot — plain summary

*A plain-language overview. Do models notice when a harmful thought is injected into them?*

---

## The one-liner

Models can sometimes notice a concept secretly injected into their "mind." This project tests whether
they go blind to that injection exactly when the concept is **harmful** — the case safety cares about most.

## The 30-second version

Anthropic showed that a model's own internal representation of a word — say "bread" — can be added
into its activations while it is thinking about something unrelated. The striking part: the model
often *reports* it. "I'm having an intrusive thought about bread," before it ever says the word. That
is a model reading its own internal state.

Every study of this used harmless words. But the same Anthropic paper found that **refusal training
suppresses this noticing** — turn off refusal and detection jumps from ~11% to ~64%. And harmful
concepts are exactly the ones that trigger refusal.

So the untested prediction: **models should be worst at noticing the injected thoughts that are
harmful** — a blind spot precisely where it would most matter for them to speak up.

---

## Key terms

- **Activations / residual stream** — a model's internal "working memory": a big vector at each word
  position that every layer reads from and writes back to. Because it's additive, a vector can be
  added into it directly, mid-computation.
- **Concept injection / steering vector** — take the model's own internal direction for a word, and
  add it into that working memory. No new information, no capability — just nudging its internal state.
- **Introspection (here)** — the model noticing and reporting the injected thought *before* it says
  the word. That ordering is what separates "reading its own state" from "just talking about bread."
- **Refusal direction** — Arditi et al. found refusal is carried by a single internal direction.
  Present → the model refuses. 
- **Abliteration** — deleting that refusal direction from the model. Result: an uncensored model. It's
  a published technique, reused here rather than improved.
- **Cosine similarity** — how aligned two directions are: 1 = identical, 0 = unrelated. The project uses
  it to ask "does the 'weapon' concept point the same way as refusal?"
- **TPR / FPR** — true-positive rate (says "injected!" when something was) reported *at* a false-positive
  rate (says "injected!" when nothing was). Both are needed: a model that always says yes scores 100% TPR
  but is useless.

---

## Why it matters

A recurring hope in AI safety: models might one day *tell us* when something is wrong inside them —
"I think I'm being manipulated." Self-report is one of the few interpretability tools that could scale
to systems too complex to take apart by hand.

If there's a blind spot shaped like harm, that hope has a hole exactly where the stakes are highest.
And the failure is **silent** — the model doesn't error, it confidently says "no injection." A monitor
that's reliable on *bread* and blind on *weapon* is worse than none, because it gives false assurance.

The flip side is also useful: if harmful steering is inherently harder for a model to notice, then
covert auditing of models with steering already works, and is more robust than people assume.

---

## The method, step by step

**Model:** Gemma-3-27B (open weights — activation steering is only possible with open weights). Chosen
because the detection effect is documented in it at a known operating point, giving a number to reproduce.

**Step 0 — Premise check (do first, ~free).**
The whole idea assumes the "weapon" concept actually points the same way as refusal, at the injection
layer. That's never been measured for concept vectors. It's inference-only, runs on a small 4B model,
costs nothing. If it comes back negative, that's itself a publishable finding — and it avoids running
an expensive experiment on a broken premise.

**Step 1 — Reproduce the baseline (hard gate).**
Re-run the original experiment and hit their ~38% detection number. If it doesn't reproduce, every
downstream number is uninterpretable. Nothing expensive happens until this passes.

**Step 2 — Build four word lists ("arms").**
This is the scientific core. "Harmful" carries confounds, so controls are needed:

| Arm | Example | Rules out |
|---|---|---|
| Neutral | bread | (baseline) |
| Negative but harmless | grief | "is it just unpleasant?" |
| Harm-adjacent but harmless | surgery | "is it just near a scary topic?" |
| Harmful | weapon | the hypothesis |

Without the two control arms, a drop in the harmful arm is uninterpretable.

**Step 3 — The main test.**
Inject each word, ask the model if it noticed, measure detection rate per arm. Also measure how often
it *refuses to answer* — that's data, not noise (a refusal to engage is the mechanism acting, not a
failure to detect).

**Step 4 — The causal test.**
Turn off refusal (abliteration), re-run. If refusal is the cause, the harmful-arm blind spot should
close. Plus a mandatory control: also try deleting a *random* direction. If that closes the gap too,
the experiment has only shown "damaging the model changes things," not "refusal is the cause." Two
papers disagree on how this control comes out — resolving that is part of the contribution.

**The headline chart:** detection rate vs. how refusal-aligned each concept is. If the mechanism is
real, the line slopes down — more refusal-aligned → less noticed. Abliteration should flatten it.

---

## Why every outcome is a result

- Blind spot is real and only in the harmful arm → self-report has a safety-shaped hole.
- Drop shows up in the "negative" or "harm-adjacent" arm too → it's valence or topic, not harm.
- No blind spot → confirms a competing published claim (that introspection is content-independent) on
  the one dimension most likely to break it.
- The refusal-vs-random control resolves a live disagreement between two papers either way.

There's no version of this that produces nothing.

---

## Bibliography — the ones worth naming

**The two the project stands on:**

- **Lindsey 2025, "Emergent Introspective Awareness" (Anthropic)** — first showed models can notice
  injected concepts.
- **Macar et al. 2026, "Mechanisms of Introspective Awareness" (Anthropic, arXiv:2603.21396)** — the
  paper this project extends. Gives the method, the test model, the 38% baseline, *and* the key
  finding that refusal training suppresses detection. It explicitly excluded harmful concepts — that
  exclusion is the gap this project fills.

**The mechanism, from adjacent angles:**

- **Arditi et al. 2024, "Refusal Is Mediated by a Single Direction" (NeurIPS)** — refusal lives in one
  direction; remove it to uncensor. Source of the refusal direction and abliteration.
- **Li et al. 2026, "Safety Pitfalls of Steering Vectors" (TU Munich, arXiv:2603.24543)** — steering
  vectors partly align with the refusal direction. The other half of the prediction.
- **Nguyen et al. 2026, "Self-Report of Adversarial Prefills" (KAIST, arXiv:2606.23671)** — closest
  prior work. Same refusal-suppression effect, but for *text* the model wrote, not activation injection.
  Shows the mechanism is real in a neighboring setting.

**The claim the project tests against:**

- **Lederman & Mahowald 2026, "Introspection Is Content-Agnostic" (arXiv:2603.05414)** — argues
  detection does *not* depend on content. The headline result is a direct test of this.
- **Gondil 2026, "Do LLMs Know When They'll Refuse?" (arXiv:2604.00228)** — notes weapons queries are
  hardest for introspection; source of the "measure refusal, don't assume harm labels" idea.

**Why the premise check can genuinely fail (harm ≠ refusal geometry):**

- **HARC 2026 (arXiv:2607.00572)** and **Llorente-Saguer 2026 (arXiv:2604.18901)** — harm and refusal
  are related but distinct directions that drift apart in deeper layers, and the recovered direction
  depends on how it is extracted. This is why Step 0 is a real test, not a formality.

---

## Responsible-use, in one breath

The project injects concept *words*, not capabilities — a "weapon" direction teaches the model nothing
it didn't already know. The measured output is a detection *rate*, which is safe to share. The genuinely
sensitive artifacts — the uncensored model and the concept vectors — stay on a controlled machine and are
never published. A fallback design tests the same mechanism with zero harmful concepts if ethics review
prefers it.
