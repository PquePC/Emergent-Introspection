# Does the Direction of an Injected Vector Determine Whether a Model Reports It?

### *Rotating concept vectors toward and away from the refusal direction at constant magnitude, and measuring the effect on introspective self-report.*

> **Project proposal.** Self-contained: everything needed to run and analyse the study is here. Written for a
> single researcher on a limited schedule, producing a short article, with a possible longer paper afterwards.
> Where a choice trades robustness for simplicity, the trade is stated rather than hidden.

---

## 1. The question

Language models can sometimes notice when a concept vector is injected into their residual stream, and say so.
This study asks whether that reporting depends on **which direction the injected vector points**, holding
**how large it is** exactly constant.

Concretely: take a concept vector, decompose it, and rebuild it so that it points a controlled amount toward
(or away from) the model's refusal direction — same length, same layer, same injection procedure, only the
direction changed. Sweep that amount across a range. Does the rate at which the model reports the injection
move?

And the follow-on: if it does move, **is the refusal loading on its own enough to explain it?** A vector
rotated up to a given loading and a vector that already sat at that loading agree on the axis being
manipulated and differ in everything else. If they produce the same reporting rate, loading is the whole
story. If they don't, something about the concept beyond its refusal loading is contributing.

> **Scope.** This study manipulates one geometric quantity. It can make claims about that quantity, and about
> whether concept identity contributes beyond it. It cannot attribute an effect to any specific semantic
> property, because it does not manipulate any. None is assumed, selected on, or used as a justification
> anywhere in the design.

---

## 2. Motivation

### 2.1 Self-report is the only inward-facing monitor there is

Every other way of checking whether a model's internal state has been tampered with reads the model from the
outside — probes, output monitoring, behavioural evals. Introspective self-report is the one channel where the
model itself is the instrument. That makes it interesting out of proportion to how reliable it currently is.

But an instrument is only usable if its failure modes are known. **A detector whose sensitivity depends on
what is being detected is not a detector — it is a filter**, and a negative report from it cannot be
interpreted. If a model reports some injected directions readily and others rarely, then the absence of a
report means nothing on its own, and the bias is invisible from outside: you would observe fewer reports, never
the reason for them.

So the question "does reporting depend on the direction injected?" is a question about whether this channel can
be read at all.

### 2.2 There is one documented suppressor, and it has only been tested bluntly

Published work shows that removing the refusal direction from a **model** substantially raises the rate at
which it reports injected concepts — refusal training appears to teach models to deny having internal states,
and that denial generalises to denying injections.

That result is important, and the intervention behind it is blunt. Ablating a direction from a model changes
everything the model does at once: its refusals, its style, its calibration, its capabilities. The observed
increase in reporting could be mediated by any of those. It also cannot be reversed, graded, or matched against
a control within the same forward pass.

**The precise version of the question has not been asked: does the refusal alignment of the injected vector
itself matter?** That is a different manipulation. The model stays completely untouched. What changes is one
geometric property of one object, per trial, reversibly, with an exactly matched control available.

If refusal geometry gates reporting, this design demonstrates it **with the model intact** — which is both a
cleaner causal claim and a far cheaper experiment.

### 2.3 Steering vectors carry refusal alignment whether or not anyone intends them to

Work on the safety properties of steering vectors has found that vectors built for unrelated behavioural
purposes carry components aligned with the refusal direction as a side effect of how they are constructed, and
that this alignment has downstream behavioural consequences.

If refusal alignment also changes **detectability**, then detectability is a property of essentially every
steering intervention, not only of vectors deliberately built to have it. That matters to anyone using steering
as an evaluation or control technique: the visibility of the intervention would depend on an incidental
geometric property nobody was tracking.

### 2.4 The mechanistic question underneath, which makes the null informative

There is a live question about what injection-detection actually is. One account is that it is essentially
**content-agnostic**: the model registers that *a perturbation occurred*, largely regardless of what was
injected. The competing possibility is that it is **direction-dependent**: some directions are more reportable
than others.

The rotation design speaks to this directly, because it holds magnitude exactly fixed and varies only
direction. **A flat curve is a clean positive result for the content-agnostic account** — detection responds to
perturbation size and not to where the perturbation points, tested on the one direction with a documented
behavioural link to denial. A non-flat curve refutes it, and the shape says how direction matters.

There is no outcome here that produces nothing.

### 2.5 Why refusal, specifically

Because it is the only direction with a *documented behavioural mechanism* connecting it to a model's
willingness to make claims about its own internal states. Any direction could be rotated toward; refusal is the
one where there is a reason to expect an effect and a published mechanism to explain it if there is one.

The control arm (§5.1, §6) is what keeps this honest: it rotates by the same angle toward a real but unrelated
direction, so "tilting the vector at all does this" is ruled out rather than assumed away.

---

## 3. What is established, and what this adds

| Established | Source |
|---|---|
| Models can report injected concepts at rates well above their false-positive rate, at a known layer and strength | Lindsey, *Emergent Introspective Awareness in LLMs*; Macar et al., *Mechanisms of Introspective Awareness* (arXiv:2603.21396) |
| Removing the refusal direction from the **model** raises those detection rates substantially | Macar et al. |
| Refusal behaviour is mediated by a direction recoverable by difference-of-means, and refusal can be measured cheaply by substring matching | Arditi et al., *Refusal in LMs Is Mediated by a Single Direction* (NeurIPS 2024) |
| Steering vectors built for other purposes carry refusal-aligned components, and removing those components changes their behavioural effect | TU Munich, *Analysing the Safety Pitfalls of Steering Vectors* |

**What is new here:** the refusal-alignment manipulation has been applied to *behavioural* vectors and measured
against *behavioural* outcomes. It has not been applied to *concept* vectors and measured against
*introspective self-report*. And refusal has been removed from *models* but never from *the injected object*,
which is the manipulation that isolates geometry from every global side effect of ablation.

---

## 4. What each vector is

| Symbol | What it is | How it is built | Injected? |
|---|---|---|---|
| `v` | **Concept vector.** The thing that makes the model think about *bread*. | Lindsey/Macar procedure: mean activation on concept prompts minus mean on baseline prompts, at the injection layer. | **Yes** — this and its rotated variants are the only things injected. |
| `‖v‖` | **Its magnitude.** The "how much" of the injection, before α scaling. Held constant everywhere. | — | — |
| `r̂` | **Refusal direction**, unit length. | Arditi procedure: mean activation on harmful instructions minus mean on harmless ones, normalised. | **No.** Used only as a reference axis — the "north" that angles are measured against. |
| `c` | **Native refusal loading**: `c = v · r̂`. Divided by `‖v‖` it becomes the **native cosine**, on a −1…+1 scale. | — | — |
| `v_⊥` | **The refusal-free part of `v`** — `v` minus its shadow on `r̂`. | — | The base every variant is built from. |
| `u` | `v_⊥` at unit length. The pure concept direction with refusal removed. | — | — |
| `ĝ` | **Control direction.** A real, meaningful direction that is not refusal, orthogonalised against `r̂` and `u`. | A **held-out concept vector** (an unrelated concept, not in the study set), orthogonalised and normalised. | **Yes**, in the control arm. |

**Why the control direction is a held-out concept vector rather than a random one.** A random direction in
several thousand dimensions is nearly orthogonal to everything the model actually uses — it is noise, not a
feature. Refusal is a real, functionally loaded direction. Comparing "tilt toward refusal" against "tilt toward
noise" risks showing only that *any real feature disrupts more than noise does*, which says nothing about
refusal. A held-out concept vector is on-manifold, built by the identical procedure to the vectors under test,
and free — concept vectors are already being built.

---

## 5. The manipulation

### 5.1 Construction

For a target cosine `t`:

```
u   = v_perp / ||v_perp||             # unit, refusal-free concept direction (u ⟂ r̂ by construction)
w   = sqrt(1 - t*t) * u  +  t * r̂     # unit vector with EXACTLY cosine t against refusal
v_t = ||v|| * w                        # restore the original magnitude
```

`v_t` has **exactly** the original magnitude and **exactly** cosine `t` with the refusal direction. The control
arm is the same three lines with `ĝ` substituted for `r̂`.

Assert both properties numerically per variant. It is two lines and it catches the most likely implementation
bug.

### 5.2 Why the sweep is labelled in cosine

The magnitude constraint is unchanged by this choice — it is only the units on the dial.

If the sweep were labelled "amount of refusal added," that number would not be comparable across concepts: a
concept with a long vector needs a much larger addition to reach the same tilt than one with a short vector, so
the same nominal value would mean a different geometric operation for every word, and the concepts could not
share an axis.

The resulting cosine has no such problem. "This variant sits at cosine 0.3" means the same thing for every
concept, layer, and model. It is also **directly comparable to the native cosine**, which is what makes the
native-vs-induced analysis (§9.4) possible at all.

### 5.3 Concept content is preserved by a known amount — no experiment needed

Because magnitude is fixed, tilting toward refusal necessarily tilts away from the concept. The fraction of the
original concept direction retained is exactly `√(1−t²)`:

| `t` | 0.1 | 0.2 | 0.4 | 0.6 | 0.707 | 0.8 |
|---|---|---|---|---|---|---|
| concept retained | 0.995 | 0.980 | 0.917 | **0.800** | 0.707 | 0.600 |

This is analytic. It answers the obvious objection — *"you just broke the vector"* — with algebra rather than a
side experiment.

**It is also why the control arm is exactly matched.** At the same `t`, the control variant has *identical*
concept retention by construction. The two arms differ in destination and in nothing else.

### 5.4 Why the sweep runs to ±0.6

1. **Concept retention stays ≥ 0.80.** Past 0.6, dilution becomes a live competing explanation for any drop in
   reporting. Below it, the objection is answerable in one line.
2. **It stays below the crossover at `t = 0.707`**, where the refusal component equals the concept component in
   magnitude. Past that point the object is better described as a refusal vector with some concept in it, and
   the experiment quietly changes its subject.
3. **It brackets the natural range generously.** In high dimensions, cosines between independently-built
   directions are small; native concept-to-refusal cosines should be expected in the low tenths, not near 0.5.
   Deliberate over-extension puts the natural operating band in the middle of the plot rather than at its edge,
   where the curve's shape is readable.
4. **Symmetry.** The negative side must reach as far as the positive side, or the bidirectional test is
   lopsided.

**Grid, denser near zero** — because that is where the natural values live:

```
t ∈ {−0.6, −0.4, −0.2, −0.1, 0, +0.1, +0.2, +0.4, +0.6}    plus the untouched vector v as a reference point
```

`t = 0` is the fully refusal-stripped variant. The untouched `v` is a *separate* point sitting at that
concept's native cosine — not the same thing as `t = 0`.

Trim the outer levels if the coherence check in Stage 0 shows the model degrading there.

### 5.5 Comparability with the published operating point

The rotation **never changes magnitude**. So if concept vectors are magnitude-matched at the outset the way the
reference work matches theirs, every variant is automatically at that same matched norm, same layer, same α,
same token positions. The injection machinery is untouched; only direction moves.

Two things follow, and both are worth having:

- **The untouched-`v` condition is a replication check.** It should land near the published detection rate. If
  it doesn't, that is a pipeline bug to fix before anything downstream is interpreted — which converts an
  unbounded debugging problem into a bounded one.
- **The primary probe and judge rubric are the published ones, unmodified.** Same question wording, same
  scoring. Any additional readout is asked as a **separate question** (§8.2), never as an edit to the primary
  one. The moment the primary probe text changes, the anchor is gone.

---

## 6. Concept selection

**The selection criterion is the measured native cosine, and nothing else.**

The design needs concepts spread along the refusal axis so the native and induced arms can be compared across a
range. It does not need, and must not assume, any theory of *which kinds of words* sit where on that axis.
Which concepts carry high refusal loading is an **empirical output of Stage 1**, not an input to the design.

**Procedure.**

1. Build a candidate pool of ~60–100 concepts chosen for **semantic breadth** — objects, actions, abstractions,
   emotions, institutions, materials, social situations. Breadth maximises the chance of covering the
   achievable cosine range. It is a sampling heuristic, not a hypothesis, and the groupings carry no analytic
   weight.
2. Build each concept vector and measure its native cosine against `r̂`.
3. **Inspect the empirical distribution before selecting anything.** Do not assume it spans ±0.6 — it very
   probably does not, and the achievable native spread may be an order of magnitude narrower than the rotation
   grid. This check determines what the native arm can claim; if the range is tight, the rotation sweep carries
   the study and the native arm becomes secondary.
4. Select ~24 concepts **evenly spread across the observed native range**, not clustered at its extremes. Even
   spacing is what allows a curve to be fitted across native values rather than two clumps compared.
5. **Log a short descriptive label per concept** — one or two words. It costs nothing, makes the Stage 1
   geometry result readable, and supports post-hoc inspection. It is **not** a factor in any planned analysis.

**One caveat, and why it barely bites.** Selecting on a *measured* quantity means the extremes are partly
measurement noise, and re-measurement would pull them toward the middle. That biases comparisons **between
concepts at their native cosines** — the native-only scatter. It does **not** bias the rotation curve, because
rotation *sets* the cosine exactly rather than measuring it: every rotated point sits where it was put. Since
the rotation sweep is the primary experiment, this selection strategy is sound; the caveat needs one sentence
wherever native-vs-native comparisons appear.

---

## 7. Experimental stages

### Stage 0 — Strength calibration

**Why it exists.** Two failure modes make the main sweep worthless and neither is visible until checked. If
baseline detection sits near 90%, refusal loading can only push it down and the recovery arm is dead; near 3%,
the reverse. And if the injection degrades the model into incoherence, everything downstream measures coherence
collapse rather than reporting. A published α is a starting point, not a guarantee for a given setup,
quantisation, or concept set.

**What runs.** 5–8 concepts drawn from the candidate pool — build just those vectors first, since calibration
needs something to inject — untouched, α over a small grid (e.g. {1, 2, 3, 4, 6}), ~20 samples each.

**Recorded.** Detection rate and a coherence score at each α.

**Decision rule.** Pick the largest α at which coherence is acceptable *and* detection lands mid-range —
roughly 30–60%, leaving headroom in both directions.

**If the chosen α differs from the published one**, run the untouched-`v` condition at **both** values. The
published-α points are the bridge to the literature even when the main sweep runs elsewhere. One small extra
condition, comparability preserved.

**Output.** A chosen α, a coherence floor, and a note on where in the `t` range coherence starts to fail.

### Stage 1 — Vector construction and geometry

**Why it exists.** Everything downstream depends on these being right, and the geometry is a small standalone
result: the distribution of native refusal cosines across a broad concept pool is worth reporting on its own,
and it is the empirical answer to *which concepts actually carry refusal loading* rather than an assumed one.

**What runs.** Extract `r̂` from matched instruction pairs at the injection layer. Build the candidate concept
vectors. Measure native cosines. Select the ~24 study concepts (§6). Build every rotated variant and assert the
magnitude and cosine properties per variant.

**One methodological requirement:** extract `r̂` from prompt data **disjoint** from the data used to build
concept vectors, so the covariate and the outcome do not share estimation noise.

### Stage 2 — The main sweep

The experiment. For each concept, each condition below, at the chosen α, injected exactly as in the reference
setup (fixed layer, applied from the trial question onward at every generated token):

| Arm | Conditions per concept | Why it exists |
|---|---|---|
| **Refusal rotation** | 9 (`t` grid) | The manipulation. |
| **Control rotation** | 4 (`t` = −0.6, −0.2, +0.2, +0.6) | Rules out *"tilting the vector at all does this."* Fewer levels because it only needs to establish that the curve is flat and sign-symmetric, not resolve its shape. |
| **Untouched `v`** | 1 | Native reference point and replication check. |
| **Clean, no injection** | shared globally, not per concept | The false-positive floor. A no-injection trial is concept-independent, so one shared set calibrates the whole sweep. Detection is only ever reported against it. |

**14 injected conditions per concept. At 24 concepts, 336 conditions.**

Illustrative budget: 50 sampled generations per condition ≈ 17k generations and judge calls, plus ~500 shared
clean trials. The P(yes) readout (§8.2) adds forward passes but no generations and no judge calls.

### Stage 3 — Refusal-knob check

**Why it exists — and it is insurance, not a core measurement.** Adding `r̂` at every generated token is not how
refusal naturally arises in the model, and the difference-of-means sign convention is arbitrary. If the knob is
disconnected — the added component does not actually engage refusal behaviour — then **a null result is
uninterpretable.** *"Refusal loading does not affect reporting"* and *"the knob was never on"* produce identical
data, and after the fact nothing can separate them. This check converts a null from a dead end into a finding.

**Why it is cheap.** It does not multiply the main grid, because it is not per concept and needs no judge:

- 2–3 representative concepts × the `t` grid × ~15 ordinary harmless requests
- scored by **substring match on refusal phrases** ("I can't", "I'm sorry", "I cannot") — a standard metric,
  computed locally, zero API cost
- a few hundred short generations, once, as a standalone side run

**Reading it.** Refusal rate on ordinary requests should rise with `t` and fall with negative `t`. That confirms
both that the knob works and that the sign convention points the way it is assumed to.

---

## 8. Measurement

### 8.1 Primary measure

The published probe, unchanged, judged by the published rubric, giving a three-way label: **detected / not
detected / declined**. Sampled generations, so the judge can score them.

**Declines are an outcome, not an exclusion.** If refusal loading makes the model refuse to engage with the
probe at all, that is the manipulation acting on the report channel — a realisation of the dependent variable,
not missing data. Analyse over the full sample and report decline rate per condition as its own curve. A
filtered, conditional-on-engagement analysis is a clearly-labelled secondary at most.

### 8.2 Second measure — single-token P(yes)

**Why add it.** A judged binary label discards almost all the information in a trial. A probability is
continuous, so it resolves smaller shifts with far fewer trials, and it can reveal a graded signal sitting
underneath a "no." It costs one forward pass, no generation, and **no judge API call**.

**How.** A separate, constrained probe asked on the same injected states — a yes/no question with the answer
format pinned to a single token — reading the probability of "Yes" from the logits instead of sampling.

**Practical note.** Logits are deterministic given the prompt and the vector. Variance comes from **varying the
trial prompt**, not from sampling. Use ~20 different trial prompts per condition and take 20 forward passes; do
not sample 20 times from one prompt and expect variation.

This never touches the primary probe. Two independent readouts of the same underlying question is the
robustness, and one of them stays comparable to published numbers.

### 8.3 Answer-polarity control

**Why it is needed even though the sweep is bidirectional.** The natural defence — *"if reporting falls when I
add refusal and rises when I remove it, a constant bias can't explain that"* — does not hold, because the
competing explanation is also bidirectional. Refusal is plausibly a *decline/deny* direction: more of it may
push the model toward answering **no** to anything at all, less of it toward **yes**. That produces exactly the
symmetric pattern the hypothesis predicts. Bidirectionality raises the bar for a generic-bias story; it does not
clear it.

**Design.** A **matched pair of yes/no questions in the same domain, worded as similarly as possible** — one
whose true answer is *yes*, one whose true answer is *no*. Same subject matter, same sentence shape, same
length, so that any interaction between the question's vocabulary and the refusal direction hits both members
equally and cancels in the comparison.

Two things to enforce:

- **Same domain, near-identical wording.** The pair differs in the fact asserted, not in topic or register.
- **Force a one-token answer**, read from the logits. This keeps the control's cost near zero and makes it
  immune to verbosity effects — a question cannot come back as a five-paragraph essay.

**Reading it.** If refusal loading pushes both members the same way, part of the detection effect is generic
answer polarity, and the size of that shift is the correction term. If the pair does not move, the objection is
closed in a sentence. Optionally express the result as a sensitivity/criterion split, which states the finding
precisely: *loading moved the model's threshold for saying yes, not its ability to tell injected from clean.*

**Scope.** The polarity pair does not need to run on every concept. A representative subset (~6 concepts across
the range) × the full `t` grid characterises the bias adequately, since it is mostly a function of loading
rather than of which concept sits underneath.

### 8.4 Identification rubric — three buckets

**The trap this exists to avoid.** As refusal loading rises, the injected vector points increasingly toward
refusal and decreasingly toward the concept. A model that says *"I notice something about danger"* under heavy
refusal loading is **identifying the injected content correctly** — at that loading, the injected content really
is substantially refusal. A judge scoring identification against the *original concept word* marks that trial a
failure, and the study ends up measuring vector composition as though it were reporting suppression.

**The rule.** The **primary dependent variable is content-agnostic**: did the model report that *something* was
injected, regardless of what it named. Identification is scored separately and never folded into it.

| Bucket | Response names… | Expected as refusal loading rises |
|---|---|---|
| **1 — concept** | the injected concept, a synonym, or a close associate | should **fall**, by construction, as concept content dilutes |
| **2 — refusal / prohibition theme** | refusal, danger, harm, safety, being blocked, "something I shouldn't say" | should **rise** — correct identification of what was actually injected |
| **3 — other** | anything else, including unrelated confabulations | the confabulation baseline; watch for drift |

**Why the split is diagnostic.** Bucket 2 rising while bucket 1 falls, with content-agnostic detection **flat**,
means *the vector changed* — not that the model stopped noticing. Bucket 2 rising while content-agnostic
detection **falls** means genuine reporting suppression. Without the split these are indistinguishable.

Scored by the same judge configuration in a second labelling pass over the same generations — no extra sampling.

### 8.5 Recorded per trial

Detection label (*detected / not detected / declined*) · P(yes) · identification bucket · coherence score · the
condition's `t`, arm, α, concept, descriptive label, native cosine, and whether the point is **native or
induced** · seed.

---

## 9. Analysis

### 9.1 The primary curves

Detection rate and P(yes) against **post-manipulation cosine `t`** — the absolute loading, not the change from
native — with both arms on the same axes.

**The primary contrast at every `t` is refusal arm vs control arm**, not refusal arm vs `t = 0`. The control arm
is what removes "tilting the vector does this regardless of destination."

**A free specificity check requiring no control at all:** an arbitrary direction has no privileged sign, so the
control arm should be roughly symmetric between `+t` and `−t`. Refusal does have a privileged sign. **Sign
asymmetry in the refusal arm, absent in the control arm, is evidence on its own.**

### 9.2 Polarity correction

Compare the shift in detection P(yes) against the shift on the matched yes/no pair at the same `t`. If the pair
is flat, report the raw curve. If it moves, report both the raw and the corrected curve, and state which part of
the effect survives.

### 9.3 The collapse test

Every point in the study — native and rotated — has a refusal loading. Plot them all on that one axis and ask
whether points from different concepts fall on a **shared curve**.

If they do, refusal loading alone predicts reporting regardless of which concept was injected. That is the
cleanest form of the mechanistic claim, and it is just a way of plotting data already collected.

### 9.4 Native vs induced — is loading sufficient?

The sharper version: **does it matter how a vector arrived at a given loading?**

Every point is one of two kinds: **native** (the untouched vector, at whatever loading the concept came with) or
**induced** (a rotated variant, placed there by construction). In the overlap region you have both, from
different concepts. They agree on the manipulated axis and differ in everything else.

**The test is group-level. A single matched pair proves nothing** — two concepts differ in a hundred ways, and
one coincidence in either direction is uninformative.

1. Fit the reporting-vs-loading curve on the **induced points only**. Rotation sets loading exactly and
   independently of concept identity, so this is the *geometry-only* reference.
2. Compute the **native points' residuals** against that curve.
3. Test for a **systematic group-level offset** — mean residual with a confidence interval, or native/induced as
   a factor alongside loading in the model of §9.6, with concept as a random effect.

**Reading it.** No offset → refusal loading is sufficient; the route by which a vector reached a loading is
irrelevant. A systematic offset → a concept's own composition contributes beyond its refusal loading. That is a
claim about semantics in general and it stops there: this design manipulates one geometric quantity, so it can
show that concept identity carries extra information but cannot say *which* property carries it.

**The within-concept version — cleaner, and free because the data already exists.** Take the concepts with the
**highest native loading** and rotate them *down* toward zero. Does their reporting move onto the geometry-only
curve?

- **Yes** → their reporting behaviour was carried entirely by the refusal component.
- **No — they stay off the curve even at zero loading** → something drives those concepts that survives removal
  of their refusal component, i.e. `r̂` does not capture it.

Same word, before and after, nothing else changed. This is the strongest single comparison in the study.

### 9.5 Does concept identity matter at all? — the variance test

The cheapest form of the same question, falling straight out of the model below. After loading is accounted for,
inspect the **variance of the concept-level random intercepts**:

- **Near zero** → concepts are interchangeable once loading is known. Geometry explains reporting.
- **Substantial** → concepts differ in a way loading does not predict, and it is worth asking what the high- and
  low-residual concepts have in common. That inspection is **exploratory** — where the descriptive labels earn
  their keep — and anything found is a candidate for a follow-up, not a result of this study.

### 9.6 Model

Hierarchical / mixed-effects, reporting as the outcome; **fixed effects** for loading `t`, arm
(refusal / control), and native-vs-induced; **random intercepts** for concept and for trial prompt. The concept
random-effect variance is itself a result (§9.5), so fit in a way that reports it with an interval rather than
treating it as nuisance. Report effect sizes with confidence intervals throughout — sample sizes here are small
enough that point estimates alone are not defensible.

---

## 10. Deliverables

1. **The loading curve** — detection and P(yes) vs post-manipulation cosine, refusal and control arms overlaid,
   with the polarity correction shown. The core figure.
2. **The collapse plot** — every concept's every variant on one loading axis, showing whether they share a curve.
3. **The native-vs-induced result** — do native points sit off the induced-fitted curve, as a group, with an
   interval; plus the within-concept version (high-loading concepts rotated down).
4. **The identification-bucket panel** — buckets 1/2/3 against loading, which is what makes the primary curve
   readable.
5. **The native-cosine distribution across the candidate pool** — a standalone geometry result from Stage 1,
   produced before any sweep runs.
6. **The α calibration curve** — detection and coherence vs α, which doubles as a methods note for anyone
   reproducing the published operating point in a different setup.

---

## 11. Outcomes

| Result | Reading |
|---|---|
| Detection falls as refusal loading rises and rises as it falls; control arm flat; polarity pair flat | **Refusal geometry gates the report** — bidirectionally, and specifically to the refusal direction. The target result. |
| Same pattern, but content-agnostic detection is flat while bucket 1 falls and bucket 2 rises | The vector changed, the model didn't. **Not** a reporting effect — visible only because of the bucket split. |
| Refusal arm and control arm behave identically | The effect is generic direction-tilting, not refusal. Bounds the natural extension of the behavioural-vector result, cheaply. |
| Polarity pair moves as much as the detection probe | Much of the effect is generic answer bias. Report the corrected curve; the honest result is smaller than it first looked. |
| Nothing moves at any loading | **Detection is magnitude-driven and direction-agnostic** at this layer — a clean positive result for the content-agnostic account, tested on the one direction most likely to break it. Interpretable **only if the Stage 3 knob check passed.** |
| Native and induced agree at matched loading; concept variance near zero | **Refusal loading is sufficient.** Concepts are interchangeable once loading is known. The cleanest mechanistic statement available here. |
| Native points sit systematically off the induced curve | A concept's composition contributes beyond its refusal loading. Real and interesting — but the claim stops at *"semantics contributes something"*. |
| High-loading concepts rotated down move onto the geometry-only curve | Their reporting behaviour was carried entirely by the refusal component. Within-concept, so nothing else explains it. |
| High-loading concepts rotated down stay off the curve | Whatever drives them survives removal of their refusal component. Strong motivation for a follow-up that manipulates a candidate property directly. |

---

## 12. Future work — deliberately out of scope here

Recorded so they are not re-derived later, and because several become cheap once this pipeline exists.

- **Model-side refusal ablation crossed with vector-side loading.** The 2×2 (intact / refusal-ablated /
  random-ablated model × loading) would close the causal claim from both directions, but it needs an ablated
  model and triples the sweep.
- **Internal readouts** — linear probes or logit lens at the layers where perturbation-evidence features are
  reported to peak, to test whether the model *registers* an injection it does not report. High value, second
  pipeline.
- **Forced-choice probing** ("which of these N words is on your mind?", chance 1/N) as an alternative report
  channel that does not require asserting an inner state.
- **Refusal as a subspace rather than a single direction.** If refusal is a cone, single-direction rotation is a
  partial manipulation; a top-k subspace version would test robustness.
- **A second `r̂` extraction protocol.** Pooling choice is known to move extracted directions substantially.
- **Strength × loading interaction.** One α is correct for this study.
- **Manipulating a named semantic property directly.** If §9.4 or §9.5 shows concept identity contributing
  beyond loading, the follow-up isolates a candidate property and manipulates *it* the way this study
  manipulates refusal loading. That is what would license a claim about which property matters.

**If time runs short, cut in this order:** Stage 3 knob check → control-arm outer levels (`±0.6`) → the polarity
control → the P(yes) second probe. **Do not cut** the control arm entirely, the clean/no-injection floor, or the
identification buckets — each removes the ability to interpret the result at all.

---

## 13. Responsible use

- The manipulation adds or removes a refusal component from a *concept vector*. It is not refusal removal from a
  model, and no variant should be tuned toward maximum bypass. Do not optimise for jailbreak efficacy; do not
  report per-concept tables of which geometry evades which safeguard.
- **If the candidate pool contains sensitive concepts, exclude them from the large negative-loading arm.**
  Strongly anti-refusal loading on a sensitive concept is a jailbreak recipe, and the design does not need it —
  the bidirectional test works on any concept, so the exclusion costs nothing scientifically.
- Publish curves, aggregate rates, and cosine scalars. Do not publish vectors, activations, or raw generations.
