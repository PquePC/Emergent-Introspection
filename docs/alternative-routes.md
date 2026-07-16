# Alternative Research Routes

Two routes were considered alongside the main plan. Both remain viable; each is documented with the
reasoning for and against, and the condition under which it becomes the better choice.

> Main plan: [`project-plan.md`](project-plan.md) · Evidence: [`background.md`](background.md)

---

## Route B — The Stealth-Efficacy Frontier

### *Can a projected steering vector still steer?*

**The question.** Take a semantically meaningful steering vector `v` (e.g. one that induces pirate-speech).
Project it onto the orthogonal complement of the detection subspace `W_k`, restore its norm, and steer with
the result. Does it evade detection **while still steering**?

Formally, the two hypotheses:

**H1 — subspace projection eliminates detection.**
```
P_{W⊥} = I − W Wᵀ
v_⊥ = P_{W⊥} v = v − Σᵢ (v · wᵢ) wᵢ
```
Each component is subtracted **independently**. (Averaging the `wᵢ` into a single direction and projecting
orthogonal to that is *incorrect* — it removes only the mean component and leaves `v` with components
inside `span(W)` along the other `k−1` dimensions, which a subspace detector still reads.)

**H2 — norm-preserving calibration recovers semantics.**
```
v_stealth = ‖v‖ · v_⊥ / ‖v_⊥‖
```

### Why this route is not the main plan

**Two independent sources predict it fails.**

1. **The angular objection.** Aparin & Gaintseva (2026) find that *"concepts are represented primarily in
   angular structure"*, with norm mattering for *"stability and downstream effects of steering."* Projection
   **changes the angle** of `v`. Renormalising restores **magnitude, not angle**. Semantic content removed
   by projection cannot be rescaled back — renormalisation makes a wrong-angled vector larger, not more
   correct. **H2 may be unrecoverable in principle.**

2. **The published failure.** *Obfuscated Activations Bypass LLM Latent-Space Defenses* reports, as an
   explicit limitation: *"**We are unable to achieve near-orthogonality while preserving behavior**."* That
   is precisely the H1 ∧ H2 conjunction. They also document the tradeoff directly: *"When obfuscating
   against probes that detect SQL code, the model still generates SQL, but its code is incorrect more often
   than without obfuscation."*

**The cross-layer problem.** Injection happens mid-stack; detection is read late. Fonseca Rivera & Africa
show the network **actively rotates** injected vectors into the detection direction:
> *"cos(Δc(ℓ), d̂) rises monotonically from ∼0.2 at injection layer to ∼0.75 at layer 41"*

Orthogonality at the injection layer need not survive to the readout layer. Worse: if `v_⊥` genuinely
steers, late-layer activations shift toward the target concept — so the detector may read the **downstream
consequence** of steering rather than the injected vector. The needle can be hidden; the behaviour change
it causes cannot be, so easily.

**The dependency.** This route's value is entirely conditional on the main plan's answer. If emergent
detection turns out to be anomaly-based, projection-based stealth is dead before it starts. **The main plan
tells us whether this route is worth attempting at all.**

### If pursued — the decisive early gate

Before any other work, compute:
```
overlap = ‖P_W v‖ / ‖v‖      # fraction of v's norm lying INSIDE the detection subspace
```

| overlap | Interpretation | Action |
|---|---|---|
| **< 5%** | `v` is already near-orthogonal to `W_k`. Projection changes almost nothing — and so changes detection almost nothing. | **Stop.** Uninformative. |
| **10–40%** | Projection meaningfully alters both detection and semantics. | **Proceed.** |
| **> 50%** | Projection will destroy the steering. | Proceed — but expect H2 to fail. *That is still a result.* |

This single number may determine the route's viability. It is cheap and must come first.

### Method sketch

1. Build and validate a CAA semantic vector `v`; fix `α` at a value that steers without garbling output.
2. **Run the overlap gate above.**
3. **Discover `W_k` via INLP**, not PCA. PCA finds maximum-*variance* directions, which are not
   necessarily detection-*relevant* — the top principal component may capture the steering effect itself or
   generic activation structure. INLP finds directions that are each independently *predictive of detection*,
   and discovers `k` rather than requiring it to be guessed:
   ```python
   W = []
   while True:
       probe = LogisticRegression().fit(acts, labels)
       if roc_auc(probe, acts, labels) < 0.55: break     # chance → stop
       w = normalize(probe.coef_[0]); W.append(w)
       acts = acts - np.outer(acts @ w, w)               # project onto nullspace of w
   k = len(W)
   ```
4. Construct `v_stealth`; verify orthogonality (`v_stealth · wᵢ ≈ 0`).
5. **Cross-layer check:** inject `v_stealth`, measure whether the readout-layer detection signal actually
   drops versus standard `v`. **If it does not, static projection is dead** — proceed to the optimisation
   fallback below.
6. Sweep `k` and `α`; measure steering efficacy and detection jointly.

### Deliverable

- **The 2×2:** {steered ✓/✗} × {detected ✓/✗}. **Does a "steered but undetected" region exist?**
- **The frontier curve:** efficacy vs detection across `k` and `α`.
- **The H2 test:** does norm-restoration recover efficacy relative to naive projection? Theory says no.

If the result is a strict tradeoff, that independently confirms the Obfuscated Activations limitation
**for steering vectors** — they tested only token/embedding suffixes. That is a legitimate contribution
even as a negative.

### Fallback — gradient-optimised stealth vector

Static projection assumes orthogonality survives the forward pass. Instead, optimise directly for the
objective:

```python
v_stealth = torch.nn.Parameter(v.clone())      # ONE vector, ~5376 floats
opt = torch.optim.Adam([v_stealth], lr=1e-3)
for step in range(300):
    out = forward_with_injection(model, v_stealth, layer=37)   # model FROZEN
    loss = L_steer(out) + lam * L_detect(out)
    loss.backward(); opt.step(); opt.zero_grad()
```

**The model's weights never change — this is not fine-tuning.** Roughly 5.4k parameters are learned.
Sweeping `λ` traces the stealth-efficacy frontier, optimising against detection **where it is actually
read** (post-rotation) rather than orthogonalising at injection and hoping.

**Prior art:** Obfuscated Activations uses this joint-loss structure
(`min ℒ_beh(M(x)) + λ_obf ℒ_obf(ℛ(x))`, model frozen) — but optimises a **token/embedding suffix**, not a
**steering vector**. The variant is defensible on that difference and must cite them.

**Verdict:** high risk, high interest. Conditional on the main plan's outcome.

---

## Route C — Is the Blind Spot Just a Training Artifact?

### *Does diverse-vector training close the CAA blind spot?*

**The question.** Fonseca Rivera & Africa trained their detector on **CAA vectors only** and obtained a
**CAA-tuned** detector, blind to orthogonal directions. Is that blind spot a fact about introspection, or
an artifact of the training distribution? Train the same detector on **diverse vector types** and see
whether the blind spot closes — and, critically, whether it generalises to vector types held out entirely.

**Why it matters.** This determines **stealth steering's shelf life**. If the blind spot closes trivially
under diverse training, then future models — or auditors hardening their detectors — will close it too, and
projection-based stealth is a short-lived trick. If it **persists stubbornly** even under diverse training,
that is a robust geometric limit on introspection, and a much stronger foundation for stealth-steering as a
durable evaluation tool.

This route tests reconciliation hypothesis (a) in [`background.md`](background.md) directly: that the
published directionality is an artifact of the CAA training distribution.

### Method

**Model:** `gemma-2-9b-it` (Fonseca Rivera's model), injection at **L28/42** (67% depth).
**Technique:** QLoRA — LoRA adapters over a 4-bit quantised frozen base. Only low-rank adapter matrices are
trained, making detector fine-tuning affordable on a single GPU.

1. **Replicate the CAA-only detector · GATE.** Training data: `(steered activation → "Yes, {concept} was
   injected")` and `(clean → "No")`. **Validate that the blind spot reproduces** — high detection on
   held-out CAA concepts, ≈0% on PCA/random. *The blind spot must be reproduced before closing it can be
   tested.*
2. **Train the diverse arm.** Identical setup, but the training set draws from **CAA + PCA + ICA + random +
   Gaussian noise**.
   > ⚠️ **Control for data volume.** Both arms must have the **same total number of training examples**,
   > or the experiment measures dataset size rather than diversity. This is the easiest way to invalidate
   > the result.
3. **Test both arms on held-out types.** Hold out **LDA** and **dropout** from *both* training sets.
   Evaluate on `{CAA, PCA, ICA, random, noise}` (seen by the diverse arm) **and** `{LDA, dropout}` (seen by
   neither).

### Analysis

1. **Does the blind spot close?** Diverse-arm detection on PCA/ICA/random vs the CAA-arm.
2. **Does it generalise?** Diverse-arm detection on **held-out LDA/dropout**. This is the real scientific
   claim — closing the spot on *trained* types is easy; generalising to *unseen construction methods* is
   not.
3. **At what cost?** Compare clean-input false-positive rate and general capability. Did the detector become
   trigger-happy, or worse at its actual job?
4. **Detection ≠ resistance.** Re-test steerability of both arms. Fonseca Rivera found detection-training
   makes models *more* steerable — does diverse training amplify or reverse that?

### Trade-offs

**For:** Well-specified — a published recipe with one variable changed. **Does not depend on emergent
introspection existing**, so it is immune to the main plan's gate risk. Uses only a modest amount of LoRA
training.

**Against:** It studies a *trained* detector, which is one step removed from the emergent capability that
matters for the risk model. It answers "can the blind spot be closed?" rather than "does the blind spot
exist where it counts?"

**Verdict:** the strongest complement to the main plan, and the best fallback if the main plan's baseline
gate fails. Together the two answer contradiction C1 completely: **does the blind spot exist in emergent
detection (main plan), and does it survive diverse training (Route C)?**
