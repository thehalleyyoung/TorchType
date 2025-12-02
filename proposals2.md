## 1. Curvature-Guided Bit Allocation in Neural Networks

**One-line pitch.**
Turn the Curvature Lower Bound Theorem into a practical algorithm that decides how many bits each layer of a neural network actually needs, and show that this beats naive quantization in accuracy–vs–precision tradeoffs.

### Core idea

The document’s curvature framework gives an invariant (\kappa_f) that lower-bounds the precision required to approximate a map (f) to error (\varepsilon): high curvature implies you *must* pay more bits or more steps. This paper specializes that to neural networks:

* Treat each layer (f_i) of a network as a map with Lipschitz constant (L_i) and curvature (\kappa_i).
* Use the curvature lower bound to derive a *layer-wise minimal mantissa precision* (p_i(\varepsilon)).
* Design a “curvature-aware quantization” (CAQ) scheme that allocates bits non-uniformly across layers (and possibly within channels) to just meet a global error budget.

### Contributions

1. **Curvature-to-precision translation for layers.**

   * Show how to estimate (\kappa_i) from minibatches via Hessian-vector products or finite differences, using small networks (2–4 layer MLPs, modest CNNs).
   * Prove a theorem of the form: if layer (i) has curvature (\kappa_i) on region (\Omega), then any fixed-point representation with (p_i < p_i^\star(\kappa_i,\varepsilon)) necessarily incurs error above (\varepsilon) on (\Omega).

2. **Global bit budgeting via the Stability Composition Theorem.**

   * Use the linear error functionals (\Phi_{f_i}(\varepsilon) = L_i \varepsilon + \Delta_i) and the composition rule to turn local precision choices (p_i) into a bound on end-to-end error.
   * Solve a discrete optimization problem: minimize total bit-operations (or memory footprint) subject to (\Phi_F(\varepsilon_0) \le \varepsilon_{\text{target}}).

3. **Algorithm: Curvature-Aware Quantization (CAQ).**

   * Simple greedy/convex heuristic: start from high precision, decrease (p_i) where curvature and downstream (L_j) suggest slack.
   * Output: per-layer (and maybe per-channel) precision schedule that can be implemented with integer/fixed-point kernels or simulated in float via “fake quantization”.

4. **Empirical study on small models.**

   * MLPs on MNIST / Fashion-MNIST, small CNNs on CIFAR-10 / SVHN, small Transformers on character-level language modeling.
   * Compare CAQ vs uniform quantization vs heuristic baselines (e.g., per-layer Hessian-based but ignoring composition) for:

     * Accuracy vs average bits per multiply.
     * Robustness of accuracy under slight hardware noise models.
     * Sensitivity to curvature mis-estimation.

### Why it’s ICML-worthy and MacBook-friendly

* **Novel angle:** Quantization paper with a *theorem-backed precision lower bound* from curvature invariants, not just empirical heuristics.
* **Compute-light:** Curvature estimates on small networks, experiments on small datasets, no massive pretraining.
* **Clear message:** “You are throwing bits away where curvature is small; here is a principled way to stop.”

---

## 2. The Stability Algebra of Learning Pipelines: Compositional Error and Generalization

**One-line pitch.**
Elevate the Stability Composition Theorem from numerical analysis to a unified view of *learning pipelines* (data preprocessing → feature extraction → model → post-processing), and show new stability-based generalization bounds that explicitly include finite-precision error.

### Core idea

The document defines error functionals (\Phi_f) for maps and proves composition rules:
[
\Phi_{g \circ f}(\varepsilon) = L_g \Phi_f(\varepsilon) + \Delta_g.
]
This is a perfect fit for ML pipelines where each stage is a Lipschitz (or approximately Lipschitz) transformation plus numerical noise.

### Contributions

1. **Formalizing ML pipelines in (\mathbf{NMet}).**

   * Model a full pipeline (F = f_k \circ \cdots \circ f_1), where each (f_i) is a numerical morphism with ((L_i, \Delta_i)).
   * Construct an error functional (\Phi_F) using the stability algebra; show explicit closed-form bounds as in the doc:
     [
     \Phi_F(\varepsilon) = \left(\prod_i L_i\right)\varepsilon + \sum_i \Delta_i \prod_{j>i} L_j.
     ]

2. **Stability-generalization connection with finite precision.**

   * Combine algorithmic stability bounds (e.g., Bousquet–Elisseeff) with numerical error: show that effective stability ( \widetilde{\beta}) for training algorithm + pipeline is bounded by a function of the (\Phi_{f_i}).
   * Derive generalization bounds where finite precision acts as an *extra source of instability*, quantified via the stability algebra.

3. **Error-aware pipeline design rules.**

   * Show that certain design practices (e.g., high-Lipschitz preprocessing, aggressive normalization, unstable post-processing) have disproportionate impact on the composed error.
   * Provide design heuristics: where to spend precision, where to insert “numerical dampers” (non-expansive maps).

4. **Empirical micro-benchmarks.**

   * Build toy but realistic pipelines: data standardization → PCA → small neural net → calibrated probabilities, for UCI tabular datasets and small image datasets.
   * Run with different floating-point formats (float64/float32/float16 and fake quantization) and compare:

     * Observed test error.
     * Measured empirical stability (leave-one-out perturbations).
     * Predicted numerical instability from the compositional (\Phi_F).

### Why it’s ICML-worthy

* **Conceptual:** Gives a clean algebraic story tying numerical analysis to generalization theory.
* **Practical:** Produces actionable rules for pipeline design and precision budgeting.
* **Feasible:** All experiments are on small datasets and models; the heavy lifting is theoretical + software instrumentation.

---

## 3. Sheaves of Precision for ML Systems: Cohomological Debugging of Numerical Inconsistencies

**One-line pitch.**
Turn the “Precision Sheaf” and its cohomology into a concrete debugging tool that finds *inconsistent precision requirements* across complex ML workflows (training + evaluation + logging + post-processing).

### Core idea

In the doc, precision requirements form a sheaf (\mathcal{P}) over a “computation graph” or parameter space, and global consistency issues show up as cohomology classes (obstructions in (H^1)). This can be reinterpreted for ML systems:

* Nodes = components (data loader, normalization, model, metrics, logging).
* Edges and overlaps = interfaces where the same quantity is represented at two precisions or with different error assumptions.
* Inconsistent precision assumptions manifest as nontrivial 1-cocycles.

### Contributions

1. **Formal ML system as a precision sheaf.**

   * Represent an ML workflow as a directed acyclic graph with typed edges (tensors, scalars, indices).
   * Define a sheaf (\Prec) with stalks = feasible precision ranges for each quantity (e.g., [8 bits, 32 bits]) and restriction maps induced by operations.
   * Show how to construct a Čech complex for a chosen cover (e.g., groups of components) as in the document.

2. **Cohomology as a bug signal.**

   * Prove a small theorem: certain classes of precision inconsistencies (e.g., mismatch between training and evaluation precision for the same embedding) correspond to nontrivial 1-cocycles in (\check{H}^1(\mathcal{U};\Prec)).
   * Provide a practical algorithm for computing a discrete surrogate of this cohomology on a finite graph—essentially solving a consistency system of inequalities around cycles.

3. **Tool: “SheafCheck” prototype.**

   * Implement a Python library that:

     * Hooks into a PyTorch or JAX computation graph.
     * Extracts precision annotations (actual dtypes + user-specified numeric assumptions).
     * Solves a small linear/quasi-linear system to identify cycles where no consistent precision assignment exists.
   * Output: flagged subgraphs and suggested patches (e.g., “layer X logs in fp64 while evaluation assumes fp32; choose a single canonical precision”).

4. **Case studies.**

   * Toy but realistic ML projects: small ResNet for CIFAR-10; a small language model with custom logging; a training-evaluation mismatch scenario.
   * Create synthetic bugs: inconsistent quantization in training vs evaluation, metrics computed at different precisions than logits, etc.
   * Show that SheafCheck detects issues that naive dtype checking misses.

### Why it’s ICML-worthy

* **Novel framing:** First use of sheaf theory / cohomology as a *practical* debugging tool in ML systems.
* **Relevance:** Precision mismatches cause silent bugs; this is a systems-meets-theory paper.
* **Compute:** Only needs small networks; the heavy part is graph analysis, which is cheap.

---

## 4. Information–Precision Tradeoffs in Representation Learning

**One-line pitch.**
Use the Numerical Information-Complexity Correspondence to quantify how much *information* a representation can actually carry under finite precision, and relate that to mutual information and compression in representation learning.

### Core idea

The doc defines numerical entropy (\mathcal{H}^{\mathrm{num}}(A,\varepsilon)) (via covering numbers) and proves:

* (\mathcal{H}^{\mathrm{num}}(f(A),\varepsilon) \le \mathcal{H}^{\mathrm{num}}(A,\varepsilon / L_f)).
* Lipschitz and condition numbers control information preservation/destroying under finite precision.

We leverage this to analyze representation learning:

* Feature map (f_\theta: X \to Z) (penultimate layer) as numerical morphism.
* Finite-precision representation (Z_H) as a discretization of (Z) with bits constrained by hardware.

### Contributions

1. **Numerical information profile of a representation.**

   * Define a “numerical capacity curve”:
     [
     C_{\text{num}}(\varepsilon) := \mathcal{H}^{\mathrm{num}}(f_\theta(X), \varepsilon),
     ]
     estimated via covering numbers or Rademacher complexity on minibatches.
   * Relate (C_{\text{num}}) to bit-budget (p) and Lipschitz constants as in the doc.

2. **Bounds connecting numerical capacity to standard representation metrics.**

   * Show how (C_{\text{num}}) upper-bounds (or correlates with) empirical mutual information proxies (e.g., noise-contrastive estimates) under certain conditions.
   * Prove that aggressive quantization that violates curvature-based bit lower bounds must collapse (C_{\text{num}}) and induce an information bottleneck.

3. **Empirical validation on toy tasks.**

   * Small CNNs / MLPs on MNIST, CIFAR-10; small Transformers on text classification (AG News, etc.) using small Colab deployments.
   * For each model:

     * Estimate Lipschitz and curvature for feature map.
     * Estimate (C_{\text{num}}(\varepsilon)) for a grid of (\varepsilon).
     * Quantize features to various bit depths and see how linear probe accuracy and mutual-information proxies degrade.

4. **Design guidelines.**

   * Show that networks trained with explicit penalties on Lipschitz/curvature (e.g., spectral normalization, Hessian regularization) achieve better information–precision tradeoffs.
   * Provide a recipe: “Given a bit budget (p), here is how to choose architecture/hyperparameters to preserve a target (C_{\text{num}}).”

### Why it’s ICML-worthy

* **Bridges theory and practice:** Connects hardcore numerical geometry (covering numbers under finite precision) to standard representation-learning questions.
* **Feasible experiments:** All on small models/datasets; information measures estimated from samples.
* **Message:** “Here’s a principled way to talk about how much information your learned representation can actually carry on real hardware.”

---

## 5. Numerical Homotopy Paths and the Geometry of Training Dynamics

**One-line pitch.**
Use Numerical Homotopy Theory to define invariants of training trajectories that are *stable under finite precision*, and show that common tricks (learning-rate schedules, weight decay, etc.) correspond to distinct numerical homotopy classes of training paths.

### Core idea

The doc defines numerical homotopy groups (\pi_n^{\mathrm{num}}(A)) for numerical spaces and proves invariance under numerical equivalence. We reinterpret:

* Parameter space (\Theta) with metric + realizability structure = numerical space.
* Training trajectory (\theta_t) (discrete in practice) approximates a path in (\Theta).
* Different optimization hyperparameters yield different homotopy classes of these paths relative to a set of low-loss regions.

### Contributions

1. **Numerical homotopy of training paths.**

   * Define numerical paths as piecewise-Lipschitz curves (\gamma: [0,1]\to \Theta) with finite precision realizations.
   * Define an equivalence relation where two discrete training trajectories are in the same numerical homotopy class if they can be connected by a path of training trajectories whose numerical error remains bounded by a threshold.

2. **Topological view of “getting stuck”.**

   * Show that in some nonconvex loss landscapes, there are homotopy classes of paths connecting initialization to low-loss basins that *require* passing through high-curvature regions, hence high precision (by curvature lower bound).
   * Argue that low-precision training may force optimization into certain “numerically cheaper” homotopy classes, changing which minima are reachable.

3. **Experimental exploration.**

   * Use tiny models and synthetic landscapes:

     * 2D toy losses with known topology.
     * Shallow networks on XOR-style tasks where different minima correspond to qualitatively different decision boundaries.
   * Train with different precisions and hyperparameters; empirically cluster training trajectories into homotopy-like classes (e.g., via path metrics or persistence-homology-style summaries).
   * Show correspondence between numerical precision, curvature along paths, and which minima are reached.

4. **Implications for robust training.**

   * Suggest using numerical homotopy invariants as *regularizers* (e.g., penalizing paths that cross high curvature under low precision).
   * Offer a speculative link to mode connectivity and loss-landscape topology, but now with finite-precision constraints.

### Why it’s ICML-worthy

* **Conceptual novelty:** Introduces a topological/numerical lens on optimization that’s new even for theory papers.
* **Compute-light:** Focused on small models and synthetic examples; heavy math, light GPU.
* **Potential to inspire:** Gives a vocabulary for talking about how precision constrains which minima we ever see.

---

## 6. Numerical Equivalence of Training Algorithms: When Are New Optimizers Just Old Ones in Disguise?

**One-line pitch.**
Use the Numerical Equivalence theory to show that many seemingly different optimizers or training tricks are *numerically equivalent* under realistic hardware, and identify truly distinct numerical behaviors.

### Core idea

The doc defines numerical equivalence (\NumEquiv) between functions when they can simulate each other via Lipschitz maps with bounded condition numbers, plus representation-theoretic constraints. Apply that to:

* Training algorithms (SGD, momentum, Adam, Adagrad, etc.) as numerical morphisms on parameter space × gradients.
* Their implementations under specific floating-point formats and update rules.

### Contributions

1. **Formal model of optimizers as numerical morphisms.**

   * Represent an update rule (U) as a map ((\theta, g, s) \mapsto (\theta', s')) where (s) is optimizer state (e.g., moments in Adam).
   * Equip domain/codomain with numerical structures (precision, realizations).

2. **Equivalence criteria.**

   * Define when two optimizers (U_1, U_2) are numerically equivalent up to tolerance (\delta): there exist numerical morphisms (f,g) such that (U_1 \approx f \circ U_2 \circ g) with controlled condition numbers and precision requirements.
   * Use the Representation Theorem from the doc to show that many “new” optimizers collapse into a small number of equivalence classes once hardware rounding is accounted for.

3. **Case studies on small tasks.**

   * Train small CNNs on CIFAR-10 / MNIST and MLPs on tabular tasks with:

     * float32 vs float16 vs quantized variants.
     * Various optimizers and learning-rate schedules.
   * Empirically show that:

     * Parameter trajectories lie within small neighborhoods in (\Theta) under different optimizers when viewed through a numerical equivalence lens.
     * Some optimizers that look different in real arithmetic become indistinguishable numerically.

4. **Positive results: genuinely distinct classes.**

   * Identify at least one optimizer (e.g., signSGD-style, or second-order preconditioners) that *cannot* be numerically equivalent to vanilla SGD under realistic precision without huge condition numbers.
   * Prove a small impossibility/canonical-form style result inspired by the “no universal canonical form” theorem in the doc.

### Why it’s ICML-worthy

* **Provocative claim:** Many optimizer papers are numerically equivalent; this gives a formal language for that skepticism.
* **Methodological:** Encourages the community to reason about *numerical* distinctness, not just algebraic.
* **Feasible:** Experiments are on small models; most work is in defining and bounding equivalence.

---

## 7. Certified Autodiff: Practical Automatic Differentiation with Error Functionals

**One-line pitch.**
Implement the document’s “Automatic Differentiation with Precision Tracking” for a subset of PyTorch/JAX operations, so every gradient comes with a *numerical error functional* certificate.

### Core idea

The doc shows how to propagate error functionals through compositions and through forward/reverse-mode AD. This paper takes that theory and builds:

* A small autodiff engine (or extension) that, instead of just computing gradients, also computes (\Phi_f) for both the forward map and the gradient.

### Contributions

1. **Error-aware AD rules.**

   * For primitive ops (add, multiply, matmul, ReLU, softmax, etc.), define:

     * Local Lipschitz bounds.
     * Local error functionals (\Phi) as in the doc, including dependence on hardware format.
   * Derive forward- and reverse-mode propagation rules to compute error functionals for gradients (\nabla_\theta \ell).

2. **Implementation: “NumGeom-AD”.**

   * Implement a mini library in Python (possibly on top of JAX or PyTorch autograd):

     * Restricted op set (enough for small MLPs/CNNs).
     * On each backward pass, compute both gradients and symbolic/parametric error bounds.
   * Provide simple interfaces: `loss_with_error(x, y)` returns `(loss, grad, error_bound)`.

3. **Validation experiments.**

   * On small networks and datasets, compare:

     * Estimated error bounds vs empirical gradient error (via high-precision baseline).
     * Impact of changing precision (float64 → float32 → float16) on bound tightness.
   * Use the tool to detect situations where gradients are numerically unreliable (e.g., saturated softmax, high-conditioned linear layers).

4. **Use cases.**

   * Show examples where NumGeom-AD automatically warns about:

     * Unstable double-backprop or higher-order gradients.
     * Catastrophic cancellation in recurrent networks.
   * Demonstrate that simple architectural tweaks (rescaling, normalizing) reduce the certified gradient error.

### Why it’s ICML-worthy

* **Practical + theoretical:** A concrete tool that implements a nontrivial theory from the doc and could be adopted by practitioners.
* **Niche but important:** Numerical reliability of AD is a growing concern; this tackles it with formal error tracking.
* **MacBook-friendly:** Operates on small models, no need for huge training runs.

---

## 8. End-to-End Quantization-Aware Training via Numerical Geometry

**One-line pitch.**
Design a quantization-aware training procedure where the *entire training loop* (forward, backward, optimizer) is treated as a numerical morphism, and bits are scheduled using the Stability Composition + Curvature tools.

### Core idea

This is more applied than Paper 1: instead of just designing inference-time bit allocations, we:

* Treat the entire training update (\theta_{t+1} = \mathcal{T}(\theta_t)) as a numerical morphism (\mathcal{T}).
* Apply the stability algebra and curvature bounds to choose precision schedules for forward pass, backward pass, and optimizer state.

### Contributions

1. **Numerical model of a training step.**

   * Factor the training step into:

     * Forward map (f_\theta(x)),
     * Loss (\ell(f_\theta(x), y)),
     * Gradient computation (g(\theta)),
     * Optimizer update (U(\theta, g, s)).
   * Compute or bound ((L_i, \Delta_i)) for each part.

2. **Precision scheduling algorithm.**

   * Given a target bound on parameter update error (\Phi_{\mathcal{T}}(\varepsilon)), derive minimal per-component precision choices:

     * Forward activations & weights.
     * Backward gradients.
     * Optimizer accumulators.
   * Use curvature estimates to decide where low precision is dangerous (e.g., high curvature layers, sharp loss regions).

3. **Training protocols.**

   * Propose:

     * “Warm-up in high precision, then gradually lower bits where curvature allows.”
     * Layer-wise mixed precision that adapts over epochs using online curvature estimates.

4. **Experiments.**

   * Train small CNNs and MLPs on MNIST/CIFAR-10 using:

     * Standard mixed-precision baselines.
     * This numerical-geometry-based scheduling.
   * Compare:

     * Final accuracy.
     * Total “bit FLOPs”.
     * Stability of training (gradient norms, loss spikes).
   * Show that numerical-geometry scheduling finds “safe” low-precision regions that heuristic methods sometimes miss.

### Why it’s ICML-worthy

* **Directly relevant to efficiency:** Quantization-aware training is a core ICML topic.
* **Stronger story:** Unlike heuristic recipes, this one is backed by explicit error-composition theorems and curvature lower bounds.
* **Compute:** Again, small models and datasets; main novelty is the scheduling algorithm.

---

## 9. Numerically Safe Scientific ML: Finite-Precision Guarantees for Neural ODEs and PINNs

**One-line pitch.**
Apply Numerical Geometry to small scientific ML models (neural ODEs, 1D PINNs) to provide *rigorous precision guarantees* for solving differential equations with learned components.

### Core idea

The doc already touches scientific computing (ODE integrators, PDEs). ICML cares about Scientific ML; we can:

* Consider simple ODEs/PDEs whose solutions are approximated by neural networks.
* Show how curvature + stability results yield lower bounds on precision for the integrator and the network evaluation to guarantee a given global error.

### Contributions

1. **Numerical geometry of neural ODE integration.**

   * Model the ODE solver (e.g., Runge–Kutta) as a numerical morphism with error functional (\Phi_{\text{ODE}}(\varepsilon)).
   * Model the neural RHS (f_\theta) or solution ansatz as a numerical morphism with curvature (\kappa_{f_\theta}).
   * Combine to derive minimal precision to ensure a global error bound (\varepsilon_{\text{global}}) on the trajectory.

2. **PINNs and residual evaluation.**

   * For physics-informed neural networks on 1D/2D toy PDEs (Poisson, heat equation), analyze:

     * Finite-difference approximations of derivatives as numerical morphisms.
     * Curvature of the loss landscape with respect to network outputs.
   * Use curvature bounds to argue about when residuals are *numerically meaningless* at low precision.

3. **Empirical verification.**

   * Implement small neural ODE models and PINNs on Colab:

     * Very low-dimensional systems (Lorenz, pendulum).
     * 1D PDEs with known closed forms.
   * Run experiments across precisions:

     * float64, float32, float16, and quantized.
   * Compare observed global error vs predicted precision lower bounds.

4. **Guidelines for scientific ML practitioners.**

   * Provide rules: for given problem regularity and integrator choice, how low can you go in precision before solution quality collapses?
   * Argue that many current neural ODE/PINN demos dangerously underbudget precision relative to curvature-based lower bounds.

### Why it’s ICML-worthy

* **Scientifically relevant:** Scientific ML is a key ICML theme.
* **Gap-filling:** Most works ignore finite precision; this one gives the first systematic finite-precision analysis.
* **Compute:** Very small systems; experiments are cheap.

---

## 10. A Numerical Geometry Benchmark Suite for Machine Learning

**One-line pitch.**
Package the whole framework into a *benchmark suite* of small tasks and metrics that stress-test ML algorithms for numerical robustness, and release code as an open-source library.

### Core idea

To make Numerical Geometry “real” for the community, we need a public artifact: a set of tasks, each annotated with:

* Known or estimated curvature invariants.
* Known stability/error-composition profiles.
* Precision stress-tests.

### Contributions

1. **Task design.**

   * Define a suite of 8–12 lightweight tasks:

     * Training small nets on ill-conditioned linear problems.
     * Classification problems with sharply curved decision boundaries.
     * Toy scientific ML tasks from Paper 9.
     * Pipelines with known precision-sensitivity from Papers 2 and 3.
   * For each task, include:

     * Baseline error vs precision curves.
     * Approximate curvature profiles.
     * Suggested “challenge variants” (e.g., low-precision only, noisy hardware).

2. **Metrics.**

   * Implement metrics derived from the doc:

     * Empirical numerical entropy (\mathcal{H}^{\mathrm{num}}(A,\varepsilon)).
     * Measured vs predicted error functionals (\Phi).
     * A “numerical robustness score” comparing performance across precision regimes.
   * Provide simple APIs (Python) for evaluating new models/algorithms.

3. **Baselines.**

   * Evaluate:

     * Standard training with float32/float16.
     * Mixed-precision heuristics.
     * (Optionally) prototypes from Papers 1, 2, 7, 8.
   * Demonstrate that numerical-geometry-aware methods outperform naive baselines on robustness metrics.

4. **Reproducible, low-compute design.**

   * Ensure every benchmark runs comfortably on a MacBook or free Colab:

     * Limit model sizes and dataset sizes.
     * Provide pre-configured Colab notebooks.

### Why it’s ICML-worthy

* **Community artifact:** ICML loves well-motivated benchmarks that crystallize an emerging concern.
* **Amplifier for the whole program:** This suite showcases all the previous papers’ ideas as part of a coherent numerical-geometry agenda.
* **Sustainable:** Other groups can extend the benchmark with their own tasks.
