## 11. Numerical Geometry of Reinforcement Learning: Curvature of the Bellman Operator

**One-line pitch.**
Use numerical curvature and stability to analyze the *Bellman operator* and show when low-precision value iteration and Q-learning are actually safe—backed by both theorems and small-grid experiments.

### Core idea

Casting RL into the HNF framework:

* State–value function (V) lives in a numerical space ((\mathcal{V}, d, \mathcal{R})).
* Bellman operator (T) is a numerical morphism with Lipschitz constant (L_T) (related to (\gamma)) and curvature (\kappa_T).
* Iterative algorithms (value iteration, Q-learning) are compositions of (T) plus numerical noise from finite precision.

We use:

* Stability Composition Theorem to bound error accumulation across Bellman updates.
* Curvature Lower Bound to get *precision lower bounds* for stable policy evaluation.

### Contributions

1. **Numerical model of Bellman updates.**

   * Prove a bound of the form
     [
     \Phi_{T^k}(\varepsilon) \le L_T^k,\varepsilon + \frac{1-L_T^k}{1-L_T},\Delta_T
     ]
     where (\Delta_T) encodes per-step rounding error.
   * Incorporate curvature: show that high curvature in value space around optimal policies forces higher precision, or else value iteration can oscillate or diverge numerically even when it is *theoretically* a contraction.

2. **Precision-safe RL protocols.**

   * Derive conditions on bit depth and learning rate for tabular Q-learning and TD(0):

     * “If precision < (p^\star(\gamma, \kappa_T)), numerical noise dominates contraction, and effective discount factor becomes ≥ 1.”
   * Provide concrete rules: e.g., for given (\gamma) and reward scale, how low you can push precision while guaranteeing numerical stability.

3. **Experiments in small environments.**

   * Gridworlds, CartPole with tiny function approximators, simple bandits—all cheap on a laptop.
   * Compare:

     * float64 / float32 / float16 / simulated fixed-point.
     * Measured divergence or slow convergence vs predicted instability from (\Phi_{T^k}).
   * Show:

     * Cases where theory predicts safe regimes and experiments confirm.
     * Cases where aggressive low precision breaks convergence exactly when predicted.

### Why ICML-worthy + laptop-friendly

* Bridges *RL theory* and *numerical analysis* in a way that hasn’t really been formalized.
* Gives practical precision guidelines for embedded/edge RL.
* All experiments are small tabular or tiny-function-approximator RL—perfect for MacBook/Colab.

---

## 12. Numerical Geometry of Sampling: Stable Low-Precision MCMC and Diffusion on Toy Models

**One-line pitch.**
Treat sampling algorithms (Langevin, HMC, toy diffusion samplers) as numerical morphisms and derive precision bounds for *sampling accuracy*, showing when low-precision sampling is trustworthy.

### Core idea

HNF gives you:

* Numerical spaces for distributions (e.g., via Wasserstein metrics).
* Lipschitz / curvature control for update maps (e.g., Langevin step, diffusion step).
* Error functionals that accumulate over steps.

We apply this to *sampling*:

* Each sampler step is a numerical morphism on state space.
* Finite precision and pseudo-randomness introduce bias and variance.
* Curvature of the log-density and the sampler map governs how sensitive samples are.

### Contributions

1. **Numerical morphisms for samplers.**

   * Formalize one-step maps for:

     * Unadjusted Langevin (ULA).
     * A simple MALA variant.
     * A low-dimensional toy diffusion sampler (e.g., 1D/2D Gaussian mixtures).
   * Derive Lipschitz constants and curvature bounds in regions of interest.

2. **Sampling error vs precision.**

   * Use the stability algebra to bound *distributional error* (e.g., Wasserstein distance (W_2)) as a function of:

     * step size,
     * number of steps,
     * floating-point precision,
     * curvature of log-density.
   * Prove that below a certain precision threshold, numerical error dominates discretization error, making additional steps useless.

3. **Experiments on tiny targets.**

   * 1D, 2D Gaussian mixtures, banana-shaped distributions, small logistic posteriors.
   * For each:

     * Run samplers in different precisions.
     * Measure empirical KL/Wasserstein distance to ground truth.
     * Compare to theoretical bounds from (\Phi_{f^k}).
   * Show how curvature of the target distribution predicts sensitivity.

4. **Implications for diffusion models (toy setting).**

   * Implement a minimal 1D diffusion-like generative model (score-based).
   * Show how curvature + precision bounds inform step-size schedules and bit-depth choices for sampling.

### Why ICML-worthy

* Sampling and generative modeling are central ICML topics.
* Provides a *precision-aware* theory of sampling that’s missing in most MCMC/diffusion work.
* Experiments are low-dimensional; no need for big GPUs.

---

## 13. Numerical Geometry Compilers: Rewriting Computation Graphs into Better-Conditioned Forms

**One-line pitch.**
Use numerical equivalence and the categorical structure in HNF to build a prototype *compiler* that rewrites ML computation graphs into numerically better-conditioned but functionally equivalent variants.

### Core idea

HNF gives:

* A notion of numerical equivalence of maps ((f \sim_\text{num} g)).
* A monoidal category of numerical spaces and morphisms.
* Algebraic laws (associativity, distributivity) plus stability/curvature metadata.

We build a “numerical geometry compiler”:

* Treat a computation graph as a morphism in the numerical category.
* Apply rewrite rules that preserve (or approximate) semantics while improving error functionals (\Phi), Lipschitz constants, or curvature.

### Contributions

1. **Catalogue of numerically safe rewrites.**

   * Identify a finite set of algebraic rewrites:

     * Fold/unfold associativity: ((AB)C \leftrightarrow A(BC)).
     * Re-parameterizations: (x \mapsto ax+b) factored in numerically stable ways.
     * Common subexpression elimination that reduces numerical cancellation.
   * For each, prove bounds on how (\Phi), (L), and curvature change using the doc’s theorems.

2. **Compiler design.**

   * Represent a PyTorch (or JAX) computation graph as a typed DAG in the numerical category.
   * Implement a rewrite engine that:

     * Searches locally for patterns where a rewrite reduces a simple numeric cost measure (e.g., condition number proxy).
     * Applies them greedily or via a small search budget.

3. **Prototype: NumGeomCompile.**

   * Restrict to small nets (2–4 layers, simple activations) and basic ops (matmul, add, ReLU, norm).
   * Show that compiled graphs:

     * Have improved estimated numerical condition and curvature.
     * Exhibit smaller empirical rounding error when run at low precision.

4. **Experiments.**

   * Compare compiled vs uncompiled graphs on:

     * Inference stability (change in outputs under precision reduction).
     * Gradient stability.
   * Task: MNIST/Fashion-MNIST classification with small nets—Colab-friendly.

### Why ICML-worthy

* The “compiler for numerical robustness” narrative fits ICML’s interest in systems + theory.
* It operationalizes HNF equivalence in something concrete and code-producing.
* Limited scope prototype is fully feasible on a laptop.

---

## 14. Numerically Certified Interpretability: Error-Aware Saliency and Attribution

**One-line pitch.**
Attach numerical error certificates to saliency/maps and feature attributions, showing where explanations are *unstable under finite precision* and how to design numerically robust interpretability methods.

### Core idea

Interpretability methods (grad-based saliency, Integrated Gradients, DeepLIFT, etc.) are themselves *numerical algorithms* composed of:

* Forward and backward passes,
* Difference quotients,
* Path integrals or discrete approximations.

We:

* Model each interpretability method as a numerical morphism with its own (\Phi), (L), and curvature.
* Use HNF’s AD + stability theory to produce explicit error bounds on attribution scores under finite precision.

### Contributions

1. **Numerical models of attribution methods.**

   * Formalize:

     * Vanilla gradient saliency.
     * Integrated Gradients (IG) along linear paths.
   * For each, derive:

     * Lipschitz/curvature bounds w.r.t. input and intermediate precision.
     * Error functionals (\Phi_{\text{saliency}}) that reflect rounding plus network numerics.

2. **Certified attribution pipeline.**

   * Extend the “NumGeom-AD” machinery (from one of your earlier papers) so that:

     * Attributions come with per-feature error bars: “this attribution is (a_i \pm \delta_i)”.
     * You can flag features whose attribution is numerically unreliable.

3. **Empirical analysis.**

   * Tasks: tiny CNNs on MNIST and CIFAR-10.
   * Compare:

     * Standard float32 saliency/IG.
     * Quantized/low-precision variants.
     * Numerically-certified versions with error bounds.
   * Show:

     * Instances where sign or ranking of attributions changes due to numeric noise.
     * HNF bounds predict when this will happen.

4. **Guidelines for robust interpretability.**

   * Derive simple heuristics:

     * Avoid IG paths through high-curvature regions unless using higher precision.
     * Use small architectural tweaks to reduce curvature in layers that most affect attributions.

### Why ICML-worthy

* Interpretability is a big ICML topic; numeric robustness of explanations is underexplored.
* The paper gives both theory and a practical tool.
* Experiments are tiny models and image patches—easy on Colab.

---

## 15. Numerical Geometry of Fairness Metrics and Constraints

**One-line pitch.**
Study how finite precision affects *fairness metrics* (e.g., demographic parity, equalized odds) and constraints, showing when fairness violations are real vs numerical artifacts—and how to design numerically stable fairness-aware training.

### Core idea

Fairness metrics are functions of:

* Confusion matrix entries,
* Probabilities/thresholds,
* Aggregated statistics.

Many are highly sensitive to small changes near decision thresholds.

HNF gives a way to:

* Model the entire fairness evaluation pipeline as a numerical morphism.
* Analyze Lipschitz constants and curvature of fairness metrics w.r.t. underlying prediction scores and counts.
* Propagate finite-precision errors through the pipeline.

### Contributions

1. **Fairness metrics as numerical morphisms.**

   * Express demographic parity, equalized odds, predictive parity as maps on:

     * Predicted probabilities,
     * Thresholds,
     * Group labels.
   * Derive Lipschitz and curvature estimates around realistic parameter regions (e.g., thresholds near 0.5).

2. **Error bounds for fairness evaluation.**

   * Use stability algebra to bound change in fairness metrics under:

     * Rounding of scores.
     * Integer count overflows / binning approximations.
   * Show cases where apparent fairness violations (e.g., 2–3% disparity) could be explained entirely by numerical noise.

3. **Numerically stable fairness training.**

   * Consider simple fairness-aware training objectives (penalties on fairness metrics).
   * Show that if you ignore numerical geometry:

     * Gradients of fairness penalties can be dominated by numerical error.
   * Propose simple stabilizers:

     * Reparameterizations,
     * Soft relaxations designed with curvature in mind,
     * Precision-aware thresholding.

4. **Empirical studies.**

   * Use well-known fairness datasets (Adult, COMPAS-like toy datasets).
   * Small models: logistic regression, tiny MLPs.
   * Evaluate:

     * Fairness metrics across precision regimes.
     * Whether HNF-derived bounds correctly predict when fairness conclusions are stable.

### Why ICML-worthy

* Fairness + reliability is a high-visibility ICML topic.
* This paper injects *numerical rigor* into fairness evaluation and training that is currently missing.
* Datasets and models are small, easily run on a laptop.

---

## 16. Numerical Geometry of Hyperparameter Optimization and AutoML

**One-line pitch.**
Analyze hyperparameter optimization (HPO) as a numerical process with a noisy objective, and show how finite precision and curvature of the validation loss warp the HPO landscape—leading to guidelines for numerically robust HPO.

### Core idea

HPO algorithms (random search, Bayesian optimization, bandits) optimize a scalar function:

* (F(\lambda) =) validation loss of model with hyperparameters (\lambda),
* computed via training runs that are themselves numerical morphisms.

Under finite precision:

* The evaluation (F(\lambda)) gains an error functional (\Phi_F(\varepsilon)).
* High curvature around optimum and noisy training plus rounding leads to “objective aliasing”.

### Contributions

1. **Numerical model of the HPO objective.**

   * View evaluation pipeline as composition:
     [
     \lambda \xrightarrow{\text{train}} \theta_\lambda
     \xrightarrow{\text{eval}} \widehat{L}*{\text{val}}(\theta*\lambda)
     ]
   * Assign error functionals to training and eval via HNF.
   * Describe an effective noise model for HPO: observed loss = true loss + numerical + stochastic error.

2. **HPO algorithm sensitivity to numerical noise.**

   * Show analytically for simple HPO schemes:

     * Random search,
     * Successive halving / Hyperband,
   * how numerical error can cause:

     * Wrong hyperparameter ranking,
     * Premature elimination of good configs.

3. **Numerically robust HPO strategies.**

   * Propose:

     * Precision-aware resampling: re-run promising configs in higher precision.
     * Curvature-aware step sizes in HPO space.
   * Derive small theorems: e.g., “If numerical noise variance < (\sigma^2), then K restarts suffice to choose hyperparameters within (\delta) of optimum with high probability.”

4. **Experiments.**

   * Tiny models on MNIST/UCIs; hyperparameters like learning rate, weight decay, dropout.
   * HPO with and without numerical-geometry-aware adjustments.
   * Show differences in:

     * Reproducibility of chosen hyperparameters.
     * Final performance.

### Why ICML-worthy

* HPO/AutoML is a standard ICML area.
* This adds a new, practically important dimension: *how much your hyperparameters depend on your floating-point choices*.
* All experiments are small-scale.

---

## 17. Topological Data Analysis under Finite Precision: Numerical Homology Guarantees

**One-line pitch.**
Apply numerical homotopy and stability tools to persistent homology and TDA, deriving precision bounds for Betti numbers and barcodes—and implement a small TDA library with certified robustness.

### Core idea

TDA (persistent homology) is sensitive to:

* Distance computations,
* Filtration thresholds.

HNF gives:

* Numerical spaces for point clouds and distance matrices.
* Stability of numerical homotopy groups.
* Error functionals for composed operations (distance → filtration → boundary matrices → ranks).

### Contributions

1. **Numerical model of persistent homology pipelines.**

   * Represent:

     * Point cloud → distance matrix;
     * Distance matrix → filtered simplicial complex;
     * Complex → boundary matrices → homology.
   * Each stage has finite-precision error and Lipschitz properties.

2. **Precision bounds for Betti stability.**

   * Derive theorems of the form:

     * If distance errors are bounded by (\varepsilon), then barcodes change by at most (2\varepsilon) in bottleneck distance (building on standard TDA stability, but now explicitly tied to numeric error).
   * Use curvature-like bounds to identify point configurations where small numerical errors cause large topological changes (e.g., near critical radii).

3. **Prototype: NumGeom-TDA.**

   * Implement a small TDA toolkit for 2D/3D point clouds:

     * Vietoris–Rips, Čech complexes.
     * Simple persistent homology computation.
   * Add HNF-inspired instrumentation to:

     * Track and report numerical error bounds for barcodes.
     * Flag potentially unstable barcodes.

4. **Experiments.**

   * Synthetic datasets: circles, tori, figure-8s, noisy manifolds.
   * Compare:

     * float64 / float32 / float16 / quantized distances.
     * Observed barcode changes vs theoretical bounds.

### Why ICML-worthy

* TDA is niche but recognized at ICML; this paper connects it tightly to numerical analysis.
* It gives practitioners a way to *trust* or *distrust* their barcodes under finite precision.
* Point clouds are small; laptops are enough.

---

## 18. Numerically Aware Probabilistic Inference: Variational and Monte Carlo with Error Functionals

**One-line pitch.**
Use numerical geometry to bound the effect of finite precision and integration error in variational inference (VI) and Monte Carlo (MC) estimators, leading to practical stopping and precision-selection rules.

### Core idea

Inference pipelines:

* Approximate expectations (\mathbb{E}_q[f]) or optimize ELBOs.
* Combine Monte Carlo sampling, gradient estimation, and optimization—each with numerical error.

HNF allows:

* Modeling the whole inference procedure as composed numerical morphisms.
* Attaching error functionals to estimators.

### Contributions

1. **Error decomposition for VI/MC.**

   * Decompose error into:

     * Approximation error (choice of family),
     * Stochastic estimation error,
     * Numerical error from finite precision.
   * Express the last term with HNF error functionals (\Phi).

2. **Precision vs sample size tradeoffs.**

   * For simple models (Gaussian posterior, logistic regression), derive:

     * Bounds on estimator variance + rounding bias.
   * Show regimes where it’s better to:

     * Take fewer samples at higher precision vs more samples at lower precision.

3. **Practical algorithms.**

   * Propose:

     * An adaptive scheme that monitors numerical error indicators and adjusts precision or sample counts.
   * Provide analytic stopping criteria that ensure total error < (\varepsilon), splitting budget between MC and numerical parts.

4. **Experiments.**

   * Toy Bayesian models (1D/2D Gaussians, small Bayesian logistic regression).
   * VI with reparameterization gradients; simple MCMC.
   * Compare:

     * Standard float32 implementations.
     * Numerically-aware versions that adjust precision/samples based on HNF bounds.

### Why ICML-worthy

* Probabilistic ML is mainstream at ICML; inference reliability is critical.
* This paper introduces a solid numerics angle with real algorithms.
* Experiments are low-dimensional; no HPC required.

---

## 19. Numerical Geometry of Dataset Distillation and Coresets

**One-line pitch.**
Use numerical entropy and curvature to define *numerically robust* coresets and distilled datasets, analyzing how finite precision limits how much you can compress training data.

### Core idea

Dataset distillation / coresets aim to approximate a large dataset with a small synthetic set.

HNF tools:

* Numerical entropy (\mathcal{H}^{\mathrm{num}}) for data manifolds.
* Curvature bounds for models on those manifolds.

We:

* Connect these to coreset size and precision requirements.
* Show that some aggressive distillation schemes essentially *overshoot* what finite precision can meaningfully represent.

### Contributions

1. **Numerical capacity of a dataset.**

   * Treat data manifold (M \subset \mathbb{R}^d) and model family (f_\theta).
   * Use (\mathcal{H}^{\mathrm{num}}(M,\varepsilon)) and curvature of (f_\theta) to bound:

     * Minimal coreset size to approximate loss within (\varepsilon) under finite precision.

2. **Precision-aware coreset construction.**

   * Modify standard coreset/distillation algorithms (k-center, gradient matching) to:

     * Incorporate numeric-entropy-based spacing.
     * Avoid selecting points in numerically unstable regions (e.g., high curvature, near degeneracies).

3. **Experiments.**

   * Small-image and tabular benchmarks (MNIST subsets, UCI datasets).
   * Compare:

     * Standard coreset/distillation vs numerically-aware variants.
   * Metrics:

     * Test performance using low-precision training.
     * Stability of performance under small changes in precision.

4. **Guidelines.**

   * Provide rules linking precision, model curvature, and achievable dataset compression.

### Why ICML-worthy

* Dataset distillation and coresets are active topics.
* Paper adds a precise story about *how much compression is even meaningful* under hardware constraints.
* All experiments involve small datasets/models.

---

## 20. Numerical Complexity Classes for ML Tasks: A Curvature-Based Taxonomy

**One-line pitch.**
Propose a numerical-geometry-inspired taxonomy of ML tasks based on curvature, condition numbers, and numerical entropy—arguing that some tasks are *intrinsically hard to learn at low precision*.

### Core idea

HNF suggests:

* Some maps require large bit-depth to approximate well (high curvature/condition).
* Datasets with high numerical entropy demand more degrees of freedom and precision.

We package this into a “numerical complexity theory” for ML tasks:

* Define classes of problems based on *minimal precision* + *model size* needed to achieve a given error.

### Contributions

1. **Formal definitions of numerical task complexity.**

   * For a task (\mathcal{T} = (P(X,Y), \mathcal{F})), define:

     * Minimal bit-depth (p^\star(\varepsilon)).
     * Minimal parameter count (n^\star(\varepsilon, p)).
   * Use curvature and numerical entropy bounds from HNF to derive lower bounds on these quantities for simple families of tasks.

2. **Toy examples and theorems.**

   * Construct synthetic tasks with:

     * Low vs high curvature decision boundaries.
     * Low vs high data manifold entropy.
   * Prove that:

     * Some tasks admit low-precision approximations with small networks.
     * Others require either:

       * High precision, or
       * Large models, even on small datasets.

3. **Empirical classification.**

   * For a set of small benchmark tasks (MNIST variants, synthetic tabular problems, simple RL tasks), empirically estimate:

     * Curvature of learned solutions,
     * Numerical entropy of data.
   * Map these tasks into a numerical complexity taxonomy and check:

     * Does observed minimal-working-precision align with theory?

4. **Implications.**

   * Argue that:

     * Some “hard” tasks are hard mostly due to numerical geometry, not sample complexity alone.
     * Precision thresholds could be used as a practical diagnostic for task difficulty.

### Why ICML-worthy

* Bold, conceptual paper: proposes a new axis of complexity theory directly grounded in numerics.
* Leverages many pieces of the HNF framework at once (curvature, entropy, equivalence).
* Experiments are all on tiny synthetic/benchmark tasks—great for MacBook/Colab.
