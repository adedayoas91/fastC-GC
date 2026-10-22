# Independent review of the fastC-GC manuscript

## Scope

This review covers the claims, proofs, agreement with the current implementation,
citation support, and manuscript writing in `../manuscript/`. The reported
stored graph-recovery values were treated as accepted results, as requested; I
did not re-adjudicate whether those values are scientifically “good.” Policy,
ridge, and null experiments were executed and analyzed during the revision.
Earlier runtime and memory measurements are retained only as historical
artifacts and are not part of the current paper or comparison plan. I checked
claims about how the estimator and its tests operate.

Evidence inspected:

- all LaTeX sources in `../manuscript/`;
- the correlation-profile estimator, conditioning plan, fit context, and
  linear dependence backend in `src/fastCGC/`;
- the current automated tests and stored experiment metadata;
- the original sources behind all eight bibliography entries; and
- a clean out-of-tree Tectonic build of the manuscript.

## Post-revision verification (18 October 2026)

The three blocking findings from the initial review have been resolved, and the
remaining experimental recommendations that were suitable for the current
codebase have been executed rather than left as proposed work.

| Initial issue | Resolution | Verification |
|---|---|---|
| Inactive conditional-screen claim | The manuscript now describes the precision-derived quantity as an analytic diagnostic. The public attributes are `analytic_score_` and `analytic_pvalue_`; the former `screen_*` names remain documented compatibility aliases. `screen_alpha` is explicitly a legacy no-op for the linear profile. | Regression tests compare `screen_alpha=0` and `1`, require full-policy shift provenance, and check full/short-circuit equivalence at fitted thresholds. |
| Fixed-ridge statistic overclaimed as a Gaussian conditional-independence test | The adjacency theorem is stated only for population-level conditional independence. The Gaussian common-cause counterexample is proved, tested from exact sample moments, and reported over a ridge grid. | The empirical residual correlations agree with the counterexample formula to a maximum absolute error of (2.3\times10^{-15}). |
| Package and test route not reproducible | The project now uses the standard `src/fastCGC/` layout, keeps estimator implementations in `fastCGC.core`, exposes a stable public namespace, and uses that namespace in tests, scripts, and notebooks. | 95 tests pass directly, both source and wheel distributions build, and an unpacked-wheel fit succeeds from outside the repository. |
| No controlled runtime or memory evidence | Superseded by the decision to exclude runtime, memory, hardware, and accelerator rankings from the study. | The active cross-method runner and notebooks record scientific outputs and provenance, not timing or memory endpoints. Older measurements under `data/outputs/linear-methodology/` are historical only. |
| No policy or ridge ablation | Full/short-circuit and ridge-grid studies were added, including sparse, dense, regular, and near-collinear regimes. | 64 policy rows and 200 ridge rows were analyzed. All paired policy decisions and graph supports agree at the fitted thresholds. |
| No calibration or graph-wide null evidence | Selected-pair IID and AR(1) null studies and a graph-wide AR(1) null study were added. | 1,000 selected-pair datasets and 200 graph-level datasets are stored with Wilson intervals, discrete-orbit binomial tests, and Holm adjustment. |

The earlier analysis bundle remains in
`data/outputs/linear-methodology/analysis/` for provenance. Its runtime and
memory material is not used by the current manuscript. The policy, ridge, and
null records remain relevant to implementation diagnostics, while the planned
causal-structure comparison is governed by
`experiments/benchmarks/CAUSAL_COMPARISON_PROTOCOL.md`.

## Verdict

**The review findings are resolved; no remaining correctness blocker was found.**

The claims now match the implementation and recorded artifacts. The
intersection--union argument, FFT identity, ridge inverse bound,
residual-recovery identity, Schur downdate, scale-invariance result, and
graph-theoretic population-level adjacency theorem check out under their stated
assumptions. The new experiments support only the bounded empirical claims now
made in the manuscript, and limitations are stated where the evidence stops.

The paper can move to final empirical validation. The proofs and implementation
claims have no remaining correctness blocker. The paper is framed as a
vectorized reimplementation of published c-GC/c-GC*, without a runtime or
memory claim. The open comparison concerns graph-recovery efficacy and paired
output agreement, not computing speed.

## Findings and their resolution

### 1. Resolved: the claimed conditional screen was inactive

The abstract, Introduction, notation table, Proposition 2, algorithm, complexity
analysis, experiment settings, and limitations all describe a screen at
`gamma = screen_alpha`: only candidates with `p_scr < gamma` are said to receive
ridge-residual recovery and a circular-shift test.

That is not what the current linear backend does:

- `screen_alpha` is declared and validated in
  `src/fastCGC/core/fast_gc_star.py:65` and
  `src/fastCGC/core/fast_gc_star.py:80-81`, but it is not passed to the linear
  backend.
- `src/fastCGC/dependence/config.py:81-82` resolves the default linear
  evaluation policy to `full`.
- With that policy, `src/fastCGC/dependence/linear.py:262-300` and
  `src/fastCGC/dependence/linear.py:341-418` evaluate every allowed conditional
  pair.
- The only implemented candidate reduction is the optional `short_circuit`
  policy in `src/fastCGC/core/fast_gc_star.py:230-236`, which selects pairs
  using the *marginal* test at `alpha`; it is not the claimed conditional
  precision screen.

A direct runtime check with identical data and `screen_alpha=0.0` versus
`screen_alpha=1.0` produced identical conditional p-values and identical
shift-tested masks. Under the default linear policy, every allowed pair had
`pvalue_source_ == "shift"`.

Consequences for the manuscript:

- `main.tex:44-45`, `1_intro.tex:39-41`, and `2_meth.tex:316-351` do not describe
  the current estimator.
- Proposition 2 is a correct statement about the *described* screened algorithm,
  but it has no operative role in the current default implementation.
- `gamma=0.2` in `3_results.tex:62-67` is currently inert.
- The phrase “screened statistical procedure” and the proposed unscreened control
  in `3_results.tex:174-179` are misleading: the current default linear run is
  already unscreened with respect to `screen_alpha`.
- The complexity variable `C`, currently defined as pairs that pass conditional
  screening, does not describe the code. For the default policy it is the number
  of all allowed pairs.
- The test at `tests/test_fast_causalised_gc.py:259-286` does not detect this
  regression: `screen_only` is empty under the default policy, so its last two
  assertions pass vacuously.

Recommended resolution: choose one contract and make code, tests, experiments,
and manuscript agree. Either implement `p_scr < gamma` candidate selection and
rerun the experiments, or remove the screening claim and describe the precision
quantity as a diagnostic score. If retaining the current code is preferred,
rename `C` to the number of evaluated conditional pairs and remove the claims
about screening recall, a gamma sweep, and mixed screen/shift decision values.

**Resolution.** The second contract was adopted. The default estimator remains
full evaluation, the manuscript uses `N_eval`, the diagnostic is published
under analytic names, and `screen_alpha` is tested and documented as inert.
The optional marginal-first `short_circuit` policy is described separately and
was evaluated in a paired sparse/dense ablation.

### 2. Resolved: positive-ridge residual correlation is not a Gaussian conditional-independence statistic

The manuscript correctly observes in `2_meth.tex:391-399` that the regularized
precision score and ridge-residual correlation need not coincide. It does not,
however, carry this distinction through to the causal claim.

Assumption 1(4) says that residual correlation represents the required
conditional independences, giving a “suitable linear-Gaussian model” as an
example. The abstract and Conclusion similarly connect a linear-Gaussian
assumption to the restricted adjacency result. For the implemented positive
ridge, ordinary linear-Gaussianity is insufficient.

A population counterexample is immediate. Let

\[
 Z,\varepsilon_X,\varepsilon_Y\stackrel{\mathrm{iid}}{\sim}\mathcal N(0,1),
 \qquad X=Z+\varepsilon_X,\qquad Y=Z+\varepsilon_Y.
\]

Then `X` and `Y` are conditionally independent given `Z`. After the manuscript's
standardization, ridge residualization with any `lambda > 0` gives

\[
 R_X=\frac{\varepsilon_X}{\sqrt 2}
     +\frac{\lambda}{\sqrt 2(1+\lambda)}Z,
 \qquad
 R_Y=\frac{\varepsilon_Y}{\sqrt 2}
     +\frac{\lambda}{\sqrt 2(1+\lambda)}Z,
\]

and therefore

\[
 \operatorname{corr}(R_X,R_Y)
 =\frac{\lambda^2}{(1+\lambda)^2+\lambda^2}>0.
\]

Ridge under-adjusts the common predictor and leaves shared variation in both
residuals. A very small fixed ridge may make the discrepancy numerically small
in a well-conditioned problem, but it does not restore the exact null. When the
unregularized covariance is singular, the vanishing-ridge perturbation lemma in
the manuscript does not provide an approximation to an ordinary inverse.

The graph-theoretic part of Theorem 4 is sound: for a faithful observed DAG and
a sink target, conditioning on all other observed vertices separates every
nonadjacent pair, while a direct edge remains active. The problem is the bridge
from that population-level conditional-independence statement to the implemented
positive-ridge statistic.

Recommended resolution:

- Present Theorem 4 explicitly as a **population-level CI theorem**, and state that it
  does not by itself justify the positive-ridge estimator.
- Remove “for example under a suitable linear-Gaussian model” unless the extra
  conditions that make ridge residual covariance vanish are stated.
- Narrow the abstract and Conclusion accordingly.
- If an estimator-level result is wanted, prove an asymptotic bridge, for
  example under an explicit `lambda_T -> 0` schedule, eigenvalue conditions,
  consistent residualization, and a valid post-estimation shift null. Otherwise,
  use unregularized residuals in the full-rank setting covered by the population-level
  statement.
- Add a regression/statistical test based on the Gaussian counterexample above.

The Baba--Shibata--Sibuya citation supports the ordinary Gaussian partial-
correlation statement. It does not extend that equivalence to positive-ridge
residual correlation.

**Resolution.** The theorem is now explicitly a population-level CI result. The abstract,
Method, and Conclusion deny an automatic fixed-ridge consistency bridge. The
counterexample appears in the Method, a deterministic regression test protects
it, and a separate exact-moment diagnostic evaluates the formula over six ridge
values.

### 3. Resolved: the numerical-verification claim and test command were stale

`3_results.tex:7-15` reports “14 deterministic test cases.” The current
`tests/test_fast_causalised_gc.py` collects **16** parameterized cases, and the
whole current test directory collects **59** cases.

The 59 cases pass when the current `fastC-GC` directory is temporarily exposed
under the import name `fastCGC`. However, the repository's direct documented
test route is not reproducible in its current layout:

- the tests import `fastCGC.src`, while the directory is named `fastC-GC`;
- running pytest from `fastC-GC` therefore fails during collection with
  `ModuleNotFoundError: No module named 'fastCGC'`; and
- pytest is not declared in `pyproject.toml`.

Recommended resolution: fix the package/import contract, declare the test
dependency, record the exact verification command and environment, and update
the manuscript to the relevant case count. If the claim is intended to cover
only the linear numerical identities, say “16 collected cases in
`test_fast_causalised_gc.py`” rather than referring ambiguously to the now much
larger suite.

**Resolution.** The linear module now has 23 cases and the full suite has 66.
The manuscript reports both counts. The documented command is
`uv run --extra dev pytest`; it succeeds without a temporary alias. Package
build and isolated-wheel smoke tests also succeed.

## Proof-by-proof assessment

### Proposition 1: intersection--union interpretation

**Correct.** `max(p^U,p^C)` is super-uniform under the union null when each
component p-value is valid for its own null. The unequal-threshold bound by
`max(alpha,beta)` is also correct and requires no independence between the two
tests. The manuscript appropriately warns that this logic does not validate the
component tests.

### Theorem 1: FFT evaluation of the circular orbit

**Correct as an algebraic identity.** The sign convention can reverse the shift
index, but the multiset of absolute correlations and the resulting tail count
are unchanged. The manuscript also correctly separates computational identity
from randomization-test validity.

The Harris citation supports conservative shift testing under stationarity
conditions, but it does not establish validity of this manuscript's complete
wrap-around orbit. The current Remark 1 already makes that distinction; retain
it.

### Lemma 1: ridge existence and perturbation bound

**Correct.** The spectral-norm bound follows directly from the eigendecomposition.
The positive-definiteness claim uses the positive ridge properly.

### Proposition 2: affine scale invariance

**Correct in exact arithmetic, provided the set of rows classified as
nonconstant does not change.** The proof with the diagonal sign matrix is sound.
The implementation, however, uses a magnitude-dependent floating-point
tolerance in `src/fastCGC/dependence/context.py:44-50`. Extreme rescalings or
offsets can change the validity mask or lose variation numerically. Qualify the
proposition as an exact-arithmetic result, or explicitly assume unchanged
nonconstant-row classification.

### Former Proposition 3: decision provenance under screening

**Removed.** The result was logically correct for the hypothetical screened
algorithm, but it did not describe the implementation. The revised manuscript
instead states and tests the narrower short-circuit contract: marginally
rejected pairs may skip conditional evaluation without changing the conjunction
at the fitted marginal threshold.

### Theorem 2: ridge-residual block identity

**Correct.** The block-inverse identity gives the stated residual recovery, and
the endpoint precision block is invertible because it is a principal block of a
positive-definite matrix.

### Theorem 3: precision after deleting a principal set

**Correct.** This is the appropriate Schur complement identity, with the
required principal blocks invertible under the positive ridge.

### Theorem 4: population-level adjacency for a present sink

**Correct as a graph-theoretic population-level statement under the stated observed-DAG,
Markov, faithfulness, and sink assumptions.** With all other vertices
conditioned on, a non-direct active path would need every internal vertex to be
a collider. Adjacent internal colliders are impossible, leaving only a
length-two collider path, which the sink assumption excludes.

Two qualifications should be explicit:

1. the theorem is about population marginal and conditional independence, not
   automatically about the implemented ridge statistic or its shift p-value;
2. the “DAG over the expanded window” must really be the graph to which the
   observed-window distribution is Markov and faithful. Unrepresented ancestors
   outside the window cannot silently act as latent common causes.

## Other claim and notation corrections

All six corrections below were applied in the revised manuscript.

1. **Undefined weight for an absent edge.** Equation (10) takes a maximum over an
   empty set when no lag is retained. The code returns zero. Define `W_ab = 0`
   when `Ahat_ab = 0`.

2. **Overloaded `C`.** The notation table uses `C` for the number of screened
   pairs, while `C_ij` is the conditional effect size and `C` is also used as a
   collider in the adjacency proof. Rename the count, for example to
   `N_eval` or `N_cond`.

3. **“Causal decision rule” is premature.** In `1_intro.tex:18`, “two-test
   decision rule” is more accurate. Causal interpretation is conditional on the
   later assumptions.

4. **Clarify “causal order.”** Assumption 1(1) should say “maximum causal lag is
   no larger than `p`” if that is what is intended.

5. **Post-estimation shift null.** Assumption 1(5) should refer specifically to
   the joint null law of the statistic actually shifted, including estimated
   ridge residuals. Ordinary stationarity of the raw process is not enough.

6. **Complexity statements depend on the chosen evaluation policy.** If
   `short_circuit` remains available, define `C`/`N_eval` in terms of that policy
   and state that the default linear policy currently evaluates all allowed
   pairs.

## Citation audit

All seven cited works exist, and the bibliographic details for Granger, Harris,
Phipson--Smyth, Baba--Shibata--Sibuya, Ledoit--Wolf, and
Benjamini--Yekutieli are consistent with their original records.

One entry needed updating:

- `main.tex:68-73` cites `arXiv:2501.02672v3`. The arXiv record is now v4
  (30 July 2026) and links the published article DOI
  `10.1142/S0219525926500037`. Cite the journal version if that is the version
  being relied on, or at least update the arXiv version. The current v4 source
  confirms the c-GC and c-GC* conditioning sets used in this manuscript.

Useful primary records checked during this review:

- source paper: <https://arxiv.org/abs/2501.02672>
- Harris shift test: <https://arxiv.org/abs/2012.06862>
- Baba et al.: <https://doi.org/10.1111/j.1467-842X.2004.00360.x>
- Ledoit--Wolf: <https://doi.org/10.1016/S0047-259X(03)00096-4>
- Phipson--Smyth: <https://doi.org/10.2202/1544-6115.1585>
- Benjamini--Yekutieli: <https://doi.org/10.1214/aos/1013699998>
- Granger: <https://doi.org/10.2307/1912791>

The proof of Theorem 4 is self-contained, but the manuscript should still cite a
standard primary treatment of d-separation, the causal Markov condition, and
faithfulness when introducing those assumptions. Do not use the source c-GC
paper as the only authority for foundational graphical-model terminology.

**Resolution.** The manuscript cites the published 2026 journal article rather
than the stale arXiv version and adds Pearl's standard treatment for the
graphical-model assumptions. No unverified reference was added.

## Writing review

The prose is already strong. It is direct, technically restrained, and notably
free of promotional language, vague field-wide attribution, canned transitions,
or generic claims of significance. The limitations are specific rather than
formulaic. I did not find a meaningful cluster of AI-writing patterns. No broad
humanization rewrite is warranted.

Particularly good choices to preserve:

- algebraic equivalence is kept separate from statistical validity;
- “complete” and “exact” are carefully limited to orbit enumeration;
- the smooth disturbance is not mislabeled as latent confounding;
- descriptive comparisons are not presented as prespecified hypothesis tests;
- the scope of the new timing, memory, calibration, and graph-wide null evidence
  is stated plainly; and
- the Conclusion introduces no new empirical evidence.

Final writing observations:

- The abstract now gives the population-level nature of the causal result in one unambiguous
  clause and distinguishes the graph-recovery, runtime, selected-pair, and
  graph-wide evidence.
- Avoid switching among “result,” “decision value,” “test,” and “p-value” unless
  the distinction is intentional. `pvalue_source_` is useful precisely because
  not every stored quantity is calibrated; state that contract once and reuse
  it consistently.
- The Method is long relative to the rest of the paper, but its subsections are
  well ordered. No separate background section is needed.

## Build and verification record

- `uv run --extra dev python -m compileall -q src/fastCGC experiments tests`: passed
  under CPython 3.14.7.
- `uv run --extra dev pytest -q`: 95 passed under CPython 3.14.7, including 23
  cases in `tests/test_fast_causalised_gc.py`.
- Package-layout checks confirm that `src/fastCGC/core/`, `dependence/`, and
  `experiments/` are included in the wheel. Canonical and compatibility imports
  resolve to the same estimator classes, and the obsolete dynamic loader and
  hyphenated module are absent.
- Causal-comparison boundary check: 26 focused tests passed. An efficacy-only
  smoke run wrote two method rows with structural and ranking metrics and no
  elapsed-time, memory, or hardware fields.
- The project now requires CPython 3.14. The ten simulation notebooks live in
  the top-level `notebooks/` directory and include Python and package versions
  in a software fingerprint while leaving hardware out of the comparison
  contract. Existing Intel/Python 3.13 results remain historical artifacts.
- The published c-GC checkout and its environment are deliberately excluded
  from this repository. The notebook loader pins and verifies the upstream URL
  and commit, while each run records the external environment and platform.
  Native import and fit smoke tests must still be run on the M5 machine before
  full experiments begin.
- Conventional c-GC/c-GC* notebooks now load `GcStar` from the authors'
  repository at commit
  `575ac1dc67a0282a5ca88bc11a2804df052f674d`. Their shared datasets are
  hash-verified, their native arrays are preserved, and completed fits resume
  from provenance-checked checkpoints. No published estimator code is copied
  into this repository.
- `uv build`: produced both `fastcgc-0.1.0.tar.gz` and
  `fastcgc-0.1.0-py3-none-any.whl`.
- Unpacked-wheel smoke test from `/tmp`: imported `fastCGC`, fitted
  `FastGcStar`, and returned a (3\times3) connectivity matrix.
- Controlled experiment bundle: 64 policy rows, 200 ridge rows, six exact-moment
  counterexample rows, 1,000 selected-pair null rows, and 200 graph-wide null
  rows. Older runtime and memory records remain historical and are not evidence
  for the current manuscript.
- `tectonic main.tex`: passed after the framing revision; generated a 15-page
  PDF with no LaTeX warning in the final run.

## Remaining empirical work and extensions

1. **Run the definitive published-versus-vectorized paired comparison.**
   Pair the pinned authors' c-GC code with fastC-GC and the pinned c-GC* mode
   with fastC-GC* on identical hash-verified datasets. Report the prespecified
   graph-recovery metrics, exact-lag agreement at order one, aggregate graph
   agreement at every order, and traceable score or decision-value
   disagreements. Because the published and vectorized implementations differ
   in standardization, ridge regularization, and shift enumeration, exact
   numerical equality is not assumed. Runtime, memory, hardware, and
   accelerator rankings are outside this comparison.
2. **Comparison study now specified and updated from recent literature (20
   October 2026).** The headline external CSL methods are PCMCI+, Bagged PCMCI+,
   and CUTS+. None is implemented in this repository. Published c-GC/c-GC* and
   every external CSL or classical control are run from pinned author or
   upstream repositories in separate environments. The local runner evaluates
   only fastC-GC/fastC-GC* and exports method-neutral datasets with hashes
   and truth arrays. CUTS+ belongs in the any-lag aggregate table because its
   official implementation returns an aggregate variable graph. The 2026
   AISTATS Stationarity-Aware Discovery method is the newest same-estimand
   candidate found, but its advertised official repository was empty when
   checked, so it is pending. DYNOTEARS is historical-only, not a
   state-of-the-art headline comparator. LPCMCI remains in a separate PAG
   track. The prospective protocol and local runner are in
   `experiments/benchmarks/CAUSAL_COMPARISON_PROTOCOL.md` and
   `experiments/benchmarks/benchmark_causal_methods.py`. Exact positive-lag
   recovery is primary for lag-resolved methods; any-lag recovery is the common
   cross-family endpoint for CUTS+ and is reported separately. Self-lags also
   remain separate. Metrics include AUPRC (primary ranking metric), AUROC
   (secondary), precision, recall, F1, MCC, specificity, balanced accuracy, FPR,
   FDR, normalized SHD, false positives per target, success rate, failure-aware
   F1, and failure-aware summaries. AUROC/AUPRC are computed only from
   continuous scores or a prespecified threshold path, never from one binary
   adjacency. The comparison deliberately excludes elapsed time, memory use,
   hardware rankings, and accelerator comparisons. Comparative manuscript
   claims remain pending until held-out seeds and the paired analysis are run.
3. Extend null calibration to the notebook threshold \(\beta=0.001\). The
   current 500-dataset regimes are intentionally reported only at 0.01 and
   0.05; a substantially larger independent run is needed at 0.001.
4. Specify and evaluate a graph-wide multiplicity procedure rather than relying
   on fixed edgewise thresholds. The present graph-wide null quantifies the
   unadjusted rule but does not change it.
5. Add external-validity regimes with shared latent causes, nonlinear links,
   heavy-tailed innovations, abrupt changes, and missing observations.
6. If an estimator-level causal consistency result is desired, supply a formal
   vanishing-ridge and shift-validity argument. The current manuscript correctly
   limits its theorem to a population-level conditional-independence statement.
7. Reduce the dense per-signature Schur cache only if memory use later becomes
   a separate implementation objective; it is not part of the current study.

Items 3--7 are extensions and are not needed for the claims currently made.
Items 1 and 2 remain required before making comparative efficacy or agreement
claims. They use pinned source, identical datasets, and paired analysis. The
manuscript makes no speed or memory claim, and none should be inferred from the
algebraic proofs or from the `fastC-` name.
