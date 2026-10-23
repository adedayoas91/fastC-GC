# Causal-structure comparison protocol

## Purpose and current status

This protocol governs the comparison of fastC-GC and fastC-GC* with the
conventional c-GC/c-GC* procedures and established time-series causal-structure
methods. It is a prospective protocol, not a results section. Pilot runs may be
used to find software or data-generation errors, but confirmatory settings must
be frozen before held-out seeds are run. Comparative claims should enter the
manuscript only after the held-out analysis is complete.

The local benchmark runner is `benchmark_causal_methods.py`. It writes the
shared datasets and their truth arrays, then evaluates only fastC-GC and
fastC-GC*. Every comparison method, including conventional c-GC/c-GC*, is run
from a pinned checkout of its authors' repository. No comparison estimator is
reimplemented in this repository.

## Methods

There is no defensible single state-of-the-art method across all time-series
causal-discovery regimes. The 2025 CausalDynamics benchmark reports substantial
changes in method ranking across its data-generating regimes. The shortlist
therefore uses recent evidence and complementary assumptions rather than venue
date alone.

The three headline external comparators are PCMCI+, Bagged PCMCI+, and CUTS+.
PCMCI+ provides a maintained, lag-resolved constraint-based reference. Bagged
PCMCI+ is the 2024 bootstrap-aggregation extension, which improved precision and
recall over PCMCI+ in the reported difficult regimes. CUTS+ is the recent neural
comparator selected for its high-dimensional and irregular-sampling results in
the 2024 paper and CausalTime benchmark.
These methods do not all return the same graph object, so they are not forced
into one leaderboard. Their repositories remain unchanged; method-specific
post-processing is limited to converting saved native outputs into the declared
graph orientation and never performs fitting or edge selection.

The 2026 AISTATS Stationarity-Aware Discovery method based on minimal separating
sets is the most recent same-estimand candidate found in the review. Its paper
reports the highest F1 and lowest SHD in its experiments, but the official
repository linked by the paper was empty when checked on 20 October 2026. It is
therefore recorded as pending, not reimplemented from pseudocode or reported as
tested. DYNOTEARS is retained only as an optional historical sensitivity
analysis; it is not described as state of the art.

The main exact-lag table contains methods that estimate directed positive-lag
edges under a common observed-variable graph. A method's assumptions remain
part of the interpretation; inclusion does not imply that all methods target
the same graph outside this controlled setting.

| Method | Family | Main role | Benchmark status |
|---|---|---|---|
| c-GC | two-test, constraint-based | Conventional reference procedure | Authors' `causalized-GC` checkout at `575ac1dc67a0282a5ca88bc11a2804df052f674d` |
| c-GC* | two-test, constraint-based | Conventional full-conditioning reference | Same pinned checkout; native `fcgc` mode |
| fastC-GC | two-test, constraint-based | Vectorized c-GC, correlation profile | Local study-method runner |
| fastC-GC* | two-test, constraint-based | Vectorized full-conditioning c-GC*, correlation profile | Local study-method runner |
| Full VAR-GC | regression and coefficient testing | Classical linear predictive baseline | Pinned upstream implementation; no local estimator code |
| Lasso-VAR | penalized regression | Optional sparse linear control | Pinned upstream implementation; no local estimator code |
| PCMCI+ with ParCorr | constraint-based | Autocorrelation-aware causal discovery | Pinned Tigramite checkout; positive lags only in the exact-lag track |
| Bagged PCMCI+ with ParCorr | bootstrap-aggregated constraint-based | Recent lag-resolved comparator | Pinned Tigramite checkout using its native bootstrap routine |
| CUTS+ | neural Granger / graph neural network | Recent high-dimensional neural comparator | Authors' pinned UNN checkout; aggregate variable graph only |
| Stationarity-Aware Discovery (MSSD) | constraint-based minimal separating sets | Newest same-estimand candidate | Pending; advertised official repository was empty on 20 October 2026 |
| VAR-LiNGAM | non-Gaussian functional model | Directional non-Gaussian baseline | Pinned authors' checkout; primary interpretation on Laplace-innovation data |
| DYNOTEARS | continuous score-based | Historical sensitivity analysis | Not a headline comparator; discontinued canonical CausalNex implementation requires a frozen legacy environment |
| LPCMCI with ParCorr | latent-variable constraint-based | Hidden-common-cause stress test | Separate PAG track; not ranked as a fully directed DAG |

PCMCI may be included as a historical secondary baseline. Alternative PCMCI+
conditional-independence tests, such as GPDC or CMIknn, may be added in a
separately labelled sensitivity analysis. Static PC or GES on raw observations are excluded:
without an explicit lag expansion they solve an IID graph problem, not the
positive-lag time-series problem. Lag-expanded PC/GES may be reported as a
sensitivity analysis if time order is enforced and overlapping-window
dependence is stated.

Neural Granger cLSTM/cMLP and TSCI remain extension candidates rather than
headline methods. Neural Granger overlaps with CUTS+'s predictive-causality
family, while TSCI is designed for deterministic dynamical systems where
Granger-style assumptions may be inappropriate. F-PCMCI is a useful
feature-filtered PCMCI extension, but it is secondary to the maintained PCMCI+
and Bagged PCMCI+ comparison unless a high-dimensional feature-selection regime
is added.

Primary records:

- c-GC/c-GC*: [published method and algorithm](https://doi.org/10.1142/S0219525926500037) and [authors' source](https://github.com/adedayoas91/causalized-GC)
- classical Granger causality: [Granger (1969)](https://doi.org/10.2307/1912791)
- PCMCI+: [Runge (2020)](https://proceedings.mlr.press/v124/runge20a.html) and [Tigramite documentation](https://jakobrunge.github.io/tigramite/)
- Bagged PCMCI+: [Debeire et al. (2024)](https://proceedings.mlr.press/v236/debeire24a.html)
- CUTS+: [Cheng et al. (2024)](https://ojs.aaai.org/index.php/AAAI/article/view/29034) and [authors' source](https://github.com/jarrycyx/UNN)
- Stationarity-Aware Discovery: [Gao et al. (2026)](https://proceedings.mlr.press/v300/gao26a.html) and [advertised source repository](https://github.com/CausalML-Lab/StationarityAwareDiscovery)
- recent cross-method evidence: [CausalTime (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/hash/0c79d6ed1788653643a1ac67b6ea32a7-Abstract-Conference.html) and [CausalDynamics (NeurIPS 2025)](https://proceedings.neurips.cc/paper_files/paper/2025/hash/39a3f7dc00e4723a5ca7808b04fc4d4f-Abstract-Datasets_and_Benchmarks_Track.html)
- empirical time-series data: [Causal Chambers (Nature Machine Intelligence, 2025)](https://doi.org/10.1038/s42256-024-00964-x) and [CausalRivers (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a205fda871b0f6c1e18a7ad7325eb6cf-Abstract-Conference.html)
- DREAM3 in-silico challenge: [Prill et al. (2010)](https://doi.org/10.1371/journal.pone.0009202)
- VAR-LiNGAM: [Hyvärinen et al. (2010)](https://www.jmlr.org/papers/v11/hyvarinen10a.html) and [official API](https://lingam.readthedocs.io/en/stable/reference/var_lingam.html)
- DYNOTEARS, historical only: [Pamfil et al. (2020)](https://proceedings.mlr.press/v108/pamfil20a.html)
- LPCMCI: [Gerhardus and Runge (2020)](https://proceedings.neurips.cc/paper_files/paper/2020/hash/94e70705efae423efda1088614128d0b-Abstract.html)

## External repository policy

For every comparison method:

1. clone the repository named in the method table and record its remote URL and
   exact commit hash;
2. create an isolated environment from that repository's dependency files, or
   from a recorded compatibility lock when those files target a different
   architecture or unsupported Python release; the lock may update
   dependencies but must not change the estimator source;
3. run the repository's native entry point on the exported `.npz` dataset, or
   on a lossless CSV copy if that is the repository's supported input;
4. preserve the command, standard output, standard error, software versions,
   and unmodified native result files;
5. convert the saved result to `lag, source, target` only after execution. The
   converter may transpose, select documented output fields, and aggregate lags
   when required, but it must not fit a model, apply a new statistical test, or
   infer missing scores.

For conventional c-GC/c-GC*, the notebook imports `GcStar` from the pinned
`src/d_CSL.py`; it does not import the local `FastGcStar` under another
name. The upstream connectivity method combines positive lags with a logical
OR. Its output is therefore an exact-lag result only when the fitted order is
one. For larger orders it is evaluated in the aggregate any-lag table. The
lag-specific upstream p-value arrays are retained as native output but are not
re-thresholded locally to construct a different estimator.

If an upstream repository cannot ingest the common dataset without changing its
estimator, use its native generator with matched dimensions, sample size, lag,
noise family, and graph density. Such results are an unpaired sensitivity
analysis and are not mixed with the paired common-dataset table.

## Shared notebook datasets

The cross-method notebooks are organized first by data source. Controlled and
in-silico data are under `notebooks/simulations/`: sparse VAR, Lorenz--96, the
autocorrelated structural causal process, and DREAM3. DREAM3 is generated in
silico with GeneNetWeaver and is not classified as a real-world dataset.
Measured data are under `notebooks/empirical/causalChamber/` and
`notebooks/empirical/causalRivers/`. Every directory contains one preparation
notebook and one notebook for each of the seven runnable methods. Simulation
directories end with a scoring notebook; empirical directories end with an
edge-collection and graph-visualization notebook. All method notebooks in a
directory consume the same saved bundles and never generate method-specific
observations.

Each preparation notebook uses one saved bundle per data-generating setting,
seed, measured run, or measured subgraph. A controlled or in-silico bundle
contains the series and either exact lag-resolved truth or the declared
aggregate graph. An empirical bundle contains the series and may include a
separately labelled physical or constructed reference network. Its adjacent
JSON file records the source, preparation settings, array layouts, identifiers,
and SHA-256 digest.

The first notebook that requests an absent bundle writes its data and metadata
through atomic file replacement.
Subsequent notebooks load it and verify its digest and generation settings.
They fail on a missing metadata file, corruption, or a conflicting seed/range
request instead of overwriting the dataset. Thus every CSL method sees the same
stored observations, rather than merely attempting to reproduce them from a
similar simulation cell.

## Common estimand and graph convention

The primary estimand is the exact positive-lag, cross-variable graph. Every
saved result is converted after execution to the canonical order
`lag_adjacency[lag, source, target]`. The simulator's regression coefficients
use `coefficients[lag, target, source]`; the explicit transpose in the shared
data object prevents an orientation error from being hidden in method-specific
scoring code.

- Exact-lag recovery is primary among methods that return lag-resolved graphs.
- Any-lag aggregate recovery is the common cross-family endpoint when CUTS+ is
  included. It is reported in a separate table and never presented as evidence
  of exact-lag recovery.
- Autoregressive self-links are scored separately.
- Contemporaneous links form a separate experiment and are not silently merged
  into the positive-lag graph.
- PAG outputs from latent-variable methods retain their endpoint marks and use
  PAG-appropriate adjacency/orientation scores rather than forced directions.

## Data-generating regimes

The main controlled study uses stable sparse VAR systems generated from support
first, followed by signed coefficient draws and companion-matrix rescaling to a
spectral radius no larger than 0.8. The initial confirmatory grid is:

- variables: 5, 10, and 20;
- observations: 250, 500, and 1,000 after burn-in;
- true and fitted maximum lags: 1, 2, and 5;
- innovations: Gaussian and variance-matched Laplace;
- at least 50 paired seeds per cell.

The Gaussian-innovation, identity-link cells are the sparse-VAR reference.
Laplace innovations test sensitivity to the innovation distribution and are
also the assumption-matched cells for VAR-LiNGAM. Known lag order and lag-order
selection must be reported separately.

The second controlled family adopts the additive autocorrelated structural
causal process from Debeire et al. (2024):

```text
X_t^j = a_j X_{t-1}^j + sum_i c_i f_i(X_{t-tau_i}^i) + eta_t^j.
```

The implementation calls `generate_random_contemp_model` and
`generate_nonlinear_contemp_timeseries` from the pinned authors' repository at
`8e16fb89b8f3b4db47ff57f2bd09dcd5c9f7e4c9`; the generator is not copied into
this repository. The reference cell uses 10 variables, 500 observations,
maximum lag 5, maximum autoregressive coefficient 0.6, identity link functions,
Gaussian innovations, and `floor(1.5 N)` cross-links. One-factor cells vary the
variable count (5 or 20), observation count (250 or 1,000), maximum
autoregressive coefficient (0.2 or 0.95), link-function set, or innovation
family. A cross-link-free cell estimates false-positive behavior when the
cross-variable graph is empty. Each cell has 50 requested seeds. Coefficients,
innovation scales, accepted model seeds, and stationarity rejections are
retained with the data.

The source study includes a 30% contemporaneous-link setting. The matched
seven-method comparison uses `contemp_fraction=0.0`, because the methods do not
all return a compatible contemporaneous graph. This deviation is recorded in
every bundle. A source-faithful contemporaneous experiment, if run, belongs in
a separate table with method-specific estimands and must not be pooled with the
positive-lag comparison.

Lorenz-96 supplies a further dynamical-system stress test. The preparation
notebook records the forcing, integration step, noise scale, burn-in, seed, and
aggregate structural graph rather than classifying the process by a broad
linear/nonlinear label.

DREAM3 Size 100 supplies five in-silico gene-regulatory systems generated with
GeneNetWeaver. The aggregate regulatory network is the recovery target. The
data are grouped with the other generated processes, not with the measured-data
case studies.

## Empirical case studies

The first measured-data study uses the `wt_walks_v1` wind-tunnel recordings
from Causal Chambers. The source-faithful primary bundle uses the
`actuators_random_walk_1` experiment, the 16 variables selected in the authors'
time-series causal-discovery case study, per-variable min--max normalization,
and maximum fitted lag 10. The physical causal model distributed by the
project is stored as a reference network. It is displayed separately from the
seven inferred graphs and is not required for running the collector.

The second study uses CausalRivers discharge measurements from eastern Germany
and Bavaria. It applies the authors' connected-subgraph routine to five-station
graphs and selects five subgraphs per region deterministically from the sorted
candidate list. The time series follow the repository's default preparation:
6-hour rounded means, per-station min--max normalization, linear interpolation,
and removal of incomplete end rows. The supplied river-connectivity subgraph is
stored as a contextual reference. It is not treated as an exhaustive
lag-resolved statistical causal graph.

Both empirical collectors write a method-by-method edge table, a run-status
table, an optional reference-edge table, and directed graph panels with common
node positions. They do not compute graph-recovery metrics. This rule also
applies to a future measured dataset with no defensible reference graph: the
reported result is the inferred network, together with method settings and
provenance.

## Metrics for controlled and in-silico recovery

No single metric is sufficient for sparse directed graphs. The following set is
recorded for exact-lag and any-lag recovery on simulated and in-silico datasets
unless stated otherwise. These metrics are not mandatory outcomes for the two
empirical case studies.

### Ranking metrics

- **AUPRC (primary ranking metric):** emphasizes performance on the rare edge
  class and is reported with the edge prevalence/random baseline.
- **AUROC (secondary ranking metric):** retained because it is widely reported,
  but interpreted cautiously under severe class imbalance.

AUROC and AUPRC require continuous edge scores. A single thresholded adjacency
is never reused as a two-level “score.” If a method exposes only decisions, a
prespecified parameter path must be run or the ranking metrics are reported as
not available. Scores must have the same direction for every method (larger
means stronger evidence for an edge).

### Thresholded graph metrics

- precision, recall, and F1;
- Matthews correlation coefficient;
- specificity, balanced accuracy, false-positive rate, and false-discovery
  rate;
- structural Hamming disagreement and normalized SHD;
- false positives per target;
- exact-lag accuracy and lag error where a unique lag comparison is defined;
- success rate and failure-aware F1, with a failed run contributing zero to the
  latter while success-only results remain available.

For PAG experiments, report skeleton precision/recall/F1, adjacency precision
and recall, endpoint-mark accuracy, and orientation precision/recall. Do not
coerce circles or bidirected marks to directed edges merely to reuse DAG
metrics.

### Calibration and multiplicity

Methods returning p-values should additionally report null false-positive
rates, p-value histograms or empirical CDFs, and family-wise or false-discovery
behavior under the declared graph-wide correction. Unadjusted edgewise and
multiplicity-adjusted results must be distinguishable. Thresholds and
regularization grids are selected on development/validation seeds only.

## Statistical comparison

For controlled and in-silico recovery, the dataset/seed is the paired
experimental unit. The notebook tables report
the mean and sample standard deviation across seeds for each method and cell.
The confirmatory analysis may also report the median and a 95% cluster-bootstrap
interval. Planned pairwise method contrasts use a paired permutation or
randomization test on the seed-level metric differences, with effect sizes and
Holm correction across the declared family of comparisons. Do not treat
individual edges as independent replicates.

AUPRC, AUROC, F1, MCC, and normalized SHD should be compared on the same paired
controlled or in-silico datasets. The empirical graph panels are descriptive
case studies and do not enter these paired tests.

Each simulation or in-silico directory ends with
`90-collect-and-score.ipynb`. The notebook
uses the shared tested scoring module to validate input and output hashes,
retain explicit missing/failed rows, and write both run-level measurements and
cell-level mean (sample SD) summaries. The summaries include attempted,
completed, and failed counts. Exact-lag and aggregate-any-lag estimates remain
in distinct rows, and failure-aware F1 assigns zero to every attempted fit that
fails. Fits that have not yet been attempted are counted as missing rather than
as failures.

Each empirical directory instead ends with
`90-collect-and-visualize.ipynb`. It uses the shared tested collector to
validate hashes, preserve missing and failed fits, export inferred edges, and
draw one directed graph per method. An available reference network is plotted
in its own panel and is not silently converted into a recovery leaderboard.

## Execution and failure protocol

- one fresh subprocess per method/dataset pair;
- deterministic seeds where the API permits them;
- identical hash-verified data for all methods;
- prespecified timeout and an explicit failure row rather than dropping the
  dataset;
- convergence warnings, Python and package versions, and source revisions
  retained in raw artifacts.

Elapsed time, memory use, hardware rankings, and accelerator comparisons are
outside this study. They are not collected or reported as comparison outcomes.

Long notebook runs checkpoint after every completed dataset fit. A checkpoint
is reusable only when its
dataset digest, method settings, upstream commit, and software fingerprint match
the current run.
Failures remain explicit records and are retried on the next run. Native arrays
are saved separately with their own SHA-256 digest. A user can cap the number
of new fits in one session or interrupt a fit and rerun the cell. Because the
conventional c-GC code has no internal checkpoint API, resumption occurs between
fits, not inside a permutation calculation.

Data-generation seeds and estimator seeds are recorded separately. The
conventional implementation uses NumPy randomness within concurrent Numba code
and does not accept a random-state object. Setting the recorded wrapper seed is
best-effort control; deterministic estimator output is not assumed until a
repeatability check establishes it.

The conventional c-GC/c-GC* thresholds (`alpha=0.01`, `beta=0.001`) and permutation
count (`N=1000`) are retained as a source-faithful arm. The finite resolution of
permutation p-values is reported explicitly. A separately labeled validation-
tuned arm may use a common threshold grid, but no setting may be chosen using
held-out test seeds.

## Software environment

The comparison orchestration and core package target CPython 3.14. The
conventional c-GC/c-GC* checkout remains at its recorded commit and runs in an
isolated environment created from the authors' repository. The checkout and
its environment are local inputs rather than vendored project files. Their
resolved package versions and platform details are recorded with each run.

Other external repositories retain isolated environments. A method that cannot
run unmodified on Python 3.14 is either reported in a clearly labeled exception
environment or excluded; it is never silently patched or presented as a native
Python 3.14 result. DYNOTEARS is historical-only because the canonical CausalNex
implementation is discontinued. MSSD remains pending until a non-empty official
source release can be pinned and verified.
