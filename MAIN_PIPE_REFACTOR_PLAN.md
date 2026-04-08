# Main Pipe Refactor Plan

## Goal

Refactor the current monolithic `main_pipe.py` into two additive, paper-aligned pipelines:

- `main_exemplars.py`
- `main_QA.py`

without rewriting the current working pipeline in place.

The first objective is structural cleanup and figure-quality improvement, not methodological change.

## Why This Refactor Is Worth Doing

The current state is workable but too coupled:

- `main_pipe.py` is a single 5742-line orchestrator mixing loading, filtering, statistics, QA modeling, cohort analysis, exemplar selection, and figure generation.
- `production_plots.py` is a 15485-line mixed plotting module containing both old and newer production-style figures.
- The GPR lane already demonstrates the architecture we want more of:
  - centralized runtime knobs
  - clean plot defaults
  - explicit plot label control
  - vector-first export defaults
  - reusable figure helpers

This makes an additive split both feasible and safer than editing the existing main pipeline directly.

## What Belongs To Each New Lane

### `main_QA.py`

This lane should own outputs used by the probabilistic QA paper.

Primary figure families confirmed from the QA paper repo:

- cohort pooled voxel histograms
- cohort DVH boxplots
- predictor-vs-delta cohort figure(s)
- Path-1 QA summary figure
- Path-1 margin-only probability figure
- Path-1 best-secondary generalized figure
- cohort dose / gradient absolute boxplot summaries
- cohort upper/lower dual-triangle summary heatmap

Core responsibilities:

- load cohort-level inputs
- build DVH metrics and threshold QA tables
- build path-1 design tables
- fit / summarize path-1 models
- write QA-facing CSVs
- write QA-facing production figures

### `main_exemplars.py`

This lane should own patient-specific and biopsy-specific outputs used by the exemplars paper.

Primary figure families confirmed from the exemplars paper repo:

- per-biopsy axial dose profile figures
- per-biopsy axial gradient profile figures
- cumulative DVH paired figures
- per-voxel summary delta overlays
- per-trial delta boxplots
- voxel-pair length-scale summary figures
- signed/absolute upper-lower voxel-pair heatmaps
- ridgeline figures

Core responsibilities:

- load cohort inputs plus patient-specific trial-level inputs
- select exemplar biopsy pairs from config
- build patient-specific summary tables
- write exemplar-facing CSVs
- write exemplar-facing production figures

## Shared Layer To Introduce

The most important new shared layer is not a giant utility file. It is a small, explicit data-loading/config layer.

Suggested modules:

- `pipeline_shared_config.py`
  - dataclasses or structured dictionaries for paths, cohort filters, export defaults, figure typography, exemplar selections
- `load_data_shared.py`
  - common path resolution
  - common simulation filtering
  - common CSV/parquet loading helpers
- `load_data_QA.py`
  - QA-only tables and derived design tables
- `load_data_exemplars.py`
  - exemplar-only patient-specific and trial-level tables

Optional alternative:

- if the overlap between QA and exemplars is smaller than it first appears, use only:
  - `load_data_QA.py`
  - `load_data_exemplars.py`

and keep shared helpers minimal.

## Plotting Layer To Introduce

Add new modules instead of trying to clean the whole of `production_plots.py` in one pass:

- `production_plots_QA.py`
- `production_plots_exemplars.py`

These should borrow the good conventions already present in `GPR_production_plots.py` and the newer sections of `production_plots.py`:

- STIX typography defaults
- vector-first export (`pdf` default; optional `svg`, `png`)
- centralized axis / tick / legend font controls
- consistent annotation box styling
- reusable save helper
- optional custom biopsy display labels
- optional anonymized labels such as `Biopsy A`, `Biopsy B`

## First-Pass Design Constraints

To keep this safe:

- do not delete or rewrite `main_pipe.py` initially
- do not break existing filenames until comparison is complete
- do not change the underlying math unless a bug is found
- keep current outputs reproducible in parallel with the new lane outputs
- prefer wrapping existing computations over re-implementing them

The first pass is a re-orchestration and plotting cleanup pass.

## Proposed Phases

### Phase 0: Freeze Current Behavior

Deliverables:

- map current figure and CSV dependencies for both papers
- identify exact exemplar biopsy selections currently used
- identify current filenames used in manuscripts

Acceptance:

- we have a definitive lane map of "QA outputs" versus "exemplar outputs"

### Phase 1: Shared Config And Data Loading

Deliverables:

- `pipeline_shared_config.py`
- `load_data_shared.py` or minimal shared helpers
- `load_data_QA.py`
- `load_data_exemplars.py`

Acceptance:

- both new lanes can obtain the required data without importing `main_pipe.py`
- path choices, filters, and exemplar selections are configurable from one place

### Phase 2: New Orchestrators

Deliverables:

- `main_QA.py`
- `main_exemplars.py`

Acceptance:

- each script has explicit runtime knobs similar to `main_pipe_GPR_analysis.py`
- each script can run independently
- each script writes to its own output subtree

### Phase 3: QA Production Plot Module

Deliverables:

- `production_plots_QA.py`
- wrappers or migrated implementations for:
  - Path-1 QA summary
  - p-pass vs margin by metric
  - best-secondary logistic family figure
  - cohort summary figures needed by the QA paper

Acceptance:

- default export format is `pdf`
- font sizes are user-configurable
- annotation boxes use a consistent production style
- titles and biopsy labels are configurable

### Phase 4: Exemplar Production Plot Module

Deliverables:

- `production_plots_exemplars.py`
- wrappers or migrated implementations for:
  - paired exemplar profile figures
  - cumulative DVH figures
  - delta overlays
  - dual boxplots
  - voxel-pair heatmaps
  - length-scale plots
  - ridgeline plots

Acceptance:

- exemplar figure creation is driven by a clean biopsy-selection config
- `Biopsy A` / `Biopsy B` headings are supported by default, with override options
- multi-format export is supported with `pdf` as default

### Phase 5: Comparison And Migration

Deliverables:

- side-by-side old/new output comparison for target figures
- filename mapping table for manuscript updates
- brief migration notes

Acceptance:

- target figures visually improve without changing underlying data unexpectedly
- manuscript figure updates can be done by swapping file paths, not manual redesign

## Fastest Safe Execution Strategy

The quickest path is:

1. split orchestration first
2. keep existing analysis functions where possible
3. introduce new plot wrappers with better defaults
4. regenerate the paper figures from the new lanes
5. only then decide whether deeper internal cleanup is worth it

This avoids a large, risky rewrite of the statistical core.

## Expected Early Wins

These should happen quickly:

- separate QA versus exemplar execution
- cleaner runtime config blocks
- cleaner output directories
- vector-first figure output
- unified font and annotation styling
- configurable biopsy labels
- removal of manuscript-specific hard-coding from core orchestration

## Known Risks

- some current figure logic is embedded in `main_pipe.py` rather than in reusable functions
- some plotting functions in `production_plots.py` already have newer behavior, but the call sites remain manuscript-specific
- some exemplar figures appear to depend on very specific patient/biopsy selections and filenames
- there may be hidden coupling between filtered cohort tables and downstream summary functions

These are manageable if we preserve the existing math and refactor additively.

## Recommended File Skeleton

Initial target structure:

- `main_QA.py`
- `main_exemplars.py`
- `pipeline_shared_config.py`
- `load_data_QA.py`
- `load_data_exemplars.py`
- `production_plots_QA.py`
- `production_plots_exemplars.py`

Possible later additions:

- `pipeline_output_layout.py`
- `selection_presets.py`
- `figure_style.py`

## Definition Of Success

Success for the first refactor pass is:

- the old pipeline still exists unchanged
- the new QA and exemplar lanes are separate and understandable
- the regenerated paper figures are at least as correct and visibly cleaner
- the figure/export settings are controlled centrally
- future revisions no longer require editing a monolithic script
