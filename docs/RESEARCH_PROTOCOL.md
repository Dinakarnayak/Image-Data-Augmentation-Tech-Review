# Research Protocol

## Objective

Investigate image data augmentation techniques using the taxonomy of Shorten & Khoshgoftaar (2019), while separating source-paper evidence from newly generated experimental evidence.

## Experimental hierarchy

### Level 1 — Visual verification
Demonstrate that an augmentation operation produces the intended image-space transformation.

### Level 2 — Pixel diagnostics
Measure image-level change using metrics such as MSE, MAE, PSNR, histogram distance and edge-density change.

These metrics describe transformation magnitude. They are not proxies for classification accuracy.

### Level 3 — Model benchmark
For a future CNN experiment:

- fix the architecture;
- fix the dataset split;
- keep the test set unaugmented;
- apply augmentation only to the designated training data;
- use multiple random seeds;
- report task-specific metrics;
- report variability across runs.

### Level 4 — Ablation and sensitivity
Compare individual augmentation families and controlled combinations. Vary augmentation magnitude while holding other factors constant.

### Level 5 — Critical comparison

Compare the new observations with the findings and limitations reported in the 2019 survey. Clearly label where the experimental setting differs.

## Threats to validity

Important considerations include:

- augmentation may not preserve semantic labels;
- unrealistic transformations can introduce distribution shift;
- improvements can depend on dataset and model architecture;
- a single random seed can produce unstable conclusions;
- pixel similarity does not imply semantic similarity;
- training-set augmentation must not leak information from the test set.

## Reproducibility standard

A result is considered reproducible only when another user can identify the data, code version, parameters, random seed and evaluation procedure needed to rerun it.
