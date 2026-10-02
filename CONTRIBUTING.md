# Research Contribution Guidelines

This repository is organised as a paper-based experimental companion.

## Evidence discipline

Every new experiment should state whether it is:

1. **Literature evidence** — directly reported by the source paper.
2. **Reproduction** — an attempt to reproduce a documented method/setting.
3. **Extension** — a new experiment designed for this repository.

Do not present extension results as results from the source paper.

## Reproducibility checklist

Record:

- dataset and version;
- train/validation/test split;
- model and software versions;
- augmentation policy;
- random seed(s);
- training configuration;
- evaluation metrics;
- number of repeated runs;
- mean and standard deviation;
- limitations.

Keep the test set isolated from augmentation operations that are intended only for training.
