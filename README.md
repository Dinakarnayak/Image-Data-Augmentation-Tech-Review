# Image Data Augmentation — Research Companion

**CO3091/CO7091 · Computational Intelligence and Software Engineering**

## Research basis

This repository is built around the source paper:

**Shorten, C. & Khoshgoftaar, T. M. (2019). _A survey on Image Data Augmentation for Deep Learning_. Journal of Big Data, 6, 60.**

The software is a **paper-based experimental companion**. It is not presented as a reproduction of the entire survey.

The survey describes augmentation as a data-space solution to limited training data and discusses a broad range of methods, including geometric transformations, colour-space augmentation, kernel filters, image mixing, random erasing, feature-space augmentation, adversarial training, GAN-based augmentation, neural style transfer and meta-learning. It also discusses test-time augmentation, resolution, dataset size and curriculum learning.

---

## Research architecture

```text
                 SOURCE PAPER
                     │
                     ▼
              PAPER TAXONOMY
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
     DATA WARPING          OVERSAMPLING
          │                     │
          ▼                     ▼
   Image-space methods     Synthetic methods
          │                     │
          └──────────┬──────────┘
                     ▼
            CONTROLLED EXPERIMENT
                     │
                     ▼
          QUANTITATIVE ANALYSIS
                     │
                     ▼
             MODEL BENCHMARK
                     │
                     ▼
          ABLATION / SENSITIVITY
                     │
                     ▼
        COMPARISON WITH LITERATURE
```

## Paper-to-project traceability

| Paper concept | Project component | Current status |
|---|---|---|
| Geometric transformations | Streamlit + Colab | Implemented |
| Colour-space transformations | Streamlit + Colab | Implemented |
| Kernel filters | Streamlit | Implemented |
| Random erasing | Streamlit + Colab | Implemented |
| Mixing images | Streamlit pipeline | Demonstration |
| Feature-space augmentation | Research documentation | Not claimed as reproduced |
| Adversarial training | Research documentation | Not claimed as reproduced |
| GAN augmentation | Research documentation | Not claimed as reproduced |
| Neural style transfer | Research documentation | Not claimed as reproduced |
| Meta-learning | Stochastic policy-search extension | Experimental extension |
| Test-time augmentation | Future benchmark | Planned |
| Dataset-size effects | Future benchmark | Planned |
| Curriculum learning | Research documentation | Not implemented |

---

## Repository structure

```text
Image-Data-Augmentation-Tech-Review/
│
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── CITATION.cff
├── CONTRIBUTING.md
│
├── notebooks/
│   └── Image_Data_Augmentation_Research_Lab_Colab.ipynb
│
├── experiments/
│   └── README.md
│
├── results/
│   └── README.md
│
└── docs/
    └── RESEARCH_PROTOCOL.md
```

---

## Experimental levels

### Level 1 — Visual verification
Confirm that an augmentation produces the intended image-space transformation.

### Level 2 — Pixel diagnostics
Measure image-level change using MSE, MAE, PSNR, histogram distance and edge-density change.

These measurements quantify transformation behaviour. They do **not** demonstrate improved classification or generalisation.

### Level 3 — Model evaluation
A future benchmark should use:

- fixed model architecture;
- fixed train/validation/test split;
- augmentation restricted to the training data;
- repeated random seeds;
- task-specific evaluation metrics;
- mean and variability across runs.

### Level 4 — Ablation and sensitivity
Evaluate individual augmentation families and controlled combinations while varying one experimental factor at a time.

### Level 5 — Literature comparison
Compare newly generated observations with the source paper while explicitly identifying differences in dataset, architecture, augmentation policy and evaluation protocol.

---

## Evidence discipline

Every result in this repository should be labelled as one of:

**Literature evidence**  
A finding or numerical result reported by the source paper.

**Reproduction**  
An experiment attempting to recreate a documented method or experimental condition.

**Extension**  
A new experiment designed for this repository.

This distinction prevents newly generated results from being presented as published results.

---

## Reproducibility requirements

Record, where applicable:

- dataset and version;
- data split;
- model architecture;
- software/library versions;
- augmentation policy;
- random seed(s);
- training configuration;
- evaluation metrics;
- number of repetitions;
- mean and standard deviation;
- confidence intervals where appropriate.

The test set should remain isolated from training augmentation to reduce the risk of data leakage.

---

## Current interfaces

### Interactive research laboratory

```bash
git clone https://github.com/Dinakarnayak/Image-Data-Augmentation-Tech-Review.git
cd Image-Data-Augmentation-Tech-Review
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### Google Colab

urlOpen the Colab research notebookhttps://colab.research.google.com/github/Dinakarnayak/Image-Data-Augmentation-Tech-Review/blob/main/notebooks/Image_Data_Augmentation_Research_Lab_Colab.ipynb

---

## Limitations

The current implementation is primarily an **image-space research laboratory**. It should not be described as implementing GAN augmentation, adversarial training, feature-space augmentation, neural style transfer or meta-learning as defined in the survey.

The policy-search module is an engineering extension for exploring stochastic augmentation policies; it is not claimed to reproduce AutoAugment or the survey's published experimental results.

Pixel-level diversity is also not equivalent to semantic validity or improved model performance.

---

## Citation

If this repository is used as a companion to the technical review, cite the source paper:

Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on Image Data Augmentation for Deep Learning. *Journal of Big Data, 6*, 60. DOI: 10.1186/s40537-019-0197-0.

See `CITATION.cff` for repository citation metadata.
