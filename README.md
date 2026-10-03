# 🧬 Image Data Augmentation — Research Companion

<p align="center">
  <strong>CO3091 / CO7091 · Computational Intelligence and Software Engineering</strong><br>
  Paper-based experimental laboratory for image augmentation research
</p>

<p align="center">
  <a href="https://github.com/Dinakarnayak/Image-Data-Augmentation-Tech-Review">GitHub</a> ·
  <a href="https://colab.research.google.com/github/Dinakarnayak/Image-Data-Augmentation-Tech-Review/blob/main/notebooks/Image_Data_Augmentation_Research_Lab_Colab.ipynb">Google Colab</a> ·
  <a href="https://doi.org/10.1186/s40537-019-0197-0">Research Paper (DOI)</a>
</p>

---

## 📚 Research basis

This repository is built around the following peer-reviewed survey paper:

> **Shorten, C. & Khoshgoftaar, T. M. (2019). _A survey on Image Data Augmentation for Deep Learning_. Journal of Big Data, 6, 60.**

🔗 **[Read the original research paper](https://doi.org/10.1186/s40537-019-0197-0)**

The project is an **experimental companion to the survey**, not a claim to reproduce every method or numerical result reported in the paper.

The survey discusses geometric transformations, colour-space transformations, kernel filters, mixing images, random erasing, feature-space augmentation, adversarial training, GAN-based augmentation, neural style transfer and meta-learning. It also discusses test-time augmentation, resolution, dataset size and curriculum learning.

---

## 📄 Final Technical Review

The completed CO7091 Technology Review is included in the repository:

**[Image Data Augmentation — Final Technical Review](reports/Image_Data_Augmentation_Tech_Review_Final.md)**

The report covers:
- overfitting and the role of augmentation;
- the ten augmentation approaches discussed by Shorten & Khoshgoftaar;
- advantages and limitations of each category;
- selected quantitative evidence from Caltech101, CIFAR-10 and liver-lesion experiments;
- methodological limitations and comparability;
- image-space diagnostics versus model-performance evidence;
- reproducibility and experimental interpretation boundaries.

The report also documents the distinction between literature evidence, repository implementation and new experimental extensions.

---

## 🚀 What the project provides

| Module | Purpose |
|---|---|
| 🎛️ **Playground** | Interactive exploration of individual augmentation operations |
| 🔗 **Policy Pipeline** | Compose stochastic multi-stage augmentation policies |
| 🧬 **Policy Search** | Monte-Carlo exploration of stochastic policies |
| 📈 **Sensitivity** | Examine how augmentation magnitude changes image-level diagnostics |
| 📦 **Dataset Generator** | Generate augmented image packages with a manifest |
| 🧪 **Experiment Matrix** | Repeated, controlled comparisons across augmentation operations |
| 🔬 **Diagnostics** | RGB statistics, histograms and image-change measurements |

---

## 🧠 Paper → Implementation traceability

| Survey concept | Repository implementation | Status |
|---|---|---|
| Geometric transformations | Streamlit + Colab | ✅ Implemented |
| Colour-space transformations | Streamlit + Colab | ✅ Implemented |
| Kernel filters | Streamlit | ✅ Implemented |
| Random erasing | Streamlit + Colab | ✅ Implemented |
| Mixing images | Policy/dataset workflow | 🟡 Demonstration |
| Feature-space augmentation | Documentation | ⚪ Not reproduced |
| Adversarial training | Documentation | ⚪ Not reproduced |
| GAN-based augmentation | Documentation | ⚪ Not reproduced |
| Neural style transfer | Documentation | ⚪ Not reproduced |
| Meta-learning | Stochastic policy-search extension | 🟡 Experimental extension |
| Test-time augmentation | Future benchmark | 🔵 Planned |
| Dataset-size effects | Future benchmark | 🔵 Planned |
| Curriculum learning | Documentation | ⚪ Not implemented |

---

## 🔬 Research workflow

```text
SOURCE PAPER → TAXONOMY → IMPLEMENTATION
       ↓
CONTROLLED EXPERIMENT → IMAGE DIAGNOSTICS
       ↓
MODEL BENCHMARK → ABLATION / SENSITIVITY
       ↓
LITERATURE COMPARISON
```

---

## 🧪 Experiment Matrix

The application now supports controlled image-level comparisons across selected augmentation operations.

It can:
- run repeated experiments;
- record random seeds;
- calculate MSE, MAE and PSNR;
- calculate histogram distance;
- measure edge-density change;
- report mean and standard deviation;
- display individual runs;
- generate an experiment identifier;
- export CSV results;
- export JSON metadata.

**Important:** these metrics measure image-space change. They do **not** establish classification accuracy, improved CNN generalisation, semantic validity, label preservation, or that one augmentation policy is superior.

---

## 🧩 Evidence discipline

The repository separates:

### Literature evidence
Findings and numerical results reported by the source paper.

### Reproduction
An implementation intended to recreate a documented method or experimental setup sufficiently to justify that claim.

### Extension
New experiments developed specifically for this repository, including stochastic policy search, sensitivity analysis and controlled experiment matrices.

Newly generated results must not be presented as results from the source paper.

---

## 📊 Reproducibility

Record, where applicable:
- dataset and version;
- train/validation/test split;
- model architecture;
- augmentation policy;
- random seed(s);
- software/library versions;
- training configuration;
- evaluation metrics;
- repetitions;
- mean and standard deviation;
- confidence intervals where appropriate.

Keep the test set isolated from training augmentation.

---

## 🏗️ Repository structure

```text
Image-Data-Augmentation-Tech-Review/
│
├── app.py
├── requirements.txt
├── README.md
├── ABOUT.md
├── LICENSE
├── CITATION.cff
├── CONTRIBUTING.md
│
├── notebooks/
│   └── Image_Data_Augmentation_Research_Lab_Colab.ipynb
│
├── reports/
│   └── Image_Data_Augmentation_Tech_Review_Final.md
│
├── experiments/
│   └── README.md
├── results/
│   └── README.md
└── docs/
    └── RESEARCH_PROTOCOL.md
```

---

## 💻 Run locally

```bash
git clone https://github.com/Dinakarnayak/Image-Data-Augmentation-Tech-Review.git
cd Image-Data-Augmentation-Tech-Review
python -m venv .venv
```

**Windows**
```bash
.venv\Scripts\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## ☁️ Google Colab

**[Open the Research Lab in Google Colab](https://colab.research.google.com/github/Dinakarnayak/Image-Data-Augmentation-Tech-Review/blob/main/notebooks/Image_Data_Augmentation_Research_Lab_Colab.ipynb)**

---

## ⚠️ Scope and limitations

The current project is primarily an **image-space research laboratory**.

It does not claim to reproduce GAN training, adversarial training, feature-space augmentation, neural style transfer, the complete meta-learning literature, or every experiment in Shorten & Khoshgoftaar (2019).

The stochastic policy-search component is an **engineering extension**, not a reproduction of AutoAugment or the survey's numerical results.

---

## 🎓 Academic use

For the accompanying technical review, distinguish:

> **What the paper reports → What this repository implements → What a new experiment measures → What conclusion the evidence supports**

---

## 📖 Reference

**Shorten, C., & Khoshgoftaar, T. M. (2019). _A survey on Image Data Augmentation for Deep Learning_. Journal of Big Data, 6, 60.**

🔗 **[Original paper — Springer / Journal of Big Data](https://doi.org/10.1186/s40537-019-0197-0)**

---

<p align="center">
  <strong>Image Data Augmentation Research Lab</strong><br>
  CO3091 / CO7091 · Experimental Research Companion
</p>
