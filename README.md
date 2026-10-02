# Image Data Augmentation — Paper-Based Research Companion

**CO3091/CO7091 Computational Intelligence and Software Engineering**

This repository is an experimental companion to:

> Shorten, C. & Khoshgoftaar, T. M. (2019). *A survey on Image Data Augmentation for Deep Learning*. Journal of Big Data, 6, 60.

The implementation is deliberately organised around the **taxonomy and research questions of the source paper** rather than being presented as an independent augmentation application.

## 1. Paper → Implementation map

| Concept in the 2019 survey | Repository component | Status |
|---|---|---|
| Geometric transformations | Streamlit + Colab experiments | Implemented |
| Colour-space transformations | Streamlit + Colab experiments | Implemented |
| Kernel filters | Streamlit experiments | Implemented |
| Random erasing | Streamlit + Colab experiments | Implemented |
| Mixing images | Streamlit pipeline | Demonstration |
| Feature-space augmentation | Research/theory extension | Not claimed as reproduced |
| Adversarial training | Research/theory extension | Not claimed as reproduced |
| GAN-based augmentation | Research/theory extension | Not claimed as reproduced |
| Neural style transfer | Research/theory extension | Not claimed as reproduced |
| Meta-learning / policy search | Stochastic policy-search framework | Experimental extension |
| Test-time augmentation | Planned benchmark extension | Not yet implemented |
| Dataset-size effects | Planned benchmark extension | Not yet implemented |
| Curriculum learning | Literature/theory extension | Not yet implemented |

The survey explicitly identifies geometric transformations, colour-space augmentation, kernel filters, mixing images, random erasing, feature-space augmentation, adversarial training, GANs, neural style transfer and meta-learning as augmentation approaches. It also discusses test-time augmentation, resolution, final dataset size and curriculum learning. 

## 2. Repository architecture

```text
Image-Data-Augmentation-Tech-Review/
│
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
│
├── notebooks/
│   └── Image_Data_Augmentation_Research_Lab_Colab.ipynb
│
├── experiments/
│   └── README.md
│
└── results/
    └── README.md
```

## 3. Research workflow

The intended workflow is:

```text
Source paper
    ↓
Taxonomy
    ↓
Technique implementation
    ↓
Controlled experiment
    ↓
Quantitative diagnostics
    ↓
CNN benchmark
    ↓
Ablation / sensitivity analysis
    ↓
Critical comparison with paper
```

The current application implements the first experimental layers. Pixel-level metrics such as MSE, MAE and PSNR describe image change; they **do not establish improved model generalisation**.

## 4. Paper evidence vs. new experiments

This repository keeps three types of evidence separate:

### A. Literature evidence
Results reported by Shorten & Khoshgoftaar (2019) are treated as findings from the published survey.

### B. Reproduction
An experiment is called a reproduction only when the repository implements the relevant method and experimental conditions sufficiently to make that claim.

### C. Extension
New policy-search, sensitivity and dataset-generation experiments are labelled as extensions rather than attributed to the 2019 paper.

This separation prevents generated experimental results from being confused with published results.

## 5. Google Colab

The Colab notebook provides a reproducible notebook-based companion:

urlOpen in Google Colabhttps://colab.research.google.com/github/Dinakarnayak/Image-Data-Augmentation-Tech-Review/blob/main/notebooks/Image_Data_Augmentation_Research_Lab_Colab.ipynb

## 6. Interactive laboratory

Run the Streamlit research interface:

```bash
git clone https://github.com/Dinakarnayak/Image-Data-Augmentation-Tech-Review.git
cd Image-Data-Augmentation-Tech-Review
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## 7. Academic use

The repository is intended as a **demonstration/experimental companion** to the technical review. It should not be presented as reproducing every experiment in the survey.

For the assessed report, distinguish:

- what the paper states;
- what the implementation demonstrates;
- what an experiment measures;
- and what conclusions are justified by the experiment.

## Reference

Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on Image Data Augmentation for Deep Learning. *Journal of Big Data, 6*, 60.
