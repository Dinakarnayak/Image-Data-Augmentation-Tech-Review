# About This Project

## Image Data Augmentation — Research Companion

This repository is a research and experimentation companion for studying **image data augmentation for deep learning**, based on the survey paper:

> Shorten, C., & Khoshgoftaar, T. M. (2019). *A survey on image data augmentation for deep learning*. Journal of Big Data, 6, 60.

### Purpose

The project provides a practical environment for exploring augmentation techniques discussed in the survey while clearly separating:

- **Literature evidence** — concepts and findings reported in the source paper.
- **Reproduction** — implementations of selected image-level augmentation techniques.
- **Extension** — additional controlled experiments and reproducible diagnostics developed for this project.

### Implemented Techniques

The research lab currently demonstrates:

- Geometric transformations
- Colour-space transformations
- Kernel filters
- Random erasing / occlusion
- Image mixing
- Stochastic augmentation policies
- Controlled experiment matrices
- Repeated experiments with fixed seeds
- Pixel-level quantitative diagnostics
- Dataset generation and experiment manifests

### Research Workflow

The workflow is designed to support:

1. Source-image inspection
2. Individual augmentation experiments
3. Quantitative image diagnostics
4. Cross-technique comparison
5. Stochastic policy experiments
6. Repeated controlled experiments
7. Dataset generation
8. Export of experiment results and metadata

### Research Scope

The survey also discusses model-level approaches such as feature-space augmentation, adversarial training, GAN-based augmentation, neural style transfer, and meta-learning. These methods require model-level implementations and are **not claimed as reproduced** by the current image-level research lab.

Pixel-level metrics such as MSE, MAE, PSNR, histogram distance, and edge-density change describe image-level differences. They do **not**, by themselves, establish improved neural-network generalisation.

### Google Colab

Run the complete research workflow in Google Colab:

[Open the Image Data Augmentation Research Lab](https://colab.research.google.com/github/Dinakarnayak/Image-Data-Augmentation-Tech-Review/blob/main/notebooks/Image_Data_Augmentation_Research_Lab_Colab.ipynb)

### Repository

[GitHub Repository](https://github.com/Dinakarnayak/Image-Data-Augmentation-Tech-Review)

### Reference

Shorten, C., & Khoshgoftaar, T. M. (2019). *A survey on image data augmentation for deep learning*. Journal of Big Data, 6, 60. https://doi.org/10.1186/s40537-019-0197-0
