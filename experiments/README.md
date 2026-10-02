# Paper-Based Experiments

This directory is reserved for experiments derived from the taxonomy in Shorten & Khoshgoftaar (2019).

## Planned experimental sequence

1. Establish a no-augmentation baseline.
2. Add geometric augmentation.
3. Add colour-space augmentation.
4. Evaluate kernel-based transformations.
5. Evaluate random erasing.
6. Evaluate image mixing.
7. Compare individual techniques with controlled settings.
8. Perform sensitivity/ablation analysis.
9. Record repeated runs and uncertainty.
10. Compare observations with the findings reported in the survey.

The current Streamlit application and Colab notebook provide image-space demonstrations. They do not by themselves establish classification improvements.

Future CNN experiments should keep the test set fixed and separate from augmentation operations to avoid leakage.
