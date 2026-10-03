# CO3091 Tech Review — Final With Images

**Source file:** `CO3091_TechReview_FINAL_WITH_IMAGES (2)(1).pdf`  
**Student:** Dinakar Nayak  
**Programme:** MSc Artificial Intelligence with Industry  
**University:** University of Leicester, United Kingdom

## Abstract

Image data augmentation provides a data-space approach to increasing training-data diversity and addressing overfitting in deep learning. This review examines the augmentation techniques covered by Shorten and Khoshgoftaar, including geometric transformations, colour-space transformations, kernel filters, mixing images, random erasing, feature-space augmentation, adversarial training, GAN-based augmentation, neural style transfer and meta-learning. Reported experimental results are considered together with the advantages and limitations of the different approaches. The analysis also considers the dependence of augmentation performance on the dataset, model and transformation magnitude. The evidence indicates that augmentation can improve performance under particular experimental conditions, but increased visual variation does not necessarily guarantee improved generalisation.

## 1. Introduction

Deep learning models can overfit when their capacity allows them to learn training examples too closely, particularly when the available data provide insufficient variation. Overfitting can produce strong training performance while reducing generalisation to unseen examples. Shorten and Khoshgoftaar [1] identify the availability of sufficiently large datasets as an important consideration for deep convolutional neural networks.

Image data augmentation provides a data-space approach to this problem. Instead of relying only on the original training examples, augmentation generates additional training instances through transformations of existing images. The objective is not simply to increase the number of images, but to introduce useful variation while preserving the semantic label. Examples include geometric transformations, colour modification, filtering, image mixing and region removal. Consequently, augmentation can act as a regularisation mechanism by exposing a model to a broader range of plausible inputs. However, its effectiveness depends on the target task because an excessive transformation can alter information required for correct classification [1].

## 2. State-of-the-Art Technologies

Shorten and Khoshgoftaar [1] survey ten image data augmentation approaches. Geometric transformations alter spatial arrangement through operations such as flipping, rotation, cropping and translation. Colour-space transformations modify visual properties such as brightness or colour. Kernel filters alter local characteristics including edges, texture and sharpness. Mixing images constructs additional examples by combining information from existing samples. Random erasing removes or masks selected image regions.

Feature-space augmentation operates on learned representations. Adversarial training introduces perturbations through an adversarial objective. GAN-based augmentation uses generative models to produce synthetic examples. Neural style transfer changes visual style while attempting to preserve relevant content. Meta-learning approaches learn augmentation strategies rather than relying entirely on manually selected transformations.

These approaches range from relatively simple data-space operations to learned strategies requiring additional model optimisation. Their suitability depends on whether the transformation preserves the semantic information required by the target task [1].

### Taxonomy figure

The PDF's Figure 1 presents a taxonomy with **Image Data Augmentation** at the top, branching into **Basic Image Manipulations** and **Deep Learning Approaches**. Basic manipulations include kernel filters, colour-space transformations, geometric transformations, random erasing and mixing images. Deep-learning approaches include adversarial training, neural style transfer and GAN data augmentation. Meta-learning is then shown connecting to neural augmentation, AutoAugment and Smart Augmentation.

## 3. Performance Evaluation

Geometric transformations are relatively simple and can provide spatial variation, but excessive transformations may violate label invariance. Colour-space transformations can improve robustness to appearance and illumination variation, although strong changes can remove task-relevant information. Kernel filters can introduce texture and sharpness variation, while excessive filtering can generate unrealistic samples.

Random erasing can improve robustness to partial occlusion and reduce dependence on individual image regions, but removing a discriminative region may alter semantic content. Image mixing can increase diversity through combinations of existing samples, although generated examples can contain semantic or label ambiguity.

Feature-space augmentation can introduce variation in learned representations, while adversarial, GAN-based and style-transfer approaches can provide more complex variation at greater computational cost. Meta-learning can support augmentation-policy selection but introduces additional optimisation requirements.

The survey reports measurable improvements under particular experimental conditions:

| Method | Dataset | Result | Metric |
|---|---|---|---|
| Flipping | Caltech101 | 48.13% → 49.73% | Top-1 |
| Rotation | Caltech101 | 48.13% → 50.80% | Top-1 |
| Cropping | Caltech101 | 48.13% → 61.95% | Top-1 |
| PatchShuffle | CIFAR-10 | 6.33% → 5.66% | Error |
| SamplePairing | CIFAR-10 | 8.22% → 6.93% | Error |
| Random Erasing | CIFAR-10 | 5.17% → 4.31% | Error |
| GAN augmentation | Liver lesions | 78.6% → 85.7% | Sensitivity |
| GAN augmentation | Liver lesions | 88.4% → 92.4% | Specificity |

These results should be interpreted within their experimental settings because dataset, model, transformation magnitude and evaluation protocol affect observed performance.

### Advantages and limitations

| Approach | Advantage | Limitation |
|---|---|---|
| Geometric | Spatial diversity and positional robustness | Large transformations may alter labels |
| Colour-space | Variation in illumination and appearance | Important colour information may be removed |
| Kernel filters | Texture, edge and sharpness variation | Strong filtering may be unrealistic |
| Mixing images | Greater sample diversity | Potential semantic or label ambiguity |
| Random erasing | Robustness to partial occlusion | Important regions may be removed |
| Feature-space | Variation in learned representations | Depends on the learned representation |
| Adversarial | Improved robustness to perturbations | Additional optimisation complexity |
| GAN-based | Can generate synthetic examples | Computationally demanding training |
| Style transfer | Introduces visual-style variation | Style changes may affect task information |
| Meta-learning | Can learn augmentation policies | Requires additional optimisation |

## 3.1 Critical Evaluation

The evidence indicates that augmentation effectiveness is conditional rather than universal. A transformation is useful only when the generated sample remains appropriate for the target task. Consequently, greater visual distortion does not necessarily produce better training data. Transformation magnitude and label preservation must be considered together.

The reported results are also dependent on experimental context. Dataset characteristics, model architecture and evaluation methodology can influence measured performance. An improvement reported for one dataset should therefore not automatically be treated as transferable to another dataset.

A further distinction exists between image-level variation and model-level generalisation. Pixel-level measures can describe how much an image has changed, but cannot independently establish that a neural network will generalise better. Learned approaches also introduce a trade-off between greater flexibility and additional computational and optimisation requirements.

## 4. Conclusion

Image data augmentation provides a practical approach for increasing training-data diversity and reducing the risk of overfitting in deep learning. Shorten and Khoshgoftaar [1] describe approaches ranging from geometric and colour-space transformations to kernel filters, random erasing, image mixing and learned methods.

The reported evidence demonstrates that different augmentation strategies can improve performance under particular experimental conditions. However, results should not be generalised independently of the dataset, model, transformation magnitude and evaluation procedure. Important remaining challenges include selecting suitable augmentation policies, preserving semantic labels, controlling computational cost and determining whether reported improvements transfer across datasets and architectures.

## References

[1] C. Shorten and T. M. Khoshgoftaar, “A survey on Image Data Augmentation for Deep Learning,” *Journal of Big Data*, vol. 6, no. 1, article 60, pp. 1–48, 2019, doi: 10.1186/s40537-019-0197-0.

## Appendix — Research Laboratory

The accompanying research laboratory implements image-space augmentation operations including geometric transformations, colour-space operations, kernel filters, random erasing and stochastic augmentation policies. The implementation separates literature evidence, reproduction and experimental extension.

The implementation includes Mean Squared Error (MSE), Mean Absolute Error (MAE), Peak Signal-to-Noise Ratio (PSNR), histogram distance, edge density and edge-density change. These measures describe image-level changes and are not independently treated as evidence of improved neural-network generalisation.

The experimental workflow supports controlled random seeds, repeated trials, mean and standard deviation reporting and experiment metadata. The research laboratory also supports generation of augmented image collections using multiple source images and stochastic augmentation policies.

## Visuals included in the uploaded PDF

- **Figure 1:** Taxonomy of image data augmentation techniques.
- **Figure 2:** Signs of overfitting and desired convergence of training and testing error.
- **Figure 3:** Examples of colour-space augmentation techniques: Contrast +20%, histogram equalisation, white balance and sharpen.
- **Figure 4:** PatchShuffle regularisation example.
- **Table I:** Advantages and limitations of augmentation approaches.
- **Table II:** Selected experimental evidence reported in the survey.

**Original uploaded PDF:** 3 pages. The PDF itself is the authoritative visual version of this report.
