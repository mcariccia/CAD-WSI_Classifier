# CAD-WSI: Pixel-Wise Semantic Segmentation of Cervical Precancerous Lesions

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)
![Groovy](https://img.shields.io/badge/Groovy-QuPath%20Scripts-4298B8)
![License](https://img.shields.io/badge/License-MIT-green)

A PyTorch-based Computer-Aided Diagnosis (CAD) system developed as my **Bachelor's Thesis** in Applied Computer Science and Data Analytics (University of Cagliari). 

This project tackles the semantic segmentation of cervical histopathological Whole Slide Images (WSI). The system classifies tissues according to the **LAST standardization criteria** (Negative, LSIL, HSIL). The research heavily focuses on mitigating domain shifts, handling extreme class imbalances (e.g., CIN1 rarity), and evaluating the model's actual clinical utility beyond standard pixel-level metrics.

## Key Engineering & Scientific Highlights

- **WSI-Level Stratified Splitting (`Splitter.py`):** Developed a constrained optimization algorithm to generate iterative stratified Train/Test splits. By operating strictly at the WSI level and enforcing maximum dominance constraints, it guarantees validability in cross-validation and completely prevents pixel-level data leakage.
- **Hierarchical & Clinical-Aware Loss (`perdite.py`):** Engineered a custom 5-term loss function integrating:
  - *Symmetric Cross-Entropy (SCE)* for robustness against manual annotation gradients (label noise).
  - *Focal Tversky Loss (FTL)* to handle spatial class imbalances.
  - *Expected Cost Penalty Matrix* (Elkan, 2001) to penalize critical clinical errors (e.g., misclassifying HSIL as Normal Mucosa carries a higher weight than misclassifying CIN1).
  - *Glass/Tissue Pseudo-labeling* based on optical density thresholds for non-annotated background regions.
- **Biologically-Grounded Stain Normalization (`PreProcess_Dataset.py`):** Implemented a robust Non-Negative Matrix Factorization (NMF) pipeline based on Vahadane (2016). It features multi-restart optimization to separate Hematoxylin and Eosin channels in the Optical Density (OD) space, dynamically computing the true white point ($I_0$) for each WSI to prevent image bleaching.
- **Scalable WSI Extraction (`estrazione_tile_up.groovy`):** Wrote custom Groovy scripts for QuPath to automate scalable, multi-threaded (ForkJoinPool) tile extraction from gigapixel WSIs.
- **Rigorous External Validation (`test_esterno.py`):** Conducted cross-dataset validation on the independent **MTCHI Task 2** public dataset. The testing pipeline utilizes Test Time Augmentation (TTA) and patient-level bootstrapping to compute reliable 95% Confidence Intervals (IC95%).

## Tech Stack

* **Deep Learning:** PyTorch, Torch.amp (Mixed Precision)
* **Architectures:** U-Net with lightweight encoders (`MobileNetV3` via *segmentation_models_pytorch*) to prevent overfitting on limited independent samples.
* **Computer Vision:** OpenCV, Albumentations, Scikit-Learn
* **Data Processing:** Groovy, QuPath, NumPy, Pandas

## Project Architecture

### 1. Data Extraction & Splitting
* `estrazione_tile_up.groovy` - QuPath script for parallel tile extraction.
* `Profila_WSI.py` - Fast pixel-level dataset profiling.
* `Splitter.py` - Constrained WSI-level train/test split generation.
* `extraction_tiles.py` - MTCHI dataset parsing and tile extraction without overlap.

### 2. Preprocessing & Normalization
* `PreProcess_Dataset.py` - Core Vahadane normalization.
* `PreProcess_MTCHI.py` - Wrapper to apply the private dataset's stain target to the MTCHI domain.
* `Confronto_prima_dopo.py` / `valutazione_OD.py` - Pre/Post normalization fading analysis in OD space.

### 3. Training Pipeline
* `modello.py` - U-Net architecture building, custom stage-unfreezing, and strict evaluation of BatchNorms.
* `perdite.py` - The custom Hierarchical Loss module.
* `hed_augment.py` - H&E OD space multiplicative perturbation.
* `addestramento.py` - Training loop with Mixed Precision and checkpoint selection based on clinical operating points (HSIL Sensitivity >= 0.90).
* `esegui.py` - Orchestrator for Stratified K-Fold cross-validation.

### 4. Evaluation & Metrics
* `test.py` - Internal testing with patient-level bootstrapping and tile-level detection rates (inspired by CAMELYON16).
* `test_esterno.py` - External validation on the MTCHI dataset.
* `metriche.py` & `metriche_esterno.py` - Clinical metric calculations mapped to the LAST criteria.
* `esempi.py` / `grafici.py` - Deterministic qualitative example selection (10th/50th/90th percentiles of IoU) to prevent cherry-picking.

## Main Findings & Results

As thoroughly discussed in the thesis chapters, evaluating the model revealed a distinct asymmetry between two clinical tasks:
1. **Lesion Detection:** The capacity to identify high-grade lesions (HSIL) remains highly stable and robust (Sensitivity > 0.93), even across radical domain shifts (MTCHI dataset).
2. **Lesion Grading:** The model's ability to correctly attribute the specific grade (LSIL vs HSIL) degrades outside the training domain. On the MTCHI dataset, the system showed a systematic tendency to overestimate severity, highlighting that *detection* and *grading* are distinct endpoints requiring separate validation strategies.

---
*Developed by Marco Cariccia - University of Cagliari (2026)*