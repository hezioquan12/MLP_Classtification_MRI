# Brain Tumor Classification using Pure MLP & Stratified K-Fold

This project implements a pure Multilayer Perceptron (MLP) built with **PyTorch** to classify brain MRI images into four distinct categories: `Glioma`, `Meningioma`, `No Tumor`, and `Pituitary`.

While vanilla MLPs traditionally face limitations in computer vision tasks, this project leverages a specialized image preprocessing pipeline combined with robust regularization and class-imbalance techniques to optimize performance on the **BRISC 2025** dataset.

---

##  Key Features & Technical Highlights

The project addresses the inherent challenges of using a pure MLP for image data through the following methodologies:

1. **Auto-Cropping (Noise Reduction):** Implements a custom transform algorithm `CropBlackBorders` with a `threshold=0.15`. This algorithm automatically detects the brain tissue region, crops out the uninformative black background surrounding the skull, and resizes the image to a standardized `128x128` resolution. This significantly minimizes "dead signals" feeding into the input layer.
<img width="1328" height="299" alt="download" src="https://github.com/user-attachments/assets/b226a745-1dc7-46d3-aa4f-2dd33b317b8b" />
<img width="1327" height="299" alt="download" src="https://github.com/user-attachments/assets/06f50f04-da8c-4dd0-9491-76b2d2e3488f" />
2. **Handling Class Imbalance:** Integrates custom class weights (`WEIGHT_CLASS = [2.0, 2.0, 1.0, 1.0]`) directly into the `CrossEntropyLoss` function. This penalizes the model more heavily for misclassifying underrepresented or more challenging classes.
3. **Stratified 5-Fold Cross Validation:** Replaces simple train/test splits with a Stratified K-Fold validation strategy (`N_SPLITS = 5`) to ensure a consistent and balanced distribution of class labels across all folds, guaranteeing evaluation stability.
4. **Robust Data Augmentation:** Applies `RandomRotation (20°)`, `ColorJitter (0.3)`, and `RandomHorizontalFlip` to enhance data diversity and prevent the MLP from overfitting or memorizing pixel positions.
5. **Adaptive Learning Rate:** Utilizes the `ReduceLROnPlateau` scheduler with a `patience=8` policy to dynamically scale down the learning rate when validation performance plateaus, facilitating smooth convergence in later epochs.

---

## Dataset

* **Source:** `briscdataset/brisc2025` (Automatically downloaded via `kagglehub`).
* **Total Images:** * Training Set (Post-Augmentation): ~5,000 images
  * Test Set: 1,000 images (Test 1) and 1,225 images (Test 2)
* **Classes (4 Categories):** Glioma, Meningioma, No Tumor, Pituitary

---

##  Hyperparameters & Training Configuration

* **Input Dimension:** 1 Channel (Grayscale) × 128 × 128 (Flattened into a 16,384-dimensional vector).
* **Batch Size:** 64
* **Initial Learning Rate:** `0.000015`
* **Maximum Epochs:** 150
* **Framework & Hardware:** PyTorch (Trained with GPU / CUDA acceleration)

---

##  Visual Results


### Loss & Accuracy Curves across Epochs
<img width="1489" height="490" alt="download" src="https://github.com/user-attachments/assets/2c5c5ea4-a154-482f-a73b-c62ada27b820" />

### Confusion Matrix
Detailed breakdown of the pure MLP's classification performance across the 4 brain tumor types:
<img width="650" height="547" alt="download" src="https://github.com/user-attachments/assets/1c61a0c8-a655-4a8e-b593-798038ec6a92" />


---

##  Quick Start Guide

1. Clone this repository and open the project in Google Colab.
2. Grant Google Drive access to save model checkpoints to `/content/drive/MyDrive/Colab Notebooks/Brics2025_MLP_kfold`.
3. Execute all cells in the `MLP_model_and_kfold_validation.ipynb` notebook to:
   * Automatically fetch the dataset from Kaggle.
   * Run the Stratified 5-Fold Cross Validation loop.
   * Export the optimal weights, which will automatically be saved as `best_mlp_model.pth`.
