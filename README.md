# Ai-project-work
CNN built from scratch in PyTorch to classify skin lesion images as benign or malignant melanoma. 92% accuracy, 0.97 ROC-AUC.
# Melanoma Skin Cancer Classification — CNN (PyTorch)

A Convolutional Neural Network built from scratch in PyTorch to classify
dermoscopic skin lesion images as benign or malignant, trained on
Google Colab (GPU).

## Overview

- **Dataset:** [Melanoma Skin Cancer Dataset of 10,000 Images](https://www.kaggle.com/) (Kaggle)
- **Task:** Binary image classification (benign vs. malignant)
- **Model:** Custom CNN — 3 convolutional blocks + fully connected layers, no pretrained weights
- **Framework:** PyTorch
- **Environment:** Google Colab (NVIDIA T4 GPU, free tier)

## Results

| Metric | Value |
|---|---|
| Test Accuracy | **92%** |
| ROC-AUC | **0.9711** |
| Precision (Benign) | 0.89 |
| Recall (Benign) | 0.95 |
| F1-Score (Benign) | 0.92 |
| Precision (Malignant) | 0.95 |
| Recall (Malignant) | 0.88 |
| F1-Score (Malignant) | 0.91 |

Trained for 15 epochs. Training and test loss/accuracy tracked closely
throughout, indicating good generalization with no significant overfitting.

**Confusion Matrix (test set, 1,000 images):**

|  | Predicted Benign | Predicted Malignant |
|---|---|---|
| **Actual Benign** | 475 | 25 |
| **Actual Malignant** | 60 | 440 |

## Repository Structure

```
.
├── notebooks/
│   └── melanoma_cnn_colab.ipynb   # Full pipeline: data loading, model, training, evaluation
├── README.md
```

## Model Architecture

```
Input (128x128x3 RGB image)
  → Conv2D(32) → BatchNorm → ReLU → MaxPool   [Block 1]
  → Conv2D(64) → BatchNorm → ReLU → MaxPool   [Block 2]
  → Conv2D(128) → BatchNorm → ReLU → MaxPool  [Block 3]
  → Flatten
  → Fully Connected (256) → ReLU → Dropout(0.5)
  → Fully Connected (1) → sigmoid (via BCEWithLogitsLoss)
Output: probability of malignant
```

## How to Run

1. Download the dataset from Kaggle and upload the `train/` and `test/`
   folders to your Google Drive.
2. Open `notebooks/melanoma_cnn_colab.ipynb` in [Google Colab](https://colab.research.google.com).
3. Enable GPU: **Runtime → Change runtime type → GPU**.
4. Update the `DATA_DIR` variable in the notebook to point to your dataset's
   location in Drive.
5. Run all cells in order (`Runtime → Run all`, or step through with `Shift+Enter`).

## Training Configuration

- Loss: Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`)
- Optimizer: Adam, learning rate = 0.001
- Batch size: 32
- Epochs: 15
- Data augmentation (training only): random horizontal flip, random rotation (±15°)

## Key Finding

While overall accuracy was strong (92%), recall on the malignant class
(88%) was lower than on the benign class (95%) — meaning the model missed
more true cancer cases than it raised false alarms. In a real clinical
screening context, false negatives are more costly than false positives,
so this is an important limitation to account for (e.g. via threshold
tuning or class-weighted loss) before any real-world use.

## Limitations

- Trained from scratch on a relatively small dataset (10,000 images) —
  transfer learning (e.g. ResNet) would likely improve performance further.
- Only 15 epochs; more epochs with early stopping could yield gains.
- **Not validated for and not intended for real clinical/medical use** —
  built strictly for educational/coursework purposes.

## Group Members

| Name | Index Number | GitHub Repository |
|---|---|---|
| [Name 1] | [Index Number] | [Link] |
| [Name 2] | [Index Number] | [Link] |
| [Name 3] | [Index Number] | [Link] |

## Course
Artificial Intelligence (Machine Learning Project) — University of Energy
and Natural Resources,Sunyani. Department of Computer Science and Informatics.
