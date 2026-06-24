#  ANN Iris Flower Classification

A multi-layer Artificial Neural Network (ANN) built with TensorFlow/Keras to classify Iris flowers into three species — Setosa, Versicolor, and Virginica — achieving **~96–100% test accuracy**.

---

##  Project Overview

This project demonstrates a complete deep learning pipeline: from data loading and preprocessing through model training, evaluation, and deployment-ready saving. The classic [Iris dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html) (150 samples, 4 features, 3 classes) is used as the classification target.

---

##  Project Structure

```
ANN__1_.ipynb             # Main Jupyter notebook (end-to-end pipeline)
iris_ann_model.keras      # Saved trained model (generated on first run)
```

---

##  Requirements

| Library | Purpose |
|---------|---------|
| `tensorflow >= 2.x` | Model building & training |
| `numpy` | Numerical operations |
| `pandas` | Data manipulation |
| `scikit-learn` | Dataset, scaling, metrics |
| `matplotlib` | Training history plots |
| `seaborn` | Pairplot & heatmap visualisations |

Install all dependencies:

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn
```

---

##  How to Run

```bash
jupyter notebook ANN__1_.ipynb
```

Run all cells top to bottom. The notebook is self-contained — no external data files needed (the Iris dataset is loaded directly from scikit-learn).

---

##  Pipeline

### 1. Data Loading & Exploration
- Loads the Iris dataset via `sklearn.datasets.load_iris`
- Produces descriptive statistics, checks for missing values and duplicates
- Visualises feature relationships with a pairplot and correlation heatmap

### 2. Preprocessing
- **StandardScaler** — normalises features to zero-mean, unit-variance
- **One-hot encoding** — converts integer class labels to categorical vectors
- **Stratified train/test split** — 80% train / 20% test, preserving class balance

### 3. Model Architecture

```
Input (4 features)
    │
Dense(16, ReLU)      ← captures primary patterns
    │
Dropout(0.2)         ← regularisation
    │
Dense(8, ReLU)       ← refines representations
    │
Dense(3, Softmax)    ← probability per class
```

| Hyperparameter | Value |
|----------------|-------|
| Optimiser | Adam |
| Loss | Categorical Cross-Entropy |
| Batch size | 8 |
| Max epochs | 100 |
| Early stopping patience | 10 |

### 4. Training
- Uses `EarlyStopping` (monitors `val_loss`, restores best weights)
- 20% of training data is held out as a validation set each epoch

### 5. Evaluation
- Accuracy & loss curves plotted for train vs. validation
- Confusion matrix with per-class breakdown
- Full `classification_report` (precision, recall, F1)

### 6. Prediction on Unseen Data
Tests four hand-crafted samples (never seen during training) with predicted species and confidence scores.

### 7. Model Saving & Loading
```python
model.save('iris_ann_model.keras')          # Save
loaded_model = load_model('iris_ann_model.keras')  # Reload
```

---

##  Results

| Metric | Value |
|--------|-------|
| Dataset | 150 samples, 4 features, 3 classes |
| Train / Test split | 80% / 20% |
| Expected test accuracy | **96–100%** |
| Regularisation | Dropout + EarlyStopping |

---

##  Target Classes

| Label | Species |
|-------|---------|
| 0 | *Iris Setosa* |
| 1 | *Iris Versicolor* |
| 2 | *Iris Virginica* |

---

##  Key Concepts Demonstrated

- Multi-class classification with a feedforward ANN
- Feature standardisation and one-hot label encoding
- Dropout regularisation to reduce overfitting
- Early stopping for optimal generalisation
- Model persistence with the Keras `.keras` format

---

