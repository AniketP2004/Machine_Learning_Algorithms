# Machine Learning Algorithms From Scratch

Twelve classical machine learning algorithms implemented from first principles, without relying on scikit-learn for the core logic. The goal is to understand how each algorithm works internally: the math, the update rules, and the trade-offs.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![NumPy](https://img.shields.io/badge/NumPy-from%20scratch-013243)

---

## Algorithms

| # | Algorithm | Type | File |
|---|---|---|---|
| 1 | K-Nearest Neighbors (KNN) | Classification | `knn.py` |
| 2 | Naive Bayes | Classification | `naive_bayes.py` |
| 3 | Decision Tree | Classification | `decision_tree.py` |
| 4 | Support Vector Machine (SVM) | Classification | `svm.py` |
| 5 | Logistic Regression | Classification | `logistic_regression.py` |
| 6 | _add algorithm_ | | |
| 7 | _add algorithm_ | | |
| 8 | _add algorithm_ | | |
| 9 | _add algorithm_ | | |
| 10 | _add algorithm_ | | |
| 11 | _add algorithm_ | | |
| 12 | _add algorithm_ | | |

---

## Why From Scratch?

Library calls hide the mechanics. Writing each model by hand makes the following explicit:

- How the **loss function** is defined and minimized.
- How **hyperparameters** (learning rate, depth, `k`, regularization) change behavior.
- Where each model **overfits, underfits, or breaks**.
- What the library is actually doing when you call `.fit()` and `.predict()`.

---

## Design

Every model follows the same interface, mirroring scikit-learn conventions:

```python
model = Algorithm(**hyperparameters)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

This keeps models interchangeable and makes side-by-side comparison with scikit-learn straightforward.

---

## Getting Started

### Install

```bash
git clone https://github.com/AniketP2004/<repo-name>.git
cd <repo-name>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Example

```python
from knn import KNN

model = KNN(k=5)
model.fit(X_train, y_train)
print(model.predict(X_test))
```

---

## Results

Compare each implementation against its scikit-learn counterpart on the same dataset and split.

| Algorithm | Dataset | From-scratch score | scikit-learn score |
|---|---|---|---|
| KNN | _dataset_ | _ | _ |
| Naive Bayes | _dataset_ | _ | _ |
| Decision Tree | _dataset_ | _ | _ |
| SVM | _dataset_ | _ | _ |
| Logistic Regression | _dataset_ | _ | _ |
| ... | | | |

---

## Key Learnings

- Distance-based methods (KNN) are simple but scale poorly and need feature scaling.
- Probabilistic models (Naive Bayes) are fast and strong baselines despite their independence assumption.
- Tree-based models overfit without depth or split constraints.
- Margin-based and gradient-based models (SVM, Logistic Regression) depend heavily on learning rate and regularization.

---
