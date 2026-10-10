# Machine Learning — Coursework & Classification Project

A collection of practical assignments and a progressively developed classification project from my **Machine Learning course at Politecnico di Torino**.

This repository documents how I learned and implemented foundational machine learning methods in Python, moving from data exploration and probabilistic modeling to model evaluation, calibration, and more advanced classifiers.

> **Repository status:** Academic coursework / learning archive. This is **not** a packaged production ML library or a single end-to-end application. The exercises were developed in a course environment, and the scripts have not all been retested with current dependency versions.

## What's inside

| Directory | Content |
| --- | --- |
| [`Labs/`](Labs/) | 11 practical lab folders with exercises, datasets, and selected reference materials |
| [`Project/`](Project/) | Successive stages of a binary classification project, from analysis to modeling and evaluation |

### Course labs

The lab folders cover:

- **Data exploration and visualisation:** NumPy arrays, statistics, histograms, scatter plots, and the Iris dataset.
- **Dimensionality reduction:** Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA).
- **Probability and generative models:** multivariate Gaussian log densities, Gaussian classifiers, and Naive Bayes variants.
- **Text classification:** simple probabilistic language models applied to text from Dante's *Divine Comedy*.
- **Decision theory and evaluation:** log-likelihood ratios, Bayes decision rules, confusion matrices, and detection cost functions.
- **Discriminative models:** logistic regression and support vector machines, including kernel methods.
- **Gaussian mixture models:** Expectation-Maximization (EM) and Linde–Buzo–Gray (LBG) style component expansion.
- **Model comparison:** validation, score calibration, and evaluation of alternative classifiers.

The exercises illustrate the mathematical foundations and implementation details of the algorithms, rather than hiding every step behind a high-level library.

### Progressive classification project

The [`Project/`](Project/) directory contains versions of a classification assignment across the course:

| Stage | Main focus |
| --- | --- |
| `Lab02_p1`–`Lab04_p3` | Data inspection and representation / dimensionality-reduction experiments |
| `Lab05_p4`–`Lab07_p5` | Generative classifiers and initial binary classification experiments |
| `Lab08_p6`–`Lab09_p7` | Logistic regression, SVMs, and detection-cost analysis |
| `Lab10_p8` | Gaussian Mixture Models and comparison of component/covariance configurations |
| `Lab11_p9` | Final-stage model evaluation, score calibration and comparison |

Different stages retain their own copies of training data and utilities so they can be read in the context of their corresponding assignment.

## Technologies

**Python**, **NumPy**, **SciPy**, **Matplotlib**, and **pandas** appear in the source files. Dependencies vary by exercise.

## How to explore the repository

1. Start with [`Labs/`](Labs/) for individual concepts or [`Project/`](Project/) to follow the classification project over time.
2. Read the corresponding lab handout (where included) and source code before executing an exercise.
3. Run a script **from its own directory**: several scripts reference datasets by relative filename.
4. Install only the libraries the selected script imports. A starting environment for many exercises is:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install numpy scipy matplotlib pandas
```

For example, from the repository root:

```bash
cd Labs/Lab02
python Lab02.py
```

**Note:** This is an illustrative launch command, not a verified clean-environment test. Some scripts may open plots, write output files, or depend on exact course data and working-directory assumptions.

## Evaluation methods

Later exercises use **minimum detection cost (minDCF)** and **actual detection cost (actDCF)**, which evaluate classification decisions at specified application priors and error costs. They are different from simply reporting accuracy. These experiments also explore likelihood-ratio scores and score calibration.

## Repository notes and limitations

- Some lab folders include **instructor/reference solutions** under `Prof_solution/`. Those files are retained as course materials and **are not claimed as my original work**.
- Coursework handouts and example datasets are included as they appeared in the archive. Their original attribution and reuse permissions should be checked before wider redistribution.
- `Project/` contains successive iterations of the same assignment; repeated datasets and helper modules are expected, not independent projects.
- Some original exploration files are named `try.py`, `trylab07.py`, or `try_later_delete.py`. These are kept to preserve the learning history.
- No comprehensive automated test suite, dependency lockfile, or reproducibility guarantee is provided.
- Generated Python bytecode caches were removed during the portfolio cleanup; substantive source files and data were preserved.

## Why I'm keeping this repository

It represents the foundations behind my later work in machine learning and AI: understanding the data, implementing core models, choosing evaluation metrics, and checking what actually improves performance.

---
*Academic learning repository — Politecnico di Torino.*
