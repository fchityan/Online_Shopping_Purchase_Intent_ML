# Online Shopping Purchase Intent ML

## Project Overview

This project predicts whether an online shopping session is likely to lead to a purchase. It uses session behaviour to explore how machine learning can support more targeted marketing decisions while managing the trade-off between missed buyers and unnecessary targeting.

## Problem

Purchase sessions are comparatively uncommon. A model that predicts most sessions will not convert may look accurate while failing to identify prospective buyers. The project therefore evaluates ranking quality and the precision–recall balance, not accuracy alone.

## Approach

- Prepare shopping-session features using a consistent preprocessing pipeline.
- Compare Logistic Regression, Random Forest, and LightGBM.
- Select the model and tune its classification threshold using validation data.
- Evaluate the chosen approach on a held-out test set.
- Track model experiments and produce monitoring reports.

## Key Features

- Repeatable model comparison and threshold selection
- Versioned predictions through a FastAPI service
- Feature-importance, data-quality, and drift reporting
- Model and experiment tracking

## Results and Limitations

The recorded **LightGBM locked-test ROC-AUC is 0.9253**. At a decision threshold of 0.50, reported precision was **0.7986** and recall **0.6126**; lowering the threshold to 0.35 increased recall to **0.6702** while reducing precision to **0.6900**.

These results apply to the evaluated dataset, **not proof of improved real-world conversions or campaign returns**.

## Technology Stack

**Python · scikit-learn · LightGBM · MLflow · FastAPI · Docker · pytest**

## Explore the Project

- [Exploratory notebook](eda.ipynb)
- [Technical guide — model results, commands, and architecture](docs/technical_guide.md)
- [Deployment and operations notes](docs/production_runbook.md)
- [Security guidance](SECURITY.md)

To reproduce training, supply a compatible shopping-session CSV and follow the instructions in the technical guide.
