# Customer Journey & Conversion Data Discovery

**Notebook:** [Customer Journey and Conversion Propensity](customer-journey-and-conversion-propensity.ipynb)

The exploratory analysis investigates shopping-session features and potential purchase intent, providing a research foundation for classification and decision-threshold evaluation.

## Input and reproducibility

- The notebook loads `online_shopping.csv` relative to its **working directory**. Supply a compatible dataset locally; it is not committed here.
- The application training workflow runs from the repository root using `src/train_model.py`; its default input file is also named `online_shopping.csv`.
- Data cleaning, validation and model comparison are documented in the [Implementation Guide](../docs/solution-design/solution-design-and-implementation-guide.md).
- Recorded model evaluation results do not demonstrate incremental campaign conversions or financial ROI.

For the model-assurance evidence and performance qualifications, see [Model Assurance](../model-assurance/README.md), and for context see the [Project Overview](../README.md).
