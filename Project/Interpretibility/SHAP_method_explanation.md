# SHAP Method Explanation

This document explains the methods used in `Project/Interpretibility/shap.ipynb`.

## Method Overview

The notebook uses **SHAP (SHapley Additive exPlanations)** to explain the predictions of a fine-tuned BERT model for depression detection.

### Key Components:

*   **Library:** `shap`
*   **Model:** A fine-tuned BERT model (`bert-base-cased-finetuned`) loaded via `transformers`.
*   **Pipeline:** A `transformers.pipeline` for text classification is created and passed to the SHAP explainer.
*   **Explainer:** `shap.Explainer` is used. Given the input is a text classification pipeline, it defaults to using the **PartitionExplainer**, which is an exact SHAP value estimator for hierarchy-structured data (like text).

## Global Word Importance Calculation

The notebook defines a function `get_global_word_importance` to aggregate SHAP values across the test dataset to find the most indicative words for the "Depressed" class.

### Process:

1.  **Compute SHAP Values:** SHAP values are computed for a subset of the test text.
2.  **Aggregation:** The code iterates through each explanation (each text sample):
    *   It extracts tokens and their corresponding SHAP values.
    *   It skips special tokens: `[CLS]`, `[SEP]`, `[PAD]`, and empty strings.
    *   It accumulates the **Total SHAP Impact** (sum of SHAP values) and **Frequency** (count) for each unique word.
3.  **Averaging with Threshold:**
    *   It calculates the **Average Attribution** for each word.
    *   **Threshold:** A frequency threshold is applied. A word must appear **more than 5 times** (`threshold = 5`) to be assigned an average score. If the frequency is 5 or less, the average attribution is set to 0.
    
    $$ \text{Average Attribution} = \begin{cases} \frac{\text{Total SHAP Impact}}{\text{Frequency}} & \text{if Frequency} > 5 \\ 0 & \text{otherwise} \end{cases} $$

4.  **Ranking:** Words are sorted by their Average Attribution in descending order to identify the top words driving the model's depression predictions.
