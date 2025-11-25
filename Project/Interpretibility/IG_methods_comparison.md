# Integrated Gradients Method Comparison

This document explains the methods used and the differences between the following two Jupyter notebooks:
1. `Project/Interpretibility/IG_averaged.ipynb`
2. `Project/Interpretibility/ig_averaged_with_threshold.ipynb`

## Methods Used

Both notebooks employ **Integrated Gradients (IG)**, an interpretability technique used to attribute the model's predictions to its input features (in this case, words).

### Key Components:
*   **Library:** `transformers-interpret` is used, specifically the `SequenceClassificationExplainer`.
*   **Model:** A fine-tuned BERT model (`bert-base-cased-finetuned`) trained for depression detection.
*   **Dataset:** The `erisk_processed` dataset (test split).
*   **Process:**
    1.  Iterate through the test set texts.
    2.  Compute attribution scores for each word in the text using Integrated Gradients.
    3.  Accumulate the **Total Attribution Score** and **Frequency** (count) for each unique word across the entire test set.
    4.  Calculate the **Average Attribution Score** for each word:
        $$ \text{Average Attribution} = \frac{\text{Total Attribution Score}}{\text{Frequency}} $$
    5.  Sort words by their average attribution to identify the most indicative words for the "Depressed" class (Label 1).

## Differences

The primary and only significant difference between the two files lies in the **Frequency Threshold** used when calculating the average attribution score. This threshold determines which words are considered significant enough to be included in the final analysis.

### 1. `IG_averaged.ipynb`
*   **Threshold:** Frequency > 1
*   **Logic:**
    ```python
    global_word_scores[key]['avg_score'] = value['total_score'] / value['frequency'] if value['frequency'] > 1 else 0
    ```
*   **Effect:** This notebook includes almost all words, excluding only those that appear exactly once in the entire test set. This results in a larger vocabulary of attributed words but may include noise from rare words.

### 2. `ig_averaged_with_threshold.ipynb`
*   **Threshold:** Frequency > 8
*   **Logic:**
    ```python
    global_word_scores[key]['avg_score'] = value['total_score'] / value['frequency'] if value['frequency'] > 8 else 0
    ```
*   **Effect:** This notebook applies a stricter filter, requiring a word to appear **more than 8 times** to be assigned an average score (otherwise it is 0). This focuses the analysis on more common words, potentially reducing noise and highlighting more robust indicators of depression, but risks missing significant rare words.

## Summary Table

| Feature | `IG_averaged.ipynb` | `ig_averaged_with_threshold.ipynb` |
| :--- | :--- | :--- |
| **Method** | Integrated Gradients (Global Average) | Integrated Gradients (Global Average) |
| **Frequency Threshold** | **> 1** (at least 2 occurrences) | **> 8** (at least 9 occurrences) |
| **Goal** | Broad analysis of word importance | Focused analysis on frequent indicators |
