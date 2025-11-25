# Attention Visualization Method Explanation

This document explains the methods used in `Project/Interpretibility/attention_visualization.ipynb`.

## Method Overview

The notebook focuses on visualizing and analyzing the **Attention Mechanism** of the fine-tuned BERT model. It uses both individual instance visualization and global aggregation of attention weights.

### Key Components:

*   **Library:** `bertviz` (specifically `head_view` and `model_view`) for interactive visualizations.
*   **Model:** A fine-tuned BERT model (`bert-base-cased-finetuned`) loaded via `transformers`.
*   **Analysis Target:** The attention weights returned by the model (`outputs.attentions`).

## Visualization (Individual Instances)

The notebook defines a function `visualize_all_bertviz` to generate interactive HTML visualizations for specific text inputs.

*   **Head View:** Visualizes attention for one or more attention heads in a specific layer.
*   **Model View:** Visualizes attention across all layers and heads.

## Global Attention Impact Calculation

The notebook defines a function `get_global_attention_impact` to aggregate attention scores across the test dataset to find which words the model focuses on most globally.

### Process:

1.  **Batch Processing:** The function iterates through the test dataset in batches.
2.  **Model Inference:** For each batch, it runs the model and retrieves the attention weights.
3.  **Layer Selection:** It specifically selects the **Last Layer** (`outputs.attentions[-1]`) of the BERT model. The last layer is often considered most relevant for the final classification decision.
4.  **Head Averaging:** It calculates the average attention across all **12 Attention Heads** in that layer.
5.  **CLS Token Focus:** It extracts the attention weights **from the `[CLS]` token to all other tokens**.
    *   In BERT for sequence classification, the `[CLS]` token is used as the aggregate sequence representation. Therefore, looking at what the `[CLS]` token attends to provides insight into what parts of the input were most important for the embedding.
6.  **Aggregation:**
    *   It iterates through the tokens, cleaning them (removing `##` subword prefixes) and skipping special tokens (`[CLS]`, `[SEP]`, `[PAD]`).
    *   It accumulates the **Total Attention Score** and **Frequency** for each unique word.

### Output

The result is a DataFrame containing:
*   **Word:** The unique word.
*   **Total_Attention:** The sum of attention scores received from the `[CLS]` token across all instances.
*   **Frequency:** How often the word appeared.

This allows for identifying the "Top Attended Words" that the model consistently relies on for its representation.
