# Novelty Method Explanation

This document explains the method used in `Project/novelty/novelty.ipynb`.

## Method Overview

The notebook implements a **Generative Explanation Framework** that uses a Large Language Model (T5) to provide psychological interpretations for the predictions of a fine-tuned BERT model.

### Key Components

1.  **Target Model (BERT):**
    *   A fine-tuned `bert-base-cased` model is used for sequence classification (likely binary classification, given labels 0 and 1).
    *   This model makes the initial prediction on the input text.

2.  **Explainability (Attributions):**
    *   **Library:** `transformers_interpret`
    *   **Method:** `SequenceClassificationExplainer` (likely based on Integrated Gradients).
    *   **Purpose:** To identify the most important words (attributions) that influenced the BERT model's prediction.

3.  **Generative Model (T5):**
    *   **Model:** `google-t5/t5-base`
    *   **Purpose:** To generate a natural language explanation based on the input text, the BERT model's output, and the computed word attributions.

## Process Flow

The method follows a pipeline approach:

1.  **Input Processing:**
    *   The input text is truncated to fit the BERT model's token limit (512 tokens).

2.  **Prediction & Attribution:**
    *   The BERT model predicts the class (Label and Score).
    *   The Explainer computes attribution scores for each word in the input.
    *   The top words (sorted by attribution score) are selected.

3.  **Prompt Construction:**
    *   A structured JSON prompt is created containing:
        *   `task`: "Explain a psychological reasoning behind the model output."
        *   `input_text`: The original (truncated) text.
        *   `model_output`: The label and confidence score from BERT.
        *   `attribution_scores`: A list of the top contributing words and their scores.
        *   `instruction`: A specific instruction to provide a "concise, evidence-based psychological interpretation" considering emotional tone, linguistic cues, etc.

4.  **Explanation Generation:**
    *   The T5 model receives this JSON prompt and generates a text response explaining *why* the BERT model might have made that prediction, effectively translating the numerical attributions and raw text into a human-readable psychological insight.

5.  **Data Generation:**
    *   The notebook iterates through a balanced subset of the test dataset (positive and negative samples).
    *   It generates these prompts for each sample and saves them to a JSON file (`bert_generated_prompts.json`).

## Goal

The "novelty" of this approach likely lies in **combining feature-based explainability (attributions) with generative AI** to produce semantic, context-aware explanations, rather than just showing a list of important words.
