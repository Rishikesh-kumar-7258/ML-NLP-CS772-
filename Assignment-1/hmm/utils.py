import nltk
from nltk.corpus import brown
from collections import Counter
import numpy as np
from hmm import HiddenMarkovModel
from tqdm import tqdm
from typing import List
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

# =========================
# Brown corpus preparation (Universal tagset)
# =========================

def split_brown_dataset(sentences, tags, test_size=0.1, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(sentences, tags, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def prepare_brown_dataset(min_freq=1, seed=42):
    nltk.download("brown", quiet=True)
    nltk.download("universal_tagset", quiet=True)
    tagged_sents = brown.tagged_sents(tagset="universal")

    # Extract raw sentences and tags
    sentences = [[w for w, t in sent] for sent in tagged_sents]
    tags = [[t for w, t in sent] for sent in tagged_sents]

    # Split dataset first
    X_train_raw, X_dev_raw, y_train_raw, y_dev_raw = split_brown_dataset(sentences, tags, test_size=0.1, random_state=seed)

    # Build word vocab from training data
    word_freq = Counter(w.lower() for sent in X_train_raw for w in sent)
    word2idx = {"<UNK>": 0}  # HMM does not need <PAD>
    for w, c in word_freq.items():
        if c >= min_freq:
            word2idx[w] = len(word2idx)
    idx2word = {i: w for w, i in word2idx.items()}

    # Build tag vocab from whole data
    tag_set = sorted({t for ts in tags for t in ts})
    tag2idx = {t: i for i, t in enumerate(tag_set)}
    idx2tag = {i: t for t, i in tag2idx.items()}


    # Encoding functions
    def encode_sentence(tokens):
        return [word2idx.get(w.lower(), word2idx["<UNK>"]) for w in tokens]

    def encode_tags(ts):
        return [tag2idx[t] for t in ts]

    X_train = [encode_sentence(s) for s in X_train_raw]
    Y_train = [encode_tags(ts) for ts in y_train_raw]

    X_dev = [encode_sentence(s) for s in X_dev_raw]
    Y_dev = [encode_tags(ts) for ts in y_dev_raw]

    return X_train, Y_train, X_dev, Y_dev, word2idx, idx2word, tag2idx, idx2tag


# =========================
# Simple evaluation (tag accuracy)
# =========================
def evaluate(model, X, Y, idx2tag, title="Model"):
    """
    Evaluates an HMM model on given sequences.

    Returns:
        tags: list of tag names (excluding <PAD>/<SOS>)
        f1_scores: list of F1 scores per tag
        overall_metrics: dict with overall precision, recall, f1, accuracy
    """
    all_preds, all_true = [], []

    for x_seq, y_seq in zip(X, Y):
        pred_seq, _ = model.predict(np.array(x_seq, dtype=np.int32))

        # Align lengths
        min_len = min(len(pred_seq), len(y_seq))
        pred_seq = pred_seq[:min_len]
        y_seq = y_seq[:min_len]

        for y_true_idx, y_pred_idx in zip(y_seq, pred_seq):
            tag_true = idx2tag[int(y_true_idx)]
            tag_pred = idx2tag[int(y_pred_idx)]

            if tag_true in ("<PAD>", "<SOS>"):
                continue

            all_true.append(tag_true)
            all_preds.append(tag_pred)

    # ---- Compute classification report ----
    labels = [t for t in idx2tag.values() if t not in ("<PAD>", "<SOS>")]
    report_dict = classification_report(all_true, all_preds, labels=labels, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report_dict).T

    # Add per-class accuracy: correct predictions for class / total instances
    total_instances = len(all_true)
    class_acc = {}
    for cls in labels:
        support = report_dict[cls]["support"]
        tp = int(report_dict[cls]["recall"] * support)
        class_acc[cls] = tp / total_instances
    report_df['accuracy'] = 0.0
    for cls, acc in class_acc.items():
        if cls in report_df.index:
            report_df.at[cls, 'accuracy'] = acc

    # ---- Overall metrics ----
    overall_accuracy = accuracy_score(all_true, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true, all_preds, average='weighted', zero_division=0
    )
    overall_metrics = {
        "precision": precision,
        "recall": recall,
        "f1-score": f1,
        "accuracy": overall_accuracy
    }

    print(f"\n===== {title} classification report =====")
    print(report_df)

    # ---- Save confusion matrix ----
    cm = confusion_matrix(all_true, all_preds, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap="Blues", ax=ax, colorbar=False)

    plt.title(f"Confusion Matrix: {title}", fontsize=16)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(fontsize=12)
    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)
    cm_path = f"outputs/hmm_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Confusion matrix saved at {cm_path}")

    # Per-class tags and F1 scores (for plotting)
    tags = labels
    f1_scores = [report_dict[tag]["f1-score"] for tag in tags]

    return tags, f1_scores, overall_metrics