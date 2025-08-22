import nltk
from nltk.corpus import brown
from collections import Counter
import numpy as np
from models.hmm import HiddenMarkovModel
from tqdm import tqdm
from typing import List
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Brown corpus preparation (Universal tagset)
# =========================
def prepare_brown_data(min_freq: int = 3):
    """
    Returns:
      X: List[np.ndarray] of word indices
      Y: List[np.ndarray] of tag indices (gold, for supervised)
      word2idx, idx2word, tag2idx, idx2tag
    """
    

    nltk.download("brown", quiet=True)
    nltk.download("universal_tagset", quiet=True)

    tagged_sents = brown.tagged_sents(tagset="universal")

    # vocab with <UNK>
    words = [w.lower() for sent in tagged_sents for (w, _) in sent]
    freqs = Counter(words)
    vocab = ["<UNK>"] + [w for w, c in freqs.items() if c >= min_freq]
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}

    # tags (fixed small set)
    tags = sorted({t for sent in tagged_sents for (_, t) in sent})
    tag2idx = {t: i for i, t in enumerate(tags)}
    idx2tag = {i: t for t, i in tag2idx.items()}

    X, Y = [], []
    for sent in tagged_sents:
        ws, ts = zip(*sent)
        x = np.array([word2idx.get(w.lower(), 0) for w in ws], dtype=np.int32)
        y = np.array([tag2idx[t] for t in ts], dtype=np.int32)
        X.append(x)
        Y.append(y)

    # print(f"Vocab size: {len(vocab)}, #Tags: {len(tags)}, #Sentences: {len(X)}")
    return X, Y, word2idx, idx2word, tag2idx, idx2tag

# =========================
# Simple evaluation (tag accuracy)
# =========================
def evaluate(hmm: HiddenMarkovModel,
             X: List[np.ndarray],
             Y: List[np.ndarray]) -> float:
    """
    Decodes with Viterbi and computes micro accuracy vs. gold tags.
    """
    correct = 0
    total = 0
    for x, y in tqdm(zip(X, Y), total=len(X)):
        path, _ = hmm.predict(x)
        L = min(len(path), len(y))
        correct += int(np.sum(path[:L] == y[:L]))
        total += L
    return correct / max(total, 1)

# -------- Metrics --------
def evaluate(model, X_dev, Y_dev, idx2tag, title="Model"):
    # Predict tags for dev sentences
    Y_pred = []
    for x in X_dev:
        path, _ = model.predict(np.array(x, dtype=np.int32))
        Y_pred.append([idx2tag[i] for i in path])

    # Flatten to token level
    y_true = [idx2tag[tag] for sent in Y_dev for tag in sent]
    y_pred = [tag for sent in Y_pred for tag in sent]

    # Overall precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    print(f"\n===== {title} =====")
    print("Overall Precision:", round(precision, 3))
    print("Overall Recall:", round(recall, 3))
    print("Overall F1:", round(f1, 3))

    # Per-tag precision, recall, F1
    report = classification_report(y_true, y_pred, digits=3, output_dict=True, zero_division=0)
    # print(report)
    report_df = pd.DataFrame(report)
    print(report_df.T)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=idx2tag.values())

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap="Blues", ax=ax, colorbar=False)

    plt.title("Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.savefig("outputs/hmm_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()


    # Extract per-tag F1 for plotting
    tags = list(report.keys())[:-3]  # drop summary rows
    f1_scores = [report[tag]['f1-score'] for tag in tags]

    return tags, f1_scores