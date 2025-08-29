import random
import matplotlib.pyplot as plt
from utils import evaluate, prepare_brown_dataset
from hmm import HiddenMarkovModel
import os, sys
import numpy as np

# -----------------------------
# 0) Repro + Device
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def train_hmm(log_file="outputs/hmm.txt", bypass=False):
    file_path = "outputs/hmm_model.pkl"
    if not bypass and os.path.exists(file_path):
        print("[INFO] Loading saved model...")
        hmm_sup = HiddenMarkovModel.load(file_path)
        return hmm_sup

    # Redirect all print outputs to log file
    os.makedirs("outputs", exist_ok=True)
    sys.stdout = open(log_file, "w")

    # ---- Prepare dataset ----
    X_train, Y_train, X_dev, Y_dev, word2idx, idx2word, tag2idx, idx2tag = prepare_brown_dataset(
        min_freq=1, seed=SEED
    )

    # ---- Train HMM ----
    print("[INFO] Training new HMM model...")
    N = len(tag2idx)   # number of hidden states
    T = len(word2idx)  # number of observation symbols

    hmm_sup = HiddenMarkovModel(
        N, T,
        index2word=idx2word, word2index=word2idx,
        index2tag=idx2tag, tag2index=tag2idx,
        init="uniform"
    )
    hmm_sup.fit_supervised(X_train, Y_train)
    hmm_sup.save(file_path)
    print(f"[INFO] Model saved at {file_path}")

    # ---- Evaluate supervised HMM ----
    tags_sup, f1_sup, overall_metrics = evaluate(hmm_sup, X_dev, Y_dev, idx2tag, title="Supervised HMM")

    # ---- Print overall metrics ----
    print("\n===== Overall metrics =====")
    print(f"Precision: {overall_metrics['precision']:.3f}")
    print(f"Recall:    {overall_metrics['recall']:.3f}")
    print(f"F1-score:  {overall_metrics['f1-score']:.3f}")
    print(f"Accuracy:  {overall_metrics['accuracy']:.3f}")

    # ---- Save F1 scores as bar plot ----
    plt.figure(figsize=(12, 6))
    x = range(len(tags_sup))
    bar_width = 0.6

    plt.bar(x, f1_sup, width=bar_width, label="Supervised", color="skyblue")
    plt.xticks(x, tags_sup, rotation=45)
    plt.xlabel("POS Tags")
    plt.ylabel("F1 Score")
    plt.title("Per-tag F1 Scores: Supervised HMM")
    plt.legend()
    plt.tight_layout()

    f1_plot_path = "outputs/hmm_f1_scores.png"
    plt.savefig(f1_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] F1 scores plot saved at {f1_plot_path}")

    # Restore printing to terminal
    sys.stdout.close()
    sys.stdout = sys.__stdout__

    print("[INFO] Training and evaluation completed.")

    return hmm_sup

if __name__ == "__main__":

    model = train_hmm(bypass=False)

    sentence = input("Enter your sentence for decoding: ")
    model.predict_raw(sentence) 
