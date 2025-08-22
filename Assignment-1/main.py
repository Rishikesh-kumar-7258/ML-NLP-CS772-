import matplotlib.pyplot as plt
from utils import evaluate, prepare_brown_data
from models.hmm import HiddenMarkovModel
from sklearn.model_selection import train_test_split
import os, sys
import numpy as np

def get_training_data():
    X, Y, word2idx, idx2word, tag2idx, idx2tag = prepare_brown_data(min_freq=1)
    # Split into train/dev
    X_train, X_dev, Y_train, Y_dev = train_test_split(
        X, Y, test_size=0.1, random_state=42, shuffle=True
    )
    return X_train, Y_train, X_dev, Y_dev, word2idx, idx2word, tag2idx, idx2tag

def train_hmm(log_file="outputs/hmm.txt", bypass=False):

    file_path = "outputs/hmm_model.pkl"
    if not bypass and os.path.exists(file_path):
        print("[INFO] Loading saved model...")
        hmm_sup = HiddenMarkovModel.load(file_path)
        return hmm_sup

    # Redirect all print outputs to log file
    sys.stdout = open(log_file, "w")

    X_train, Y_train, X_dev, Y_dev, word2idx, idx2word, tag2idx, idx2tag = get_training_data()

    print("[INFO] Training new HMM model...")
    N = len(tag2idx)    # number of hidden states
    T = len(word2idx)   # number of observation symbols
    hmm_sup = HiddenMarkovModel(N, T, index2word=idx2word, word2index=word2idx, index2tag=idx2tag, tag2index=tag2idx, init="uniform")
    hmm_sup.fit_supervised(X_train, Y_train)
    hmm_sup.save(file_path)
    print(f"[INFO] Model saved at {file_path}")

    # ---- Evaluate supervised HMM ----
    tags_sup, f1_sup = evaluate(hmm_sup, X_dev, Y_dev, idx2tag, title="Supervised HMM")

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

    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/hmm_f1_scores.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[INFO] F1 scores plot saved at outputs/hmm_f1_scores.png")

    # Restore printing to terminal
    sys.stdout.close()
    sys.stdout = sys.__stdout__

    return hmm_sup


if __name__ == "__main__":
    
    print("Choose model:\n1. Hidden Markov Model\n2. Encoder Decoder\n3. LLM model\n4. Exit")

    model_choice = input("Enter your choice (1-4): ")
    while model_choice not in ["1", "2", "3", "4"]:
        print("Invalid choice. Please enter a number between 1 and 4.")
        model_choice = input("Enter your choice (1-4): ")

    if model_choice == "1":
        model = train_hmm(bypass=False)
    elif model_choice == "2":
        # model = train_encoder_decoder()
        pass
    elif model_choice == "3":
        # model = train_llm()
        pass
    elif model_choice == "4":
        print("Exiting...")
        sys.exit(0)

    sentence = input("Enter your sentence for decoding: ")
    model.predict_raw(sentence) 
