import os
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F # Import torch.nn.functional
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# -----------------------------
# 0) Repro + Device
# -----------------------------
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 1) Data: Brown (Universal tags)
# -----------------------------
import nltk
from nltk.corpus import brown

def prepare_brown(max_len=40, min_freq=1, use_universal=True):
    nltk.download("brown", quiet=True)
    if use_universal:
        nltk.download("universal_tagset", quiet=True)
        tagged_sents = brown.tagged_sents(tagset="universal")
    else:
        tagged_sents = brown.tagged_sents()

    # sentences: list[list[str]], tags: list[list[str]]
    sentences = [[w for w, t in sent] for sent in tagged_sents]
    tags = [[t for w, t in sent] for sent in tagged_sents]

    # word vocab
    word_freq = Counter(w.lower() for sent in sentences for w in sent)
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for w, c in word_freq.items():
        if c >= min_freq:
            word2idx[w] = len(word2idx)
    idx2word = {i: w for w, i in word2idx.items()}

    # tag vocab (include <PAD>, <SOS>)
    tag_set = sorted({t for ts in tags for t in ts})
    tag2idx = {"<PAD>": 0, "<SOS>": 1}
    for t in tag_set:
        tag2idx[t] = len(tag2idx)
    idx2tag = {i: t for t, i in tag2idx.items()}

    def encode_sentence(tokens):
        return [word2idx.get(w.lower(), word2idx["<UNK>"]) for w in tokens]

    def encode_tags(ts):
        return [tag2idx[t] for t in ts]

    # encode
    X = [encode_sentence(s) for s in sentences]
    Y = [encode_tags(ts) for ts in tags]

    # split (80/10/10)
    N = len(X)
    idxs = list(range(N))
    random.shuffle(idxs)
    n_train = int(0.8 * N)
    n_val = int(0.1 * N)
    train_idx = idxs[:n_train]
    val_idx = idxs[n_train:n_train + n_val]
    test_idx = idxs[n_train + n_val:]

    def subset(lst, ids):
        return [lst[i] for i in ids]

    X_tr, Y_tr = subset(X, train_idx), subset(Y, train_idx)
    X_va, Y_va = subset(X, val_idx), subset(Y, val_idx)
    X_te, Y_te = subset(X, test_idx), subset(Y, test_idx)

    return (X_tr, Y_tr, X_va, Y_va, X_te, Y_te,
            word2idx, idx2word, tag2idx, idx2tag, max_len)

def pad_to_len(seq, L, pad):
    if len(seq) >= L:
        return seq[:L]
    return seq + [pad] * (L - len(seq))

class BrownPOSDataset(Dataset):
    """
    For MT-style training:
    - encoder input: X  (word ids, padded)
    - decoder input: Y_in  = <SOS> + gold tags (shifted right)
    - decoder target: Y_out = gold tags + <PAD> (aligned with Y_in)
    """
    def __init__(self, X, Y, word2idx, tag2idx, max_len=40):
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.max_len = max_len

        PADw = word2idx["<PAD>"]
        PADt = tag2idx["<PAD>"]
        SOS = tag2idx["<SOS>"]

        self.X = [pad_to_len(x, max_len, PADw) for x in X]
        self.Y_in = [pad_to_len([SOS] + y, max_len, PADt) for y in Y]
        self.Y_out = [pad_to_len(y + [PADt], max_len, PADt) for y in Y]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return (torch.tensor(self.X[i], dtype=torch.long),
                torch.tensor(self.Y_in[i], dtype=torch.long),
                torch.tensor(self.Y_out[i], dtype=torch.long))
# -----------------------------
# 6) Training / Evaluation utils
# -----------------------------
def token_accuracy(logits, targets, pad_idx=0):
    """
    logits: [B, T, C], targets: [B, T]
    ignores positions where targets == pad_idx
    """
    with torch.no_grad():
        preds = logits.argmax(dim=-1)  # [B, T]
        mask = (targets != pad_idx)
        correct = ((preds == targets) & mask).sum().item()
        total = mask.sum().item()
        return correct / max(1, total)

def evaluate(model, loader, criterion, pad_idx):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    with torch.no_grad():
        for x, y_in, y_out in loader:
            x, y_in, y_out = x.to(device), y_in.to(device), y_out.to(device)
            logits = model(x, y_in, teacher_forcing_ratio=0.0)  # no TF at eval
            loss = criterion(logits.view(-1, model.tag_size), y_out.view(-1))
            acc = token_accuracy(logits, y_out, pad_idx)
            total_loss += loss.item()
            total_acc += acc
            n_batches += 1
    return total_loss / n_batches, total_acc / n_batches

def greedy_decode(model, sentence_tokens, word2idx, idx2tag, max_len, max_steps=None):
    """
    sentence_tokens: list[str] (already tokenized)
    Returns list[str] POS tags predicted.
    """
    model.eval()
    if max_steps is None:
        max_steps = max_len

    PADw = word2idx["<PAD>"]
    PADt = 0
    SOS = 1

    # Convert sentence to indices and pad
    x = [word2idx.get(w.lower(), word2idx["<UNK>"]) for w in sentence_tokens]
    x = pad_to_len(x, max_len, PADw)
    x = torch.tensor(x, dtype=torch.long, device=device).unsqueeze(0)  # [1, T]

    with torch.no_grad():
        enc_h, enc_c = model.encoder(x)
        # start token for decoder
        inp = torch.tensor([SOS], dtype=torch.long, device=device)  # [1]
        h, c = enc_h, enc_c
        tags = []
        for t in range(max_steps):
            # NEW: one-hot encode the input index instead of embedding
            one_hot = F.one_hot(inp, num_classes=model.tag_size).float()  # [1, C]

            h, c = model.decoder.cell(one_hot, h, c)
            logit = model.decoder.fc_out(h)        # [1, C]
            pred = logit.argmax(dim=1)             # [1]
            tags.append(pred.item())
            inp = pred  # feed previous prediction as next input

        tags = tags[:len(sentence_tokens)]
        tags_str = [idx2tag.get(i, "UNK") for i in tags]
        return tags_str


def evaluate_model(model, test_loader, tag2idx, idx2tag, device="cuda"):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for words, y_in, y_out in test_loader:  # unpack all 3
            words, y_in, y_out = words.to(device), y_in.to(device), y_out.to(device)

            # Forward pass
            outputs = model(words, y_in)  # [batch, seq_len, num_tags]
            predictions = torch.argmax(outputs, dim=-1)

            # Collect predictions and labels (ignoring <PAD>)
            for i in range(words.size(0)):
                for j in range(words.size(1)):
                    if y_out[i, j].item() != tag2idx["<PAD>"]:
                        y_true.append(y_out[i, j].item())
                        y_pred.append(predictions[i, j].item())

    return y_true, y_pred

def main():
    # Hyperparams
    MAX_LEN = 40
    MIN_FREQ = 1
    BATCH_SIZE = 64
    EMB_DIM = 128
    HID_DIM = 256
    DROPOUT = 0.2
    EPOCHS = 100
    LR = 1e-3
    TEACHER_FORCING = 0.6
    CLIP = 1.0
    SAVE_PATH = "pos_seq2seq_new_decoder.pt"

    # Data
    (X_tr, Y_tr, X_va, Y_va, X_te, Y_te,
     word2idx, idx2word, tag2idx, idx2tag, max_len) = prepare_brown(
        max_len=MAX_LEN, min_freq=MIN_FREQ, use_universal=True
    )

    train_ds = BrownPOSDataset(X_tr, Y_tr, word2idx, tag2idx, max_len=MAX_LEN)
    val_ds   = BrownPOSDataset(X_va, Y_va, word2idx, tag2idx, max_len=MAX_LEN)
    test_ds  = BrownPOSDataset(X_te, Y_te, word2idx, tag2idx, max_len=MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Model - **CHANGE HERE**
    vocab_size = len(word2idx)
    num_tags = len(tag2idx)
    model = Seq2SeqTagger(
        vocab_size=vocab_size,
        tag_size=num_tags,
        emb_dim=EMB_DIM,
        hidden_dim=HID_DIM,
        dropout=DROPOUT
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tag2idx["<PAD>"])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"Device: {device}")
    print(f"Vocab size: {vocab_size}, Tags: {num_tags}")
    print(f"Train/Val/Test sizes: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    best_val = float("inf")

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for x, y_in, y_out in train_loader:
            x, y_in, y_out = x.to(device), y_in.to(device), y_out.to(device)

            # **FORWARD WITH NEW DECODER**
            logits = model(x, y_in, teacher_forcing_ratio=TEACHER_FORCING)
            # logits: [B, T, num_tags]

            loss = criterion(logits.view(-1, num_tags), y_out.view(-1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss, val_acc = evaluate(model, val_loader, criterion, tag2idx["<PAD>"])
        print(f"Epoch {epoch:02d} | Train Loss {train_loss:.4f} | "
              f"Val Loss {val_loss:.4f} | Val Acc {val_acc*100:.2f}%")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "word2idx": word2idx,
                "idx2tag": idx2tag,
                "params": {
                    "EMB_DIM": EMB_DIM, "HID_DIM": HID_DIM, "DROPOUT": DROPOUT,
                    "MAX_LEN": MAX_LEN, "num_tags": num_tags, "vocab_size": vocab_size
                }
            }, SAVE_PATH)

    # Final Test
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device)["model_state"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, tag2idx["<PAD>"])
    print(f"Test Loss {test_loss:.4f} | Test Acc {test_acc*100:.2f}%")

    # Demo
    sample = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "."]
    pred_tags = greedy_decode(model, sample, word2idx, idx2tag, max_len=MAX_LEN)
    print("Sentence:", " ".join(sample))
    print("Pred POS:", pred_tags)

    # Metrics
    y_true, y_pred = evaluate_model(model, test_loader, tag2idx, idx2tag, device=device)

    acc = accuracy_score(y_true, y_pred)
    print("Overall Accuracy:", acc)

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(tag2idx.values()), zero_division=0
    )
    for tag, p, r, f, s in zip(tag2idx.keys(), prec, rec, f1, support):
        print(f"{tag:10s}  P={p:.3f}  R={r:.3f}  F1={f:.3f}  Support={s}")

    cm = confusion_matrix(y_true, y_pred, labels=list(tag2idx.values()))
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=True, fmt='d',
                cmap="Blues",
                xticklabels=tag2idx.keys(),
                yticklabels=tag2idx.keys())
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("POS Tagging Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    main()