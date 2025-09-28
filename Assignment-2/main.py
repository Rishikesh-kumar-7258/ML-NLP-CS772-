# week1_train_and_eval.py
import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import Dataset
from models.lstm_model import Encoder, Decoder, Seq2Seq, Attention
from utils.data_loader import load_aksharantar_hindi
from utils.evaluation import evaluate_model

# === 1. Load Data ===
print("1️⃣ Loading data...")
dataset = load_aksharantar_hindi(max_train=100_000, seed=42)
train_data = dataset["train"]
test_data = dataset["test"]

# === 2. Build Vocab ===
SPECIALS = ["<pad>", "<unk>", "<sos>", "<eos>"]
def build_vocab(data, key):
    chars = set()
    for ex in data:
        chars.update(list(ex[key]))
    vocab = {tok: i for i, tok in enumerate(SPECIALS)}
    for ch in sorted(chars):
        vocab[ch] = len(vocab)
    return vocab

src_vocab = build_vocab(train_data, "source")
trg_vocab = build_vocab(train_data, "target")
inv_trg_vocab = {v: k for k, v in trg_vocab.items()}

def collate_fn(batch):
    src_list, trg_list = [], []
    for ex in batch:
        src_ids = [src_vocab.get(c, src_vocab["<unk>"]) for c in ex["source"]]
        trg_ids = [trg_vocab["<sos>"]] + [trg_vocab.get(c, trg_vocab["<unk>"]) for c in ex["target"]] + [trg_vocab["<eos>"]]
        src_list.append(torch.tensor(src_ids))
        trg_list.append(torch.tensor(trg_ids))
    return (
        torch.nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=0),
        torch.nn.utils.rnn.pad_sequence(trg_list, batch_first=True, padding_value=0)
    )

train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, collate_fn=collate_fn)

# === 3. Model Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HID_DIM = 512
EMB_DIM = 256
N_LAYERS = 2
DROPOUT = 0.3

enc = Encoder(len(src_vocab), EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
attn = Attention(HID_DIM)
dec = Decoder(len(trg_vocab), EMB_DIM, HID_DIM, N_LAYERS, DROPOUT, attn)
model = Seq2Seq(enc, dec, device).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

# === 4. Training Loop ===
print("2️⃣ Training LSTM...")
epochs = 10
losses = []

model.train()
for epoch in range(epochs):
    total_loss = 0
    for src, trg in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg)
        loss = criterion(output[:, 1:].reshape(-1, len(trg_vocab)), trg[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

# === 5. Plot Training Loss ===
plt.figure(figsize=(8, 4))
plt.plot(losses, marker='o')
plt.title("LSTM Training Loss (Week 1)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("lstm_training_loss.png")
print("📈 Training loss plot saved as 'lstm_training_loss.png'")

# === 6. Greedy Decoding Function ===
def greedy_decode(src_text, max_len=50):
    model.eval()
    src_ids = [src_vocab.get(c, src_vocab["<unk>"]) for c in src_text]
    src_tensor = torch.LongTensor([src_ids]).to(device)
    with torch.no_grad():
        enc_out, hidden, cell = model.encoder(src_tensor)
    trg_ids = [trg_vocab["<sos>"]]
    for _ in range(max_len):
        input_token = torch.LongTensor([trg_ids[-1]]).to(device)
        with torch.no_grad():
            output, hidden, cell = model.decoder(input_token, hidden, cell, enc_out)
        pred_id = output.argmax(1).item()
        trg_ids.append(pred_id)
        if pred_id == trg_vocab["<eos>"]:
            break
    # Convert to string
    chars = []
    for idx in trg_ids[1:]:
        if idx in [trg_vocab["<sos>"], trg_vocab["<eos>"], trg_vocab["<pad>"]]:
            continue
        chars.append(inv_trg_vocab.get(idx, "?"))
    return "".join(chars)

# === 7. Evaluate on Full Test Set ===
print("3️⃣ Evaluating on full test set...")
results = evaluate_model(greedy_decode, test_data)
print(f"✅ Test Results (LSTM, Greedy Decoding):")
print(f"   - Character-level F1: {results['char_f1']:.4f}")
print(f"   - Word-level Accuracy: {results['word_acc']:.4f}")
print(f"   - Total test examples: {results['total']}")

# Save results
with open("week1_results.txt", "w") as f:
    f.write(f"Character-level F1: {results['char_f1']:.4f}\n")
    f.write(f"Word-level Accuracy: {results['word_acc']:.4f}\n")
    f.write(f"Total test examples: {results['total']}\n")

# === 8. Save Model & Vocab ===
torch.save(model.state_dict(), "lstm_week1.pt")
torch.save({"src_vocab": src_vocab, "trg_vocab": trg_vocab}, "vocab_week1.pt")
print("💾 Model and vocab saved.")