#https://www.youtube.com/watch?v=zbdong_h-x4
#https://www.youtube.com/watch?v=KiL74WsgxoA

import torch
import torch.nn as nn
import random

class MyLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)

    def forward(self, x_t, h_prev, c_prev):
        # x_t: [B, D], h_prev/c_prev: [B, H]
        z = torch.cat([x_t, h_prev], dim=1)
        i, f, g, o = torch.chunk(self.W(z), 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)
        return h_t, c_t

# -----------------------------
# 3) Encoder
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.cell = MyLSTMCell(emb_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: [B, T]
        B, T = src.size()
        h = src.new_zeros((B, self.hidden_dim), dtype=torch.float32).to(src.device)
        c = src.new_zeros((B, self.hidden_dim), dtype=torch.float32).to(src.device)

        emb = self.embedding(src)  # [B, T, E]
        for t in range(T):
            x_t = self.dropout(emb[:, t, :])
            h, c = self.cell(x_t, h, c)
        # return final states (no attention; MT-style context = final states)
        return h, c

# -----------------------------
# 4) Decoder (autoregressive, MT-style)
# -----------------------------
class Decoder(nn.Module):
    def __init__(self, num_tags, hidden_dim, dropout=0.1):
        super().__init__()
        self.num_tags = num_tags
        self.cell = MyLSTMCell(num_tags, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, num_tags)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_h, enc_c, trg_in, teacher_forcing_ratio=0.5):
        """
        enc_h, enc_c: [B, H] encoder final states
        trg_in: [B, T] shifted gold (starts with <SOS>)
        """
        B, T = trg_in.size()
        h, c = enc_h, enc_c
        logits_list = []

        input_token = trg_in[:, 0]  # first input should be <SOS>

        for t in range(T):
            # Convert token indices to one-hot vectors
            one_hot = F.one_hot(input_token, num_classes=self.num_tags).float()  # [B, num_tags]
            one_hot = self.dropout(one_hot)

            h, c = self.cell(one_hot, h, c)
            logits = self.fc_out(h)  # [B, num_tags]
            logits_list.append(logits.unsqueeze(1))

            if t + 1 < T:
                use_teacher = random.random() < teacher_forcing_ratio
                next_gold = trg_in[:, t + 1]
                next_pred = logits.argmax(dim=1)
                input_token = next_gold if use_teacher else next_pred

        return torch.cat(logits_list, dim=1)  # [B, T, num_tags]


# -----------------------------
# 5) Seq2Seq wrapper
# -----------------------------
class Seq2SeqTagger(nn.Module):
    def __init__(self, vocab_size, tag_size, emb_dim=128, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(vocab_size, emb_dim, hidden_dim, dropout)
        self.decoder = Decoder(tag_size, hidden_dim, dropout)
        self.tag_size = tag_size

    def forward(self, src, trg_in, teacher_forcing_ratio=0.5):
        enc_h, enc_c = self.encoder(src)
        return self.decoder(enc_h, enc_c, trg_in, teacher_forcing_ratio)