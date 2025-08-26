

import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.W = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("Expected input shape (batch_size, seq_len, input_dim)")

        batch_size, seq_len, _ = x.size()
        device = x.device

        h_t = torch.zeros(batch_size, self.hidden_dim, device=device)
        c_t = torch.zeros(batch_size, self.hidden_dim, device=device)

        all_ht = []

        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch_size, input_dim)
            combined = torch.cat((x_t, h_t), dim=1)  # (batch_size, input_dim + hidden_dim)

            gates = self.W(combined)
            i_t, f_t, g_t, o_t = torch.chunk(gates, 4, dim=1)

            i_t = torch.sigmoid(i_t)
            f_t = torch.sigmoid(f_t)
            g_t = torch.tanh(g_t)
            o_t = torch.sigmoid(o_t)

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            all_ht.append(h_t)

        outputs = torch.stack(all_ht, dim=1)

        return outputs


class Encoder(nn.Module):
  def __init__(self, input_dim, hidden_dim):
    super().__init__()

    self.encoder = LSTM(input_dim, hidden_dim)
    self.fc = nn.Linear(hidden_dim, hidden_dim)

  def forward(self, x):
    
    x = self.encoder(x)
    x = self.fc(x)


    return x
  

class Decoder(nn.Module):
  def __init__(self, hidden_dim, output_dim):
    super().__init__()

    self.decoder = LSTM(hidden_dim, hidden_dim)
    self.fc = nn.Linear(hidden_dim, output_dim)

  def forward(self, x):
    x = self.decoder(x)
    x = self.fc(x)

    return x