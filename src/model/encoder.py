import torch
import torch.nn as nn

class Conv1DEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=256, out_dim=256):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, emb_dim)

        self.conv_net = nn.Sequential(
            nn.Conv1d(emb_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, out_dim, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # x: (B, L)
        x = self.embedding(x)          # (B, L, E)
        x = x.transpose(1, 2)          # (B, E, L)
        z = self.conv_net(x)           # (B, D, L)
        z = z.transpose(1, 2)          # (B, L, D)
        return z
