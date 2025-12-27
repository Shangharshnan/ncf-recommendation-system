import torch
import torch.nn as nn

# Sample dummy NCF model
class DummyNCF(nn.Module):
    def __init__(self, num_users=5, num_items=5, embed_dim=8):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim*2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, user, item):
        u = self.user_embed(user)
        i = self.item_embed(item)
        x = torch.cat([u, i], dim=-1)
        return self.fc(x)

# Initialize model
num_users = 5
num_items = 5
model = DummyNCF(num_users, num_items)

# Dummy user-item pairs
users = torch.tensor([0, 1, 2, 3, 4])
items = torch.tensor([0, 1, 2, 3, 4])

print("Predicted interaction scores (0-1):\n")
for u in users:
    for i in items:
        pred = model(torch.tensor([u]), torch.tensor([i]))
        print(f"User {u} -> Item {i}: {pred.item():.4f}")
