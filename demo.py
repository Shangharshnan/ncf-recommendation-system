import torch
import torch.nn as nn
import pandas as pd

# Dummy NCF Model Definition
class DummyNCF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=8):
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

# Load Sample Dataset
data = pd.read_csv("data/sample_data.csv")
print("Sample Data:\n", data)

# Get unique users and items
unique_users = data['user_id'].unique()
unique_items = data['item_id'].unique()

num_users = len(unique_users)
num_items = len(unique_items)

# Initialize Model
model = DummyNCF(num_users=num_users, num_items=num_items)

# Predict Interaction Scores
print("\nPredicted interaction scores (0-1):\n")
for _, row in data.iterrows():
    user = torch.tensor([row['user_id']])
    item = torch.tensor([row['item_id']])
    pred = model(user, item)
    print(f"User {row['user_id']} -> Item {row['item_id']}: {pred.item():.4f}")
