import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from ncf_model import NCF

# Dummy interaction data (safe)
data = pd.DataFrame({
    "user_id": [0, 1, 2, 3, 4],
    "item_id": [0, 1, 2, 3, 4],
    "rating": [1, 1, 1, 1, 1]
})

num_users = data["user_id"].max() + 1
num_items = data["item_id"].max() + 1

user_ids = torch.tensor(data["user_id"].values)
item_ids = torch.tensor(data["item_id"].values)
ratings = torch.tensor(data["rating"].values, dtype=torch.float32)

model = NCF(num_users, num_items)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    optimizer.zero_grad()
    preds = model(user_ids, item_ids).squeeze()
    loss = criterion(preds, ratings)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "ncf_model.pt")
print("Model trained and saved.")
