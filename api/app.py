from flask import Flask, request, jsonify
import torch
import os
import sys

# Add project root to sys.path so Python can find model module
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from model.ncf_model import NCF

app = Flask(__name__)

# Set number of users and items (should match your train.py setup)
NUM_USERS = 5
NUM_ITEMS = 5

# Initialize the model
model = NCF(NUM_USERS, NUM_ITEMS)

# Load the trained model dynamically from project root
model_path = os.path.join(PROJECT_ROOT, "ncf_model.pt")
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

@app.route("/")
def home():
    return jsonify({"status": "NCF Recommendation API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    user_id = torch.tensor([data["user_id"]])
    item_id = torch.tensor([data["item_id"]])

    with torch.no_grad():
        score = model(user_id, item_id).item()

    return jsonify({"prediction_score": score})

if __name__ == "__main__":
    app.run(debug=True)
