import json
import shutil
from pathlib import Path

# Read the champion registry
with open("registry/champion.json", "r") as f:
    champion = json.load(f)

print(f"Champion model: {champion['model_name']}")
print(f"Champion F1 score: {champion['f1_score']}")

# Ensure deployment folder exists
Path("deployment").mkdir(parents=True, exist_ok=True)

# Copy champion model into deployment folder
shutil.copy("outputs/model.pkl", "deployment/model.pkl")
print("Copied model.pkl to deployment/")

# Copy API code into deployment folder
shutil.copy("api/app.py", "deployment/app.py")
print("Copied app.py to deployment/")

print("Deployment folder ready.")
