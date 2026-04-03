import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# 1. Load Data (GitHub-friendly)
# -----------------------------
df = pd.read_csv("data/generated_ml_dataset_surrogate_ready_corrected.csv")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

# -----------------------------
# 2. Features and Targets
# -----------------------------
feature_cols = ["Tin", "DNI", "Mhtf", "Tamb", "Pressurehtf", "K"]
target_col = "Eff"

X = df[feature_cols].values
y = df[target_col].values.reshape(-1, 1)

# -----------------------------
# 3. Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 5. Tensors
# -----------------------------
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# -----------------------------
# 6. Model
# -----------------------------
class ThermalNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)

model = ThermalNN(input_dim=X_train.shape[1])

# -----------------------------
# 7. Training setup
# -----------------------------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# 8. Train
# -----------------------------
epochs = 1000

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.3f}")

# -----------------------------
# 9. Train metrics
# -----------------------------
model.eval()
with torch.no_grad():
    y_pred_train = model(X_train_tensor).numpy()

train_mae = mean_absolute_error(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
train_r2 = r2_score(y_train, y_pred_train)

print("\nTrain Results")
print(f"MAE  = {train_mae:.3f}")
print(f"RMSE = {train_rmse:.3f}")
print(f"R²   = {train_r2:.3f}")

# -----------------------------
# 10. Test metrics
# -----------------------------
with torch.no_grad():
    y_pred_test = model(X_test_tensor).numpy()

test_mae = mean_absolute_error(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_r2 = r2_score(y_test, y_pred_test)

print("\nTest Results")
print(f"MAE  = {test_mae:.3f}")
print(f"RMSE = {test_rmse:.3f}")
print(f"R²   = {test_r2:.3f}")

# -----------------------------
# 11. External validation
# -----------------------------
ext_df = pd.read_csv("data/validation_1000_random_combinations.csv")

X_ext = ext_df[feature_cols].values
y_ext = ext_df[target_col].values.reshape(-1, 1)

X_ext_scaled = scaler.transform(X_ext)
X_ext_tensor = torch.tensor(X_ext_scaled, dtype=torch.float32)

with torch.no_grad():
    y_ext_pred = model(X_ext_tensor).numpy()

ext_mae = mean_absolute_error(y_ext, y_ext_pred)
ext_rmse = np.sqrt(mean_squared_error(y_ext, y_ext_pred))
ext_r2 = r2_score(y_ext, y_ext_pred)

print("\nExternal Validation Results")
print(f"MAE  = {ext_mae:.3f}")
print(f"RMSE = {ext_rmse:.3f}")
print(f"R²   = {ext_r2:.3f}")
