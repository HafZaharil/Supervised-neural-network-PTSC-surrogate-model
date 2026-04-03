import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# 1. Load Data
# -----------------------------
df = pd.read_csv(TRAIN_PATH)

# 1.1 python display settings
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
# 4. Scale inputs
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 5. Convert to tensors
# -----------------------------
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# -----------------------------
# 6. Build neural network
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
# 7. Loss function and optimiser
# -----------------------------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# 8. Train the model
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
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# -----------------------------
# 9. Evaluate on train set
# -----------------------------
model.eval()
with torch.no_grad():
    y_pred_train = model(X_train_tensor).numpy()

train_mae = mean_absolute_error(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
train_r2 = r2_score(y_train, y_pred_train)

print("\nTrain Results")
print(f"MAE  = {train_mae:.4f}")
print(f"RMSE = {train_rmse:.4f}")
print(f"R²   = {train_r2:.4f}")

# -----------------------------
# 10. Evaluate on test set
# -----------------------------
with torch.no_grad():
    y_pred_test = model(X_test_tensor).numpy()

test_mae = mean_absolute_error(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_r2 = r2_score(y_test, y_pred_test)

print("\nTest Results")
print(f"MAE  = {test_mae:.4f}")
print(f"RMSE = {test_rmse:.4f}")
print(f"R²   = {test_r2:.4f}")

# -------------------------------------
# 11. Save and export test prediction values
# -------------------------------------
results_df = pd.DataFrame({
    "Actual": y_test.flatten(),
    "Predicted": y_pred_test.flatten()
})
results_df.to_csv(TEST_PREDICTIONS_PATH, index=False)

# -----------------------------
# 12. External validation
# -----------------------------
ext_df = pd.read_csv(VALIDATION_PATH)
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
print(f"MAE  = {ext_mae:.4f}")
print(f"RMSE = {ext_rmse:.4f}")
print(f"R²   = {ext_r2:.4f}")

results_validation_df = pd.DataFrame({
    "Actual": y_ext.flatten(),
    "Predicted": y_ext_pred.flatten()
})
results_validation_df.to_csv(VALIDATION_PREDICTIONS_PATH, index=False)

# -----------------------------
# 13. Export summary statistical data to one CSV file
# -----------------------------
summary_metrics = pd.DataFrame([
    {
        "Dataset": "Train",
        "MAE": train_mae,
        "RMSE": train_rmse,
        "R2": train_r2
    },
    {
        "Dataset": "Test",
        "MAE": test_mae,
        "RMSE": test_rmse,
        "R2": test_r2
    },
    {
        "Dataset": "Validation",
        "MAE": ext_mae,
        "RMSE": ext_rmse,
        "R2": ext_r2
    }
])

summary_metrics.to_csv(SUMMARY_METRICS_PATH, index=False)

print("\nSummary metrics file exported:")
print(summary_metrics)

# -----------------------------
# 14. Interactive calculator
# -----------------------------
FIXED_PRESSUREHTF = 20000.0

TIN_MIN, TIN_MAX = 350.0, 850.0
MHTF_MIN, MHTF_MAX = 0.50, 5.00

TIN_STEP_COARSE = 5.0
MHTF_STEP_COARSE = 0.10

TIN_WINDOW_FINE = 20.0
MHTF_WINDOW_FINE = 0.50
TIN_STEP_FINE = 0.2
MHTF_STEP_FINE = 0.02

TIN_REPORT_MIN = 350.0
TIN_REPORT_MAX = 850.0
TIN_REPORT_STEP = 50.0
MHTF_REPORT_STEP = 0.02

DNI_MIN, DNI_MAX = 100.0, 1000.0
TAMB_C_MIN, TAMB_C_MAX = 10.0, 50.0
K_MIN, K_MAX = 0.0, 1.0

def c_to_k(temp_c: float) -> float:
    return float(temp_c) + 273.15

def ask_float_in_range(name: str, min_value: float, max_value: float) -> float:
    while True:
        try:
            value = float(input(f"Enter {name} [{min_value} to {max_value}]: ").strip())
            if min_value <= value <= max_value:
                return value
            print(f"{name} must be between {min_value} and {max_value}. Please input again within the limit.")
        except ValueError:
            print(f"Invalid input for {name}. Please enter a numeric value within the limit.")

def make_grid_predictions(model, scaler, dni, tamb_k, k_factor, tin_vals, mhtf_vals):
    Tin_grid, Mhtf_grid = np.meshgrid(tin_vals, mhtf_vals, indexing="xy")
    n = Tin_grid.size

    X_grid = pd.DataFrame(
        {
            "Tin": Tin_grid.ravel(),
            "DNI": np.full(n, float(dni)),
            "Mhtf": Mhtf_grid.ravel(),
            "Tamb": np.full(n, float(tamb_k)),
            "Pressurehtf": np.full(n, float(FIXED_PRESSUREHTF)),
            "K": np.full(n, float(k_factor)),
        }
    )[feature_cols]

    X_grid_scaled = scaler.transform(X_grid.values)
    X_grid_tensor = torch.tensor(X_grid_scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        y_pred = model(X_grid_tensor).numpy().flatten()

    out = X_grid.copy()
    out["Predicted_Eff"] = y_pred
    return out

def find_best_for_ambient(model, scaler, dni, tamb_c, k_factor):
    tamb_k = c_to_k(tamb_c)

    tin_coarse = np.arange(TIN_MIN, TIN_MAX + 1e-9, TIN_STEP_COARSE)
    mhtf_coarse = np.arange(MHTF_MIN, MHTF_MAX + 1e-9, MHTF_STEP_COARSE)

    df_coarse = make_grid_predictions(model, scaler, dni, tamb_k, k_factor, tin_coarse, mhtf_coarse)
    best_coarse = df_coarse.loc[df_coarse["Predicted_Eff"].idxmax()]

    best_tin = float(best_coarse["Tin"])
    best_mhtf = float(best_coarse["Mhtf"])

    tin_fine_min = max(TIN_MIN, best_tin - TIN_WINDOW_FINE)
    tin_fine_max = min(TIN_MAX, best_tin + TIN_WINDOW_FINE)
    mhtf_fine_min = max(MHTF_MIN, best_mhtf - MHTF_WINDOW_FINE)
    mhtf_fine_max = min(MHTF_MAX, best_mhtf + MHTF_WINDOW_FINE)

    tin_fine = np.arange(tin_fine_min, tin_fine_max + 1e-9, TIN_STEP_FINE)
    mhtf_fine = np.arange(mhtf_fine_min, mhtf_fine_max + 1e-9, MHTF_STEP_FINE)

    df_fine = make_grid_predictions(model, scaler, dni, tamb_k, k_factor, tin_fine, mhtf_fine)
    best = df_fine.loc[df_fine["Predicted_Eff"].idxmax()].copy()

    return best, tamb_k

def best_for_each_tin(model, scaler, dni, tamb_k, k_factor):
    tin_targets = np.arange(TIN_REPORT_MIN, TIN_REPORT_MAX + 1e-9, TIN_REPORT_STEP)
    mhtf_vals = np.arange(MHTF_MIN, MHTF_MAX + 1e-9, MHTF_REPORT_STEP)

    rows = []
    for tin in tin_targets:
        X_grid = pd.DataFrame(
            {
                "Tin": np.full_like(mhtf_vals, float(tin), dtype=float),
                "DNI": np.full_like(mhtf_vals, float(dni), dtype=float),
                "Mhtf": mhtf_vals,
                "Tamb": np.full_like(mhtf_vals, float(tamb_k), dtype=float),
                "Pressurehtf": np.full_like(mhtf_vals, float(FIXED_PRESSUREHTF), dtype=float),
                "K": np.full_like(mhtf_vals, float(k_factor), dtype=float),
            }
        )[feature_cols]

        X_grid_scaled = scaler.transform(X_grid.values)
        X_grid_tensor = torch.tensor(X_grid_scaled, dtype=torch.float32)

        model.eval()
        with torch.no_grad():
            y_pred = model(X_grid_tensor).numpy().flatten()

        best_idx = int(np.argmax(y_pred))

        rows.append(
            {
                "Tin": float(tin),
                "Mhtf": float(mhtf_vals[best_idx]),
                "Pressurehtf": float(FIXED_PRESSUREHTF),
                "DNI": float(dni),
                "Tamb": float(tamb_k),
                "K": float(k_factor),
                "Predicted_Eff": float(y_pred[best_idx]),
            }
        )

    return pd.DataFrame(rows)

def run_surrogate_max_finder(model, scaler):
    print("\n============================================================")
    print("SURROGATE MAX FINDER — Neural Network (Eff)")
    print("Inputs: DNI, Tamb (°C), K")
    print(f"Allowed ranges: DNI = {DNI_MIN} to {DNI_MAX}, Tamb = {TAMB_C_MIN} to {TAMB_C_MAX} °C, K = {K_MIN} to {K_MAX}")
    print(f"Pressurehtf is fixed to {FIXED_PRESSUREHTF} in the search.")
    print("============================================================")

    while True:
        dni = ask_float_in_range("DNI", DNI_MIN, DNI_MAX)
        tamb_c = ask_float_in_range("Tamb (°C)", TAMB_C_MIN, TAMB_C_MAX)
        k_factor = ask_float_in_range("K", K_MIN, K_MAX)

        best, tamb_k = find_best_for_ambient(model, scaler, dni, tamb_c, k_factor)

        print("\n===== BEST (MAX PREDICTED EFF) =====")
        print(f"Predicted_Eff: {best['Predicted_Eff']:.6f}")
        print(f"Tin (K)      : {best['Tin']:.3f}")
        print(f"Mhtf (kg/s)  : {best['Mhtf']:.3f}")
        print(f"Pressurehtf  : {best['Pressurehtf']:.3f}")
        print(f"DNI          : {best['DNI']:.3f}")
        print(f"Tamb (°C)    : {tamb_c:.3f}")
        print(f"Tamb (K)     : {tamb_k:.3f}")
        print(f"K            : {best['K']:.3f}")

        print(f"\n===== BEST FOR EACH Tin ({int(TIN_REPORT_MIN)}–{int(TIN_REPORT_MAX)}, step {int(TIN_REPORT_STEP)}) =====")
        per_tin = best_for_each_tin(model, scaler, dni, tamb_k, k_factor)
        print(per_tin.to_string(index=False))

        ans = input("\nTry another set of ambient conditions? (Y/N): ").strip().upper()
        if ans != "Y":
            break

run_surrogate_max_finder(model, scaler)
