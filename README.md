# PTSC Surrogate Modelling (Neural Network: Eff / EffEX)

A supervised machine learning project for fast surrogate models of Parabolic Trough Solar Collector (PTSC) performance using a fowardfeed neural network, trained on validated simulation-generated data from Engineering Equation Solver (EES).

The objective is to replace slow physics-based simulations with near-instant predictions and enable rapid operating-point exploration.

---

## Project Overview

The neural network model learns the relationship between operating conditions and PTSC performance.

### Input Features

- **Mhtf** — Mass flow rate  
- **Pressurehtf** — Heat transfer fluid pressure  
- **Tin** — Inlet temperature  
- **DNI** — Direct normal irradiance  
- **Tamb** — Ambient temperature  
- **K** — Incident angle factor  

### Target Outputs

- **Eff** — Thermal efficiency  
- **EffEX** — Exergetic efficiency  

After training, the model predicts efficiency directly from operating conditions without requiring repeated thermodynamic simulations.

---

## Why Neural Network Surrogate?

Detailed PTSC simulations based on thermodynamic and heat-transfer models can become computationally expensive during:

- Large parametric sweeps  
- Sensitivity analysis  
- Optimisation loops  

The neural network replaces repeated physics evaluations with a learned mapping:

```
(Mhtf, Pressurehtf, Tin, DNI, Tamb, K)
                ↓
     Predicted Eff / EffEX
```
Once trained, predictions are extremely fast and suitable for:

- Dense grid searches  
- Sensitivity studies  
- Rapid operating-point exploration  
- Decision-support workflows  

---

## Methodology

### 1. Data Generation

Training and validation datasets are generated using a validated EES simulation pipeline.

Expected CSV columns:

#### Features

- Mhtf  
- Pressurehtf  
- Tin  
- DNI  
- Tamb  
- K  

#### Targets

- Eff  
- EffEX  

---

### 2. Data Preprocessing

- Replace `inf` and `-inf` with `NaN`  
- Remove invalid rows  
- Standardise input features using `StandardScaler`  

---

### 3. Neural Network Architecture

Fully connected feedforward network:

Input → 128 → 64 → 32 → 16 → Output

- Activation: ReLU  
- Loss function: Mean Squared Error  
- Optimiser: Adam  
- Epochs: 1000  

---

### 3.1 Neural Network Methodological Design Justification

The neural network architecture and training settings were selected to achieve a balance between predictive accuracy and computational efficiency. A multi-layer feedforward structure (128–64–32–16) was adopted as it is sufficiently expressive to capture the nonlinear relationships in PTSC performance while remaining lightweight and stable to train. Increasing depth or width beyond this configuration did not yield meaningful improvements in accuracy but led to longer training times and a higher risk of overfitting.

The model was trained for 1000 epochs using the Adam optimiser and mean squared error loss. This choice ensures convergence to a stable solution with consistently low error, while maintaining reasonable computational cost. Empirical testing showed that fewer epochs resulted in underfitting, whereas significantly more epochs provided negligible gains.

Overall, the selected architecture and training parameters provide a practical trade-off, delivering high predictive accuracy (R² ≈ 0.99+) with efficient training and inference, making the model suitable for rapid optimisation and large-scale parametric studies.

### 4. Train/Test Evaluation

Dataset split: **80% train / 20% test**

Metrics used:

- MAE  
- RMSE  
- R²  

---

### 5. External Validation

An independent dataset (1000 random combinations from EES) is used to evaluate generalisation performance.

---

## Results

### Thermal Efficiency (Eff)

| Dataset     | MAE       | RMSE      | R²        |
|------------|-----------|-----------|-----------|
| Train      | 1.819238675 | 4.053853916 | 0.990248851 |
| Test       | 1.847099385 | 4.127711542 | 0.989870657 |
| Validation | 1.274407278 | 2.230647980 | 0.994942532 |

---

### Exergetic Efficiency (EffEX)

| Dataset     | MAE        | RMSE       | R²        |
|------------|------------|------------|------------|
| Train      | 0.438264238 | 0.708081485 | 0.999096189 |
| Test       | 0.442825377 | 0.763602328 | 0.998949436 |
| Validation | 0.432968556 | 0.623658851 | 0.998417546 |

---

## Results and Discussion

The neural network demonstrates very strong predictive performance across all datasets.

For thermal efficiency, both training and test results show high accuracy (R² ≈ 0.99), indicating that the model successfully captures the nonlinear relationship between operating parameters and system performance.

External validation shows even stronger performance (R² ≈ 0.995), suggesting that the model generalises well to unseen operating conditions rather than overfitting the training data.

The relatively small difference between training and test errors indicates stable learning behaviour. In contrast to tree-based models, the neural network provides a smooth and continuous mapping of the input space, making it well-suited for optimisation and interpolation tasks.

Overall, the neural network serves as a reliable surrogate for PTSC performance prediction and is suitable for rapid evaluation and optimisation workflows.

---

## Interactive Operating-Point Search

The script includes an interactive calculator.

User inputs:

- DNI (100–1000)  
- Tamb (10–50 °C)  
- K (0–1)  

The model:

1. Performs coarse grid search over Tin and Mhtf  
2. Refines locally around the optimum  
3. Outputs:
   - Global maximum predicted efficiency  
   - Best Mhtf for each Tin (350–850 K)  

This transforms the model into a practical optimisation tool.

---

## Repository Structure

```
├── .gitignore
├── LICENSE
├── README.md
├── eff_train.csv
├── eff_validation.csv
├── nn_Eff.py
└── nn_EffEx.py
```
---

## Description

### nn_eff.py

- Neural network training for **thermal efficiency**
- Train/test evaluation
- External validation
- Summary metrics export
- Interactive calculator

---

### nn_effex.py

- Neural network training for **exergetic efficiency**
- Same workflow as `nn_eff.py`

---


## Intended Use

This project demonstrates how neural networks can accelerate thermodynamic modelling workflows.

Applications include:

- Engineering optimisation  
- Surrogate modelling research  
- Parametric studies  
- Energy system modelling  

---

## License

This project is released under the MIT License.
