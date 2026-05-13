#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

#Phase 1: Data Automation
def loadandcleandata(filepath="master_markbook.csv"):
    df = pd.read_csv(filepath)
    df.dropna(inplace=True)
    score_columns = [c for c in df.columns if c not in ("Student", "Gender", "Attendance")]
    for col in score_columns:
        df = df[(df[col] >= 0) & (df[col] <= 100)]
    
    df.reset_index(drop=True, inplace=True)
    print(f"Cleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.head())
    return df

#Phase 2 + 3: Baseline AI + Mathematical Proof
def runlevel1(df):
    X = df[["Maths_ Advanced"]]
    Y = df["Software_Engineering_Final"]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = MarkPredictor()
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(Y_test, predictions)
    r2  = r2_score(Y_test, predictions)
    print(f"\n── Phase 2: Baseline AI ──")
    print(f"MSE: {mse:.2f}  |  R²: {r2:.4f}")

    w = model.coef_[0]
    b = model.intercept_
    print(f"\n── Phase 3: Mathematical Proof ──")
    print(f"Equation: y = {w:.4f}x + {b:.4f}")

    x_line = np.linspace(X["Maths_Advanced"].min(), X["Maths_Advanced"].max(), 100)
    y_line = w * x_line + b

    plt.figure(figsize=(8, 5))
    plt.scatter(X_test, Y_test, color="steelblue", label="Actual", alpha=0.7)
    plt.plot(x_line, y_line, color="tomato", linewidth=2, label=f"y = {w:.2f}x + {b:.2f}")
    plt.xlabel("Maths Advanced")
    plt.ylabel("Software Engineering Final")
    plt.title("Phase 3 – Linear Regression Fit")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return model

#Phase:4
class MarkPredictor:
    def __init__(self):
        self.model = LinearRegression()
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    def predict(self, X_input):
        return self.model.predict(X_input)
    @property
    def coef_(self):
        return self._model.coef_
    @property
    def intercept_(self):
        return self._model.intercept_

#Phase:5
def run_level_2(df):
    X = df[["Maths_Advanced", "Physics"]]
    Y = df["Software_Engineering_Final"]
    

#Phase:6
def bias_audit(df):

#Phase:7
def cross_validate(df):

#Phase:8
def predict_alex(df):

#Phase:extension(band 6)
def neural_network_comparison(df):
    
