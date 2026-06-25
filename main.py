import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

#phase 1
df_raw = pd.read_csv('Master_Markbook.csv', encoding="latin-1")
alex_row = df_raw[df_raw['Student_Name'] == 'Alex Anderson'].copy()

df_clean = df_raw.dropna()
df_clean = df_clean[
    df_clean['Maths_Advanced'].between(0, 100) &
    df_clean['Physics'].between(0, 100) &
    df_clean['Software_Engineering_Final'].between(0, 100)
]
print(f"Phase 1: cleaned dataset has {len(df_clean)} valid rows (started with {len(df_raw)})")

#phase 2
X1, y1 = df_clean[['Maths_Advanced']], df_clean['Software_Engineering_Final']
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2)

model = LinearRegression()
model.fit(X1_train, y1_train)
prediction = model.predict(X1_test)
rmse_l1 = np.sqrt(mean_squared_error(y1_test, prediction))
print(f"\nPhase 2: Level 1 RMSE = {rmse_l1:.2f}")

#phase 3
weight = model.coef_[0]
bias = model.intercept_
print(f"\nPhase 3: y = {weight:.4f}x + {bias:.4f}")

x_line = np.linspace(X1['Maths_Advanced'].min(), X1['Maths_Advanced'].max(), 100)
y_line = weight * x_line + bias

plt.figure(figsize=(8, 5))
plt.scatter(X1_test, y1_test, color='blue', label='Actual')
plt.plot(x_line, y_line, color='red', label=f"y = {weight:.2f}x + {bias:.2f}")
plt.xlabel('Maths Advanced')
plt.ylabel('Software Engineering Final')
plt.title('Level 1 AI: Maths Advanced vs Software Engineering Final')
plt.legend()
plt.tight_layout()
plt.show()

#phase 4
class MarkPredictor:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


my_ai = MarkPredictor()
my_ai.fit(X1_train, y1_train)
predictions = my_ai.predict(X1_test)
rmse_oop = np.sqrt(mean_squared_error(y1_test, predictions))
print(f"\nPhase 4: OOP-wrapped RMSE = {rmse_oop:.2f} (should match Phase 2)")

#phase 5
X2, y2 = df_clean[["Maths_Advanced", "Physics"]], df_clean["Software_Engineering_Final"]
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2)

scaler_l2 = StandardScaler()
X2_train_scaled = scaler_l2.fit_transform(X2_train)
X2_test_scaled = scaler_l2.transform(X2_test)

my_ai2 = MarkPredictor()
my_ai2.fit(X2_train_scaled, y2_train)
y_pred = my_ai2.predict(X2_test_scaled)
rmse_l2 = np.sqrt(mean_squared_error(y2_test, y_pred))
print(f"\nPhase 5: Level 2 RMSE (single split) = {rmse_l2:.2f}")

#phase 6
group_a = df_clean[df_clean['Physics'] > 70]
group_b = df_clean[df_clean['Modern_History'] > 70]
group_a_pass_rate = (group_a['Software_Engineering_Final'] >= 50).mean()
group_b_pass_rate = (group_b['Software_Engineering_Final'] >= 50).mean()
print(f"\nPhase 6: Group A (STEM) pass rate = {group_a_pass_rate:.2%}")
print(f"Phase 6: Group B (Humanities) pass rate = {group_b_pass_rate:.2%}")

ratio = group_b_pass_rate / group_a_pass_rate
print(f"Disparate Impact Ratio: {ratio:.4f}")
if ratio < 0.8:
    print("WARNING: bias detected (ratio below the 80% rule threshold)")
else:
    print("Audit passed.")

#phase 7
scaler_cv = StandardScaler()
X2_scaled = scaler_cv.fit_transform(X2)
cv_scores = cross_val_score(
    LinearRegression(), X2_scaled, y=y2, cv=5, scoring='neg_root_mean_squared_error'
)
cv_rmse = -cv_scores.mean()
print(f"\nPhase 7: Cross-validation mean RMSE: {cv_rmse:.2f}")
print(f"Fold RMSEs: {[round(-s, 2) for s in cv_scores]}")

#phase 8
def check_data_reliability(attendance_percentage):
    if attendance_percentage < 50.0:
        print("Access denied")
        return False
    else:
        print("Access granted")
        return True


print("\nTesting 45% attendance:")
check_data_reliability(45)
print("\nTesting 92% attendance:")
check_data_reliability(92)

alex_maths = alex_row['Maths_Advanced'].iloc[0]
alex_physics = alex_row['Physics'].iloc[0]
print(f"\nAlex's recorded marks -> Maths_Advanced: {alex_maths}, Physics: {alex_physics}")

ALEX_ATTENDANCE_PLACEHOLDER = 92
print("\nGatekeeper check before releasing Alex's predicted grade:")
cleared = check_data_reliability(ALEX_ATTENDANCE_PLACEHOLDER)

if cleared:
    alex_features = pd.DataFrame([[alex_maths, alex_physics]], columns=['Maths_Advanced', 'Physics'])
    alex_scaled = scaler_l2.transform(alex_features)
    alex_predicted = my_ai2.predict(alex_scaled)[0]
    print(f"Alex's predicted Software Engineering Final score: {alex_predicted:.1f}/100")
else:
    print("Prediction withheld pending an attendance review.")