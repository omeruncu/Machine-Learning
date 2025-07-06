#################################
# Error Evaluation for Regression Models
#################################
import os
os.environ['MPLBACKEND'] = 'TkAgg'
import matplotlib
matplotlib.use('TkAgg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

# Dataset: Employees' years of experience and salary information

data = [
    (5, 600),
    (7, 900),
    (3, 550),
    (3, 500),
    (2, 400),
    (7, 950),
    (3, 540),
    (10, 1200),
    (6, 900),
    (4, 550),
    (8, 1100),
    (1, 460),
    (1, 400),
    (9, 1000),
    (1, 380)
]

df = pd.DataFrame(data, columns=["Experience(x)", "Salary(y)"])

################################################
# Exploratory Data Analysis
################################################
print(df.head())
print(df.shape)
print(df.describe().T)

# Determined bias and weight values
# bias
b = 275
# weight
w = 90

sample_experience = 5

salary_prediction_model = lambda x: b + w * x

print(f"Predicted salary for someone with {sample_experience} year of experience : {salary_prediction_model(sample_experience)}")

# Salary prediction for all years of experience in the dataframe based on the model equation
df["Predicted_Salary(y')"] = df["Experience(x)"].apply(salary_prediction_model)

print(df.head())

# Calculate error (y - y')
df["Error( y-y' )"] = df["Salary(y)"] - df["Predicted_Salary(y')"]

# Squared Error
df["SquaredError"] = df["Error( y-y' )"] ** 2

# Absolute Error
df["AbsoluteError( |y-y'| )"] = df["Error( y-y' )"].abs()

print(df.head())

# MSE, RMSE, MAE scores for model

mse = df["SquaredError"].mean()
rmse = np.sqrt(mse)
mae = df["AbsoluteError( |y-y'| )"].mean()

print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")


# Visualization
plt.figure(figsize=(12, 6))

# Real salaries (scatter)
sns.scatterplot(data=df, x="Experience(x)", y="Salary(y)", label="Actual Salary", s=100, color="blue")

# Predicted salaries (line)
sns.lineplot(data=df, x="Experience(x)", y="Predicted_Salary(y')", label="Predicted Salary", color="orange", linewidth=2)

plt.title("Actual vs Predicted Salary", fontsize=14)
plt.xlabel("Experience (Years)")
plt.ylabel("Salary")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()