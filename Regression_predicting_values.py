# Task 1
# Create a synthetic dataset with at least 50 records containing these features: area_sqft, num_bedrooms, age_years, and a target column price_lakhs. 
# Build a multiple linear regression model using scikit-learn, print the intercept and each feature's coefficient, and display the first five 
# actual vs. predicted values.


# Task1 Code

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

np.random.seed(42)
n = 50

area = np.random.randint(500, 3000, n)
bedrooms = np.random.randint(1, 6, n)
age = np.random.randint(1, 30, n)
price = 0.05 * area + 3 * bedrooms - 0.2 * age + np.random.normal(10, 5, n)

df = pd.DataFrame({'area_sqft': area, 'num_bedrooms': bedrooms,
                   'age_years': age, 'price_lakhs': price})

X = df[['area_sqft', 'num_bedrooms', 'age_years']]
y = df['price_lakhs']

model = LinearRegression()
model.fit(X, y)

print("Intercept:", model.intercept_)
print("Coefficients:", dict(zip(X.columns, model.coef_)))

df['predicted'] = model.predict(X)
print(df[['price_lakhs', 'predicted']].head())








# Task 2
# Evaluate your model using MAE, RMSE, and R². Print all three metrics. 
# Then write a comment in your code (2–3 lines) explaining what each metric value tells you about your model's performance.


# Task 2 Code

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = model.predict(X)

mae = mean_absolute_error(y, y_pred)
rmse = mean_squared_error(y, y_pred) ** 0.5
r2 = r2_score(y, y_pred)

print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.2f}")


# Task 2 comments

# MAE - Average error of the model
# RMSE - Average error of the model where the large errors are punished
# R² - How well the model explains the variance in the data






# Task 3
# Compute the residuals for all predictions. Plot a histogram of residuals using Matplotlib. 
# Add a title and axis labels. Below the plot, write a markdown cell (or a comment) explaining what a residual is and what the shape of your histogram 
# suggests about your model.


# Task3 Code

import matplotlib.pyplot as plt

residuals = y - y_pred

plt.hist(residuals, bins=15, color='steelblue', edgecolor='black')
plt.title('Distribution of Residuals')
plt.xlabel('Residual (Actual − Predicted)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# Task3 comments:

# residual - The difference between the actual and predicted value. Rasidual can be either positive or negative
# By plotting the residual, the shape of histogram suggests that the frequency is more near to residual value 0. Hence the model is a good model.
