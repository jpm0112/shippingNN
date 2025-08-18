import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load and prepare data
df = pd.read_csv("test_daily.csv")
df['FECHA'] = pd.to_datetime(df['FECHA'])
df = df.sort_values('FECHA')

# Parameters
target_col = 'CONTENEDOR 40'
test_size = 30
feature_cols = [
    col for col in df.columns
    if col not in ['FECHA', target_col] and target_col not in col
]


# Split into train and test
train_df = df[:-test_size]
test_df = df[-test_size:]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
exog_train = scaler.fit_transform(train_df[feature_cols])
exog_test = scaler.transform(test_df[feature_cols])

# Target variables
y_train = train_df[target_col]
y_test = test_df[target_col]





# Fit SARIMAX model
model = SARIMAX(
    endog=y_train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 0, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)
results = model.fit(disp=False)

# Forecast
forecast = results.predict(start=len(train_df), end=len(df)-1)

# Evaluation
mse = mean_squared_error(y_test, forecast)
print("MSE:", mse)

print("test values")
print(y_test.values)
print("forecast values")
print(forecast.values)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Real')
plt.plot(forecast.values, label='SARIMAX Prediction')
plt.title('SARIMAX Forecast')
plt.xlabel('Days')
plt.ylabel('Prediction target')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(ticks=range(0, len(y_test), max(1, len(y_test)//30)))
plt.show()


# import seaborn as sns
# import matplotlib.pyplot as plt

# plt.figure(figsize=(12, 10))
# sns.heatmap(df.corr(), cmap="coolwarm", square=True, annot=False)
# plt.title("Correlation Matrix")
# plt.show()


# Calculate correlation matrix
corr = df.corr()

# Filter for high correlation with target
high_corr = corr[target_col][abs(corr[target_col]) > 0.95]
high_corr = high_corr.drop(labels=[target_col])  # exclude the target itself

# Print
print("Columns with high correlation to target:")
print(high_corr)



