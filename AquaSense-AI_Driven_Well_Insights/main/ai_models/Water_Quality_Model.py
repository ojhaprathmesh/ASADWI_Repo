import joblib
import numpy as np
import pandas as pd
import pickle

dataset = pd.read_csv("raw_data\\water_quality_data_processed.csv")

y = dataset['TDS']
x = dataset.drop(['TDS', 'X'], axis=1)

x.to_excel("Excel.xlsx")

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=1000)

from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(max_depth=100, random_state=1000)
rf_model.fit(x_train, y_train)

y_rf_train_pred = rf_model.predict(x_train)
y_rf_test_pred = rf_model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score

rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)

rf_model_results = pd.DataFrame(['Random Forest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_model_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']

# dataset_models = pd.concat([lr_model_results, rf_model_results], axis=0)
# dataset_models.reset_index(drop=True)

print("Random Forest Training MSE:", rf_train_mse)
print("Random Forest Training R2 Score:", rf_train_r2)
print("Random Forest Test MSE:", rf_test_mse)
print("Random Forest Test R2 Score:", rf_test_r2)

import matplotlib.pyplot as plt
plt.figure(figsize=(8,8))
plt.scatter(x=y_train, y=y_rf_train_pred, alpha=0.2)

z = np.polyfit(y_train,y_rf_train_pred, 1)
p = np.poly1d(z)

plt.plot(y_train, p(y_train), '#FFBB99')
plt.ylabel('Predict LogS')
plt.xlabel('Experimental LogS')
plt.show()

with open("Water_Quality_Model.pkl", 'wb') as file:
    pickle.dump(rf_model, file)

model = joblib.load("Water_Quality_Model.pkl")

new_data = pd.DataFrame(data=[[8.45, 1110.0, 364.0, 60.0, 165.0, 43.0, 14.03, 209.0, 12.8]],
                        columns=['pH', 'EC in μS/cm', 'HCO3', 'Cl', 'TH', 'Ca', 'Mg', 'Na', 'K'])

print(model.predict(new_data))