import joblib
import numpy as np
import pandas as pd
import pickle

dataset = pd.read_excel("raw_data\\Lithology Bihar.xlsx")

print(dataset)

y = dataset['Lithology']
x = dataset.drop(['Lithology'], axis=1)

dataset.to_csv("Excel.csv")

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

with open("Lithology_Model.pkl", 'wb') as file:
    pickle.dump(rf_model, file)

model = joblib.load("Lithology_Model.pkl")

new_data = pd.DataFrame(data=[[245.0,252.0,7.0]],
                        columns=['From', 'To', 'Thickness'])

print(model.predict(new_data))