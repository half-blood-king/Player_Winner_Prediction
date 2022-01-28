import numpy as np
import pandas as pd
from sklearn import linear_model

from sklearn.model_selection import train_test_split, cross_val_predict
import matplotlib.pyplot as plt
import os
from sklearn import metrics

dataset = pd.read_csv("csgo_round_snapshots.csv")

X = dataset[[
    'time_left', 
    'ct_score', 
    't_score', 
    'bomb_planted', 
    'ct_health',
    't_health', 
    'ct_armor', 
    't_armor', 
    'ct_money', 
    't_money', 
    'ct_helmets',
    't_helmets', 
    'ct_defuse_kits', 
    'ct_players_alive', 
    't_players_alive'
]].values


y = dataset[['round_winner']]

label_mapping = {
    "CT":1,
    "T":0
}

y["round_winner"] = y["round_winner"].map(label_mapping)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size = 0.2)

linear_regression_model = linear_model.LinearRegression()

#plt.scatter(X.T[14], y)
#plt.show()

model = linear_regression_model.fit(X_train, y_train)
prediction = model.predict(X_test)
os.system("CLS")





print("R^2 score: ", linear_regression_model.score(X,y))
print("coedd: ", linear_regression_model.coef_)
print("intercept: ", linear_regression_model.intercept_)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, prediction))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, prediction))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))

i = 100
AV = y[i] == 0 and "Terroists" or "Counter terrorists"
PV = model.predict(X)[i] < 1 and "Terroists" or "Counter terrorists"

print("According to actual value, {} wins.".format(AV))
print("According to prediction, {} wins.".format(PV))

predicted = cross_val_predict(model, X, y, cv=5)

fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=3)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

