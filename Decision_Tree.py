# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(X_train, y_train)
prediction = model.predict(X_test)

#print("predictions: ", prediction)
print("accuracy: ", accuracy_score(y_test, prediction))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, prediction))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, prediction))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))

i = 100
AV = y[i] == 0 and "Terroists" or "Counter terrorists"
PV =  model.predict(X)[i] < 1 and "Terroists" or "Counter terrorists"


print("According to actual value, {} wins.".format(AV))
print("According to prediction, {} wins.".format(PV))