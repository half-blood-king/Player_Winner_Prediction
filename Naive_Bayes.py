

from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
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

gnb = GaussianNB()


prediction = gnb.fit(X_train, y_train).predict(X_test)
print("accuracy: ", accuracy_score(y_test, prediction))

