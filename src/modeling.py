import pandas as pd 
import numpy as np
import os
import pickle

df = pd.read_csv("C:\\Users\\Aman\\Desktop\\kifyaw4\\data\\processed\\train_processed.csv")
x = df.drop('Sales', axis=1)
y = df['Sales']

from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(x,y, test_size=0.2)

from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(random_state=42)

reg.fit(x_train,y_train)
pickle.dump(reg,open("random_model_1.pkl", "wb"))
