# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 13:45:58 2023

@author: Ahmet
"""

#breast cancer prediction knn

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("Breast_cancer_data.csv")
y = df.diagnosis.values
x_raw = df.drop(["diagnosis"], axis=1)
x = ((x_raw)-np.min(x_raw))/(np.max(x_raw)-(np.min(x_raw)))
x_train, x_test, y_train, y_test =train_test_split(x,y, test_size=0.2, random_state=1)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print("accurity: ", knn.score(x_test, y_test))
