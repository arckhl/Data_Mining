# Muhammad Archibald W.A.B
# A11.2021.13578
# A11.4606


import numpy as ny
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=ny.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(), [0])], remainder='passthrough')
X = ny.array(ct.fit_transform(X))

print(X)
  # OUTPUT
    [[1.0 0.0 0.0 44.0 72000.0]
    [0.0 0.0 1.0 27.0 48000.0]
    [0.0 1.0 0.0 30.0 54000.0]
    [0.0 0.0 1.0 38.0 61000.0]
    [0.0 1.0 0.0 40.0 63777.77777777778]
    [1.0 0.0 0.0 35.0 58000.0]
    [0.0 0.0 1.0 38.77777777777778 52000.0]
    [1.0 0.0 0.0 48.0 79000.0]
    [0.0 1.0 0.0 50.0 83000.0]
    [1.0 0.0 0.0 37.0 67000.0]]
    
print(Y)
  # OUTPUT
    ['No' 'Yes' 'No' 'No' 'Yes' 'Yes' 'No' 'Yes' 'No' 'Yes']
