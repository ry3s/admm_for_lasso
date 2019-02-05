import admm

import numpy as np
import pandas as pd

# Loads data.
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()

# Standardizes data.
mean = np.mean(diabetes.data, axis=0)
std = np.std(diabetes.data, axis=0)
X_train = (diabetes.data - mean)/std
y_train = diabetes.target # 一年後の疾患の進行状況
diabetes_labels = ("age", "sex", "bmi", "map", "tc", "ldl", "hdl", "tch", "ltg", "glu")


print("Data Sets")
print(pd.DataFrame(diabetes.data, columns=diabetes_labels))
print()

# ADMM for Lasso
model = admm.Admm(lambda_=1.0, rho=1.0, max_iter=1000)
model.fit(X_train, y_train)
print("ADMM for Lasso")
print(pd.DataFrame({"Name":diabetes_labels
                    , "Coefficients":model.coef_}))

print()

# by sci-kit learn
from sklearn import linear_model

model = linear_model.Lasso(alpha=1.0, max_iter=1000)
model.fit(X_train, y_train)
print("By sickit-learn")
print(pd.DataFrame({"Name":diabetes_labels
                    , "Coefficients":model.coef_}))
