
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

bank = pd.read_csv("bank-full.csv", sep = ";", na_values = "unknown")

bank.head()

bank.shape
bank.columns

bank["default"] = bank["default"].map({"no":0,"yes":1})
bank["housing"] = bank["housing"].map({"no":0,"yes":1})
bank["loan"] = bank["loan"].map({"no":0,"yes":1})
bank["y"] = bank["y"].map({"no":0,"yes":1})
bank.education = bank.education.map({"primary": 0, "secondary":1, "tertiary":2})
bank.month = pd.to_datetime(bank.month, format = "%b").dt.month

bank.isnull().sum()

bank.drop(["poutcome", "contact"], axis = 1, inplace = True)
bank.dropna(inplace = True)
bank = pd.get_dummies(bank, drop_first = True)

bank.y.value_counts()

X = bank.drop("y", axis = 1)
y = bank.y

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1, stratify=y)
y_train.value_counts()
lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

confusion_matrix(y_test, y_pred)

accuracy_score(y_test, y_pred)

recall_score(y_test, y_pred)

#SMOTE-----------------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1, stratify=y)

smt = SMOTE()
X_train, y_train = smt.fit_sample(X_train, y_train)
np.bincount(y_train)

lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

confusion_matrix(y_test, y_pred)

accuracy_score(y_test, y_pred)

recall_score(y_test, y_pred)

#NearMiss--------------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1, stratify=y)

nr = NearMiss()
X_train, y_train = nr.fit_sample(X_train, y_train)
np.bincount(y_train)

lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

confusion_matrix(y_test, y_pred)

accuracy_score(y_test, y_pred)

recall_score(y_test, y_pred)
