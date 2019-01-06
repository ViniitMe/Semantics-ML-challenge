import pandas as pd
import numpy as pn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

test=pd.read_csv("data_sematics_test.csv")
train=pd.read_csv("data_semantics_training.csv")

x=train.drop(['C7'],1)
y=train['C7']

x_train=pd.get_dummies(x)
x_test=pd.get_dummies(test)

my_imputer = SimpleImputer()
x_train = my_imputer.fit_transform(x_train)
x_test = my_imputer.transform(x_test)

sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

from xgboost.sklearn import XGBClassifier
xgb=XGBClassifier(n_estimators=200,max_depth=4,learning_rate=0.1)
xgb.fit(x_train,y_train)
y_pred=xgb.predict(y)

s=pd.DataFrame({'serial_no':test['serial_no'],'C7':y_pred})
s.to_csv("xgb.csv",index=False)
