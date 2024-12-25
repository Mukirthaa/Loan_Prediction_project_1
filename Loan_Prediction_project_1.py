# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import pandas as pd

df=pd.read_csv('/kaggle/input/loan-approval-prediction-dataset/loan_approval_dataset.csv')

df

df.head()

df.info()

df.isnull().sum()

df.duplicated()

df.columns = df.columns.str.strip()

df.describe()

X = df.drop([ 'loan_status'], axis=1)
y = df['loan_status']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2)

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

num_cols = ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score',
                    'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']
cat_cols = ['education', 'self_employed']
num_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])
cat_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

model = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', RandomForestClassifier(random_state=42))])
model.fit(X_train, y_train)

y_predict = model.predict(X_test)
from sklearn.metrics import  accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
c=confusion_matrix(y_test, y_predict)
print('Accuracy:', accuracy_score(y_test, y_predict))
print('Confusion Matrix:\n', c)
print('Classification Report:\n', classification_report(y_test, y_predict))


import seaborn as sns
cf=sns.heatmap(c,annot=True,linecolor='White')

model = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', DecisionTreeClassifier(random_state=42))])
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
from sklearn.metrics import  accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
c=confusion_matrix(y_test, y_pred)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:\n', c)
print('Classification Report:\n', classification_report(y_test, y_pred))

model = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', LogisticRegression(random_state=42))])
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
from sklearn.metrics import  accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
c=confusion_matrix(y_test, y_pred)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:\n', c)
print('Classification Report:\n', classification_report(y_test, y_pred))

from sklearn.ensemble import GradientBoostingClassifier
model = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', GradientBoostingClassifier(random_state=42))])
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
from sklearn.metrics import  accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
c=confusion_matrix(y_test, y_pred)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:\n', c)
print('Classification Report:\n', classification_report(y_test, y_pred))
