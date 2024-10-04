import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

cal = fetch_california_housing()

cal

df = pd.DataFrame(cal.data, columns=cal.feature_names)

df.head(5)

df['Price'] = cal.target

print(df.info())
print(df.describe())
print(df.isna().sum)
print(df.head())

sns.pairplot(df)
plt.show

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

x = df.drop('Price', axis = 1)
y=df['Price']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)

sc=StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# build the model
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#Initialize models
models = {
    'Lin_Reg':LinearRegression(),
    'Rd': Ridge(),
    'Ls': Lasso()
}

# train and evaluate the models
results={}
for name, model in models.items():
    model.fit(x_train, y_train)
    y_train_pred=model.predict(x_train)
    y_test_pred = model.predict(x_test)

    results[name]={'Train_mse':mean_squared_error(y_train, y_train_pred),
                   'Test_mse': mean_squared_error(y_test, y_test_pred),
                   'Train_mae': mean_absolute_error(y_train, y_train_pred),
                   'Test_mse': mean_absolute_error(y_test, y_test_pred),
                   'Train_r2':r2_score(y_train, y_train_pred),
                   'Test_r2': r2_score(y_test, y_test_pred),}

for i,j in results.items():
    print(f"{i}")

