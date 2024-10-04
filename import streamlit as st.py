import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# loading dataset
cal = fetch_california_housing()
df = pd.DataFrame(cal.data, columns=cal.feature_names)
df['Price'] = cal.target

# Title of the app
st.write('### Data Overview')
st.write(df.head(10))

# split the dataset
X=df.drop('Price', axis=1)
y=df['Price']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)
sc=StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Model selection



#Initialize models
models = {
    'Lin_Reg':LinearRegression(),
    'Rd': Ridge(),
    'Ls': Lasso()
}

