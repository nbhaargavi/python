#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[11]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
df = pd.read_csv("C:\\Users\\Bhaargavi N\\Downloads\housing (3)\\housing.csv")
#cleaning
df = df.dropna()

# Create new features 
df['rooms_per_house'] = df['total_rooms'] / df['households']
features = ['median_income', 'rooms_per_house']

# Split the data into training set and a test set
X_train, X_test, y_train, y_test = train_test_split(df[features], df['median_house_value'], test_size=0.2)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# making predictions on test set
y_pred = model.predict(X_test)

# mean squared error calculating
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


# In[ ]:




