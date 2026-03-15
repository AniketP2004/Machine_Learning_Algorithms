import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split

df = pd.read_csv("home_data.csv")
# df = pd.read_csv("dummy_data.csv")

# full_data = pd.concat([df[['bedrooms','bathrooms','sqft_living','floors','condition','sqft_above','sqft_basement','yr_built','yr_renovated','sqft_living15','sqft_lot15']]])

single_data = pd.concat([df[['bedrooms']]])
price = pd.concat([df[['price']]])

# print(single_data)
# print(price)

X, y = single_data, price
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# plt.scatter(single_data, price)
# plt.show()


# df = pd.read_csv("dummy_data.csv")

# Visualization
# plt.xlabel('area(sqr ft)')
# plt.ylabel('price(US$)')
# plt.scatter(df.area, df.price)
# plt.show()

reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
predictions = reg.predict([[3300]])

print(predictions)